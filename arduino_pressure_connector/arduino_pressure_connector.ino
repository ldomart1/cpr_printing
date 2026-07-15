/*
  Duet 3HC -> Arduino GIGA R1 -> MAX3232 -> Nordson Ultimus V

  Duet outputs:
    1.io0.out -> GIGA D49 : dispense enable
    1.io1.out -> GIGA D51 : pressure PWM setpoint
    1.io2.out -> GIGA D53 : optional fixed vacuum pulse trigger
    1.io3.out -> GIGA D47 : variable vacuum PWM setpoint

  GIGA Serial2:
    D18/TX1 -> MAX3232 T1IN
    D19/RX1 <- MAX3232 R1OUT

  Ultimus side:
    MAX3232 RS232 TX -> Ultimus RS232 RX
    MAX3232 RS232 RX <- Ultimus RS232 TX
    GND              -> Ultimus RS232 GND

  Electrical notes:
    - Arduino GIGA R1 inputs are 3.3 V logic.
    - Do NOT feed 5 V or 24 V directly into GIGA input pins.
    - Share Duet GND and Arduino GIGA GND.
*/

#include <Arduino.h>
#include <math.h>
#include <string.h>

// ---------------- USER SETTINGS ----------------

const uint8_t PIN_DISPENSE_ENABLE = 49;
const uint8_t PIN_PRESSURE_PWM    = 51;
const uint8_t PIN_VACUUM_TRIGGER  = 53;  // optional pulse trigger
const uint8_t PIN_VACUUM_PWM      = 47;  // variable vacuum setpoint

// GIGA has hardware Serial2 on D18/TX1 and D19/RX1.
HardwareSerial &nordson = Serial2;

// Ultimus V default is usually 115200 baud.
// If you changed the Ultimus comm port setting, change this to match.
const uint32_t NORDSON_BAUD = 115200;

// If your signals are inverted by optocouplers/transistor interfaces, set this true.
const bool INVERT_DUET_INPUTS = false;

// Pressure mapping.
// Duet M42 P11 S0.35 -> 35% duty -> 35 psi if MAX_PRESSURE_PSI = 100.
const float MAX_PRESSURE_PSI = 100.0f;
const float PRESSURE_UPDATE_THRESHOLD_PSI = 0.25f;

// Variable vacuum mapping.
// Duet M42 P13 S0.167 -> 16.7% duty -> 3.0 inH2O if MAX_VACUUM_H2O = 18.
const float MAX_VACUUM_H2O = 18.0f;
const float VACUUM_UPDATE_THRESHOLD_H2O = 0.10f;

// Optional vacuum pulse at end of print / path.
// This is separate from the variable vacuum setpoint.
// Rising edge on PIN_VACUUM_TRIGGER applies this vacuum briefly,
// then restores the current PWM-commanded vacuum setpoint.
const bool ENABLE_VACUUM_PULSE = true;
const float VACUUM_PULSE_H2O = 10.0f;
const uint32_t VACUUM_PULSE_MS = 800;

// Set this true if you want the Arduino to force the Ultimus units on boot:
// pressure = psi, vacuum = inches H2O.
const bool SET_UNITS_ON_BOOT = true;

// Set this true if you want the Arduino to force the Ultimus to Steady Mode on boot.
const bool SET_STEADY_MODE_ON_BOOT = true;

// How often to sample PWM commands.
const uint32_t PRESSURE_SAMPLE_INTERVAL_MS = 200;
const uint32_t VACUUM_SAMPLE_INTERVAL_MS   = 200;

// Duet config should use 100 Hz PWM, so the PWM period is about 10 ms.
// 30 ms timeout gives margin.
const uint32_t PWM_PULSE_TIMEOUT_US = 30000;

// ---------------- ULTIMUS CONTROL BYTES ----------------

const uint8_t STX = 0x02;
const uint8_t ETX = 0x03;
const uint8_t EOT = 0x04;
const uint8_t ENQ = 0x05;
const uint8_t ACK = 0x06;

// ---------------- STATE ----------------

bool lastDispenseSignal = false;
bool assumedDispensing = false;

bool lastVacuumPulseSignal = false;

float lastPressurePsi = -999.0f;
float lastVacuumH2O   = -999.0f;

uint32_t lastPressureSampleMs = 0;
uint32_t lastVacuumSampleMs   = 0;

// ---------------- FUNCTION PROTOTYPES ----------------

bool readDuetPin(uint8_t pin);
void configureInputPin(uint8_t pin);
float readPwmDuty(uint8_t pin);

void handleDispense();
void handlePressure();
void handleVacuumSetpoint();
void handleVacuumPulse();

void setPressurePsi(float psi);
void setVacuumH2O(float h2o);
void setPressureUnitsPsi();
void setVacuumUnitsH2O();
void setSteadyMode();
bool toggleDispense();

bool sendWriteCommand(const char command4[5], const char *data);
void sendPacket(const char command4[5], const char *data);
bool waitForByte(uint8_t target, uint32_t timeoutMs);
int readPacket(char *out, int outSize, uint32_t timeoutMs);
void flushNordsonInput();

// ---------------- SETUP ----------------

void setup() {
  Serial.begin(115200);
  delay(1000);

  Serial.println();
  Serial.println("Duet -> GIGA -> Ultimus V interface starting");

  configureInputPin(PIN_DISPENSE_ENABLE);
  configureInputPin(PIN_PRESSURE_PWM);
  configureInputPin(PIN_VACUUM_TRIGGER);
  configureInputPin(PIN_VACUUM_PWM);

  nordson.begin(NORDSON_BAUD);
  delay(500);

  flushNordsonInput();

  if (SET_UNITS_ON_BOOT) {
    Serial.println("Setting Ultimus units: psi / inches H2O");
    setPressureUnitsPsi();
    delay(50);
    setVacuumUnitsH2O();
    delay(50);
  }

  if (SET_STEADY_MODE_ON_BOOT) {
    Serial.println("Setting Ultimus to Steady Mode");
    setSteadyMode();
    delay(50);
  }

  // Safe startup state.
  setPressurePsi(0.0f);
  delay(50);
  setVacuumH2O(0.0f);
  delay(50);

  lastDispenseSignal = readDuetPin(PIN_DISPENSE_ENABLE);
  lastVacuumPulseSignal = readDuetPin(PIN_VACUUM_TRIGGER);

  Serial.println("Ready");
}

// ---------------- LOOP ----------------

void loop() {
  handleDispense();
  handlePressure();
  handleVacuumSetpoint();

  if (ENABLE_VACUUM_PULSE) {
    handleVacuumPulse();
  }
}

// ---------------- DUET INPUT HANDLING ----------------

void configureInputPin(uint8_t pin) {
#if defined(INPUT_PULLDOWN)
  pinMode(pin, INPUT_PULLDOWN);
#else
  // If INPUT_PULLDOWN is not supported by your core,
  // use an external pulldown resistor, e.g. 10k to GND.
  pinMode(pin, INPUT);
#endif
}

bool readDuetPin(uint8_t pin) {
  bool v = digitalRead(pin);
  return INVERT_DUET_INPUTS ? !v : v;
}

void handleDispense() {
  bool dispenseSignal = readDuetPin(PIN_DISPENSE_ENABLE);

  if (dispenseSignal != lastDispenseSignal) {
    lastDispenseSignal = dispenseSignal;

    // Ultimus DI command toggles dispense in Steady Mode.
    // We track assumedDispensing so Duet high = ON and Duet low = OFF.
    if (dispenseSignal && !assumedDispensing) {
      Serial.println("Dispense ON request");

      if (toggleDispense()) {
        assumedDispensing = true;
      }
    } else if (!dispenseSignal && assumedDispensing) {
      Serial.println("Dispense OFF request");

      if (toggleDispense()) {
        assumedDispensing = false;
      }
    }
  }
}

void handlePressure() {
  uint32_t now = millis();

  if (now - lastPressureSampleMs < PRESSURE_SAMPLE_INTERVAL_MS) {
    return;
  }

  lastPressureSampleMs = now;

  float duty = readPwmDuty(PIN_PRESSURE_PWM);
  float psi = duty * MAX_PRESSURE_PSI;

  if (fabsf(psi - lastPressurePsi) >= PRESSURE_UPDATE_THRESHOLD_PSI) {
    lastPressurePsi = psi;

    Serial.print("Pressure command: ");
    Serial.print(psi, 1);
    Serial.println(" psi");

    setPressurePsi(psi);
  }
}

void handleVacuumSetpoint() {
  uint32_t now = millis();

  if (now - lastVacuumSampleMs < VACUUM_SAMPLE_INTERVAL_MS) {
    return;
  }

  lastVacuumSampleMs = now;

  float duty = readPwmDuty(PIN_VACUUM_PWM);
  float h2o = duty * MAX_VACUUM_H2O;

  if (fabsf(h2o - lastVacuumH2O) >= VACUUM_UPDATE_THRESHOLD_H2O) {
    lastVacuumH2O = h2o;

    Serial.print("Vacuum command: ");
    Serial.print(h2o, 1);
    Serial.println(" inH2O");

    setVacuumH2O(h2o);
  }
}

void handleVacuumPulse() {
  bool vacuumPulseSignal = readDuetPin(PIN_VACUUM_TRIGGER);

  // Rising edge trigger only.
  if (vacuumPulseSignal && !lastVacuumPulseSignal) {
    Serial.println("Vacuum pulse request");

    if (assumedDispensing) {
      Serial.println("Dispense was assumed ON, turning OFF before vacuum pulse");

      if (toggleDispense()) {
        assumedDispensing = false;
      }
    }

    // Save current variable vacuum setpoint.
    float restoreH2O = lastVacuumH2O;

    if (restoreH2O < -100.0f) {
      restoreH2O = 0.0f;
    }

    setVacuumH2O(VACUUM_PULSE_H2O);
    delay(VACUUM_PULSE_MS);

    // Restore PWM-commanded vacuum value instead of forcing 0.
    setVacuumH2O(restoreH2O);

    Serial.println("Vacuum pulse complete");
  }

  lastVacuumPulseSignal = vacuumPulseSignal;
}

// ---------------- PWM READING ----------------

float readPwmDuty(uint8_t pin) {
  uint32_t highUs = pulseIn(pin, HIGH, PWM_PULSE_TIMEOUT_US);
  uint32_t lowUs  = pulseIn(pin, LOW,  PWM_PULSE_TIMEOUT_US);

  // Constant low or disconnected with pulldown.
  if (highUs == 0 && lowUs > 0) {
    return 0.0f;
  }

  // Constant high.
  if (highUs > 0 && lowUs == 0) {
    return 1.0f;
  }

  // No useful reading.
  // Preserve previous command when possible.
  if (highUs == 0 && lowUs == 0) {
    if (pin == PIN_PRESSURE_PWM) {
      if (lastPressurePsi < -100.0f) {
        return 0.0f;
      }
      return lastPressurePsi / MAX_PRESSURE_PSI;
    }

    if (pin == PIN_VACUUM_PWM) {
      if (lastVacuumH2O < -100.0f) {
        return 0.0f;
      }
      return lastVacuumH2O / MAX_VACUUM_H2O;
    }

    return 0.0f;
  }

  float duty = (float)highUs / (float)(highUs + lowUs);

  if (duty < 0.0f) {
    duty = 0.0f;
  }

  if (duty > 1.0f) {
    duty = 1.0f;
  }

  return duty;
}

// ---------------- ULTIMUS HIGH-LEVEL COMMANDS ----------------

void setPressurePsi(float psi) {
  if (psi < 0.0f) {
    psi = 0.0f;
  }

  if (psi > 100.0f) {
    psi = 100.0f;
  }

  // Ultimus pressure in psi:
  // 0.0 to 100.0 psi -> 0000 to 1000
  int value = (int)roundf(psi * 10.0f);

  char data[5];
  snprintf(data, sizeof(data), "%04d", value);

  sendWriteCommand("PS  ", data);
}

void setVacuumH2O(float h2o) {
  if (h2o < 0.0f) {
    h2o = 0.0f;
  }

  if (h2o > 18.0f) {
    h2o = 18.0f;
  }

  // Ultimus vacuum in inches H2O:
  // 0.0 to 18.0 H2O -> 0000 to 0180
  int value = (int)roundf(h2o * 10.0f);

  char data[5];
  snprintf(data, sizeof(data), "%04d", value);

  sendWriteCommand("VS  ", data);
}

void setPressureUnitsPsi() {
  // E6--uu, where 00 = PSI
  sendWriteCommand("E6  ", "00");
}

void setVacuumUnitsH2O() {
  // E7--uu, where 01 = inches H2O
  sendWriteCommand("E7  ", "01");
}

void setSteadyMode() {
  // MT-- = Steady Mode
  sendWriteCommand("MT  ", "");
}

bool toggleDispense() {
  // DI-- toggles dispense in Steady Mode.
  return sendWriteCommand("DI  ", "");
}

// ---------------- ULTIMUS PACKET LAYER ----------------

bool sendWriteCommand(const char command4[5], const char *data) {
  flushNordsonInput();

  // Step 1: ENQ
  nordson.write(ENQ);
  nordson.flush();

  // Step 2: wait for ACK
  if (!waitForByte(ACK, 1000)) {
    Serial.print("ERROR: No ACK after ENQ for command ");
    Serial.println(command4);
    nordson.write(EOT);
    nordson.flush();
    return false;
  }

  // Step 3: send text packet
  sendPacket(command4, data);
  nordson.flush();

  // Step 4: read response packet, normally A0 on success
  char response[32];
  int n = readPacket(response, sizeof(response), 1000);

  // Step 5: end sequence
  nordson.write(EOT);
  nordson.flush();

  if (n <= 0) {
    Serial.print("ERROR: No response packet for command ");
    Serial.println(command4);
    return false;
  }

  // Response includes No. Bytes + Command + Checksum.
  // Success command should contain A0.
  if (strstr(response, "A0") != NULL) {
    return true;
  }

  Serial.print("ERROR: Ultimus response for ");
  Serial.print(command4);
  Serial.print(" = ");
  Serial.println(response);

  return false;
}

void sendPacket(const char command4[5], const char *data) {
  char body[260];

  // Command is always 4 characters.
  // In the manual, command hyphens "--" mean ASCII spaces 0x20.
  snprintf(body, sizeof(body), "%c%c%c%c%s",
           command4[0], command4[1], command4[2], command4[3], data);

  uint8_t bodyLen = strlen(body);

  char lenAscii[3];
  snprintf(lenAscii, sizeof(lenAscii), "%02X", bodyLen);

  // Checksum = 0 - sum of ASCII bytes from No. Bytes through Data.
  uint8_t sum = 0;

  sum += (uint8_t)lenAscii[0];
  sum += (uint8_t)lenAscii[1];

  for (uint8_t i = 0; i < bodyLen; i++) {
    sum += (uint8_t)body[i];
  }

  uint8_t checksum = (uint8_t)(0 - sum);

  char chkAscii[3];
  snprintf(chkAscii, sizeof(chkAscii), "%02X", checksum);

  nordson.write(STX);
  nordson.print(lenAscii);
  nordson.print(body);
  nordson.print(chkAscii);
  nordson.write(ETX);

  Serial.print("Sent packet command=");
  Serial.print(command4);
  Serial.print(" data=");
  Serial.println(data);
}

bool waitForByte(uint8_t target, uint32_t timeoutMs) {
  uint32_t start = millis();

  while (millis() - start < timeoutMs) {
    if (nordson.available()) {
      uint8_t b = nordson.read();

      if (b == target) {
        return true;
      }
    }
  }

  return false;
}

int readPacket(char *out, int outSize, uint32_t timeoutMs) {
  uint32_t start = millis();
  bool inPacket = false;
  int idx = 0;

  while (millis() - start < timeoutMs) {
    if (!nordson.available()) {
      continue;
    }

    uint8_t b = nordson.read();

    if (b == STX) {
      inPacket = true;
      idx = 0;
      continue;
    }

    if (b == ETX && inPacket) {
      out[idx] = '\0';
      return idx;
    }

    if (inPacket && idx < outSize - 1) {
      out[idx++] = (char)b;
    }
  }

  out[0] = '\0';
  return -1;
}

void flushNordsonInput() {
  while (nordson.available()) {
    nordson.read();
  }
}