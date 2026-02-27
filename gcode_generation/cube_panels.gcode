; --- Cube printing (Duet/RRF) with tube B/C compensation ---
; edge=15.0  center=(25.0,25.0,-25.0)
; F_print=300.0 mm/min  F_travel=600.0 mm/min
; U_per_mm=0.0131665306122449  retract=0.1  dwell=5.0s
; Tube: B90=-4.0  offset_xy_mm=25.0  offset_z_mm=18.0  C in turns (0.5=180deg)
; Assumes printer is already homed before running this file.

G90                         ; absolute positioning (XYZBC)
M83                         ; relative extrusion (U)
G92 U0                      ; reset U
M400                        ; wait

; --- Bottom horizontal square (tip path in XY @ Z=z0) ---
G1 B0.0000 C0.00000 F2000    ; set tube bend/rotation
G1 U-0.100 F300       ; retract U to stop flow
G4 S5                ; dwell to let material stop
G1 X17.500 Y17.500 Z-32.500 F600 ; travel above bottom start
G1 X32.500 Y17.500 Z-32.500 U0.19750 F300 ; bottom edge 1
G1 X32.500 Y32.500 Z-32.500 U0.19750 F300 ; bottom edge 2
G1 X17.500 Y32.500 Z-32.500 U0.19750 F300 ; bottom edge 3
G1 X17.500 Y17.500 Z-32.500 U0.19750 F300 ; bottom edge close

; --- Side panels (each face perimeter), printed with 90deg angle of attack ---

; Face X+  (B=-4.0, C=0.75 turns)
G1 U-0.100 F300       ; retract U to stop flow
G4 S5                ; dwell to let material stop
G1 B-4.0000 C0.75000 F2000    ; set tube bend/rotation
G1 X32.500 Y42.500 Z-50.500 F600 ; travel above X+ start
G1 X32.500 Y57.500 Z-50.500 U0.19750 F300 ; X+ edge 1
G1 X32.500 Y57.500 Z-35.500 U0.19750 F300 ; X+ edge 2
G1 X32.500 Y42.500 Z-35.500 U0.19750 F300 ; X+ edge 3
G1 X32.500 Y42.500 Z-50.500 U0.19750 F300 ; X+ edge close

; Face Y+  (B=-4.0, C=0.0 turns)
G1 U-0.100 F300       ; retract U to stop flow
G4 S5                ; dwell to let material stop
G1 B-4.0000 C0.00000 F2000    ; set tube bend/rotation
G1 X-7.500 Y32.500 Z-50.500 F600 ; travel above Y+ start
G1 X7.500 Y32.500 Z-50.500 U0.19750 F300 ; Y+ edge 1
G1 X7.500 Y32.500 Z-35.500 U0.19750 F300 ; Y+ edge 2
G1 X-7.500 Y32.500 Z-35.500 U0.19750 F300 ; Y+ edge 3
G1 X-7.500 Y32.500 Z-50.500 U0.19750 F300 ; Y+ edge close

; Face X-  (B=-4.0, C=0.25 turns)
G1 U-0.100 F300       ; retract U to stop flow
G4 S5                ; dwell to let material stop
G1 B-4.0000 C0.25000 F2000    ; set tube bend/rotation
G1 X17.500 Y-7.500 Z-50.500 F600 ; travel above X- start
G1 X17.500 Y7.500 Z-50.500 U0.19750 F300 ; X- edge 1
G1 X17.500 Y7.500 Z-35.500 U0.19750 F300 ; X- edge 2
G1 X17.500 Y-7.500 Z-35.500 U0.19750 F300 ; X- edge 3
G1 X17.500 Y-7.500 Z-50.500 U0.19750 F300 ; X- edge close

; Face Y-  (B=-4.0, C=0.5 turns)
G1 U-0.100 F300       ; retract U to stop flow
G4 S5                ; dwell to let material stop
G1 B-4.0000 C0.50000 F2000    ; set tube bend/rotation
G1 X42.500 Y17.500 Z-50.500 F600 ; travel above Y- start
G1 X57.500 Y17.500 Z-50.500 U0.19750 F300 ; Y- edge 1
G1 X57.500 Y17.500 Z-35.500 U0.19750 F300 ; Y- edge 2
G1 X42.500 Y17.500 Z-35.500 U0.19750 F300 ; Y- edge 3
G1 X42.500 Y17.500 Z-50.500 U0.19750 F300 ; Y- edge close

; --- Top horizontal square (straight tool) ---
G1 U-0.100 F300       ; retract U to stop flow
G4 S5                ; dwell to let material stop
G1 B0.0000 C0.00000 F2000    ; set tube bend/rotation
G1 X17.500 Y17.500 Z-17.500 F600 ; travel above top start
G1 X32.500 Y17.500 Z-17.500 U0.19750 F300 ; top edge 1
G1 X32.500 Y32.500 Z-17.500 U0.19750 F300 ; top edge 2
G1 X17.500 Y32.500 Z-17.500 U0.19750 F300 ; top edge 3
G1 X17.500 Y17.500 Z-17.500 U0.19750 F300 ; top edge close

; --- Finish / retract / home ---
G1 U-0.100 F300       ; retract U to stop flow
G4 S5                ; dwell to let material stop
G1 B0.0000 C0.00000 F2000    ; set tube bend/rotation
G1 Z-15.500 F600 ; lift before home
G28                         ; home all axes
M400                        ; wait
; --- End ---
