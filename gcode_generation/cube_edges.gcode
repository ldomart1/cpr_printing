; --- Cube edge G-code (Duet/RRF) ---
; edge=30.0mm center=(15.0,15.0,-20.0)
; F_print=400.0 mm/min, F_travel=800.0 mm/min
; U_per_mm=0.00131665306122449, U_mult=50.0
; Assumes machine is already homed.

G90                         ; absolute positioning (XYZ)
M83                         ; relative extrusion (U)
G92 U0.000                 ; reset extrusion axis position
M400                        ; wait for moves to finish

; --- Bottom plane (square) ---
G1 X0.000 Y0.000 Z-35.000 F800 ; travel above start
G1 X30.000 Y0.000 Z-35.000 U1.97498 F400 ; bottom edge 1
G1 X30.000 Y30.000 Z-35.000 U1.97498 F400 ; bottom edge 2
G1 X0.000 Y30.000 Z-35.000 U1.97498 F400 ; bottom edge 3
G1 X0.000 Y0.000 Z-35.000 U1.97498 F400 ; bottom edge close

; --- Vertical edges ---
G1 X0.000 Y0.000 Z-35.000 F800 ; travel to vertical edge 1 base
G1 X0.000 Y0.000 Z-5.000 U1.97498 F400 ; vertical edge 1
G1 X30.000 Y0.000 Z-35.000 F800 ; travel to vertical edge 2 base
G1 X30.000 Y0.000 Z-5.000 U1.97498 F400 ; vertical edge 2
G1 X30.000 Y30.000 Z-35.000 F800 ; travel to vertical edge 3 base
G1 X30.000 Y30.000 Z-5.000 U1.97498 F400 ; vertical edge 3
G1 X0.000 Y30.000 Z-35.000 F800 ; travel to vertical edge 4 base
G1 X0.000 Y30.000 Z-5.000 U1.97498 F400 ; vertical edge 4

; --- Top plane (square) ---
G1 X0.000 Y0.000 Z-5.000 F800 ; travel above top start
G1 X30.000 Y0.000 Z-5.000 U1.97498 F400 ; top edge 1
G1 X30.000 Y30.000 Z-5.000 U1.97498 F400 ; top edge 2
G1 X0.000 Y30.000 Z-5.000 U1.97498 F400 ; top edge 3
G1 X0.000 Y0.000 Z-5.000 U1.97498 F400 ; top edge close

; --- Finish / return to origin ---
G1 Z-3.000 F800 ; lift before return
G1 X0.000 Y0.000 Z0.000 F800 ; return to origin
M400                        ; wait for moves to finish
; --- End ---
