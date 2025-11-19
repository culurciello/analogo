* 5 kHz RC Low-Pass Filter

Vin in 0 AC 1

R1 in out 1k
C1 out 0 33n

* Frequency sweep: 1 Hz to 20 kHz
.ac dec 50 1 20k

.control
  run
  print ac vm(out)
  write lp5k.raw
.endc

.end
