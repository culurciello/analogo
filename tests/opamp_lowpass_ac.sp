* Active 5 kHz low-pass filter using ideal op-amp

* Input source: small-signal AC for frequency sweep
Vin in 0 AC 1

* Op-amp power rails
Vcc vcc 0 12
Vee vee 0 -12

* Non-inverting active low-pass
* + input
Rin_src in plus 1     ; tiny series resistor to avoid node tying issues
                       ; plus node is effectively Vin

* Feedback network
Rf out neg 10k        ; feedback resistor
Rg neg 0 10k          ; to ground, sets gain
Cf out neg 3.3n       ; in parallel with Rf, creates low-pass pole

* Ideal op-amp modeled as high-gain VCVS
Eop out 0 plus neg 1e6

* AC frequency sweep: 100 Hz to 200 kHz
.ac dec 50 100 200k

.control
  run
  * Print magnitude of output vs frequency
  print ac vm(out)
  * Also write raw data for external tools/AI agent
  write active_lp5k.raw
.endc

.end
