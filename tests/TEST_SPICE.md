# test ngspice 


## Mac OS X 

Install with

```
https://formulae.brew.sh/formula/ngspice
```


Confirm your installation: just to be sure Homebrew installed everything sanely:

```
brew info ngspice
which ngspice
ngspice -v
```

you may get an error:

```
ERROR: (external)  no graphics interface;
 please check if X-server is running,
 or ngspice is compiled properly (see INSTALL)
```

The message is just a warning that the graphical plotting backend is not available. This happens on macOS when ngspice tries to use an X11-based GUI (historically used for waveforms) but no X-server is running.

––––––––––––––––––––––

## Running

Run simualation and get image of in, out:

```
python tests/run_test.py tests/active_opamp_lowpass.sp
```


## ngspice 


### 1. Running ngspice on macOS without GUI

To avoid the GUI/X11 requirement, run ngspice in batch mode:

```
ngspice -b mycircuit.sp
```

Or interactive without expecting GUI:

```
ngspice -i mycircuit.sp
```

If you actually want graphical plots, you’d need to install XQuartz on macOS:

```
brew install --cask xquartz
```

After installation, restart and run:

```
ngspice mycircuit.sp
```

––––––––––––––––––––––

### 2. Minimal runnable example

Create a file called `rc_lowpass.sp` on your Desktop:

```
* Simple RC low-pass filter
V1 in 0 SIN(0 1 1k)
R1 in out 1k
C1 out 0 1uF

.tran 0.1ms 5ms
.control
  run
  print v(in) v(out)
.endc
.end
```

Run it in batch mode:

```
ngspice -b rc_lowpass.sp
```

You should see output similar to numerical time-series for input and output voltages.

If you want to see a waveform plot (and have XQuartz installed), run interactively:

```
ngspice rc_lowpass.sp
plot v(in) v(out)
```

