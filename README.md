# SpecCADO

SpecCADO is a prototype for the spectroscopic mode in SimCADO, the instrument data simulator for MICADO@ELT.

Install the package by doing
```sh
python setup.py install
```

The directory `example/` includes a script to demonstrate how to use SpecCADO. Run it by doing
```sh
python example.py
```
in the shell, or
```python
%run example.py
```
within an `ipython` session. As delivered, `example.py` simulates a single MICADO chip. In order to simulate the full focal-plane array, comment out the line starting `sc.do_all_chips`. 

## Requirements

The following python packages are needed to run SpecCADO:
* SimCADO
* astropy
* numpy
* scipy
