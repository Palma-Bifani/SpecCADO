# SpecCADO

SpecCADO is a prototype for the spectroscopic mode in SimCADO, the instrument data simulator for MICADO@ELT.

Install the package by doing
```sh
pip install .
```
or
```sh
pip install --user .
```
within the unpacked speccado directory.

If `pip` and `python` point to a python-2.7 installation, please try
the commands `pip3` and `python3` instead.

The directory `example/` includes a script to demonstrate how to use SpecCADO. Run it by doing
```sh
python simulate_example.py
```
in the shell, or
```python
%run simulate_example.py
```
within an `ipython` session. As delivered, `simulate_example.py` simulates a single MICADO chip. In order to simulate the full focal-plane array, comment out the line starting `sc.do_all_chips`.

There is also a script that recifies the simulated spectra using the
known spectral mapping. Run it with
```sh
python rectify_example.py <filename>
```
where ``filename`` refers to a SpecCADO simulation covering the full
focal-plane array.

## Requirements

The following python packages are needed to run SpecCADO:
* SimCADO
* astropy
* numpy
* scipy

You can check which versions of these packages, if any, you have
installed by running
```
>>> speccado.bug_report()
```
Please always include the output of this command when reporting bugs.
