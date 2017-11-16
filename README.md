### Singular Spectrum Analisys, python3

This is a Python project which realizes Singular Spectrum Analisys algorithm.
The project provides the user interface on Python 3 and PyQt5.
It allows to analyse SSA parameters for different window L and offsets within
time series.

User should select configuration which is a file with a simple structure.
Data for time series has CSV format with fields
'date','time','open','high','low','close','vol'.
The first line of a CSV file should include the field names.

How to use:

- Run ./ssa.py
- Select a configuration file
- Click the 'Calc' button.
- Select required number of SVD eigenvectors and click 'Calc' once more.

The site with a description: [link](rtaubes.pythonanywhere.com)


