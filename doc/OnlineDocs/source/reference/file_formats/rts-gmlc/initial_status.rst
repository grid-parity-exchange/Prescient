initial_status.csv
------------------

This file holds the initial state of each generator. It is an optional
file; defaults are used if the file is not present. The file contains a
header row and 1 to 3 data rows.

The header row consists of one column per generator, with the column
name being the name of the generator, as specified in the :col:`GEN UID`
column of gen.csv.

The first data row is the status of each generator at the start of the
simulation period, where a positive number indicates how many time
periods the generator has been running, and a negative number indicates
how many time periods since the generator was shut down. The first row
must contain a value for every generator.

The second data row is the power output of each generator in the time
period just before the start of the simulation. This row can be left
blank for all generators, or should be populated for all generators.

The third data row is the reactive power of the generator in the time
period just before the start of the simulation. This row can be left
blank, or should be populated for all generators. If the second row was
left blank, then the third row must also be left blank. In other words,
the third row can hold data only if the second row also holds data.
