The CSV Input File Format
=========================

The system being modeled by Prescient is read from a set of CSV files. The
CSV files and their format is based on the 
`RTS-GMLC format <https://github.com/GridMod/RTS-GMLC/blob/master/RTS_Data/SourceData/README.md>`_.
Prescient uses only a subset of the columns present in RTS-GMLC format. This
document identifies the columns read by Prescient, their meaning, and how they
are represented in the Egret model used by Prescient at runtime. Any additional
columns be present in the input are ignored.

There are six required CSV files and two optional CSV file. Timeseries
data is stored in an additional set of files you specify in
timeseries_pointers.csv. Documentation for each of the files is found
below:

Required Files
   .. toctree::
      :maxdepth: 1
   
      bus
      branch
      gen
      reserves
      simulation_objects

Optional Files
   .. toctree::
      :maxdepth: 1

      timeseries_pointers
      dc_branch
      initial_status
