Input Data
==========

.. toctree::
   :maxdepth: 2
   :caption: Contents:

   file_formats/rts-gmlc/index.rst

Overview of Input Data
----------------------

Data is read into Prescient from a collection of CSV files in a format similar
to that used by  
`RTS-GMLC <https://github.com/GridMod/RTS-GMLC/blob/master/RTS_Data/SourceData/README.md>`_.
See :doc:`./file_formats/rts-gmlc/index` for details. The data in CSV input files includes
the definition of the system under study including system elements such as generators and buses,
as well as timeseries data such as variable loads and renewable generator outputs.

Timeseries data in CSV files often covers a larger time period than is included in the
study. The dates provided to Prescient as configuration parameters are used to trim down
the data read from input files.

Internally, Prescient stores data in the `Egret <https://github.com/grid-parity-exchange/Egret>`_ format.
