Input Data
==========

Standard Input
--------------

Prescient requires information about the system being studied, such as the generators,
buses, loads, and so on. This information is typically read into Prescient from a
collection of CSV files in a format similar to that used by
`RTS-GMLC <https://github.com/GridMod/RTS-GMLC/blob/master/RTS_Data/SourceData/README.md>`_.
See :doc:`../reference/file_formats/rts-gmlc/index` reference for a detailed description
of CSV input files and their contents.

Input files are placed together into a single directory. When running Prescient, the
directory containing the input files is specified using the `--data-path` configuration option.
Prescient will look in the input directory for files that follow the standard naming convention,
such as `gen.csv`, `branch.csv`, and so on.

.. _custom-data-providers:

Custom Data Providers
---------------------

As an alternative to reading data from the standard CSV input files, it is
possible to provide data from other sources using a custom data provider.

Internally, Prescient stores data in the `Egret <https://github.com/grid-parity-exchange/Egret>`_
format. A custom data provider is a python module that populates an Egret
model with initial data, and updates it with timeseries data as the simulation
progresses. For details, see :doc:`../reference/file_formats/custom-input` in
the reference section.

To use a custom data provider, set the `--data-provider` configuration option
to the name or path of the desired python module.
