timeseries_pointers.csv
-----------------------

This file identifies where to find timeseries values, and which model
elements they apply to. Each row in the file identifies a model element
(such as a particular generator's power output, or an area's load),
whether the values are forecast or actual values, and what file holds
the values. The CSV file has the following columns:

.. list-table:: timeseries_pointers.csv Columns
   :header-rows: 1

   * - **Column Name**
     - **Description**
   * - :col:`Simulation`
     - Either *DAY_AHEAD* or *REAL_TIME*. If *DAY_AHEAD*, the values are forecasts
       that inform RUC formulations. If *REAL_TIME*, the values are actual
       values used in SCED formulations.
   * - :col:`Category`
     - What kind of object the data is for. Supported values are:

          * *Generator*
          * *Area*
          * *Reserve*
   * - :col:`Object`
     - The name of the specific object the data is for
   * - :col:`Parameter`
     - The specific attribute of the object that the data is for
   * - :col:`Data File`
     - The path to the file holding the timeseries values.

The model element the data applies to is identified by the :col:`Category`,
:col:`Object`, and :col:`Parameter`. Which parameters are supported depend on the
:col:`Category`.

   *  If :col:`Category` is *Generator*, then :col:`Object` must be the name
      of a generator as specified in the :col:`GEN UID` column of gen.csv.
      :col:`Parameter` must be either *PMax MW* or *PMin MW*.

   *  If :col:`Category` is *Area*, then :col:`Object` must be an area name referenced in
      bus.csv, and :col:`Parameter` must be *MW Load*. The timeseries values specify
      the load imposed on the area at each timestep.

   *  If :col:`Category` is *Reserve*, then :col:`Object` is a reserve product name in
      *\<category\>_R\<area\>* format, and :col:`Parameter` must be *Requirement*. The
      timeseries values specify the magnitude of the reserve requirement
      for the reserve product.

The :col:`Data File` is the path to the CSV file holding timeseries values. The
path can be relative or absolute. If it is relative, it is relative to
the folder containing timeseries_pointers.csv.

Timeseries File Formats
~~~~~~~~~~~~~~~~~~~~~~~

There are two supported formats for timeseries files, columnar and 2D. A
columnar file has a row for each value in the timeseries, while a 2D
file has a row for each day and a column for each value within the day.
A columnar file can have multiple data columns for each row, allowing
data for multiple model elements to be stored in the same file. A 2D
file can only hold a single timeseries.

Both file formats store data at equally spaced time intervals. Each day
is split into periods, numbered 1 through N. The first period of each
day starts at midnight. The duration of each period is specified by the
Period_Resolution row in simulation_objects.csv. The number of periods
per day must add up to 24 hours per day. Note that *DAY_AHEAD* periods and
*REAL_TIME* periods often have different durations, so the appropriate the
number of periods per day may depend on whether the data are forecasts
or actuals.

Each file's data must cover the time period from *DATE_FROM* to *DATE_TO*,
as specified in simulation_objects.csv, including the extra look-ahead
periods after *DATE_TO*.

Columnar Timeseries Files
^^^^^^^^^^^^^^^^^^^^^^^^^

A columnar timeseries file has one row per period. It has 4 columns that
identify the date and period of the row's data, followed by any number
of data columns. The name of each data column must match the name of the
object the data pertains to, such as the name of the appropriate
generator. Here is an example of the first few rows of a columnar
timeseries file with data for two generators named *Hydro1* and *Hydro2*:

.. list-table:: Example Columnar Timeseries File
   :header-rows: 1

   * - **Year**
     - **Month**
     - **Day**
     - **Period**
     - **Hydro1**
     - **Hydro2**
   * - 2023
     - 4
     - 1
     - 1
     - 2.0152
     - 11.958
   * - 2023
     - 4
     - 1
     - 2
     - 2.3055
     - 12.616
   * - ...
     - ...
     - ...
     - ...
     - ...
     - ...


Note that the :col:`Year`, :col:`Month`, :col:`Day`, and :col:`Period` are
entered as integer values.

2D Timeseries Files
^^^^^^^^^^^^^^^^^^^

A 2D timeseries file holds data for a single timeseries in a 2D layout.
The file has :col:`Year`, :col:`Month`, and :col:`Day` columns, followed by
one column per period in each day. For example, a file with hourly data
will have 27 columns: the :col:`Year`, :col:`Month`, and :col:`Day` columns
followed by 24 period columns:

.. list-table:: Example 2D Timeseries Data File
   :header-rows: 1

   * - **Year**
     - **Month**
     - **Day**
     - **1**
     - **2**
     - **...**
     - **24**
   * - 2023
     - 4
     - 1
     - 1.989
     - 2.0152
     - ...
     - 1.958
   * - 2023
     - 4
     - 1
     - 2 .015
     - 2.3055
     - ...
     - 2.616
   * - ...
     - ...
     - ...
     - ...
     - ...
     - ...
     - ...

The name of each period column must be the period number, from 1 to N.
