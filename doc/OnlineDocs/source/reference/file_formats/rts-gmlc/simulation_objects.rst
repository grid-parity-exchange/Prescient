simulation_objects.csv
----------------------

This file is used to enter data about the data set as a whole. Each row
specifies a global parameter with two values, one that applies to
forecasts and another that applies to real-time data (actuals). The file
has three columns:

.. list-table:: simulation_objects.csv Columns
   :header-rows: 1

   * - **Column Name**
     - **Description**
   * - :col:`Simulation_Parameters`
     - Which global parameter is set by this row
   * - :col:`DAY_AHEAD`
     - The row's value for forecast data and/or RUC plans
   * - :col:`REAL_TIME`
     - The row's value as it applies to real-time data and/or SCED operations


The following values of :col:`Simulation_Parameter` are supported:

.. list-table:: Supported values of :col:`Simulation_Parameter` in simulation_objects.csv
   :header-rows: 1

   * - **Simulation_Parameter**
     - **Required?**
     - **Parameter Description**
     - **DAY_AHEAD**
     - **REAL_TIME**
   * - *Period_Resolution*
     - Yes
     - The number of seconds between values in timeseries data files
     - The number of seconds between values in *DAY_AHEAD* timeseries data
       files
     - The number of seconds between values in *REAL_TIME* timeseries data
       files
   * - *Date_From*
     - Yes
     - The date and time of the first value in each timeseries data file.
       Most reasonable formats are accepted.
     - The date and time of the first value in each DAY_AHEAD timeseries data
       file
     - The date and time of the first value in each REAL_TIME timeseries data
       file
   * - *Date_To*
     - Yes
     - The latest date and time for which we have enough data in timeseries
       files to formulate a RUC or SCED. Most reasonable formats are
       accepted. See `Date_To Details`_ below.
     - The latest date and time for which we have enough data in DAY_AHEAD
       timeseries data files to formulate a RUC
     - The latest date and time for which we have enough data in REAL_TIME
       timeseries data files to formulate a SCED
   * - *Look_Ahead_Periods_Per_Step*
     - Yes
     - The default number of look-ahead periods to use in RUC or SCED formulations
     - The default number of look-ahead periods to use in RUC formulations
     - The default number of look-ahead periods to use in SCED formulations
   * - *Look_Ahead_Resolution*
     - Yes
     - The default number of seconds between each look-ahead period used in
       RUC or SCED formulations. See `Look_Ahead_Resolution Details`_ below.
     - The default number of seconds between each look-ahead period in RUC
       formulations
     - The default number of seconds between each look-ahead period in SCED
       formulations
       formulations
   * - *Reserve_Products*
     - No
     - Which reserve products to enforce for RUC plans or SCED operations.
       See `Reserve_Products Details`_ below.
     - Which reserve products to enforce for RUC plans
     - Which reserve products to enforce for SCED operations


Date_To Details
~~~~~~~~~~~~~~~

The value of *Date_To* identifies the latest time for which there is
enough data to formulate or RUC (for :col:`DAY_AHEAD`) or SCED (for
:col:`REAL_TIME`), including look-ahead periods. This is not the date
and time of the final value in timeseries data files. Instead, the
*Date_To* is *Look_Ahead_Periods_Per_Step \* Look_Ahead_Resolution*
before the date and time of the final value in the timeseries data files.

For example, consider a data set with 24 look-ahead periods with a
look-ahead resolution of 1 hour. If the final value in a timeseries is
for April 10\ :sup:`th` at midnight, then *Date_To* is April 9\ :sup:`th`
at midnight, because that is the latest time for which we have enough
data to satisfy the 24 hour look-ahead requirement.

Look_Ahead_Resolution Details
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The *Look_Ahead_Resolution* parameter is used to determine the date and time of the
final value in timeseries data files, as described in `Date_To Details`_. Despite its
name, it is not used to specify the look-ahead resolution used during simulation. The
actual look-ahead resolution used during simulation is determined by configuration parameters
passed to Prescient. Prescient will interpolate the available data as necessary to
honor the look-ahead resolution specified in its configuration parameters.

Reserve_Products Details
~~~~~~~~~~~~~~~~~~~~~~~~

Some categories of reserve products may apply to RUC formulations, while
others may apply to SCED formulations. This row allows you to configure
which reserve product categories apply to each formulation type. Reserve
product categories listed in the :col:`DAY_AHEAD` column impose their
requirements on RUC formulations, and reserve product categories listed
in the :col:`REAL_TIME` column impose their requirements on SCED formulations.

Specify applicable reserve product categories as a comma-separated list.
Only listed reserve product categories will be imposed on corresponding
formulations. Supported reserve products are *Spin_Up*, *Reg_Up*, *Reg_Down*,
*Flex_Up*, and *Flex_Down*.

This row is optional. If you leave the row out, all reserve categories
apply to both RUCs and SCEDs.
