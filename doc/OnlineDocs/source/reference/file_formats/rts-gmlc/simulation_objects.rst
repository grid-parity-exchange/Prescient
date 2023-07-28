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
   * - *Reserve_Products*
     - No
     - Which reserve products to enforce for RUC plans or SCED operations.
       See `Reserve_Products Details`_ below.
     - Which reserve products to enforce for RUC plans
     - Which reserve products to enforce for SCED operations


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
