storage.csv
===========

This file is where storage elements are defined. Add one row for each
storage component in the model.

Note that Prescient supports more storage properties than what is
supported by the standard RTS-GMLC input format. Prescient extends
the standard RTS-GMLC file format by supporting a set of additional
columns that are not part of the RTS-GMLC standard. Some of these
non-standard columns are required by Prescient; Prescient is unable
to use standard RTS-GMLC storage.csv files without a few extra
columns.

Optional properties can be left out of the CSV file, either by leaving
the entire column out of the file, or by leaving values blank. Any
missing values will use default values.

.. note::
   Be aware that some default values may prevent a storage element from
   being utilized as intended. For example, the default `End State of
   Charge` defaults to the initial state of charge. If the storage
   element starts fully charged, the default `End State of Charge` is set
   to 1.0, requiring the storage element to be fully charged at the end
   of each RUC and SCED.

.. note::
   Note for reviewers: As implemented, default values are handled by Egret. If a value is omitted from
   the CSV file, it is also omitted from the Egret model. Egret then handles the
   missing value using its own default value logic. The defaults listed in the table
   below were taken from the Egret source code.


.. list-table:: storage.csv Columns
   :header-rows: 1
   :widths: auto
   :class: table-top-align-cells

   * - **Column Name**
     - **Description**
     - **Egret**
     - **Default Value**
   * - :col:`GEN UID`
     - The name of a generator on the same bus as this storage element
     - Used to identify the bus this storage element is attached to.
       The storage will be placed on the same bus as the named generator.
       No relationship between the generator and the storage is reflected
       in the Egret model, other than being placed on the same bus.
     - *Required*
   * - :col:`Storage`
     - A unique name for the storage element
     - Used as the storage element name in Egret. Data for this storage component is placed in a
       dictionary located at :samp:`['elements']['generator'][{<Storage>}]`.
     - *Required*
   * - :col:`Max Volume GWh`
     - The maximum storage capacity, in GWh
     - Converted to MWh, then placed in the storage dictionary as ``energy_capacity``
     - *Required*
   * - :col:`Initial Volume GWh`
     - The quantity of energy stored in the storage element at the beginning
       of the simulation, in GWh.
     - Converted to a fraction of *Max Volume GWh*, then placed in the storage dictionary as 
       ``initial_state_of_charge``
     - 0.0
   * - :col:`Start Energy`
     - The rate at which energy is being drawn from the storage element (if positive),
       or the rate at which energy is being injected into the storage element (if negative),
       at the start of the simulation. Units are GW.
     - First converted to MW.

       If positive, placed in the storage dictionary as ``initial_discharge_rate``.

       If negative, placed in the storage dictionary as ``initial_charge_rate``.

       This column serves the same purpose as either the :col:`Initial Charge Rate MW`
       or the :col:`Initial Discharge Rate MW` column, depending on sign. If a row
       has a value in the :col:`Start Energy` column and in the corresponding initial
       charge/discharge column, the initial charge/discharge column takes precedence and
       the value in :col:`Start Energy` is ignored.
     - 0.0
   * - :col:`Initial Charge Rate MW`
     - The rate at which energy is being injected into the storage element
       at the start of the simulation. Units are MW.
     - Placed in the storage dictionary as ``initial_charge_rate``.

       Takes precedence over negative values in the :col:`Start Energy` column.
     - 0.0
   * - :col:`Initial Discharge Rate MW`
     - The rate at which energy is being drawn from the storage element
       at the start of the simulation. Units are MW.
     - Placed in the storage dictionary as ``initial_discharge_rate``.

       Takes precedence over positive values in the :col:`Start Energy` column.
     - 0.0
   * - :col:`Inflow Limit GWh`
     - The maximum rate at which energy can be injected into the storage element, in GW

       .. note::
          (The column name says GWh. Should this really be GW instead???). Also, the
          Egret default is 0.0. Is that a good default? I guess it means "not rechargeable"?
     - Converted to MW, then placed in the storage dictionary as ``max_charge_rate``
     - 0.0
   * - :col:`Rating MVA`
     - The maximum rate at which energy can be drawn from the storage element
     - Placed in the storage dictionary as ``max_discharge_rate``
     - 0.0
   * - :col:`Min Discharge Rate MW`
     - The minimum rate at which energy can be drawn from the storage element
     - Placed in the storage dictionary as ``min_discharge_rate``
     - 0.0
   * - :col:`Min Charge Rate MW`
     - The minimum rate at which energy can be injected into the storage element
     - Placed in the storage dictionary as ``min_charge_rate``
     - 0.0
   * - :col:`Max Hourly Discharge Ramp Up MW`
     - The maximum increase in the discharge rate within a 60 minute period
     - Placed in the storage dictionary as ``ramp_up_output_60min``
     - *Required*

       .. note::
          Required by Egret, has no default. Consider updating Egret to make this
          optional, with some sort of "Unlimited" default.
   * - :col:`Max Hourly Discharge Ramp Down MW`
     - The maximum decrease in the discharge rate within a 60 minute period
     - Placed in the storage dictionary as ``ramp_down_output_60min``
     - *Required*

       .. note::
          Required by Egret, has no default. Consider updating Egret to make this
          optional, with some sort of "Unlimited" default.
   * - :col:`Max Hourly Charge Ramp Up MW`
     - The maximum increase in the charging rate within a 60 minute period
     - Placed in the storage dictionary as ``ramp_up_input_60min``
     - *Required*

       .. note::
          Required by Egret, has no default. Consider updating Egret to make this
          optional, with some sort of "Unlimited" default.
   * - :col:`Max Hourly Charge Ramp Down MW`
     - The maximum decrease in the charging rate within a 60 minute period
     - Placed in the storage dictionary as ``ramp_down_input_60min``
     - *Required*

       .. note::
          Required by Egret, has no default. Consider updating Egret to make this
          optional, with some sort of "Unlimited" default.
   * - :col:`Min SoC`
     - The minimum state of charge the storage element is allowed to be drawn down to.
       Below this point, the system is not allowed to draw additional energy from
       the storage element. Expressed as a number between 0 and 1 that indicates the
       fraction of the maximum storage capacity below which additional energy
       may not be drawn. A value of 0 means all energy is allowed to be drawn from the
       storage element; a value of 0.5 means the system must stop drawing energy 
       from the storage element once its stored energy drops below half of its capacity.
     - Placed in the storage dictionary as ``minimum_state_of_charge``
     - 0.0
   * - :col:`Charge Efficiency`
     - The fraction of injected energy that is stored in the storage
       element. Between 0 and 1.
     - Placed in the storage dictionary as ``charge_efficiency``
     - 1.0
   * - :col:`Discharge Efficiency`
     - The fraction of drawn energy that is injected onto the bus. 
       Between 0 and 1.
     - Placed in the storage dictionary as ``discharge_efficiency``
     - 1.0
   * - :col:`Hourly Retention Rate`
     - The fraction of stored energy that is still stored in the storage
       element after 60 minutes of idle time. Between 0 and 1.
     - Placed in the storage dictionary as ``retention_rate_60min``
     - 1.0
   * - :col:`Charge Cost`
     - The cost per MW of inflow, before losses due to charge efficiency.
     - Placed in the storage dictionary as ``charge_cost``
     - 0.0
   * - :col:`Discharge Cost`
     - The cost per MW of outflow, before losses due to discharge efficiency.
     - Placed in the storage dictionary as ``discharge_cost``
     - 0.0
   * - :col:`End State of Charge`
     - The minimum state of charge at the end of each RUC and SCED.
       Between 0 and 1.
     - Placed in the storage dictionary as ``end_state_of_charge``.
     - Defaults to the initial state of charge fraction implied by :col:`Initial Volume GWh`,
       if specified. If that value is also omitted, defaults to 0.5.
