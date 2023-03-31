.. _bus.csv:

bus.csv
=======

This file is used to define buses. Add one row for each bus in the system. Each
row in the CSV file will cause a bus dictionary object to be added to 
``['elements']['bus']`` in the Egret model.

Each row with a non-zero :col:`MW Load` and/or non-zero :col:`MVAR Load` will also cause a
load to be added to ``['elements']['load']`` in Egret, and each row with a non-zero 
:col:`MW Shunt G` and/or non-zero :col:`MVAR Shunt B` will cause a shunt to be added
to ``['elements']['shunt']`` in Egret.

.. list-table:: bus.csv Columns
   :header-rows: 1

   * - **Column Name**
     - **Description**
     - **Egret**
   * - :col:`Bus ID`
     - A unique string identifier for the bus. This string is used to refer
       to this bus in other CSV files.
     - Not used by Egret except during parsing of CSV files.
   * - :col:`Bus Name`
     - A human-friendly unique string for this bus.
     - Used as the bus name in Egret. Data for this bus is stored in a bus
       dictionary stored at :samp:`['elements']['bus'][{<Bus Name>}]`.
       
       This is also the name of the load, if a load is added for the bus (a load
       is added if MW Load or MVAR Load is non-zero). The load dictionary is
       stored at :samp:`['elements']['load'][{<Bus Name>}]`.
       
       This is also the name of the shunt, if a shunt is added for the bus (a bus
       is added if :col:`MW Shunt G` or :col:`MVAR Shunt G` is non-zero). The shunt
       dictionary is stored at :samp:`['elements']['shunt'][{<Bus Name>}]`.
   * - :col:`BaseKV`
     - The bus base voltage. Must be non-zero positive.
     - Stored in the Egret bus dictionary as :code:`base_kv`.
   * - :col:`Bus Type`
     - The type of bus. Can be one of the following:

       * *PQ*
       * *PV*
       * *Ref*
     - Stored in Egret bus dictionary as ``matpower_bustype``. The *Ref* bus type
       is stored in Egret in all lower case (*ref*).
   * - :col:`MW Load`
     - Magnitude of the load on the bus.
     - Stored in the Egret bus dictionary as ``p_load``. A non-zero value causes
       a load to be added; see `Bus Loads`_.
   * - :col:`MVAR Load`
     - Magnitude of the reactive load on the bus.
     - Stored in the Egret bus dictionary as ``q_load``. A non-zero value causes
       a load to be added; see `Bus Loads`_.
   * - :col:`V Mag`
     - Voltage magnitude setpoint
     - Stored in the Egret bus dictionary as ``vm``.
   * - :col:`V Angle`
     - Voltage angle setpoint in degrees
     - Stored in the Egret bus dictionary as ``va``. If the Bus Type is *Ref*, this
       value must be *0.0*.
   * - :col:`Area`
     - The area the bus is in.
     - Stored in the Egret bus dictionary as ``area``. An area dictionary is added to the
       Egret model for each unique area mentioned in the file. The Egret area dictionary
       is found at :samp:`['elements']['area'][{<Area>}]`. See `Areas`_.
   * - :col:`Zone`
     - The zone the bus is in.
     - Stored in the Egret bus dictionary as ``zone``.
   * - :col:`MW Shunt G`
     - Optional.
     - Stored in the shunt dictionary as ``gs``. See `Shunts`_.
   * - :col:`MVAR Shunt B`
     - Optional.
     - Stored in the shunt dictionary as ``bs``. See `Shunts`_.
   * - :col:`va`
     - Reference bus angle. If the :col:`Bus Type` is *Ref*, :col:`va` is required and must
       be zero.
     - 

Additional Bus Values
~~~~~~~~~~~~~~~~~~~~~

The following values are automatically added to the bus dictionary:

-  ``v_min`` = *0.95*

-  ``v_max`` = *1.05*

Bus Loads
~~~~~~~~~

If a bus has a non-zero :col:`MW Load` or :col:`MVAR Load`, a load dictionary is added
to Egret at :samp:`['elements']['load'][{<Bus Name>}]`. The load dictionary will
have the following values taken from bus.csv:

-  ``bus`` = :col:`Bus Name`

-  ``p_load`` = :col:`MW Load`

-  ``q_load`` = :col:`MVAR Load`

-  ``area`` = :col:`Area`

-  ``zone`` = :col:`Zone`

An additional property is automatically added, always with the same value:

-  ``in_service`` = *true*

Loads can (and usually do) vary throughout the study horizon. Variable loads are
defined using a timeseries (see :doc:`timeseries_pointers`).

Shunts
~~~~~~

If a bus has a non-zero :col:`MW Shunt G` or a non-zero :col:`MVAR Shunt B`, a shunt
dictionary is added to Egret at :samp:`['elements']['shunt'][{<Bus Name>}]`. The shunt
dictionary will have the following values taken from bus.csv:

-  ``bus`` = :col:`Bus Name`

-  ``gs`` = :col:`MW Shunt G`

-  ``bs`` = :col:`MVAR Shunt B`

An additional property is automatically added, always with the same value:

-  ``shunt_type`` = *fixed*

Areas
~~~~~

Each unique area mentioned in bus.csv leads to an area being created in
the Egret model at :samp:`['elements']['area'][{<Area>}]`, using the area
name as it appears in bus.csv.
