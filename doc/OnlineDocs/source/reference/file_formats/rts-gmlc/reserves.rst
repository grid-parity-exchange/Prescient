reserves.csv
------------

This file defines the reserve products to be included in the model.
Reserve products impose requirements on surplus generation capacity
within a particular area under certain conditions. Each reserve product
has a category and an area. The reserve product's category identifies
the conditions under which its requirements apply, and its area
identifies the region where the requirements apply.

There are 5 supported reserve product categories. The table below shows
the name of reserve product categories on the left as they appear in CSV
input files, and the corresponding name in Egret on the right.

.. list-table:: Reserve Product Categories
   :header-rows: 1

   * - **CSV Reserve Product Category**
     - **Egret reserve product name**
   * - *Spin_Up*
     - spinning_reserve_requirement
   * - *Reg_Up*
     - regulation_up_requirement
   * - *Reg_Down*
     - regulation_down_requirement
   * - *Flex_Up*
     - flexible_ramp_up_requirement
   * - *Flex_Down*
     - flexible_ramp_down_requirement


Each reserve product's category and applicable area are embedded in its
name, as *\<category\>_R\<area\>*. For example, a spinning reserve requirement
for an area named *\"Area 1\"* would be named *\"Spin_Up_RArea 1\"*.

.. list-table:: reserves.csv Columns
   :header-rows: 1

   * - **Column Name**
     - **Description**
     - **Egret**
   * - :col:`Reserve Product`
     - The name of the reserve product, following the *\<category\>_R\<area\>*
       naming convention.
     - Added to the area's Egret dictionary as the Egret reserve product
       name.
   * - :col:`Requirement (MW)`
     - Magnitude of the reserve requirement. This value is ignored if there
       is a timeseries associated with the reserve product.
     - If honored, it is used as the value of the Egret reserve product name entry
       in the area's dictionary.


Reserve Requirement Magnitudes
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The magnitude of each reserve requirement may be constant throughout the
entire simulation, or it may change as specified by a timeseries in
timeseries_pointers.csv. If the magnitude is constant, enter it in this
file as the :col:`Requirement (MW)`. If it varies during the study period,
associate a timeseries with the reserve product (see
:doc:`timeseries_pointers`). In this case, the magnitude entered in this
file is discarded and is replaced with appropriate timeseries values.

Applicability to RUCs and SCEDs
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Each category of reserve product may be configured to apply to RUC
plans, to SCED operations, or both. This is designated in
simulation_objects.csv. See that file's `documentation <simulation_objects>`_
for details.
