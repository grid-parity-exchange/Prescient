dc_branch.csv
-------------

This file is where DC branches are defined. Prescient has limited support
for DC branches, as indicated by the small number of columns in this file.

This file is optional; if the file does not exist, no DC branches are added
to the model. If the file exists, add a row for each DC branch in the model.
Each row in the file will cause a DC branch dictionary to be added to
``['elements']['dc_branch']`` in the Egret model.

.. list-table:: dc_branch.csv Columns
   :header-rows: 1

   * - **Column Name**
     - **Description**
     - **Egret**
   * - :col:`UID`
     - A unique string identifier for the DC branch.
     - Used as the branch name in Egret. Data for this branch is stored in a
       branch dictionary located at :samp:`['elements']['dc_branch'][{<UID>}]`.
   * - :col:`From Bus`
     - The :col:`Bus ID` of one end of the branch
     - The :col:`Bus Name` of the bus with the matching :col:`Bus ID`, as
       entered in bus.csv, is stored in the Egret branch dictionary as
       ``from_bus``.
   * - :col:`To Bus`
     - The :col:`Bus ID` of the other end of the branch
     - The :col:`Bus Name` of the bus with the matching :col:`Bus ID`, as
       entered in bus.csv, is stored in the Egret branch dictionary as
       ``to_bus``.
   * - :col:`MW Load`
     - Power Demand in MW
     - This value is repeated 3 times in the Egret dc_branch dictionary, as
       ``rating_short_term``, ``rating_long_term``, and ``rating_emergency``.
