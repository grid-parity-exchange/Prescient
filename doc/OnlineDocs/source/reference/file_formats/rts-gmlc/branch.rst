branch.csv
==========

This file defines branches - flow pathways between pairs of buses -
including lines and transformers. Add a row for each branch in the
system. Each row in the CSV file will cause a branch dictionary to be
added to ``['elements']['branch']`` in the Egret model.

.. list-table:: ListTable
   :header-rows: 1

   * - **Column Name**
     - **Description**
     - **Egret**
   * - :col:`UID`
     - A unique string identifier for the branch.
     - Used as the branch name in Egret. Data for this branch is stored in a
       branch dictionary stored at :samp:`['elements']['branch'][{<UID>}]`.
   * - :col:`From Bus`
     - The :col:`Bus ID` of one end of the branch
     - The :col:`Bus Name` of the bus with the corresponding :col:`Bus ID`,
       as entered in bus.csv, is stored in the Egret branch dictionary as
       ``from_bus``.
   * - :col:`To Bus`
     - The :col:`Bus ID` of the other end of the branch
     - The :col:`Bus Name` of the bus with the corresponding :col:`Bus ID`,
       as entered in bus.csv, is stored in the Egret branch dictionary as
       ``to_bus``.
   * - :col:`R`
     - Branch resistance p.u.
     - Stored in Egret bus dictionary as ``resistance``.
   * - :col:`X`
     - Branch reactance p.u.
     - Stored in the Egret bus dictionary as ``reactance``.
   * - :col:`B`
     - Charging susceptance p.u.
     - Stored in the Egret bus dictionary as ``charging_susceptance``.
   * - :col:`Cont Rating`
     - Continuous flow limit in MW
     - Stored in the Egret bus dictionary as ``rating_long_term``. Optional.
   * - :col:`LTE Rating`
     - Non-continuous long term flow limit in MW
     - Stored in the Egret bus dictionary as ``rating_short_term``. Optional.
   * - :col:`STE Rating`
     - Short term flow limit in MW
     - Stored in the Egret bus dictionary as ``rating_emergency``. Optional.
   * - :col:`Tr Ratio`
     - Transformer winding ratio.
     - If non-zero, branch is treated as a transformer. If blank or zero,
       branch is considered a line. See `Lines and Transformers`_ below.


Additional Branch Values
~~~~~~~~~~~~~~~~~~~~~~~~

The following values are automatically added to every branch dictionary
in the Egret model:

-  ``in_service`` = *true*
-  ``angle_diff_min`` = *-90*
-  ``angle_diff_max`` = *90*
-  ``pf`` = *null*
-  ``qf`` = *null*
-  ``pt`` = *null*
-  ``qt`` = *null*

Lines and Transformers
~~~~~~~~~~~~~~~~~~~~~~

Each branch is either a line or a transformer. The type of branch is
determined by the :col:`Tr Ratio`. If this field is blank or zero, the branch
is a line and the following property is added to the branch dictionary:

-  ``branch_type`` = *line*

If the :col:`Tr Ratio` is a non-zero value, the following properties are added
to the branch dictionary:

-  ``branch_type`` = *transformer*

-  ``transformer_tap_ratio`` = :col:`Tr Ratio`

-  ``transformer_phase_shift`` = *0*
