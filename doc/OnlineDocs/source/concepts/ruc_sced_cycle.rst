The Prescient Simulation Cycle
=============================================

.. note::

   This was taken from a previous write-up and needs to be revisited.

Prescient simulates the operation of the network throughout a study horizon, finding
the set of operational choices that satisfy demand at the lowest possible cost.

Prescient loops through two repeating phases, the reliability unit commitment (RUC) phase and 
the security constrained economic dispatch (SCED) phase.  The RUC phase determines which
dispatchable generators will be active in upcoming operational time periods. For each operational
period within a RUC cycle, the SCED phase selects the dispatch level of each committed thermal 
generator.

The RUC phase occurs one or more times per day.  Each time the RUC phase occurs, Prescient generates
a unit commitment schedule that indicates which generators will be brought online or taken offline
within the RUC's time horizon. The SCED phase occurs one or more times per hour.  Each SCED selects
a thermal dispatch level for each committed generator.

The RUC Phase
-------------
More detailed description of the RUC... 

The RUC phase occurs one or more times per day.  Each time the RUC phase occurs, Prescient generates
a unit commitment schedule that indicates which generators will be brought online or taken offline
within the RUC's time horizon.  The RUC schedule may begin immediately, or it may begin a number of
hours after the RUC is generated.

The SCED Phase
----------------
More detailed description of the SCED, including a high level description of the optimization problem
being solved, and possibly a conversational description of some things that can be tweaked (such as
how often a SCED runs).

.. _future-times-in-sceds:

Future Values in SCEDs
``````````````````````
.. warning::
   Coming soon.