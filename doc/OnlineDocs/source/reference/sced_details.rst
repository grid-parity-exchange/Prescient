SCED Details
============

A Security Constrained Economic Dispatch plan, or SCED, determines the power output
level of each dispatchable generator during a single timestep of the simulation.
SCEDs work in conjunction with :doc:`RUCs<ruc_details>` (Reliability Unit
Commitment plans) to simulate operation of the power network.

Each SCED determines operational parameters of each dispatchable generator for a single time step. The SCED coordinates
changes to generator setpoints to minimize total costs for the system as a whole.
The decisions in a
SCED are made by building a model which reflects the state of the power network
at the current simulation time, forecasts for future loads and future renewable
power generation, and unit commitments as dictated by the most recently activated
RUC. The model is solved to find the most cost-efficient way to satisfy
current and forecasted loads while honoring system constraints such as reserve requirements
and line limits. SCEDs always honor unit commitment decisions made by the active
RUC. The number of hours of forecast data to include in the SCED model is
determined by the :ref:`SCED horizon<config_sced-horizon>`.

If :ref:`market settlement<config_compute-market-settlements>` is enabled,
additional market-related statistics are calculated with each SCED. These
statistics report performance against day-ahead commitments and reserve
requirements and the resulting impact on generator revenue.

SCEDs are generated more frequently than RUCs. Where a new RUC is typically
generated between 1 and 4 times a day, SCEDs occur at least hourly. The :ref:`SCED
frequency<config_sced-frequency-minutes>` determines how often a SCED is
generated, and also serves as the size of the simulation time step.

Prescient provides several plugin points to allow the SCED generation process to
be observed or modified. These are documented in the :doc:`detailed_lifecycle`.