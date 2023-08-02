RUC Details
===========

A Reliability Unit Commitment plan, or RUC, determines which dispatchable
generators will be active during a portion of the simulation. RUCs work in
conjunction with :doc:`SCEDs<sced_details>` (Security Constraint Economic
Dispatch plans) to simulate operation of the power network.

Each RUC covers a specific period within the simulation. For each hour within its
applicable period, a RUC dictates whether each dispatchable generator is on or off.
The unit commitment decisions in a RUC are made by building a model which reflects
the current state of the power network and forecasts for future loads and future
renewable power generation. The model is solved to find the most cost-efficient way
to satisfy forecasted loads while honoring system constraints such as reserve
requirements and line limits.

RUCs may also include pricing schedules. This option is enabled when the
:ref:`compute-market-settlements<config_compute-market-settlements>` option
is set to true. The pricing schedule sets the contract price for expected power
delivery and for reserves (ancillary service products).

A new RUC is generated at regular intervals. The number of hours between RUCs is
called the :ref:`RUC interval<config_ruc-every-hours>`. The RUC interval also
dictates how many hours each RUC is active. The RUC interval must be between 1 and
24 hours and must divide evenly into 24 hours.

The number of hours of forecast data to include in the RUC model is determined by
the :ref:`RUC horizon<config_ruc-horizon>`. The RUC horizon must be at least
equal to the RUC interval, but typically extends further into the future to avoid
poor choices at the end of the plan ("end effects"). A commonly used RUC horizon is
48 hours.

Each RUC may be generated just as its applicable period is about to begin, or it
may be generated in advance. For this reason, Prescient splits RUC management
into two phases: RUC generation and RUC activation. In the RUC generation phase, a
RUC model is created and optimized, resulting in a RUC plan. In the RUC activation
phase, the commitment decisions identified in the RUC plan begin to take effect.

A new RUC is always activated at the beginning of each day, and at each time that
is a multiple of the RUC interval. For example, if the RUC interval is 8 hours,
then a new RUC is activated each day at midnight, 8:00 a.m., and 4:00 p.m.

To generate RUCs in advance of their activation time, set the :ref:`RUC execution
hour<config_ruc-execution-hour>` to indicate the time of day that one of the
day's RUC should be generated. If the specified time falls on a scheduled RUC
activation time, then RUCs will not be generated in advance. Otherwise, the
specified time is interpreted as the time to generate the next scheduled RUC. For
example, if the RUC interval is 8 hours and the RUC execution hour is 14 (2:00
p.m.), then each RUC will be generated 2 hours before its activation time (because
the next RUC activation time ater 2:00 p.m. is 4:00 p.m.). The gap between RUC
generation and RUC activation is called the RUC delay.

When there is a non-zero RUC delay, generating a RUC model includes an additional
step at the beginning of the RUC generation process. In this first step, a SCED
model is created and solved for the period starting with the current simulation
time and ending after the RUC activation time. Next, a RUC model is created using
the future system state predicted by the SCED as its initial conditions.

The very first RUC of the simulation is always generated with zero RUC delay, even
if Prescient has been configured to generate other RUCs in advance.

Prescient provides several plugin points to allow the RUC generation and activation
process to be observed or modified. These are documented in the
:doc:`detailed_lifecycle`.
