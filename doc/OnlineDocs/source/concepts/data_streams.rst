Time Series Data Streams
========================

Prescient uses time series data from two data streams, the real-time stream
(i.e., actuals) and the forecast stream. As their names imply, the real-time
stream includes data that the simulation should treat as actual values that
occur at specific times in the simulation, and the forecast stream includes
forecasts for time periods that have not yet occured in the simulation.

Both streams consist of time-stamped values for loads and non-dispatchable
generation data.

Real-Time Data (Actuals)
------------------------

The real-time data stream provides data that the simulation should treat as
actual values. Real-time values are typically used only when the simulation reaches
the corresponding simulation time.

Real-time data can be provided at any time interval. The real-time data interval
generally matches the SCED interval
(see :ref:`sced-frequency-minutes<config_sced-frequency-minutes>`), but this is
not a requirement. If the SCED interval does not match the real-time interval
then real-time data will be interpolated or discarded as needed to match the SCED
interval.

Forecasts
---------

Forecast data are provided by the forecast data stream. The frequency of data
provided through the forecast stream must be hourly.

New forecasts are retrieved each time a new RUC plan is generated. The
forecasts retrieved in a given batch are those required to satisfy the RUC horizon
(see :ref:`ruc-horizon<config_ruc-horizon>`), starting with the RUC activation time.


.. _forecast_smoothing:

Forecast Smoothing
~~~~~~~~~~~~~~~~~~

As forecasts are retrieved from the forecast data stream, they may be adjusted so that
near-term forecasts are more accurate than forecasts further into the future. This serves
two purposes: first, to avoid large jumps in timeseries values due to inaccurate forecasts;
and second, to model how forecasts become more accurate as their time approaches.

The number of forecasts to be smoothed is determined by the
:ref:`ruc-prescience-hour<config_ruc-prescience-hour>` configuration option. Values for
the current simulation time are set equal to their actual value, ignoring data read from
the forecast stream. Values for ``ruc-prescience-hour`` hours after the current simulation
time are set equal to data read from the forecast stream. Between these two times,
values are a weighted average of the values provided by the actuals and forecast data
streams. The weights vary linearly with where the time falls between the current time
and the ruc prescience hour. For example, if ``ruc-prescience-hour`` is 8, then the adjusted
forecast for 2 hours after the current simulation time will be ``0.25*forecast + 0.75*actual``.

Note that blending weights are determined relative to the current simulation time when
the RUC is generated, not relative to the time the RUC goes into effect.

Real-Time Forecast Adjustments
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Forecasts are adjusted further each time a SCED is run. This is done by comparing the forecast
for the current time with the actual value for the current time. The ratio of these two
values is calculated, then used as a scaling factor for forecast values. For example, if the
forecast for a value was 10% too high, all future forecasts for the same value are reduced by 10%.

.. note::

	If :ref:`run-sced-with-persistent-forecast-errors<config_run-sced-with-persistent-forecast-errors>`
	is false, then SCEDs will use actual values for all time periods. Forecasts will still be used
	for RUCs, but SCEDs will be based entirely on actual values, even for future time periods.