Custom Input Data Providers
===========================

A custom data provider is a python module that provides data to Prescient
throughout its run. It is an alternative to the standard CSV input file
format typically used by Prescient.

The custom data provider python module must have a function called
`get_data_provider()`. This function must return an object that implements
the `prescient.data.DataProvider` abstract base class.

Internally, Prescient stores data in the `Egret <https://github.com/grid-parity-exchange/Egret>`_
format. Each function in the `prescient.data.DataProvider` abstract base class
generates or manipulates an Egret model. Prescient will call these methods
to acquire initial data, and to request updates to data for specific time
periods.

For an example or a custom data provider, see the
`example <https://github.com/grid-parity-exchange/Prescient/blob/main/prescient/simulator/tests/custom_data_provider.py>`_ 
in the source code, or examine one of the `standard data providers
<https://github.com/grid-parity-exchange/Prescient/tree/main/prescient/data/providers>`_.

To use a custom data provider, set the `--data-provider` configuration option
to the custom provider's python module.