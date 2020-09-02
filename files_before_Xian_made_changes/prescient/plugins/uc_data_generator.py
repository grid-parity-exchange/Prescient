
from pyomo.environ import value
## utility for constructing pyomo data dictionary from the passed in parameters to use
def _default_get_data_dict( data_model, time_horizon, demand_dict, reserve_dict, reserve_factor, \
                    min_nondispatch_dict, max_nondispatch_dict, \
                    UnitOnT0Dict, UnitOnT0StateDict, PowerGeneratedT0Dict, StorageSocOnT0Dict, hour):
    
    print("In _defualt_get_data_dict")
    if hour is None:
        print("Setting up DA UC")
    else:
        print("Setting up RT SCED, hour={}".format(hour))
    ## get some useful generators
    Buses = sorted(data_model.Buses)
    TransmissionLines = sorted(data_model.TransmissionLines)
    Interfaces = sorted(data_model.Interfaces)
    ThermalGenerators = sorted(data_model.ThermalGenerators)
    AllNondispatchableGenerators = sorted(data_model.AllNondispatchableGenerators)
    Storage = sorted(data_model.Storage)

    data = {None: { 'Buses': {None: [b for b in Buses]},
                    'StageSet': {None: ["Stage_1", "Stage_2"]},
                    'TimePeriodLength': {None: 1.0},
                    'NumTimePeriods': {None: time_horizon},
                    'CommitmentTimeInStage': {"Stage_1": list(range(1, time_horizon+1)), "Stage_2": []},
                    'GenerationTimeInStage': {"Stage_1": [], "Stage_2": list(range(1, time_horizon+1))},
                    'TransmissionLines': {None : list(TransmissionLines)},
                    'BusFrom': dict((line, data_model.BusFrom[line])
                                    for line in TransmissionLines),
                    'BusTo': dict((line, data_model.BusTo[line])
                                  for line in TransmissionLines),
                    'Impedence': dict((line, data_model.Impedence[line])
                                      for line in TransmissionLines),
                    'ThermalLimit': dict((line, data_model.ThermalLimit[line])
                                         for line in TransmissionLines),
                    'Interfaces': {None : list(Interfaces)},
                    'InterfaceLines': dict((interface, [line for line in data_model.InterfaceLines[interface]])
                                             for interface in Interfaces),
                    'InterfaceFromLimit': dict((interface, data_model.InterfaceFromLimit[interface])
                                             for interface in Interfaces),
                    'InterfaceToLimit': dict((interface, data_model.InterfaceToLimit[interface])
                                             for interface in Interfaces),
                    'ThermalGenerators': {None: [gen for gen in data_model.ThermalGenerators]},
                    'ThermalGeneratorType': dict((gen, data_model.ThermalGeneratorType[gen])
                                                 for gen in ThermalGenerators),
                    'ThermalGeneratorsAtBus': dict((b, [gen for gen in sorted(data_model.ThermalGeneratorsAtBus[b])])
                                                   for b in Buses),
                    'QuickStart': dict((g, value(data_model.QuickStart[g])) for g in ThermalGenerators),
                    'QuickStartGenerators': {None: [g for g in sorted(data_model.QuickStartGenerators)]},
                    'AllNondispatchableGenerators': {None: [g for g in AllNondispatchableGenerators]},
                    'NondispatchableGeneratorType': dict((gen, data_model.NondispatchableGeneratorType[gen])
                                                         for gen in AllNondispatchableGenerators),
                    'MustRunGenerators': {None: [g for g in sorted(data_model.MustRunGenerators)]},
                    'NondispatchableGeneratorsAtBus': dict((b, [gen for gen in sorted(data_model.NondispatchableGeneratorsAtBus[b])])
                                                           for b in Buses),
                    'Demand': demand_dict,
                    'ReserveRequirement': reserve_dict,
                    'MinimumPowerOutput': dict((g, value(data_model.MinimumPowerOutput[g]))
                                               for g in ThermalGenerators),
                    'MaximumPowerOutput': dict((g, value(data_model.MaximumPowerOutput[g]))
                                               for g in ThermalGenerators),
                    'MinNondispatchablePower': min_nondispatch_dict,
                    'MaxNondispatchablePower': max_nondispatch_dict,
                    'NominalRampUpLimit': dict((g, value(data_model.NominalRampUpLimit[g]))
                                               for g in ThermalGenerators),
                    'NominalRampDownLimit': dict((g, value(data_model.NominalRampDownLimit[g]))
                                                 for g in ThermalGenerators),
                    'StartupRampLimit': dict((g, value(data_model.StartupRampLimit[g]))
                                             for g in ThermalGenerators),
                    'ShutdownRampLimit': dict((g, value(data_model.ShutdownRampLimit[g]))
                                              for g in ThermalGenerators),
                    'MinimumUpTime': dict((g, value(data_model.MinimumUpTime[g]))
                                          for g in ThermalGenerators),
                    'MinimumDownTime': dict((g, value(data_model.MinimumDownTime[g]))
                                            for g in ThermalGenerators),
                    'UnitOnT0': UnitOnT0Dict,
                    'UnitOnT0State': UnitOnT0StateDict,
                    'PowerGeneratedT0': PowerGeneratedT0Dict,
                    'ProductionCostA0': dict((g, value(data_model.ProductionCostA0[g]))
                                             for g in ThermalGenerators),
                    'ProductionCostA1': dict((g, value(data_model.ProductionCostA1[g]))
                                             for g in ThermalGenerators),
                    'ProductionCostA2': dict((g, value(data_model.ProductionCostA2[g]))
                                             for g in ThermalGenerators),
                    'CostPiecewisePoints': dict((g, [point for point in data_model.CostPiecewisePoints[g]])
                                                for g in ThermalGenerators),
                    'CostPiecewiseValues': dict((g, [value for value in data_model.CostPiecewiseValues[g]])
                                                for g in ThermalGenerators),
                    'FuelCost': dict((g, value(data_model.FuelCost[g]))
                                     for g in ThermalGenerators),
                    'NumGeneratorCostCurvePieces': {None:value(data_model.NumGeneratorCostCurvePieces)},
                    'StartupLags': dict((g, [point for point in data_model.StartupLags[g]])
                                        for g in ThermalGenerators),
                    'StartupCosts': dict((g, [point for point in data_model.StartupCosts[g]])
                                         for g in ThermalGenerators),
                    'ShutdownFixedCost': dict((g, value(data_model.ShutdownFixedCost[g]))
                                              for g in ThermalGenerators),
                    'Storage': {None: [s for s in Storage]},
                    'StorageAtBus': dict((b, [s for s in sorted(data_model.StorageAtBus[b])])
                                         for b in Buses),
                    'MinimumPowerOutputStorage': dict((s, value(data_model.MinimumPowerOutputStorage[s]))
                                                      for s in Storage),
                    'MaximumPowerOutputStorage': dict((s, value(data_model.MaximumPowerOutputStorage[s]))
                                                      for s in Storage),
                    'MinimumPowerInputStorage': dict((s, value(data_model.MinimumPowerInputStorage[s]))
                                                     for s in Storage),
                    'MaximumPowerInputStorage': dict((s, value(data_model.MaximumPowerInputStorage[s]))
                                                     for s in Storage),
                    'NominalRampUpLimitStorageOutput': dict((s, value(data_model.NominalRampUpLimitStorageOutput[s]))
                                                            for s in Storage),
                    'NominalRampDownLimitStorageOutput': dict((s, value(data_model.NominalRampDownLimitStorageOutput[s]))
                                                              for s in Storage),
                    'NominalRampUpLimitStorageInput': dict((s, value(data_model.NominalRampUpLimitStorageInput[s]))
                                                           for s in Storage),
                    'NominalRampDownLimitStorageInput': dict((s, value(data_model.NominalRampDownLimitStorageInput[s]))
                                                             for s in Storage),
                    'MaximumEnergyStorage': dict((s, value(data_model.MaximumEnergyStorage[s]))
                                                 for s in Storage),
                    'MinimumSocStorage': dict((s, value(data_model.MinimumSocStorage[s]))
                                              for s in Storage),
                    'InputEfficiencyEnergy': dict((s, value(data_model.InputEfficiencyEnergy[s]))
                                                  for s in Storage),
                    'OutputEfficiencyEnergy': dict((s, value(data_model.OutputEfficiencyEnergy[s]))
                                                   for s in Storage),
                    'RetentionRate': dict((s, value(data_model.RetentionRate[s]))
                                          for s in Storage),
                    'EndPointSocStorage': dict((s, value(data_model.EndPointSocStorage[s]))
                                               for s in Storage),
                    'StoragePowerOutputOnT0': dict((s, value(data_model.StoragePowerOutputOnT0[s]))
                                                   for s in Storage),
                    'StoragePowerInputOnT0': dict((s, value(data_model.StoragePowerInputOnT0[s]))
                                                  for s in Storage),
                    'StorageSocOnT0': StorageSocOnT0Dict,
                    'LoadMismatchPenalty': {None: value(data_model.LoadMismatchPenalty)},
                    'ReserveShortfallPenalty': {None: value(data_model.ReserveShortfallPenalty)}
                  }
           }
    if reserve_factor > 0.0:
        data[None]["ReserveFactor"] = {None: reserve_factor}

    return data
