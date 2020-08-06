#  ___________________________________________________________________________
#
#  Prescient
#  Copyright 2020 National Technology & Engineering Solutions of Sandia, LLC
#  (NTESS). Under the terms of Contract DE-NA0003525 with NTESS, the U.S.
#  Government retains certain rights in this software.
#  This software is distributed under the Revised BSD License.
#  ___________________________________________________________________________

## modified version of the script from RTS-GMLC/RTS_Data/FormattedData/Prescient/topysp.py

import sys
import os
import pandas as pd
import math

from collections import namedtuple


def write_template( rts_gmlc_dir, file_name, copper_sheet = False, reserve_factor = None ):

    base_dir = os.path.join(rts_gmlc_dir,'RTS_Data','SourceData')
    
    Generator = namedtuple('Generator',
                           ['ID', # integer 
                            'Bus',
                            'UnitGroup',
                            'UnitType',
                            'Fuel',
                            'MinPower',
                            'MaxPower',
                            'MinDownTime',
                            'MinUpTime',
                            'RampRate',         # units are MW/minute
                            'StartTimeCold',    # units are hours
                            'StartTimeWarm',    # units are hours
                            'StartTimeHot',     # units are hours
                            'StartCostCold',    # units are MBTU 
                            'StartCostWarm',    # units are MBTU
                            'StartCostHot',     # units are MBTU
                            'NonFuelStartCost', # units are $
                            'FuelPrice',        # units are $ / MMBTU
                            'OutputPct0',  
                            'OutputPct1',
                            'OutputPct2',
                            'OutputPct3',
                            'HeatRateAvg0',
                            'HeatRateIncr1',
                            'HeatRateIncr2',
                            'HeatRateIncr3'],
                           )
    
    Bus = namedtuple('Bus',
                     ['ID', # integer
                      'Name',
                      'BaseKV',
                      'Type',
                      'MWLoad',
                      'Area',
                      'SubArea',
                      'Zone',
                      'Lat',
                      'Long'],
                     )
    
    Branch = namedtuple('Branch',
                        ['ID',
                         'FromBus',
                         'ToBus',
                         'R',
                         'X', # csv file is in PU, multiple by 100 to make consistent with MW
                         'B', 
                         'ContRating'],
                       )
    
    
    generator_dict = {} # keys are ID
    bus_dict = {} # keys are ID
    branch_dict = {} # keys are ID
    timeseries_pointer_dict = {} # keys are (ID, simulation-type) pairs
    
    generator_df = pd.read_table(os.path.join(base_dir,"gen.csv"), header=0, sep=',')
    bus_df = pd.read_table(os.path.join(base_dir,"bus.csv"), header=0, sep=',')
    branch_df = pd.read_table(os.path.join(base_dir,"branch.csv"), header=0, sep=',')
    
    for generator_index in generator_df.index.tolist():
        this_generator_dict = generator_df.loc[generator_index].to_dict()
        new_generator = Generator(this_generator_dict["GEN UID"],
                                  int(this_generator_dict["Bus ID"]),
                                  this_generator_dict["Unit Group"],
                                  this_generator_dict["Unit Type"],
                                  this_generator_dict["Fuel"],
                                  float(this_generator_dict["PMin MW"]),
                                  float(this_generator_dict["PMax MW"]),
                                  # per Brendan, PLEXOS takes the ceiling at hourly resolution for up and down times.
                                  int(math.ceil(this_generator_dict["Min Down Time Hr"])),
                                  int(math.ceil(this_generator_dict["Min Up Time Hr"])),
                                  this_generator_dict["Ramp Rate MW/Min"],
                                  int(this_generator_dict["Start Time Cold Hr"]),
                                  int(this_generator_dict["Start Time Warm Hr"]),
                                  int(this_generator_dict["Start Time Hot Hr"]),
                                  float(this_generator_dict["Start Heat Cold MBTU"]),
                                  float(this_generator_dict["Start Heat Warm MBTU"]),
                                  float(this_generator_dict["Start Heat Hot MBTU"]),
                                  float(this_generator_dict["Non Fuel Start Cost $"]),
                                  float(this_generator_dict["Fuel Price $/MMBTU"]),
                                  float(this_generator_dict["Output_pct_0"]),
                                  float(this_generator_dict["Output_pct_1"]),
                                  float(this_generator_dict["Output_pct_2"]),
                                  float(this_generator_dict["Output_pct_3"]),
                                  float(this_generator_dict["HR_avg_0"]),
                                  float(this_generator_dict["HR_incr_1"]),
                                  float(this_generator_dict["HR_incr_2"]),
                                  float(this_generator_dict["HR_incr_3"]))
        
        generator_dict[new_generator.ID] = new_generator
    
    bus_id_to_name_dict = {}
    
    for bus_index in bus_df.index.tolist():
        this_bus_dict = bus_df.loc[bus_index].to_dict()
        new_bus = Bus(int(this_bus_dict["Bus ID"]),
                      this_bus_dict["Bus Name"],
                      this_bus_dict["BaseKV"],
                      this_bus_dict["Bus Type"],
                      float(this_bus_dict["MW Load"]),
                      int(this_bus_dict["Area"]),
                      int(this_bus_dict["Sub Area"]),
                      this_bus_dict["Zone"],
                      this_bus_dict["lat"],
                      this_bus_dict["lng"])
        bus_dict[new_bus.Name] = new_bus
        bus_id_to_name_dict[new_bus.ID] = new_bus.Name
    
    for branch_index in branch_df.index.tolist():
        this_branch_dict = branch_df.loc[branch_index].to_dict()
        new_branch = Branch(this_branch_dict["UID"],
                            this_branch_dict["From Bus"],
                            this_branch_dict["To Bus"],
                            float(this_branch_dict["R"]),
                            float(this_branch_dict["X"]) / 100.0, # nix per unit
                            float(this_branch_dict["B"]),
                            float(this_branch_dict["Cont Rating"]))
        branch_dict[new_branch.ID] = new_branch
    
    
    unit_on_time_df = pd.read_table(os.path.join(base_dir,"../FormattedData/PLEXOS/PLEXOS_Solution/DAY_AHEAD Solution Files/noTX/on_time_7.12.csv"),
                                    header=0,
                                    sep=",")
    unit_on_time_df_as_dict = unit_on_time_df.to_dict(orient="split")
    unit_on_t0_state_dict = {} 
    for i in range(0,len(unit_on_time_df_as_dict["columns"])):
        gen_id = unit_on_time_df_as_dict["columns"][i]
        unit_on_t0_state_dict[gen_id] = int(unit_on_time_df_as_dict["data"][0][i])
    
    #print("Writing Prescient template file")
    
    minutes_per_time_period = 60
    ## we'll bring the ramping down by this factor
    ramp_scaling_factor = 1.
    
    dat_file = open(file_name,"w")
    
    print("param NumTimePeriods := 48 ;", file=dat_file)
    print("", file=dat_file)
    print("param TimePeriodLength := 1 ;", file=dat_file)
    print("", file=dat_file)
    print("set StageSet := Stage_1 Stage_2 ;", file=dat_file)
    print("", file=dat_file)
    print("set CommitmentTimeInStage[Stage_1] := 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31 32 33 34 35 36 37 38 39 40 41 42 43 44 45 46 47 48 ;", file=dat_file)
    print("", file=dat_file)
    print("set CommitmentTimeInStage[Stage_2] := ;", file=dat_file)
    print("", file=dat_file)
    print("set GenerationTimeInStage[Stage_1] := ;", file=dat_file)
    print("", file=dat_file)
    print("set GenerationTimeInStage[Stage_2] := 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31 32 33 34 35 36 37 38 39 40 41 42 43 44 45 46 47 48 ;", file=dat_file)
    print("", file=dat_file)

    if reserve_factor is not None:
        print("param ReserveFactor := %10.8f ;" % (reserve_factor), file=dat_file)
    
    print("set Buses := ", file=dat_file)
    if copper_sheet:
        print("CopperSheet", file=dat_file)
    else:
        for bus_id in bus_dict.keys():
            print("%s" % bus_id, file=dat_file)
    print(";", file=dat_file)
    
    print("", file=dat_file)
    
    print("set TransmissionLines := ", file=dat_file)
    if copper_sheet:
        pass
    else:
        for branch_id in branch_dict.keys():
            print("%s" % branch_id, file=dat_file)
    print(";", file=dat_file)
    
    print("", file=dat_file)
    
    print("param: BusFrom BusTo ThermalLimit Impedence :=", file=dat_file)
    if copper_sheet:
        pass
    else:
        for branch_id, branch_spec in branch_dict.items():
            print("%15s %15s %15s   %10.8f      %10.8f" % (branch_spec.ID, bus_id_to_name_dict[branch_spec.FromBus], bus_id_to_name_dict[branch_spec.ToBus], branch_spec.ContRating, branch_spec.X), file=dat_file)
    
    print(";", file=dat_file)
    
    print("", file=dat_file)
    
    print("set ThermalGenerators := ", file=dat_file)
    for gen_id, gen_spec in generator_dict.items():
        if gen_spec.Fuel == "Oil" or gen_spec.Fuel == "Coal" or gen_spec.Fuel == "NG" or gen_spec.Fuel == "Nuclear":
            print("%s" % gen_id, file=dat_file)
    print(";", file=dat_file)
    
    print("", file=dat_file)
    
    if copper_sheet:
        print("set ThermalGeneratorsAtBus[CopperSheet] := ", file=dat_file)
        for gen_id, gen_spec in generator_dict.items():
            if gen_spec.Fuel == "Oil" or gen_spec.Fuel == "Coal" or gen_spec.Fuel == "NG" or gen_spec.Fuel == "Nuclear":
                print("%s" % gen_id, file=dat_file)
        print(";", file=dat_file)
        print("", file=dat_file)
    else:
        for bus_name in bus_dict.keys():
            print("set ThermalGeneratorsAtBus[%s] := " % bus_name, file=dat_file)
            for gen_id, gen_spec in generator_dict.items():
                if bus_dict[bus_name].ID == gen_spec.Bus:
                    if gen_spec.Fuel == "Oil" or gen_spec.Fuel == "Coal" or gen_spec.Fuel == "NG" or gen_spec.Fuel == "Nuclear":
                        print("%s" % gen_id, file=dat_file)
            print(";", file=dat_file)
            print("", file=dat_file)
    
    print("set NondispatchableGenerators := ", file=dat_file)
    for gen_id, gen_spec in generator_dict.items():
        if gen_spec.Fuel == "Solar" or gen_spec.Fuel == "Wind" or gen_spec.Fuel == "Hydro":
            print("%s" % gen_id, file=dat_file)
    print(";", file=dat_file)
    
    print("", file=dat_file)
    
    if copper_sheet:
        print("set NondispatchableGeneratorsAtBus[CopperSheet] := ", file=dat_file)
        for gen_id, gen_spec in generator_dict.items():
            if gen_spec.Fuel == "Solar" or gen_spec.Fuel == "Wind" or gen_spec.Fuel == "Hydro":
                print("%s" % gen_id, file=dat_file)
        print(";", file=dat_file)
        print("", file=dat_file)
    else:
        for bus_name in bus_dict.keys():
            print("set NondispatchableGeneratorsAtBus[%s] := " % bus_name, file=dat_file)
            for gen_id, gen_spec in generator_dict.items():
                if bus_dict[bus_name].ID == gen_spec.Bus:
                    if gen_spec.Fuel == "Solar" or gen_spec.Fuel == "Wind" or gen_spec.Fuel == "Hydro":
                        print("%s" % gen_id, file=dat_file)
            print(";", file=dat_file)
            print("", file=dat_file)
    
    print("param MustRun := ", file=dat_file)
    for gen_id, gen_spec in generator_dict.items():
        if gen_spec.Fuel == "Nuclear":
            print("%s 1" % gen_id, file=dat_file)
    print(";", file=dat_file)
    
    print("", file=dat_file)
    
    print("param ThermalGeneratorType := ", file=dat_file)
    for gen_id, gen_spec in generator_dict.items():
        if gen_spec.Fuel == "Nuclear":
            print("%s N" % gen_id, file=dat_file)
        elif gen_spec.Fuel == "NG":
            print("%s G" % gen_id, file=dat_file)
        elif gen_spec.Fuel == "Oil":
            print("%s O" % gen_id, file=dat_file)
        elif gen_spec.Fuel == "Coal":
            print("%s C" % gen_id, file=dat_file)
        else:
            pass
    print(";", file=dat_file)
    
    print("", file=dat_file)
    
    print("param NondispatchableGeneratorType := ", file=dat_file)
    for gen_id, gen_spec in generator_dict.items():
        if gen_spec.Fuel == "Wind":
            print("%s W" % gen_id, file=dat_file)
        elif gen_spec.Fuel == "Solar":
            print("%s S" % gen_id, file=dat_file)
        elif gen_spec.Fuel == "Hydro":
            print("%s H" % gen_id, file=dat_file)
        else:
            pass
    print(";", file=dat_file)
    
    print("", file=dat_file)
    
    print("param: MinimumPowerOutput MaximumPowerOutput MinimumUpTime MinimumDownTime NominalRampUpLimit NominalRampDownLimit StartupRampLimit ShutdownRampLimit := ", file=dat_file)
    for gen_id, gen_spec in generator_dict.items():
        if gen_spec.Fuel == "Oil" or gen_spec.Fuel == "Coal" or gen_spec.Fuel == "NG" or gen_spec.Fuel == "Nuclear":
            print("%15s %10.2f %10.2f %2d %2d %10.2f %10.2f %10.2f %10.2f" % (gen_id, 
                                                                              gen_spec.MinPower,
                                                                              gen_spec.MaxPower,
                                                                              gen_spec.MinUpTime,
                                                                              gen_spec.MinDownTime,
                                                                              gen_spec.RampRate * float(minutes_per_time_period) / ramp_scaling_factor,
                                                                              gen_spec.RampRate * float(minutes_per_time_period) / ramp_scaling_factor,
                                                                              gen_spec.MinPower,
                                                                              gen_spec.MinPower),
                  file=dat_file)
    print(";", file=dat_file)
    
    print("", file=dat_file)
    
    for gen_id, gen_spec in generator_dict.items():
        if gen_spec.Fuel == "Oil" or gen_spec.Fuel == "Coal" or gen_spec.Fuel == "NG" or gen_spec.Fuel == "Nuclear":
            if (gen_spec.StartTimeCold <= gen_spec.MinDownTime) or \
                (gen_spec.StartTimeCold == gen_spec.StartTimeWarm == gen_spec.StartTimeHot): ## in this case, only one startup cost
                print("set StartupLags[%s] := %d ;" % (gen_id, gen_spec.MinDownTime), file=dat_file)
                print("set StartupCosts[%s] := %12.2f ;" % (gen_id, gen_spec.StartCostCold * gen_spec.FuelPrice + gen_spec.NonFuelStartCost), file=dat_file)        
            elif (gen_spec.StartTimeWarm <= gen_spec.MinDownTime): ## in this case, only two startup costs
                print("set StartupLags[%s] := %d %d ;" % (gen_id, gen_spec.MinDownTime, gen_spec.StartTimeCold), file=dat_file)
                print("set StartupCosts[%s] := %12.2f %12.2f ;" % (gen_id, gen_spec.StartCostWarm * gen_spec.FuelPrice + gen_spec.NonFuelStartCost, gen_spec.StartCostCold * gen_spec.FuelPrice + gen_spec.NonFuelStartCost), file=dat_file)
            else: # if we got here, we have all three startup costs
                print("set StartupLags[%s] := %d %d %d ;" % (gen_id, gen_spec.MinDownTime, gen_spec.StartTimeWarm, gen_spec.StartTimeCold), file=dat_file)
                print("set StartupCosts[%s] := %12.2f %12.2f %12.2f ;" % (gen_id, gen_spec.StartCostHot * gen_spec.FuelPrice + gen_spec.NonFuelStartCost, gen_spec.StartCostWarm * gen_spec.FuelPrice + gen_spec.NonFuelStartCost, gen_spec.StartCostCold * gen_spec.FuelPrice + gen_spec.NonFuelStartCost), file=dat_file)

    print("", file=dat_file)
    
    for gen_id, gen_spec in generator_dict.items():
        if gen_spec.Fuel == "Oil" or gen_spec.Fuel == "Coal" or gen_spec.Fuel == "NG" or gen_spec.Fuel == "Nuclear":
            # round the power points to the nearest 10kW
            # IMPT: These quantities are MW
            x0 = round(gen_spec.OutputPct0 * gen_spec.MaxPower,1)
            x1 = round(gen_spec.OutputPct1 * gen_spec.MaxPower,1)
            x2 = round(gen_spec.OutputPct2 * gen_spec.MaxPower,1)
            x3 = round(gen_spec.OutputPct3 * gen_spec.MaxPower,1)

            # NOTES:
            # 1) Fuel price is in $/MMBTU
            # 2) Heat Rate quantities are in BTU/KWH 
            # 3) 1+2 => need to convert both from BTU->MMBTU and from KWH->MWH
            y0 = gen_spec.FuelPrice * ((gen_spec.HeatRateAvg0 * 1000.0 / 1000000.0) * x0)
            y1 = gen_spec.FuelPrice * (((x1-x0) * (gen_spec.HeatRateIncr1 * 1000.0 / 1000000.0))) + y0
            y2 = gen_spec.FuelPrice * (((x2-x1) * (gen_spec.HeatRateIncr2 * 1000.0 / 1000000.0))) + y1
            y3 = gen_spec.FuelPrice * (((x3-x2) * (gen_spec.HeatRateIncr3 * 1000.0 / 1000000.0))) + y2

            ## for the nuclear unit
            if y0 == y3:
                ## PRESCIENT currently doesn't gracefully handle generators with zero marginal cost
                print("set CostPiecewisePoints[%s] := %12.1f %12.1f ;" % (gen_id, x0, x3),
                      file=dat_file)
                print("set CostPiecewiseValues[%s] := %12.2f %12.2f ;" % (gen_id, y0, y3+0.01),
                      file=dat_file)
            else:
                print("set CostPiecewisePoints[%s] := %12.1f %12.1f %12.1f %12.1f ;" % (gen_id, x0, x1, x2, x3),
                      file=dat_file)
                print("set CostPiecewiseValues[%s] := %12.2f %12.2f %12.2f %12.2f ;" % (gen_id, y0, y1, y2, y3),
                      file=dat_file)
    
    print("", file=dat_file)
    
    print("param: UnitOnT0State PowerGeneratedT0 :=", file=dat_file)
    for gen_id, gen_spec in generator_dict.items():
        if gen_spec.Fuel == "Sync_Cond" or gen_spec.Fuel == "Hydro" or gen_spec.Fuel == "Wind" or gen_spec.Fuel == "Solar":
            continue
        if gen_id not in unit_on_t0_state_dict:
            print("***WARNING - No T0 initial condition found for generator=%s" % gen_id)
            continue
        if unit_on_t0_state_dict[gen_id] < 0:
            power_generated_t0 = 0.0
        else:
            power_generated_t0 = gen_spec.MinPower 
        print("%15s %3d %12.2f" % (gen_id, unit_on_t0_state_dict[gen_id], power_generated_t0),
              file=dat_file)
    print(";", file=dat_file)
    
    print("", file=dat_file)
    
    dat_file.close()
    
    print("dat file written to "+str(file_name))
    
    print("")
    


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("rts_gmlc_dir", help="location of the RTS-GMLC data", type=str)
    parser.add_argument("output_file", help="specify the file name to write", type=str)
    parser.add_argument("--copper-sheet", help="don't create network", action="store_true")

    args = parser.parse_args()

    write_template( args.rts_gmlc_dir, args.output_file, args.copper_sheet, )

if __name__ == "__main__":
    sys.exit(main())


