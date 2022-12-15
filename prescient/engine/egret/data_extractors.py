#  ___________________________________________________________________________
#
#  Prescient
#  Copyright 2020 National Technology & Engineering Solutions of Sandia, LLC
#  (NTESS). Under the terms of Contract DE-NA0003525 with NTESS, the U.S.
#  Government retains certain rights in this software.
#  This software is distributed under the Revised BSD License.
#  ___________________________________________________________________________

from __future__ import annotations
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from typing import Iterable, Dict, Tuple
    from prescient.engine.abstract_types import *
from typing import NamedTuple

import numpy as np

from prescient.util.math_utils import round_small_values

from prescient.engine.data_extractors import ScedDataExtractor as BaseScedExtractor
from prescient.engine.data_extractors import RucDataExtractor as BaseRucExtractor
from prescient.engine.data_extractors import ReserveIdentifier

from egret.model_library.transmission.tx_utils import ancillary_services

class ScedDataExtractor(BaseScedExtractor):

    @staticmethod
    def get_sced_duration_minutes(sced: OperationsModel) -> int:
        return sced.data['system']['time_period_length_minutes']

    @staticmethod
    def get_buses(sced: OperationsModel) -> Iterable[B]:
        return sced.data['elements']['bus'].keys()

    @staticmethod
    def get_loads(sced: OperationsModel) -> Iterable[L]:
        return sced.data['elements']['load'].keys()    

    @staticmethod
    def get_transmission_lines(sced: OperationsModel) -> Iterable[L]:
        yield from sced.data['elements']['branch'].keys()
        if 'dc_branch' in sced.data['elements']:
            yield from sced.data['elements']['dc_branch'].keys()

    @staticmethod
    def get_all_storage(sced: OperationsModel) -> Iterable[S]:
        return sced.data['elements']['storage'].keys()

    @staticmethod
    def get_all_generators(sced: OperationsModel) -> Iterable[G]:
        return sced.data['elements']['generator'].keys()

    @staticmethod
    def get_thermal_generators(sced: OperationsModel) -> Iterable[G]:
        return (g for g,_ in \
                sced.elements(element_type='generator', generator_type='thermal'))

    @staticmethod
    def get_virtual_generators(sced: OperationsModel) -> Iterable[G]:
        return (g for g,_ in \
                sced.elements(element_type='generator', generator_type='virtual'))

    @staticmethod
    def get_nondispatchable_generators(sced: OperationsModel) -> Iterable[G]:
        return (g for g,_ in \
                sced.elements(element_type='generator', generator_type='renewable'))

    @staticmethod
    def get_thermal_generators_at_bus(sced: OperationsModel, b: B) -> Iterable[G]:
        return (g for g,_ in \
                sced.elements(element_type='generator', generator_type='thermal', bus=b))

    @staticmethod
    def get_virtual_generators_at_bus(sced: OperationsModel, b: B) -> Iterable[G]:
        return (g for g,_ in \
                sced.elements(element_type='generator', generator_type='virtual', bus=b))

    @staticmethod
    def get_nondispatchable_generators_at_bus(sced: OperationsModel, b: B) -> Iterable[G]:
        return (g for g,_ in \
                sced.elements(element_type='generator', generator_type='renewable', bus=b))

    @staticmethod
    def get_quickstart_generators(sced: OperationsModel) -> Iterable[G]:
        return (g for g,_ in \
                sced.elements(element_type='generator', fast_start=True))

    @staticmethod
    def get_generator_bus(sced: OperationsModel, g: G) -> B:
        return sced.data['elements']['generator'][g]['bus']

    @staticmethod
    def is_generator_on(sced: OperationsModel, g: G) -> bool:
        g_dict = sced.data['elements']['generator'][g]
        if 'fixed_commitment' in g_dict:
            if isinstance(g_dict['fixed_commitment'], dict):
                return g_dict['fixed_commitment']['values'][0] > 0
            else:
                return g_dict['fixed_commitment'] > 0
        elif 'commitment' in g_dict:
            return g_dict['commitment']['values'][0] > 0
        else:
            raise RuntimeError(f"Can't find commitment status for generator {g}")

    @staticmethod
    def generator_was_on(sced: OperationsModel, g: G) -> bool:
        return sced.data['elements']['generator'][g]['initial_status'] > 0

    @staticmethod
    def get_fixed_costs(sced: OperationsModel) -> float:
        total = 0.
        for g,g_dict in sced.elements(element_type='generator', generator_type='thermal'):
            total += g_dict['commitment_cost']['values'][0]
        return total

    @staticmethod
    def get_variable_costs(sced: OperationsModel) -> float:
        total = 0.
        for g,g_dict in sced.elements(element_type='generator', generator_type='thermal'):
            total += g_dict['production_cost']['values'][0]
        return total

    @staticmethod
    def get_power_generated(sced: OperationsModel, g: G) -> float:
        return sced.data['elements']['generator'][g]['pg']['values'][0]

    @staticmethod
    def get_power_generated_T0(sced: OperationsModel, g: G) -> float:
        return sced.data['elements']['generator'][g]['initial_p_output']

    @staticmethod
    def get_load_mismatch(sced: OperationsModel, b: B) -> float:
        return sced.data['elements']['bus'][b]['p_balance_violation']['values'][0]

    @staticmethod
    def get_positive_load_mismatch(sced: OperationsModel, b: B):
        val = ScedDataExtractor.get_load_mismatch(sced, b)
        if val > 0:
            return val
        return 0

    @staticmethod
    def get_negative_load_mismatch(sced: OperationsModel, b: B):
        val = ScedDataExtractor.get_load_mismatch(sced, b)
        if val < 0:
            return -val
        return 0

    @staticmethod
    def get_max_power_output(sced: OperationsModel, g: G) -> float:
        p_max = sced.data['elements']['generator'][g]['p_max']
        if isinstance(p_max, dict):
            p_max = p_max['values'][0]
        return p_max

    @staticmethod
    def get_max_power_available(sced: OperationsModel, g: G) -> float:
        gdata = sced.data['elements']['generator'][g]
        val = gdata['headroom']['values'][0] \
                + ScedDataExtractor.get_power_generated(sced, g)*int(ScedDataExtractor.is_generator_on(sced,g))
        if 'reserve_supplied' in gdata:
            val += gdata['reserve_supplied']['values'][0]
        return val

    @staticmethod
    def get_thermal_headroom(sced: OperationsModel, g: G) -> float:
        gdata = sced.data['elements']['generator'][g]
        val = gdata['headroom']['values'][0]
        if 'reserve_supplied' in gdata:
            val += gdata['reserve_supplied']['values'][0]
        return val

    @staticmethod
    def get_implicit_thermal_headroom(sced: OperationsModel, g: G) -> float:
        return sced.data['elements']['generator']['headroom']['values'][0]

    @staticmethod
    def get_thermal_reserve_provided(sced: OperationsModel,
                                     res: ReserveIdentifier, g: G) -> float:
        if not ScedDataExtractor.generator_is_in_scope(g, res.region_type, res.region_name):
            return 0.0
        attname = f'{res.reserve_name}_provided'
        gdata = sced.data['elements']['generator'][g]
        return gdata.get(attname, 0.0)

    @staticmethod
    def get_min_downtime(sced: OperationsModel, g: G) -> float:
        return sced.data['elements']['generator'][g]['min_down_time']

    @staticmethod
    def get_scaled_startup_ramp_limit(sced: OperationsModel, g: G) -> float:
        return sced.data['elements']['generator'][g]['startup_capacity']

    @staticmethod
    def get_generator_fuel(sced: OperationsModel, g: G) -> str:
        return sced.data['elements']['generator'][g].get('fuel', 'Other')
    
    @staticmethod
    def generator_is_in_scope(g_dict:dict, region_type:str, region_name:str):
        ''' Whether a generator is part of a scope (system, zone, or area)
        '''
        if region_type == 'system':
            return True
        return hasattr(g_dict, region_type) and g_dict[region_type] == region_name

    @staticmethod
    def get_reserve_products(sced: OperationsModel) -> Iterable[ReserveIdentifier]:
        def get_scope_reserve_products(region_type, region_name, data):
            for svc in ancillary_services:
                if f'{svc}_requirement' in data:
                    yield ReserveIdentifier(region_type, region_name, svc)

        yield from get_scope_reserve_products('system', None, sced.data['system'])
        for region_type in ('area', 'zone'):
            for name, data in sced.elements(element_type=region_type):
                yield from get_scope_reserve_products(region_type, name, data)

    @staticmethod
    def _get_reserve_parent(sced: OperationsModel, 
                            reserve_id: ReserveIdentifier
                           ) -> dict:
        ''' Get the data dict holding properties related to this reserve
        '''
        if reserve_id.region_type == 'system':
            return sced.data['system']
        else:
            return sced.data['elements'][reserve_id.region_type][reserve_id.region_name]


    @staticmethod
    def _get_reserve_property(sced: OperationsModel, 
                              reserve_id: ReserveIdentifier,
                              suffix: str) -> float:
        ''' Get the value of a particular reserve property.

            Reserve property name must follow a standard convention,
            f'{reserve_id.reserve_name}{suffix}'.
        ''' 
        data = ScedDataExtractor._get_reserve_parent(sced, reserve_id)
        attr = f'{reserve_id.reserve_name}{suffix}'
        if attr in data:
            if isinstance(data[attr], dict):
                return round_small_values(data[attr]['values'][0])
            else:
                return round_small_values(data[attr])
        else:
            return 0.

    @staticmethod
    def get_reserve_requirement(
                                sced: OperationsModel, 
                                reserve_id: ReserveIdentifier) -> float:
        return ScedDataExtractor._get_reserve_property(sced, reserve_id, "_requirement")

    @staticmethod
    def get_reserve_shortfall(
                              sced: OperationsModel, 
                              reserve_id: ReserveIdentifier
                              ) -> float:
        return ScedDataExtractor._get_reserve_property(sced, reserve_id, "_shortfall")

    @staticmethod
    def get_reserve_RT_price(
                             lmp_sced: OperationsModel, 
                             reserve_id: ReserveIdentifier) -> float:
        return ScedDataExtractor._get_reserve_property(lmp_sced, reserve_id, "_price")

    @staticmethod
    def get_max_nondispatchable_power(sced: OperationsModel, g: G) -> float:
        p_max = sced.data['elements']['generator'][g]['p_max']
        if isinstance(p_max, dict):
            p_max = p_max['values'][0]
        return p_max

    @staticmethod
    def get_min_nondispatchable_power(sced: OperationsModel, g: G) -> float:
        p_min = sced.data['elements']['generator'][g]['p_min']
        if isinstance(p_min, dict):
            p_min = p_min['values'][0]
        return p_min

    @staticmethod
    def get_nondispatchable_power_used(sced: OperationsModel, g: G) -> float:
        return sced.data['elements']['generator'][g]['pg']['values'][0]

    @staticmethod
    def get_total_demand(sced: OperationsModel) -> float:
        total = 0.
        for l, l_dict in sced.elements(element_type='load'):
            total += l_dict['p_load']['values'][0]
        return total

    @staticmethod
    def get_generator_cost(sced: OperationsModel, g: G) -> float:
        return sced.data['elements']['generator'][g]['commitment_cost']['values'][0] + \
               sced.data['elements']['generator'][g]['production_cost']['values'][0]

    @staticmethod 
    def _get_line_dict(sced: OperationsModel, line: L) -> dict:
        return (sced.data['elements']['branch'].get(line)
                or sced.data['elements']['dc_branch'].get(line)
        )

    @staticmethod
    def get_flow_level(sced: OperationsModel, line: L) -> float:
        return ScedDataExtractor._get_line_dict(sced, line)['pf']['values'][0]

    @staticmethod
    def get_flow_violation_level(sced: OperationsModel, line: L) -> float:
        pf_violation = ScedDataExtractor._get_line_dict(sced, line).get('pf_violation', 0.)
        if pf_violation != 0.:
            pf_violation = pf_violation['values'][0]
        return pf_violation

    @staticmethod
    def get_all_contingency_flow_levels(sced: OperationsModel) -> Dict[Tuple[L,L], float]:
        contingency_dict = {}
        for c_dict in sced.data['elements'].get('contingency', {}).values():
            line_out = c_dict['branch_contingency']
            monitored_branches = c_dict.get('monitored_branches',{'values':[{}]})
            for bn, b_dict in monitored_branches['values'][0].items():
                contingency_dict[line_out, bn] = b_dict['pf']
        return contingency_dict

    @staticmethod
    def get_all_contingency_flow_violation_levels(sced: OperationsModel) -> Dict[Tuple[L,L], float]:
        contingency_viol = {}
        for c_dict in sced.data['elements'].get('contingency', {}).values():
            line_out = c_dict['branch_contingency']
            monitored_branches = c_dict.get('monitored_branches',{'values':[{}]})
            for bn, b_dict in monitored_branches['values'][0].items():
                contingency_viol[line_out, bn] = b_dict.get('pf_violation', 0.)
        return contingency_viol

    @staticmethod
    def get_bus_mismatch(sced: OperationsModel, bus: B) -> float:
        return ScedDataExtractor.get_load_mismatch(sced, bus)

    @staticmethod
    def get_storage_input_dispatch_level(sced: OperationsModel, storage: S) -> float:
        return sced.data['elements']['storage'][s]['p_charge']['values'][0]

    @staticmethod
    def get_storage_output_dispatch_level(sced: OperationsModel, storage: S) -> float:
        return sced.data['elements']['storage'][s]['p_discharge']['values'][0]

    @staticmethod
    def get_storage_soc_dispatch_level(sced: OperationsModel, storage: S) -> float:
        return sced.data['elements']['storage'][s]['state_of_charge']['values'][0]

    @staticmethod
    def get_storage_type(sced: OperationsModel, storage: S) -> str:
        if 'fuel' in sced.data['elements']['storage'][s]:
            return sced.data['elements']['storage'][s]['fuel']
        return 'Other'

    @staticmethod
    def get_bus_demand(sced: OperationsModel, bus: B) -> float:
        ''' get the demand on a bus in a given time period '''
        return sced.data['elements']['bus'][bus]['pl']['values'][0]

    @staticmethod
    def get_load_bus(sced: OperationsModel, load: L) -> float:
        ''' get the bus associated with a given load '''
        return sced.data['elements']['load'][load]['bus']

    @staticmethod
    def get_load_demand(sced: OperationsModel, load: L) -> float:
        ''' get the demand associated with a load in a given time period '''
        return sced.data['elements']['load'][load]['p_load']['values'][0]    

    @staticmethod
    def get_bus_LMP(lmp_sced: OperationsModel, bus: B) -> float:
        return lmp_sced.data['elements']['bus'][bus]['lmp']['values'][0]


class RucDataExtractor(BaseRucExtractor):
    """
    Extracts information from RUC instances
    """

    def get_num_time_periods(self, ruc: RucModel) -> int:
        ''' Get the number of time periods for which data is available.
            
            Time periods are numbered 1..N, where N is the value returned by this method.
        '''
        return len(ruc.data['system']['time_keys'])

    def get_buses(self, ruc: RucModel) -> Iterable[B]:
        ''' Get all buses in the model '''
        return ruc.data['elements']['bus'].keys()

    def get_bus_demand(self, ruc: RucModel, bus: B, time: int) -> float:
        ''' get the demand on a bus in a given time period '''
        return ruc.data['elements']['bus'][bus]['pl']['values'][time-1]

    def get_nondispatchable_generators(self, ruc: RucModel) -> Iterable[G]:
        return (g for g,_ in \
                ruc.elements(element_type='generator', generator_type='renewable'))

    def get_min_nondispatchable_power(self, ruc: RucModel, gen: G, time: int) -> float:
        p_min = ruc.data['elements']['generator'][gen]['p_min']
        if isinstance(p_min, dict):
            p_min = p_min['values'][time-1]
        return p_min

    def get_max_nondispatchable_power(self, ruc: RucModel, gen: G, time: int) -> float:
        p_max = ruc.data['elements']['generator'][gen]['p_max']
        if isinstance(p_max, dict):
            p_max = p_max['values'][time-1]
        return p_max
