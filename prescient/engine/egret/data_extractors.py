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

import numpy as np

from prescient.util.math_utils import round_small_values

from prescient.engine.data_extractors import ScedDataExtractor as BaseScedExtractor
from prescient.engine.data_extractors import RucDataExtractor as BaseRucExtractor


class ScedDataExtractor(BaseScedExtractor):

    def get_sced_duration_minutes(self, sced: OperationsModel) -> int:
        return sced.data['system']['time_period_length_minutes']

    def get_buses(self, sced: OperationsModel) -> Iterable[B]:
        return sced.data['elements']['bus'].keys()

    def get_loads(self, sced: OperationsModel) -> Iterable[L]:
        return sced.data['elements']['load'].keys()    

    def get_transmission_lines(self, sced: OperationsModel) -> Iterable[L]:
        return sced.data['elements']['branch'].keys()

    def get_all_storage(self, sced: OperationsModel) -> Iterable[S]:
        return sced.data['elements']['storage'].keys()

    def get_all_generators(self, sced: OperationsModel) -> Iterable[G]:
        return sced.data['elements']['generator'].keys()

    def get_thermal_generators(self, sced: OperationsModel) -> Iterable[G]:
        return (g for g,_ in \
                sced.elements(element_type='generator', generator_type='thermal'))

    def get_virtual_generators(self, sced: OperationsModel) -> Iterable[G]:
        return (g for g,_ in \
                sced.elements(element_type='generator', generator_type='virtual'))

    def get_nondispatchable_generators(self, sced: OperationsModel) -> Iterable[G]:
        return (g for g,_ in \
                sced.elements(element_type='generator', generator_type='renewable'))

    def get_thermal_generators_at_bus(self, sced: OperationsModel, b: B) -> Iterable[G]:
        return (g for g,_ in \
                sced.elements(element_type='generator', generator_type='thermal', bus=b))

    def get_virtual_generators_at_bus(self, sced: OperationsModel, b: B) -> Iterable[G]:
        return (g for g,_ in \
                sced.elements(element_type='generator', generator_type='virtual', bus=b))

    def get_nondispatchable_generators_at_bus(self, sced: OperationsModel, b: B) -> Iterable[G]:
        return (g for g,_ in \
                sced.elements(element_type='generator', generator_type='renewable', bus=b))

    def get_quickstart_generators(self, sced: OperationsModel) -> Iterable[G]:
        return (g for g,_ in \
                sced.elements(element_type='generator', fast_start=True))

    def get_generator_bus(self, sced: OperationsModel, g: G) -> B:
        return sced.data['elements']['generator'][g]['bus']

    def is_generator_on(self, sced: OperationsModel, g: G) -> bool:
        g_dict = sced.data['elements']['generator'][g]
        if 'fixed_commitment' in g_dict:
            return g_dict['fixed_commitment']['values'][0] > 0
        elif 'commitment' in g_dict:
            return g_dict['commitment']['values'][0] > 0
        else:
            raise RuntimeError(f"Can't find commitment status for generator {g}")

    def generator_was_on(self, sced: OperationsModel, g: G) -> bool:
        return sced.data['elements']['generator'][g]['initial_status'] > 0

    def get_fixed_costs(self, sced: OperationsModel) -> float:
        total = 0.
        for g,g_dict in sced.elements(element_type='generator', generator_type='thermal'):
            total += g_dict['commitment_cost']['values'][0]
        return total

    def get_variable_costs(self, sced: OperationsModel) -> float:
        total = 0.
        for g,g_dict in sced.elements(element_type='generator', generator_type='thermal'):
            total += g_dict['production_cost']['values'][0]
        return total

    def get_power_generated(self, sced: OperationsModel, g: G) -> float:
        return sced.data['elements']['generator'][g]['pg']['values'][0]

    def get_power_generated_T0(self, sced: OperationsModel, g: G) -> float:
        return sced.data['elements']['generator'][g]['initial_p_output']

    def get_load_mismatch(self, sced: OperationsModel, b: B) -> float:
        return sced.data['elements']['bus'][b]['p_balance_violation']['values'][0]

    def get_positive_load_mismatch(self, sced: OperationsModel, b: B):
        val = self.get_load_mismatch(sced, b)
        if val > 0:
            return val
        return 0

    def get_negative_load_mismatch(self, sced: OperationsModel, b: B):
        val = self.get_load_mismatch(sced, b)
        if val < 0:
            return -val
        return 0

    def get_max_power_output(self, sced: OperationsModel, g: G) -> float:
        p_max = sced.data['elements']['generator'][g]['p_max']
        if isinstance(p_max, dict):
            p_max = p_max['values'][0]
        return p_max

    def get_max_power_available(self, sced: OperationsModel, g: G) -> float:
        gdata = sced.data['elements']['generator'][g]
        val = gdata['headroom']['values'][0] \
                + self.get_power_generated(sced, g)*int(self.is_generator_on(sced,g))
        if 'rg' in gdata:
            val += sced.data['elements']['generator'][g]['rg']['values'][0]
        return val

    def get_thermal_headroom(self, sced: OperationsModel, g: G) -> float:
        gdata = sced.data['elements']['generator'][g]
        val = gdata['headroom']['values'][0]
        if 'rg' in gdata:
            val += gdata['rg']['values'][0]
        return val

    def get_min_downtime(self, sced: OperationsModel, g: G) -> float:
        return sced.data['elements']['generator'][g]['min_down_time']

    def get_scaled_startup_ramp_limit(self, sced: OperationsModel, g: G) -> float:
        return sced.data['elements']['generator'][g]['startup_capacity']

    def get_generator_fuel(self, sced: OperationsModel, g: G) -> str:
        return sced.data['elements']['generator'][g].get('fuel', 'Other')

    def get_reserve_shortfall(self, sced: OperationsModel) -> float:
        if 'reserve_shortfall' in sced.data['system']:
            return round_small_values(sced.data['system']['reserve_shortfall']['values'][0])
        else:
            return 0.

    def get_max_nondispatchable_power(self, sced: OperationsModel, g: G) -> float:
        p_max = sced.data['elements']['generator'][g]['p_max']
        if isinstance(p_max, dict):
            p_max = p_max['values'][0]
        return p_max

    def get_min_nondispatchable_power(self, sced: OperationsModel, g: G) -> float:
        p_min = sced.data['elements']['generator'][g]['p_min']
        if isinstance(p_min, dict):
            p_min = p_min['values'][0]
        return p_min

    def get_nondispatchable_power_used(self, sced: OperationsModel, g: G) -> float:
        return sced.data['elements']['generator'][g]['pg']['values'][0]

    def get_total_demand(self, sced: OperationsModel) -> float:
        total = 0.
        for l, l_dict in sced.elements(element_type='load'):
            total += l_dict['p_load']['values'][0]
        return total

    def get_reserve_requirement(self, sced: OperationsModel) -> float:
        if 'reserve_requirement' in sced.data['system']:
            return sced.data['system']['reserve_requirement']['values'][0]
        else:
            return 0. 

    def get_generator_cost(self, sced: OperationsModel, g: G) -> float:
        return sced.data['elements']['generator'][g]['commitment_cost']['values'][0] + \
               sced.data['elements']['generator'][g]['production_cost']['values'][0]

    def get_flow_level(self, sced: OperationsModel, line: L) -> float:
        return sced.data['elements']['branch'][line]['pf']['values'][0]

    def get_flow_violation_level(self, sced: OperationsModel, line: L) -> float:
        pf_violation = sced.data['elements']['branch'][line].get('pf_violation', 0.)
        if pf_violation != 0.:
            pf_violation = pf_violation['values'][0]
        return pf_violation

    def get_all_contingency_flow_levels(self, sced: OperationsModel) -> Dict[Tuple[L,L], float]:
        contingency_dict = {}
        for c_dict in sced.data['elements'].get('contingency', {}).values():
            line_out = c_dict['branch_contingency']
            monitored_branches = c_dict.get('monitored_branches',{'values':[{}]})
            for bn, b_dict in monitored_branches['values'][0].items():
                contingency_dict[line_out, bn] = b_dict['pf']
        return contingency_dict

    def get_all_contingency_flow_violation_levels(self, sced: OperationsModel) -> Dict[Tuple[L,L], float]:
        contingency_viol = {}
        for c_dict in sced.data['elements'].get('contingency', {}).values():
            line_out = c_dict['branch_contingency']
            monitored_branches = c_dict.get('monitored_branches',{'values':[{}]})
            for bn, b_dict in monitored_branches['values'][0].items():
                contingency_viol[line_out, bn] = b_dict.get('pf_violation', 0.)
        return contingency_viol

    def get_bus_mismatch(self, sced: OperationsModel, bus: B) -> float:
        return self.get_load_mismatch(sced, bus)

    def get_storage_input_dispatch_level(self, sced: OperationsModel, storage: S) -> float:
        return sced.data['elements']['storage'][s]['p_charge']['values'][0]

    def get_storage_output_dispatch_level(self, sced: OperationsModel, storage: S) -> float:
        return sced.data['elements']['storage'][s]['p_discharge']['values'][0]

    def get_storage_soc_dispatch_level(self, sced: OperationsModel, storage: S) -> float:
        return sced.data['elements']['storage'][s]['state_of_charge']['values'][0]

    def get_storage_type(self, sced: OperationsModel, storage: S) -> str:
        if 'fuel' in sced.data['elements']['storage'][s]:
            return sced.data['elements']['storage'][s]['fuel']
        return 'Other'

    def get_bus_demand(self, sced: OperationsModel, bus: B) -> float:
        ''' get the demand on a bus in a given time period '''
        return sced.data['elements']['bus'][bus]['pl']['values'][0]

    def get_load_bus(self, sced: OperationsModel, load: L) -> float:
        ''' get the bus associated with a given load '''
        return sced.data['elements']['load'][load]['bus']

    def get_load_demand(self, sced: OperationsModel, load: L) -> float:
        ''' get the demand associated with a load in a given time period '''
        return sced.data['elements']['load'][load]['p_load']['values'][0]    

    def get_reserve_RT_price(self, lmp_sced: OperationsModel) -> float:
        if 'reserve_price' in lmp_sced.data['system']:  
            return lmp_sced.data['system']['reserve_price']['values'][0]
        else:
            return 0.

    def get_bus_LMP(self, lmp_sced: OperationsModel, bus: B) -> float:
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
