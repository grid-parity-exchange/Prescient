#  ___________________________________________________________________________
#
#  Prescient
#  Copyright 2020 National Technology & Engineering Solutions of Sandia, LLC
#  (NTESS). Under the terms of Contract DE-NA0003525 with NTESS, the U.S.
#  Government retains certain rights in this software.
#  This software is distributed under the Revised BSD License.
#  ___________________________________________________________________________

from collections import namedtuple

PTDFOptions = namedtuple('PTDFOptions', ['ruc', 'dap', 'la_ed', 'mip_ed', 'lmp_ed'])

LowConfidenceOptions = PTDFOptions(ruc = {},
                                   dap = {},
                                   la_ed = {},
                                   mip_ed = {'pre_lp_iteration_limit':0},
                                   lmp_ed = {'pre_lp_iteration_limit':0},
                                   )

MediumConfidenceOptions = PTDFOptions(ruc = {'pre_lp_iteration_limit':0, 'lp_iteration_limit':5},
                                      dap = {'pre_lp_iteration_limit':0},
                                      la_ed = {'pre_lp_iteration_limit':0, 'lp_iteration_limit':3},
                                      mip_ed = {'pre_lp_iteration_limit':0, 'lp_iteration_limit':0},
                                      lmp_ed = {'pre_lp_iteration_limit':0},
                                      )

HighConfidenceOptions = PTDFOptions(ruc = {'pre_lp_iteration_limit':0, 'lp_iteration_limit':0},
                                    dap = {'pre_lp_iteration_limit':0},
                                    la_ed = {'pre_lp_iteration_limit':0, 'lp_iteration_limit':0},
                                    mip_ed = {'pre_lp_iteration_limit':0, 'lp_iteration_limit':0},
                                    lmp_ed = {'pre_lp_iteration_limit':0},
                                    )

class PTDFManager:
    ''' helper for managing the initial set of active
        constraints handed to EGRET '''
    def __init__(self, inactive_limit=5):
        # stored PTDF_matrix_dict for Egret
        self.PTDF_matrix_dict = None

        # dictionary with activate constraints
        # keys are some immutable unique constraint 
        # idenifier (e.g., line name), values are
        # the number of times since last active
        self._active_branch_constraints = {}
        self._active_interface_constraints = {}

        # constrants leave the active set if we
        # haven't seen them after this many cycles
        self._inactive_limit = inactive_limit

        self._ptdf_options = LowConfidenceOptions
        self._calls_since_last_miss = 0

    def mark_active(self, md):
        for bn, branch in md.elements(element_type='branch'):
            if bn in self._active_branch_constraints:
                branch['lazy'] = False
        for i_n, interface in md.elements(element_type='branch'):
            if i_n in self._active_interface_constraints:
                interface['lazy'] = False

    def update_active(self, md):
        # increment active
        for bn in self._active_branch_constraints:
            self._active_branch_constraints[bn] += 1
        for i_n in self._active_interface_constraints:
            self._active_interface_constraints[i_n] += 1

        misses = 0
        for bn, branch in md.elements(element_type='branch'):
            if _at_limit(branch['pf']['values'],branch['rating_long_term']):
                # we're seeing it now, so reset its counter
                # or make a new one
                if bn not in self._active_branch_constraints:
                    misses += 1
                self._active_branch_constraints[bn] = 0
        for i_n, interface in md.elements(element_type='interface'):
            if _at_limit(interface['pf']['values'],interface['lower_limit'],interface['upper_limit']):
                if i_n not in self._active_interface_constraints:
                    misses += 1
                self._active_interface_constraints[i_n] = 0

        self._update_confidence(misses)

        _del_inactive(self._active_branch_constraints, self._inactive_limit)
        _del_inactive(self._active_interface_constraints, self._inactive_limit)

        #print(f"Current set of activite lines: {self._active_branch_constraints.keys()}")

    def _update_confidence(self, misses):
        if misses == 0:
            self._calls_since_last_miss += 1
            if self._calls_since_last_miss > 100:
                #print("SETTING HIGH CONFIDENCE")
                self._ptdf_options = HighConfidenceOptions
            else:
                #print("SETTING MEDIUM CONFIDENCE")
                self._ptdf_options = MediumConfidenceOptions
        elif misses < 5:
            self._calls_since_last_miss = 0
            #print("SETTING MEDIUM CONFIDENCE")
            self._ptdf_options = MediumConfidenceOptions
        else:
            self._calls_since_last_miss = 0
            #print("SETTING LOW CONFIDENCE")
            self._ptdf_options = LowConfidenceOptions

    @property
    def ruc_ptdf_options(self):
        return self._ptdf_options.ruc

    @property
    def damarket_ptdf_options(self):
        return self._ptdf_options.dap

    @property
    def look_ahead_sced_ptdf_options(self):
        return self._ptdf_options.la_ed

    @property
    def sced_ptdf_options(self):
        return self._ptdf_options.mip_ed

    @property
    def lmpsced_ptdf_options(self):
        return self._ptdf_options.lmp_ed

def _at_limit(power_flow_list, limit, limit_eps=1e-2):
    limit = limit-limit_eps
    for f in power_flow_list:
        if abs(f) > limit:
            return True
    return False

def _at_two_sided_limit(power_flow_list, lb, ub, limit_eps=1e-2):
    lb = lb+limit_eps
    ub = ub-limit_eps
    for f in power_flow_list:
        if lb > f or f > ub:
            return True
    return False

def _del_inactive(tracker, limit):
    hit_list = [ k for k,v in tracker.items() if v > limit ]
    for k in hit_list:
        del tracker[k]
