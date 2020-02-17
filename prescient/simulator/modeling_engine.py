from __future__ import annotations
from abc import ABC, abstractmethod
from typing import TypeVar, Iterable, Optional, Mapping, Tuple, Union

from .options import Options
import prescient.stats.stats_extractors

OperationsModel = TypeVar('OperationsModel')
RucModel = TypeVar('RucModel')
ScenarioTree = TypeVar('ScenarioTree')

Bus = TypeVar('Bus')
Generator = TypeVar('Generator')

class ModelingEngine(ABC):
    '''
    Provides model manipulation and solving capabilities
    '''

    @abstractmethod
    def initialize(self, options:Options) -> None:
        pass

    @abstractmethod
    def create_and_solve_deterministic_ruc(self,
            options: Options,
            uc_date:str,
            uc_hour: int,
            next_uc_date: Optional[str],
            prior_ruc_instance: RucModel,
            prior_scenario_tree: ScenarioTree,
            output_ruc_initial_conditions: bool,
            projected_sced_instance: OperationsModel,
            sced_schedule_hour: int,
            ruc_horizon: int,
            run_ruc_with_next_day_data: bool
           ) -> Tuple[RucModel, ScenarioTree]:
        pass


    @abstractmethod
    def create_ruc_instance_to_simulate_next_period(
            self,
            options: Options,
            uc_date: str,
            uc_hour: int,
            next_uc_date: Optional[str]
           ) -> RucModel:
        pass


    @abstractmethod
    def create_and_solve_sced_instance(self,
            deterministic_ruc_instance_for_this_period: RucModel,
            scenario_tree_for_this_period: ScenarioTree,
            deterministic_ruc_instance_for_next_period: RucModel,
            scenario_tree_for_next_period: ScenarioTree,
            ruc_instance_to_simulate_this_period: RucModel,
            prior_sced_instance: OperationsModel,
            actual_demand: Mapping[Tuple[Bus, int], float],
            demand_forecast_error: Mapping[Tuple[Bus, int], float],
            actual_min_renewables: Mapping[Tuple[Generator, int], float],
            actual_max_renewables: Mapping[Tuple[Generator, int], float],
            renewables_forecast_error: Mapping[Tuple[Generator, int], float],
            hour_to_simulate: int,
            reserve_factor: float,
            options: Options,
            hours_in_objective: int=1,
            sced_horizon: int=24,
            ruc_every_hours: int=24,
            initialize_from_ruc: bool=True,
            use_prescient_forecast_error: bool=True,
            use_persistent_forecast_error: bool=False,
            write_sced_instance: bool = False,
            output_initial_conditions: bool = False,
            output_demands: bool = False
            ) -> Tuple[OperationsModel, float]:
        '''
        Create a new operations model and solve it.

        Although the last three arguments are present in the options argument, they should not be taken 
        directly from that object.  That is because they only apply to a "normal" sced, not a projected 
        sced, and this method is called for both (and could conceivably be called for other purposes as
        well).  The engine doesn't have any context to know which type of sced is being requested.

        Returns
        -------
        The operations model with results, and the solve time in seconds.
        '''
        pass

    def enable_quickstart_and_solve(self,
            sced_instance: OperationsModel,
            options: Options
           ) -> OperationsModel:
        '''
        Re-solve the passed in model with quickstart generators enabled.  The existing model 
        is allowed to be modified in-place rather than creating a new model if it is efficient
        to do so.  The returned model may be the existing model with quick start modifications
        and results, or it may be a new model.
        '''
        pass


    @abstractmethod
    def create_and_solve_lmp(self,
            sced_instance: OperationsModel,
            options:Options,
           ) -> OperationsModel:
        '''
        Create and solve a variation of the passed in model that honors price thesholds
        '''
        pass

    @property
    @abstractmethod
    def RucStatsExtractor(self) -> stats_extractors.RucStatsExtractor:
        ''' An object that extracts statistics from a RUC model '''
        pass

    @property
    @abstractmethod
    def OperationsStatsExtractor(self) -> stats_extractors.OperationsStatsExtractor:
        ''' An object that extracts statistics from a solved operations model '''
        pass
