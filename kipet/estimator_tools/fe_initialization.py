"""
FIInit is a class used to initilize a KIPET pyomo model using the newly added
incidence analysis methods in Pyomo 6.1.2. This replaces the previously used
fe_factory and fe_simulator classes.

FEInit also handles the changes to state constraints when dosing occurs.
"""

# Third party imports
from pyomo.dae import ContinuousSet
from pyomo.environ import Constraint, ConstraintList, Var, VarList
from pyomo.opt import SolverFactory

# KIPET library imports
from kipet.estimator_tools.initialize2 import initialize_by_time_element
from kipet.estimator_tools.pyomo_simulator import PyomoSimulator
from kipet.general_settings.settings import solver_path
from kipet.general_settings.variable_names import VariableNames
from kipet.model_tools.visitor_classes import ReplacementVisitor
from kipet.input_output.kipet_io import supress_stdout


class FEInit(PyomoSimulator):
    
    """Step by step initialization of a Pyomo model"""
    
    def __init__(self, model, dosing_points=None):
        
        super(FEInit, self).__init__(model)
        self.model_orig = model
        self.dosing_points = dosing_points
        self.__var = VariableNames()
        self.volume_name = self.__var.volume_name
        
        time_index = None
        for i in self.model_orig.component_objects(ContinuousSet):
            time_index = i
            break
        if time_index is None:
            raise Exception('No ContinuousSet')
        self.time_set = time_index.name
        
    
    @supress_stdout
    def initialize(self):
        """
        Calls the solve_strongly_connected_components from the incidence_analysis
        toolbox to solve element by element
        
        """
        solver = SolverFactory(solver_path("ipopt"))
        initialize_by_time_element(self.model_orig, getattr(self.model_orig, self.time_set), solver, solve_kwds={"tee": False})

        return None
        
    def add_constraints(self):
        """
        This method adds constraints to handle dosing points - this greatly
        updates the original method by not adding any new variables or
        constraints.
        
        The model must be discretized before using this method.
        
        """
        replacer = ReplacementVisitor()
        model_time_set = getattr(self.model_orig, self.time_set)
        ncp = model_time_set.get_discretization_info()['ncp']
        comp_dict = self._make_comp_dict()
        dosing_constraints = self._dosing_point_aggregation()

        for dosing_time in dosing_constraints:
            jump_fe, _ = _fe_cp(model_time_set, dosing_time)
            for kcp in range(1, ncp + 1):
                cp_time = _t_ij(model_time_set, jump_fe + 1, kcp)
                for comp in comp_dict:
                    model_var = comp_dict[comp]
                    state_constraint_name = f'd{model_var}dt_disc_eq'
                    state_constraint_object = getattr(self.model_orig, state_constraint_name)
                    replacer.change_replacement(getattr(self.model_orig, model_var)[dosing_time, comp] + dosing_constraints[dosing_time][comp])
                    target_var_object = getattr(self.model_orig, model_var)[dosing_time, comp]
                    replacer.change_suspect(id(target_var_object))
                    old_expr = state_constraint_object[cp_time, comp].expr
                    new_expr = replacer.dfs_postorder_stack(old_expr)
                    state_constraint_object[cp_time, comp].set_value(new_expr)
                    
        return None
    
       
    def _make_comp_dict(self):
        """Creates a list of tuples to pair model variables with component
        and volume variables
        
        :return list comp_list: A list of tuples
        
        """
        comp_dict = {}
        for comp in self.model_orig.mixture_components.data():
            comp_dict[comp] = self.__var.concentration_model
        comp_dict[self.volume_name] = self.__var.state_model
    
        return comp_dict
    
    
    def _dosing_point_aggregation(self):
        """
        This method checks the dosing points for time and aggregates the them into
        a single constraint change for the model. This only converts concentrations
        at the moment.

        Returns:
        
        :return dosing_constraints:
            A dictionary of the states with the state change expressions
        :rtype: dict

        """
        dosing_constraints = {}
        all_dosing_times = set()
        if self.dosing_points is not None:
            for dp in self.dosing_points[self.__var.concentration_model]:
                all_dosing_times.add(dp.time)
            
            for time in all_dosing_times:
                vol_obj = getattr(self.model_orig, self.__var.state_model)[(time,) + (self.volume_name,)]
                
                conc_obj = {}
                for comp in list(self.model_orig.mixture_components.data()):
                    conc_obj[comp] = getattr(self.model_orig, self.__var.concentration_model)[(time,) + (comp,)]
                
                moles_expr = {}
                for comp in conc_obj:
                    moles_expr[comp] = vol_obj * conc_obj[comp]
                
                volume_change_in_dosing_point = sum([dp.vol[0] for dp in self.dosing_points[self.__var.concentration_model] if dp.time == time])
                vol_obj += volume_change_in_dosing_point
            
                for dp in self.dosing_points[self.__var.concentration_model]:
                    if dp.time == time:
                        conc_change = dp.conc[0]
                        vol_change = dp.vol[0]
                        moles_expr[dp.component] += conc_change * vol_change
                        delta_conc_obj = {k: v / vol_obj - conc_obj[k] for k, v in moles_expr.items()}
                        delta_conc_obj[self.volume_name] = volume_change_in_dosing_point
                        dosing_constraints[time] = delta_conc_obj
                    
        return dosing_constraints
      
        
def _fe_cp(time_set, feedtime):
    """Return the corresponding fe and cp for a given time

    :param ContinuousSet time_set: The time index
    :param float feedtime: The time of the dosing point

    :return: tuple of the finite element and the collocation point

    """
    fe_l = time_set.get_lower_element_boundary(feedtime)
    fe = None
    j = 0
    for i in time_set.get_finite_elements():
        if fe_l == i:
            fe = j
            break
        j += 1
    h = time_set.get_finite_elements()[1] - time_set.get_finite_elements()[0]
    tauh = [i * h for i in time_set.get_discretization_info()['tau_points']]
    j = 0
    cp = None
    for i in tauh:
        if round(i + fe_l, 6) == feedtime:
            cp = j
            break
        j += 1
        
    return fe, cp


def _t_ij(time_set, i, j):
    """Return the corresponding time(continuous set) based on the i-th finite element and j-th collocation point
    From the NMPC_MHE framework by @dthierry.

    :param ContinuousSet time_set: Parent Continuous set
    :param int i: finite element
    :param int j: collocation point

    :return: Corresponding index of the ContinuousSet
    :rtype: float

    """
    if i < time_set.get_discretization_info()['nfe']:
        h = time_set.get_finite_elements()[i + 1] - time_set.get_finite_elements()[i]
    else:
        h = time_set.get_finite_elements()[i] - time_set.get_finite_elements()[i - 1]
    tau = time_set.get_discretization_info()['tau_points']
    fe = time_set.get_finite_elements()[i]
    time = fe + tau[j] * h
    
    return round(time, 6)
