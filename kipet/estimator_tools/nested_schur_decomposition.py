"""
This is a new NSD module for use with IPOPT


TODO: 
    
    1. define the global and local parameters - DONE
    2. check if scaling still works - may not be needed
    
    3. get cyipopt working with this
    
    4. test cyipopt with multiprocessing
    5. speed improvements?
    6. performance with bigger problems (spectra?)
    7. clean and wrap into final problem
    8. add other metrics for optimization (Tom's problems)
    9. report findings
    10. write paper

"""
# Standard library imports
from multiprocessing import set_start_method, cpu_count
import os
import platform
import re
import time

# Third party imports
import cyipopt
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.offline import plot
from pyomo.contrib.pynumero.interfaces.pyomo_nlp import PyomoNLP

from pyomo.environ import (
    Suffix,
    Var,
    Param,
    SolverFactory,
    )
from scipy.optimize import (
    Bounds,
    minimize,
    )
from scipy.sparse.linalg import spsolve
from scipy.sparse import coo_matrix, csc_matrix, lil_matrix, hstack, vstack, bmat, triu

# KIPET library imports
from kipet.estimator_tools.multiprocessing_kipet import Multiprocess
#from kipet.estimator_tools.reduced_hessian_methods import add_global_constraints
from kipet.estimator_tools.results_object import ResultsObject        
from kipet.model_tools.pyomo_model_tools import get_vars

from kipet.input_output.kipet_io import print_margin
#from kipet.estimator_tools.reduced_hessian_methods import reduced_hessian, build_matrices, calculate_reduced_hessian, prepare_global_constraints, optimize_model, optimize_model_cyipopt, generate_parameter_object_lists, generate_parameter_index_lists
from kipet.general_settings.solver_settings import solver_path, SolverSettings

import kipet.estimator_tools.reduced_hessian_methods as rhm
from kipet.estimator_tools.nsd_step import NSD_Model

DEBUG = True


class NSD:

    """This handles all methods related to solving multiple scenarios with NSD strategies"""
    
    def __init__(self, 
                 models_dict,
                 strategy='ipopt',
                 init=None, 
                 global_parameters=None,
                 kwargs=None,
                 scaled=False,
                 parallel=True,
                 rh_method='pynumero',
                 rs_min_variance=1,
                 all_params=None,
                 free_params=None,
                 local_as_global=False,
                 proj=True
                 ):
        
        print_margin('Nested Schur Decomposition (NSD)')
        
        print(f'# NSD: Initializing models and variables')
        
        self.use_cyipopt = False
        self.use_full_KKT = False
        self.use_k_aug = False
        
        if self.use_cyipopt:
            self.use_k_aug = False
        
        self.model_vars = ['P', 'S']
        
        kwargs = kwargs if kwargs is not None else {}
        #parameter_name = kwargs.get('parameter_name', 'P')
        self.objective_multiplier = kwargs.get('objective_multiplier', 1)
        self.rs_min_variance = rs_min_variance
        
        self.is_projected = proj
        self.isKipetModel = kwargs.get('kipet', True)
        self.scaled = scaled
        self.reduced_hessian_kwargs = {}
        self.reaction_models = models_dict
        #pre_solve = False
        self._model_preparation(use_duals=True)
        self.model_dict = {name: r.p_model for name, r in models_dict.items()}
        self._nlps = [m._nlp for m in self.reaction_models.values()]
        
        self.all_params = all_params
        #self.free_params = free_params
        self.local_as_global = local_as_global
        self.final_param = {}
        
        if len(self.all_params.unique_globals()) == 0:
            raise ValueError('No global variables are declared.')
        #if not self.local_as_global:
            #self.free_params = [param for param in self.all_params.parameters if param.is_global]
        self.free_params = self.all_params.unique_globals(self.local_as_global)
            
        avg_param = calculate_parameter_averages(self.model_dict)
        self.strategy = strategy
        self.method = 'trust-constr'
        
        all_parameters = []
        for model in self.model_list:
            for param in model.P.keys():
                if param not in all_parameters:
                    all_parameters.append(param)
        
        self.parameter_names = all_parameters
        self.parameter_global = [v.identity for v in self.free_params.values()]
        self.d_init = {p.identity: [p.pyomo_var.value, p.pyomo_var.lb, p.pyomo_var.ub] for p in self.free_params.values()}
        self.d_init_unscaled = None
        self.d_iter = []
        self.parallel = parallel
        
        self.rh_method = rh_method
        
        if init is not None:
            for k, v in init.items():
                if k in global_parameters:
                    self.d_init[k][0] = v
        
        # Initialization of NSD algorithm variables
        self.model_number = {k: i for i, k in enumerate(self.reaction_models.keys())}
        self.x = [0 for k, v in self.d_init.items()]
        self.d = {k : 0 for k in self.reaction_models}
        self.M = pd.DataFrame(np.zeros((len(self.parameter_global), len(self.parameter_global))), index=self.parameter_global, columns=self.parameter_global)
        self.m = pd.DataFrame(np.zeros((len(self.parameter_global), 1)), index=self.parameter_global, columns=['dual'])
        self.g = pd.DataFrame(np.zeros((len(self.parameter_global), 1)), index=self.parameter_global, columns=['grad'])
        self.duals = [0 for i in range(len(self.reaction_models))]
        self.grad = [0 for i in range(len(self.reaction_models))]
        self.rh = [0 for i in range(len(self.reaction_models))]
        self.obj = [0 for i in range(len(self.reaction_models))]
        #self.stub = [None for v in self.reaction_models.values()]
        self.name_list_mp = [None for v in self.reaction_models.values()]
        
        self.kkt_dict = {0: None}
        
        self.J = [None for v in self.reaction_models.values()]
        self.H = [None for v in self.reaction_models.values()]
        
        self.varList = {i: model._nlp.get_pyomo_variables() for i, model in self.reaction_models.items()}
        self.conList = {i: model._nlp.get_pyomo_constraints() for i, model in self.reaction_models.items()}
        self.zLList = {i: [v for v in self.varList[i] if v.lb is not None] for i in self.reaction_models}
        self.zUList = {i: [v for v in self.varList[i] if v.ub is not None] for i in self.reaction_models}
        
        self._has_global_constraints = False
        #self.add_global_parameter_constraints_2()
        self.initialize_global_variables()
        self.final_param = {}
        
        self.nsd_models = {}
        self._make_nsd_models()
        
        self.iter_data = {}
        self.iter_counter = 0
        self.objective = 1e10
        
        if self.parallel:
            if platform.system() == 'Darwin':
                try:
                    set_start_method('fork')
                    print('# Changing Multiprocessing start method to "fork"')
                except:
                    print('# Multiprocessing start method already fixed')    
        
    def __str__(self):
        
        return 'Nested Schur Decomposition Object'
    
    def set_initial_value(self, init):
        """Add custom initial values for the parameters
        
        Args:
            init (dict): keys are parameters, values are floats
            
        Returns:
            None
            
        """
        for k, v in init.items():
            if k in self.d_init:
                self.d_init[k][0] = v
    
        return None
    
    
    def _make_nsd_models(self):
        
        for name, r_model in self.reaction_models.items():
        
            nsd = NSD_Model(r_model, 
                            #iter_count=50, 
                            use_k_aug=self.use_k_aug, 
                            use_cyipopt=self.use_cyipopt, 
                            use_full_method=self.use_full_KKT, 
                            model_vars=self.model_vars
                            )
            
            self.nsd_models[name] = nsd
            
        return None
    
    
    def initialize_global_variables(self):

        self.global_parameters = {}
        self.local_parameters = {}
        self.labels = {}
        counter = 0
        
        for model in self.reaction_models.values():
        
            model_name = model.name
            if not self.local_as_global:
                global_params = self.all_params.parameters_model(model_name, query='global')
                local_params = self.all_params.parameters_model(model_name, query='local')
            else:
                global_params = self.all_params.unique_globals(True, model_name)
                local_params = []
           
            labels = {}
            for param in self.all_params.unique_globals(False).values():
                #print(f'{param = }')
                valid_models = self.all_params.find_parameter(param.name)
                #print(f'{valid_models = }')
                if param.reaction in valid_models and param.is_global or param.reaction == model_name:
                    labels[param.name] = param.identity
            
            
            self.global_parameters[counter] = global_params
            self.local_parameters[counter] = local_params
            self.labels[model.name] = labels
            
            for key, param in global_params.items():
                if hasattr(model.p_model, param.model_var) and param.index in getattr(model.p_model, param.model_var):
                    var_obj = getattr(model.p_model, param.model_var)[param.index]
                    var_obj.setlb(None)
                    var_obj.setub(None)
            
            counter += 1
        
        print(self.labels)
        
        return None
    
    # def add_global_parameter_constraints_2(self):
        
    #     from kipet.estimator_tools.reduced_hessian_methods import prepare_global_constraints, generate_parameter_object_lists
        
    #     use_cyipopt = self.use_cyipopt
    #     model_vars = self.model_vars #['P', 'S']
    #     self.global_param_objs = {}
    #     self.model_object_lists = {}
    #     self.constraint_map = {}
    #     self.reverse_constraint_map = {}
        
    #     self.projected_nlp_list = []
    #     self.nlp_list = {}
    #     self.projected_nlp_list = {}
    #     self.mdict = {}
        
    #     for key, r_model in self.reaction_models.items():
            
    #         r_model.start_parameter_manager(model_vars=self.model_vars)
    #         global_parameter_set, local_parameter_set = r_model.globals_locals
    #         model = self.model_dict[key]
    #         self.constraint_map[key] = prepare_global_constraints(model,
    #                                                               global_parameter_set, 
    #                                                               local_parameter_set, 
    #                                                               use_cyipopt, 
    #                                                               )
            
    #         self.reverse_constraint_map[key] = {v: k for k, v in self.constraint_map[key].items()}
    
    #         r_model._create_pynumero_model_object()
    #         self.reaction_models[key] = r_model
    #         nlp = self.reaction_models[key]._nlp
    #         self.nlp_list[key] = nlp
    #         varNames = r_model._nlp.primals_names()
    #         projected_nlp = ProjectedNLP(nlp, varNames[:-len(getattr(model, 'd'))])
    #         self.projected_nlp_list[key] = projected_nlp
    #         model_object_lists = generate_parameter_object_lists(r_model)
    #         global_param_objs = model_object_lists[0]
    #         self.model_object_lists[key] = model_object_lists
    #         self.global_param_objs[key] = global_param_objs
            
    #         nlp_object = nlp
    #         if self.use_cyipopt:
    #             nlp_object = projected_nlp
            
    #         self.mdict[key] = rhm.generate_model_data(model, nlp_object)
    
    
    # def add_global_parameter_constraints(self):
    #     """Adds the global constraints to the p_models instead of in each
    #     iteration
        
    #     This is not important for k_aug, but critical for pynumero's start
        
    #     """
    #     self.conNames = {}
    #     self.varNames = {}
    #     self.allVarNames = {}
    #     self.projected_nlp_list = []
    #     self.nlp_list = []
    #     self.varList = []
        
    #     self.col_dict_global = {}
    #     self.col_dict_local = {}
    #     self.col_dict_all = {}
    #     self.con_dict = {}
        
        
    #     counter = 0
    #     #%%
    #     #self = lab.nsd
    #     for key, model in self.model_dict.items():
            
    #         #params = [p.name for p in self.all_params.parameters if p.identity in self.parameter_global]# in ['global', key]]
    #         #print(params)
    #         params = [obj.name for p, obj in self.all_params.parameters_model(key).items() if obj.is_global]
    #         #print(f'{params = }')
    #         global_params = self.all_params.parameters_model(key, query='global')
    #         #print(f'{params = }')
    #         #%%
        
    #         self.constraint_map = add_global_constraints(model, global_params, as_var=self.is_projected)
    #         self.reverse_constraint_map = {v: k for k, v in self.constraint_map.items()}
    #         self.reaction_models[key]._create_pynumero_model_object()
    #         nlp = self.reaction_models[key]._nlp
    #         self.varNames[counter] = self.reaction_models[key]._nlp.primals_names()
            
    #         self.col_dict_global[counter] = {}
    #         for k, param in self.all_params.parameters_model(key, query='global').items():
    #             if not param.pyomo_var.fixed:
    #                 self.col_dict_global[counter][param.name] = nlp.get_primal_indices([param.pyomo_var])[0]
            
    #         self.col_dict_local[counter] = {}
    #         for k, param in self.all_params.parameters_model(key, query='local').items():
    #             if not param.pyomo_var.fixed:
    #                 self.col_dict_local[counter][param.name] = nlp.get_primal_indices([param.pyomo_var])[0]
                    
    #         self.col_dict_all[counter] = {}
    #         for k, param in self.all_params.parameters_model(key, query='all').items():
    #             if not param.pyomo_var.fixed:
    #                 self.col_dict_all[counter][param.name] = nlp.get_primal_indices([param.pyomo_var])[0]
                    
    #         global_param_name = 'd'
    #         global_constraint_name = 'fix_params_to_global'
    #         d_vars = getattr(model, global_constraint_name)
    #         self.con_dict[counter] = [nlp.get_constraint_indices([obj])[0] for d, obj in d_vars.items()]
            
    #         varList = nlp.get_pyomo_variables()[:-len(getattr(model, global_param_name))]
    #         self.varList.append(varList)
            
    #         projected_nlp = ProjectedNLP(nlp, self.varNames[counter][:-len(getattr(model, global_param_name))])
    #         self.nlp_list.append(nlp)
    #         self.projected_nlp_list.append(projected_nlp)
    #         counter += 1
            
    #     self._has_global_constraints = True
        
    #     return None
    
    
    def run_opt(self):
        
        strategy_title = {
            'tr': 'Trust-Region',
            'ns': 'Newton-Step',
            'ipopt': 'IPOPT'}
        
        print(f'# NSD: Outer-Problem solution strategy is {strategy_title[self.strategy]}')

        if self.strategy == 'ipopt':
            results = self.ipopt_method()
        
        elif self.strategy in ['trust-region', 'tr']:
            results = self.trust_region()
            
        elif self.strategy in ['newton-step', 'ns']:
            results = self.run_simple_newton_step(alpha=0.15, iterations=1) 
        else:
            pass
        
        for param in self.all_params.unique_globals(True).values():
            self.final_param[param.identity] = self.results[param.reaction].P[param.index]
        
        if self.parallel:
            self.solve_mp_simulation()
            
        print()          
        print_margin('NSD: Parameter estimation finished!')
        
        return None
    
    
    def _model_preparation(self, use_duals=True):
        """Creates the list of models for use with NSD algorithms

        """
        self.model_list = []
        
        for name, model in self.reaction_models.items():
            model.settings.parameter_estimator.covariance = 'k_aug'
            self.model_list.append(model.p_model)
            
        return None
    
    
    def _generate_bounds_object(self):
        """Creates the Bounds object needed by SciPy for minimization
        
        Returns:
            bounds (scipy Bounds object): returns the parameter bounds for the
                trust-region method
        
        """
        lower_bounds = []
        upper_bounds = []
        
        for k, v in self.d_init.items():
            lower_bounds.append(v[1])
            upper_bounds.append(v[2])
        
        lower_bounds = np.array(lower_bounds, dtype=float)
        upper_bounds = np.array(upper_bounds, dtype=float) 
        bounds = Bounds(lower_bounds, upper_bounds, True)
        
        return bounds
    

    def _update_x_impl(self, x):
        
        """Yet another wrapper for the objective function"""
        
        # print(f'{self.parameter_global = }')
        objective_value = 0
        
        if self.parallel:
            objective_value = self.solve_mp('objective', [x, [k for k in self.reaction_models.keys()]])
        else:
            for name, model in self.reaction_models.items():
                objective_value += self.solve_element('obj', model, x, True)
                
       # print(f'$$$$$$$$$$$$ The new obj val is {objective_value}')
                
        self.objective = objective_value
        self.x = x
        self.iter_counter += 1
        
        return None
    
    
    def _update_m_impl(self, x, optimize=False):
        
        mod_param_names = [p.identity for p in self.free_params.values()]
        m = pd.DataFrame(np.zeros((len(mod_param_names), 1)), index=mod_param_names, columns=['dual'])
        for name, model in self.reaction_models.items():
            
            duals = self.solve_element('duals', model, x, optimize)
            
            #print(f'{duals = }')
            for param in m.index:
                if param in duals.keys():
                    m.loc[param] = m.loc[param] + duals[param]
            
            #print(f'{m = }')

        self.m = m
        
    def _update_grad_impl(self, x, optimize=False):
        
        mod_param_names = [p.identity for p in self.free_params.values()]
        grad = pd.DataFrame(np.zeros((len(mod_param_names), 1)), index=mod_param_names, columns=['grad'])
        for name, model in self.reaction_models.items():
            grad_i = self.solve_element('grad', model, x, optimize)
            for param in grad.index:
                if param in grad_i.keys():
                    grad.loc[param] = grad.loc[param] + grad_i[param]

        self.g = grad
        
    def _update_M_impl(self, x, optimize=False):
        
        mod_param_names = [p.identity for p in self.free_params.values()]
        #mod_param_names = ['P[k1]', 'P[k2]', 'P[k3]']
        #M = pd.DataFrame(np.zeros((len(self.parameter_names), len(self.parameter_names))), index=mod_param_names, columns=mod_param_names)
        M = pd.DataFrame(np.zeros((len(mod_param_names), len(mod_param_names))), index=mod_param_names, columns=mod_param_names)
        
        for name, model in self.reaction_models.items(): 
            reduced_hessian = self.solve_element('rh', model, x, optimize)
            
            # rh_raw = reduced_hessian.values
            # rh_cor = inertia_correction(rh_raw)
            
            # df_rh = pd.DataFrame(rh_cor, columns=reduced_hessian.columns, index=reduced_hessian.index)
            
            # print(f'Iteration {i}')
            #print(f'{reduced_hessian = }')
            
            M = M.add(reduced_hessian).combine_first(M)
            M = M[mod_param_names]
            M = M.reindex(mod_param_names)
            
            #print(f'{np.linalg.eigh(M) = }')
            #print(f'{min(min(np.linalg.eigh(M))) < 0 = }')
            
            #print(f'{M = }')
                
        #modified_M = inertia_correction(M.values)
        self.M = M #pd.DataFrame(modified_M, columns=reduced_hessian.columns, index=reduced_hessian.index)
        #pg = [f'P[{p}]' for p in self.parameter_global]
            
        #self.M = M #.loc[pg, pg]
    
    
    def objective_function(self, x):
        """Inner problem calculation for the NSD
        
        Args:
            x (np.array): array of parameter values
            scenarios (list): list of reaction models
            parameter_names (list): list of global parameters
            
        Returns:
            
            objective_value (float): sum of sub-problem objectives
            
        """
        # print(f'### Iter: {self.iter_counter} - Starting objective calculation')
        if not np.array_equal(x, self.x):
            
            # print(f'### Iter: {self.iter_counter} - The x values do not match ')
            # print(f'###\n\t     {x = }\n\t{self.x = }')
            self._update_x_impl(x)
        # else:
        #     print(f'### Iter: {self.iter_counter} - The x values match ')
            
        # print('### The current objective is:\n')
        # print(f'{self.objective}\n')

        return self.objective


    def calculate_m(self, x, *args):
        """Calculate the vector of duals for the NSD
        
        Args:
            x (np.array): array of parameter values
            
            scenarios (list): list of reaction models
            
            parameter_names (list): list of global parameters
            
        Returns:
            
            m (np.array): vector of duals
            
        """
        #print(f'### Iter: {self.iter_counter} - Starting m calculation ')
        
        optimize = False
        
        if not np.array_equal(x, self.x):
            
            # print(f'### Iter: {self.iter_counter} - The x values do not match ')
            # print(f'###\n\t     {x = }\n\t{self.x = }')
            optimize = True
        # else:
        #     print(f'### Iter: {self.iter_counter} - The x values match ')
        
        self._update_m_impl(x, optimize=optimize)
        m = self.m.values.flatten()
        
        # print('### The current m is:\n')
        # print(f'{m}\n')
        
        return m
    
    def calculate_M(self, x, *args):
        """Calculate the sum of reduced Hessians for the NSD
        
        Args:
            x (np.array): array of parameter values
            
            scenarios (list): list of reaction models
            
            parameter_names (list): list of global parameters
            
        Returns:
            
            M (np.array): sum of reduced Hessians
        """
        
        #print(f'### Iter: {self.iter_counter} - Starting M calculation ')
        
        optimize = False
        if not np.array_equal(x, self.x):
        #    print(f'### Iter: {self.iter_counter} - The x values do not match ')
        #    print(f'###\n\t     {x = }\n\t{self.x = }')
            optimize = True
        #else:
        #    print(f'### Iter: {self.iter_counter} - The x values match ')
            
        self._update_M_impl(x, optimize=optimize)
        M = self.M.values# + np.eye(M_size)*0.1
        
        # print('The current M is:\n')
        # print(f'{M}\n')
          
        return M
    
    def calculate_grad(self, x, *args):
        """Calculate the average of the gradients for the NSD
        
        Args:
            x (np.array): array of parameter values
            
            scenarios (list): list of reaction models
            
            parameter_names (list): list of global parameters
            
        Returns:
            
            grad (np.array): sum of gradients
        """
        optimize = False
        if not np.array_equal(x, self.x):
            optimize = True
            
        self._update_grad_impl(x, optimize=optimize)
        grad = self.g.values
        
        return grad
    

    def parameter_initialization(self):
        """Sets the initial parameter values in each scenario to d_init
        """
        
        d_vals =  [d[0] for k, d in self.d_init.items()]
        if self.scaled:
            self.d_init_unscaled = d_vals
            d_vals = [1 for p in d_vals]
             
        return d_vals
    
    
    def parameter_scaling_conversion(self, results):
        
        if self.scaled:
            s_factor = {k: self.d_init_unscaled[k] for k in self.d_init.keys()}
        else:
            s_factor = {k: 1 for k in self.d_init.keys()}
        
        self.parameters_opt = {k: results[i]*s_factor[k] for i, k in enumerate(self.d_init.keys())}
        
        return None 
        
        
    def ipopt_method(self, callback=None, options=None, **kwargs):
        """ Minimization of scalar function of one or more variables with
            constraints
    
        Args:
            m : PyomoNLP Model or equivalent
    
            callback  : callable
                Called after each iteration.
    
                    ``callback(xk, state) -> bool``
                
                where ``xk`` is the current parameter vector. and ``state`` is
                an optimization result object. If callback returns True, the algorithm
                execution is terminated.
    
            options : IPOPT options
        
        Returns:
            result : Optimization result
        
        """
        d_vals = self.parameter_initialization()
    
        kwargs = {
                'scenarios': self.model_list,
                'parameter_names': self.parameter_names,
                'parameter_number': len(d_vals)
                 }
    
        problem_object = Optproblem(objective=self.objective_function,
                                    hessian=self.calculate_M,
                                    gradient=self.calculate_m,
                                    jacobian=self.calculate_grad,
                                    kwargs=kwargs,
                                    callback=self.callback)
        
        bounds = self._generate_bounds_object()
        
        print('# NSD: Starting IPOPT algorithm\n')
        
        nlp = cyipopt.Problem(n = len(d_vals),
                              m = 0,
                              problem_obj = problem_object,
                              lb = bounds.lb,
                              ub = bounds.ub,
                              cl = [],
                              cu = [],
                              )
    
        options = {#'tol': 3e-1,
                   #'acceptable_tol' : 1e-4,
                 #  'bound_relax_factor': 1.0e-8, 
                   'max_iter': 10,
                   #'print_user_options': 'yes', 
                   #'nlp_scaling_method': 'none',
                   'print_level': 5,
                   #'corrector_type': 'primal-dual',
                   #'alpha_for_y': 'full',
               #    'accept_every_trial_step': 'yes',
                   'linear_solver': 'ma57',
                   #'hsllib': 'libcoinhsl.dylib',
                   'print_info_string': 'yes',
                   'output_file': 'ipopt_output_nsd.txt',
                   'file_print_level' : 6,
                   }

        solver_settings = SolverSettings()
        if solver_settings.custom_solvers_lib['hsllib'] is not None:
            nlp.add_option('hsllib', solver_settings.custom_solvers_lib['hsllib'])

        if options: 
            for key, value in options.items():
                nlp.add_option(key, value)
        
        x, results = nlp.solve(d_vals)
        
        print('\n# NSD: Finished IPOPT algorithm')
        print(f'# NSD: Solver Status: {results["status"]}')
        print(f'# NSD: Solver Message: {results["status_msg"].decode()}')
        self.d_iter.append(x)
        
        results_kipet = self.get_results()
        return results_kipet
    
    def trust_region(self, debug=False):
        """This is the outer problem controlled by a trust region solver 
        running on scipy. This is the only method that the user needs to 
        call after the NSD instance is initialized.
        
        Returns:
            results (scipy.optimize.optimize.OptimizeResult): The results from the 
                trust region optimation (outer problem)
                
            opt_dict (dict): Information obtained in each iteration (use for
                debugging)
                
        """
        d_vals = self.parameter_initialization()
        
        # Start TR Routine
        if self.method not in ['trust-exact', 'trust-constr']:
            raise ValueError('The chosen Trust Region method is not valid')

        tr_options={
            'xtol': 1e-10,
            'verbose': 3,
            }
    
        print('# NSD: Starting Trust-Region algorithm\n')
        
        results = minimize(
            self.objective_function, 
            d_vals,
            args=(), 
            method=self.method,
            jac=self.calculate_m,
            hess=self.calculate_M,
            callback=self.callback,
            bounds=self._generate_bounds_object(),
            options=tr_options,
        )
            
        self.opt_results = results
        
        print('\n# NSD: Finished Trust-Region algorithm')
        print(f'# NSD: Solver Status: {results.status}')
        # End internal methods
        self.parameter_scaling_conversion(results.x)
        results_kipet = self.get_results()
        
        return results_kipet
    
    def run_simple_newton_step(self, debug=False, alpha=1, iterations=1, opt_tol=1e-8):
        """Performs NSD using simple Newton Steps
        
        This is primarily for testing purposes
        
        """
        d_vals = self.parameter_initialization()
        
        # Start internal methods   
        self.callback(d_vals)

        print('# NSD: Starting Newton-Step algorithm\n')

        for i in range(iterations):
            
            self.objective_function(
                d_vals,
            )
           
            # Get the M matrices to determine search direction
            M = self.calculate_M(d_vals)
            m = self.calculate_m(d_vals)
            
            # Calculate the search direction
            print('Pre-NS:')
            
            print(f'{M = }')
            print(f'{m = }')
            
            
            d = np.linalg.inv(M) @ -(m)
            d_vals = d*alpha + d_vals
            print(f'{d = }')
            print(f'{d_vals = }')
            self.callback(d_vals)
            
            print(f'Current Parameters in Iteration {i}: {d_vals}')

            for name, model in self.reaction_models.items():
                for j, param in enumerate(self.d_init.keys()):
                    if param.split('-')[-1] in ['global', name]:
                        model_param_name = '-'.join(param.split('-')[:-1])
                        model.p_model.P[model_param_name].set_value(d[j]*alpha + model.p_model.d[model_param_name].value)
                    
            if max(abs(d)) <= opt_tol:
                print('Tolerance reached')
                break

        # Only delete after checking scaling
        # Update the correct, final parameters taking scaling into account
        # for m, reaction in enumerate(self.reaction_models.values()):
        #     for k, v in reaction.p_model.P.items():
        #         if scaled:
        #             reaction.p_model.P[k].set_value(self.model_list[m].K[k].value*self.model_list[m].P[k].value)
        #         else:
        #             reaction.p_model.P[k].set_value(self.model_list[m].P[k].value)
            
        
        print('# NSD: Finished Newton-Step algorithm')
        # End internal methods
        
        self.parameter_scaling_conversion(d_vals)
        results_kipet = self.get_results()
        return results_kipet
    
    def callback(self, x, opt_obj=None):
        """Method to record the parameters in each iteration
        """
        self.d_iter.append(x)
        
        
    def _calculate_covariance(self):
        """Displays the covariance results to the console

        :param dict variances_p: The dict of parameter variances to display

        :return: None

        """
        #%%
        # self = lab.nsd
        # import pandas as pd
        # import numpy as np
        
        self.rm_variances = {}
        self.rm_cov = {}
        
        for j, (name, rm) in enumerate(self.reaction_models.items()):
            
            all_model_params = self.all_params.parameters_model(name)
            
            cov_all = pd.DataFrame(np.zeros((len(all_model_params), len(all_model_params))), columns=[p.index for p in all_model_params.values()], index=[p.index for p in all_model_params.values()])
            
            col = []
            col_name = []
            #cov_mat = pd.DataFrame(np.linalg.inv(self.M.values)*self.rs_min_variance, columns=self.M.columns, index=self.M.index)
            cov_mat = pd.DataFrame(np.linalg.inv(self.M.values)*self.rs_min_variance, columns=self.M.columns, index=self.M.index)
            self.rm_variances[name] = {key: 0 for key in rm.p_model.P}
            
            for param in self.all_params.unique_globals(True).values(): #self.local_as_global):
                
                valid_models = self.all_params.find_parameter(param.name)
                # print(name, param.identity)
                if param.reaction in valid_models and param.is_global or param.reaction == name:
                    #print(name, param.identity)
                #if param.model in ['global', name]:
                    
                    if param.identity in cov_mat:
                        
                        self.rm_variances[name][param.index] = cov_mat.loc[param.identity, param.identity]
                        col.append(param.identity)
                        col_name.append(param.index)
                        
                    else:
                        
                        self.rm_variances[name][param.index] = np.nan
                        #col.append(param.identity)
                        #col_name.append(param.name)
                    
                    
            self.cov = cov_mat
            
            intermediate_cov = cov_mat.loc[col, col]# if col in cov_mat else 0
            intermediate_cov.columns = col_name
            intermediate_cov.index = col_name
            
            self.rm_cov[name] = cov_all + intermediate_cov 
            #self.rm_cov[name] = self.rm_cov[name].dropna(axis=1, how='all').dropna(axis=0, how='all')
            
            # print(self.rm_cov[name])
            # print(self.rm_variances[name])
            
            #%%
        
    def _display_covariance(self):
        """Displays the covariance results to the console

        :param dict variances_p: The dict of parameter variances to display

        :return: None

        """
        import scipy.stats as st
        
        if not hasattr(self, 'confidence') or self.confidence is None:
            confidence = 0.95
        
        number_of_stds = st.norm.ppf(1 - (1 - confidence)/2)
        margin = max([len(v.name) for v in self.free_params.values()]) + 2
        
        print()
        print_margin('NSD Results', sub_phrase=f'Parameter Values (Confidence: {int(confidence*100)}%)')
        
        for fp in self.all_params.unique_globals(self.local_as_global).values():
    
            model = self.reaction_models[fp.reaction].p_model
            value = model.P[fp.index].value
            
            percent_deviation = f'(  n/a  )'
            interval = '+/- ??????????'
            
            if fp.name in self.rm_variances[fp.reaction]:
            
                variance = self.rm_variances[fp.reaction][fp.index]
                percent_deviation = f'({100*number_of_stds*(variance**0.5)/abs(value):6.1f}%)'
                interval = f'+/- {number_of_stds*(variance**0.5):0.4e}'
            
            param_location = fp.model if not fp.is_global else 'global'
            #if k in self.parameter_global:
            print(f'{fp.name.rjust(margin)} = {int(value > 0)*" "}{value:0.4e} {interval}   {percent_deviation}   {param_location}')
                
        # for exp, r_model in self.reaction_models.items():
        #     model = r_model.p_model
        #     print(f'Experiment - {exp}:\n')
        #     for i, (k, p) in enumerate(model.P.items()):
        #         if p.is_fixed():
        #             continue
        #         is_global = '(local)'
        #         if k in self.parameter_global:
        #             is_global = '(global)'
                
        #         #key = f'P[{k}]'
        #         variance = self.rm_variances[exp][k]
        #         #if k in self.parameter_global:
        #         print(f'{k.rjust(margin)} = {int(p.value > 0)*" "}{p.value:0.4e} +/- {number_of_stds*(variance**0.5):0.4e}   ({100*number_of_stds*(variance**0.5)/abs(p.value):6.1f}%)   {is_global}')
                
        #     # These need to be set equal to the local values
        #     if hasattr(model, 'Pinit'):
        #         for i, k in enumerate(model.Pinit.keys()):
        #             model.Pinit[k] = model.init_conditions[k].value
        #             key = f'Pinit[{k}]'
        #             value = model.Pinit[k].value
        #             variance = self.rm_variances[exp][k]
        #             print_name = f'{k} (init)'
        #             print(f'{print_name.rjust(margin)} = {value:0.4e} +/- {number_of_stds*(variance)**0.5:0.4e}    (local)')
                        
        #     # These need to be set equal to the local values
        #     if hasattr(model, 'time_step_change'):
        #         for i, k in enumerate(model.time_step_change.keys()):
        #             model.time_step_change[k] = model.time_step_change[k].value
        #             key = f'time_step_change[{k}]'
        #             value = model.time_step_change[k].value
        #             variance = self.rm_variances[exp][k]
        #             print(f'{k.rjust(margin)} = {value:0.4e} +/- {number_of_stds*(variance)**0.5:0.4e}    (local)')
        print()
            
        return None
    
    def get_results(self):
        
        # n = 'reaction-1'
        # m = self.nsd_models[n].model
        # print(f'{m.P["k1"].value = }')
        
        # res = self.nsd_models[n]._get_results()
        # print(f'{res = }')
            
        # for nlp_obj in self.nlp_list.values():
        #     nlp_obj.load_state_into_pyomo()
            
        self._calculate_covariance()
        
        if self.parallel:
            for name, param_data in self.d.items():
                for param, value in param_data.items():
                    self.reaction_models[name].p_model.P[param].set_value(value)
        
        solver_results = {}
        for name, model in self.reaction_models.items():
            #model.p_estimator._get_results(display=False)
            solver_results[name] = self.nsd_models[name]._get_results()
            #solver_results[name] = ResultsObject()
            #solver_results[name].load_from_pyomo_model(self.nsd_models[name].model)
            #print(f'{self.nsd_models[name].model.P["k1"].value = }')
            #print('adding cov')
            solver_results[name].parameter_covariance = self.rm_cov[name]
            model.results = solver_results[name]
            
        self.results = solver_results 
        self._display_covariance()
        
        return solver_results
    
    def plot_paths(self, filename=''):
        """Plot the parameter paths through parameter space during the NSD
        algorithm. For diagnostic purposes.
        
        """
        all_data = np.r_[self.d_iter]
        y_data = []
        
        for i in range(len(all_data)):
            if i > 0:
                if all(all_data[i, :] == all_data[i - 1, :]):
                    continue
            y_data.append(all_data[i, :])

        x_data = list(range(len(y_data)))
        y_data = np.asarray(y_data)
        
        fig = go.Figure()    
        
        for i, params in enumerate(self.d_init.keys()):
            
            fig.add_trace(
                go.Scatter(x = x_data,
                           y = y_data[:, i],
                           name = params,
                   )
                )
        
        fig.update_layout(
            title='Parameter Paths in NSD',
            xaxis_title='Iterations',
            yaxis_title='Parameter Values',
            )
    
        plot(fig)
    
        return None
    
    def _arrange_model_specific_x(self, r_model, x):
        
        model_name = r_model.name
        model = r_model.p_model
        file_number = self.model_number[model_name]
        param_value_dict = {}
        
        for param in self.global_parameters[file_number].values():
            #print(f'{param = }')
            if param.index in model.P or param.index in model.S:
                for i, fp in enumerate(self.free_params.values()):
                    #print(i, fp)
                    if fp.index == param.index:
                        if fp.reaction == model_name or fp.is_global:
                            if not fp.pyomo_var.fixed:
                                param_value_dict[fp.index] = x[i] 
                                
        #print(f'{param_value_dict = }')
        return list(param_value_dict.values())
    
    
    def _model_update(self, model, x):
        
        #print(f'{x = }')
        
        nsd = self.nsd_models[model.name]
        if nsd.iter_last is not None:    
            nsd.update_primals(x)
        nsd.optimize()
        nsd.update_results()
        
        rh = nsd.rh[nsd.iter_last]
        duals = nsd.duals[nsd.iter_last]
        obj_val = nsd.obj_val[nsd.iter_last]
        
        new_labels = [self.labels[model.name][v] for v in rh.columns]
        rh.columns = new_labels
        rh.index = new_labels
        
        duals_dict = {}
        dual_modifier = 1 - 2*(not self.use_cyipopt)
        
        for i, (key, val) in enumerate(getattr(nsd.model, 'fix_params_to_global').items()):
            local_var_name = nsd.reverse_constraint_map[key]
            dict_key = self.labels[model.name][local_var_name]
            dual_value = -1*dual_modifier*duals.loc[local_var_name].value
            duals_dict[dict_key] = dual_value
        
        return rh, duals_dict, None, None, obj_val, model.name
    

    def general_update(self, model, x):
        """Updates the model using the reduced hessian method using global,
        fixed parameters. This may need to be updated to handle parameters that
        are not fixed, such as local effects.
        
        :param ConcreteModel model: The current model of the system used in optimization
        :param np.ndarra x: The current parameter array
        :param int file_number: The index of the model
        
        :return: None
        
        """
        
        self.iter_data[self.iter_counter] = x
        
        rh, duals, grad, param_values, obj_val, _ = self._model_update(model, x)        
        
        file_number = self.model_number[model.name]
        self.rh[file_number] = rh
        self.grad[file_number] = grad
        self.duals[file_number] = duals
        self.obj[file_number] = obj_val
        
        return None
            
    
    def solve_element(self, element, model, x, optimize=True):
        
        if optimize:
            self.general_update(model, x)
        
        number = self.model_number[model.name]
        values = getattr(self, element)[number]
        
        return values
    
    
    def solve_model_objective_mp(self, model, x):
        """Wrapper for obtaining the objective function value
        
        """
        rh, stub, duals, grad, param_values, name = self._model_update(model, x)
        
        return model.objective.expr(), rh, stub, duals, grad, param_values, name
    

    def obj_func(self, q, i, args):
        """This takes the input and passes it to the target function
        
        """
        data = self.solve_model_objective_mp(self.model_list[i - 1], args[0])
        q.put(data)
        
        return None
    
    
    def solve_mp(self, func, args):

        mp = Multiprocess(self.obj_func)
        data = mp([args[0]], num_processes = len(self.model_list))
        obj = 0 
        model_names = args[1]
    
        for i, d in enumerate(data.values()):    
            
            model_name = d[-1]
            
            obj += d[0]
            self.rh[i] = d[1]
            # self.stub[i] = d[2]#
            self.duals[i] = d[2]
            self.grad[i] = d[3]
            self.d[model_name] = d[4]
            self.name_list_mp[i] = d[5]
        
        return obj

        
    def func_simulation(self, q, i):
        
        model_names = [k for k in self.reaction_models.keys()]
        model_to_solve = list(self.reaction_models.values())[i - 1]
        d_index = self.name_list_mp.index(model_names[i - 1])
        model_to_solve.parameters.update('value', self.d[model_names[i - 1]])
        data = self.solve_simulation(model_to_solve)
        q.put(data)


    def solve_simulation(self, model):
        """Uses the ReactionModel framework to calculate parameters instead of
        repeating this in the MEE
    
        # Add mp here
    
        """
        model.simulate(self.d[model.name])
        print('Model has been simulated')

        attr_list = ['name', 'results_dict']
    
        model_dict = {}
        for attr in attr_list:
            model_dict[attr] = getattr(model, attr)
    
        return model_dict['results_dict'], model.name
    
    def solve_mp_simulation(self):
        """Method for simulating the models using the final parameter sets
        
        """
        mp = Multiprocess(self.func_simulation)
        data = mp(num_processes = len(self.reaction_models))
        
        self.mp_results = data
        simulator = 'simulator'
        estimator = 'p_estimator'
        
        for i in range(len(data)):
            
            model_data, model_name = self.mp_results[i + 1]
            model = self.reaction_models[model_name]
            model._pe_set_up(solve=False)
            setattr(model, 'results_dict', self.mp_results[i + 1][0])
            setattr(model, 'results', self.mp_results[i + 1][0][simulator])
            model.results.parameter_covariance = self.rm_cov[model_name]
            vars_to_init = get_vars(model.p_model)
            
            for var in vars_to_init:
                if hasattr(model.results_dict[simulator], var) and var != 'S':
                    getattr(model, estimator).initialize_from_trajectory(var, getattr(model.results_dict[simulator], var))
                elif var == 'S' and hasattr(model.results_dict[simulator], 'S'):
                    getattr(model, estimator).initialize_from_trajectory(var, getattr(model.results_dict[simulator], var))
                else:
                    pass
                    #print(f'Variable: {var} is not updated')
            
        return None
    
    
class Optproblem(object):
    """Optimization problem

    This class defines the optimization problem which is callable from cyipopt.

    """
    def __init__(self, 
                 objective=None, 
                 hessian=None, 
                 jacobian=None, 
                 gradient=None, 
                 kwargs={}, 
                 callback=None):
        
        self.fun = objective
        self.grad = gradient
        self.hess = hessian
        self.jac = jacobian
        self.kwargs = kwargs
        self.callback = callback

    def objective(self, x):
        
        return self.fun(x)
    
    def gradient(self, x):
        
        scenarios = self.kwargs.get('scenarios', None)
        parameter_names = self.kwargs.get('parameter_names', None)
        
        return self.grad(x, scenarios, parameter_names)

    def constraints(self, x):
        """The problem is unconstrained in the outer problem excluding
        parameters
        
        """
        return np.array([])

    def jacobian(self, x):

        scenarios = self.kwargs.get('scenarios', None)
        parameter_names = self.kwargs.get('parameter_names', None)

        return self.jac(x, scenarios, parameter_names)        

    def hessianstructure(self):
        
        global hs
        nx = self.kwargs['parameter_number']
        hs = coo_matrix(np.tril(np.ones((nx, nx))))
        
        return (hs.col, hs.row)
        
    def hessian(self, x, a, b):
        
        scenarios = self.kwargs.get('scenarios', None)
        parameter_names = self.kwargs.get('parameter_names', None)
        H = self.hess(x, scenarios, parameter_names)
        
        return H[hs.row, hs.col]

    def intermediate(self, alg_mod, iter_count, obj_value, inf_pr, inf_du, mu,
                     d_norm, regularization_size, alpha_du, alpha_pr,
                     ls_trials):
        
        pass


def calculate_parameter_averages(model_dict):
    
    p_dict = {}
    lb = {}
    ub = {}
    c_dict = {}
    
    for key, model in model_dict.items():
        for param, obj in getattr(model, 'P').items():
            if param not in p_dict:
                p_dict[param] = obj.value
                c_dict[param] = 1
            else:
                p_dict[param] += obj.value
                c_dict[param] += 1
                
            lb[param] = obj.lb
            ub[param] = obj.ub
                
    avg_param = {param: [p_dict[param]/c_dict[param], lb[param], ub[param]] for param in p_dict.keys()}
    
    return avg_param

def print_debug(value):
    
    if DEBUG:
        print(f'{value = }')
        

# def generate_model_data(model, nlp):
    
#     if isinstance(nlp, ProjectedNLP):
#         nlp = nlp._original_nlp
        
#     dummy_constraints = [v for v in getattr(model, 'fix_params_to_global').values()]
#     dummy_vars = [id(v) for v in getattr(model, 'd').values()]
#     attr_dict = {}
#     attr_dict['varList'] = [v for v in nlp.get_pyomo_variables() if id(v) not in dummy_vars]
#     attr_dict['conList'] = nlp.get_pyomo_constraints()
#     attr_dict['zLList'] = [v for v in attr_dict['varList'] if v.lb is not None]
#     attr_dict['zUList'] = [v for v in attr_dict['varList'] if v.ub is not None]
#     attr_dict['conList_dummy'] = [v for v in attr_dict['conList'] if v in dummy_constraints]
#     attr_dict['conList_red'] = [v for v in attr_dict['conList'] if v not in dummy_constraints]
#     attr_dict['var_index_names'] = [v.name for v in attr_dict['varList']]
#     attr_dict['init_data'] = {v.name : v.value for v in attr_dict['varList']}
        

#     return attr_dict

# def get_E_matrix(nlp, mdict):

#     _J_dum = nlp.extract_submatrix_jacobian(pyomo_variables=mdict['varList'], pyomo_constraints=mdict['conList_dummy'])
#     _z = coo_matrix((len(mdict['conList_red']), len(mdict['conList_dummy'])))
#     E = vstack([_J_dum.T, _z], format='csc')
#     return E

# def _KKT(nlp, mdict, use_cyipopt=True, index_list=None):
    
#     if isinstance(nlp, PyomoNLP):
#         J = nlp.extract_submatrix_jacobian(pyomo_variables=mdict['varList'], pyomo_constraints=mdict['conList_red'])
#         H = nlp.extract_submatrix_hessian_lag(pyomo_variables_rows=mdict['varList'], pyomo_variables_cols=mdict['varList'])
   
#     else:
        
#         H, J, grad = build_matrices(nlp, index_list[0], use_cyipopt=use_cyipopt)
#         J_c = delete_from_csr(J.tocsr(), row_indices=index_list[2]).tocsc()
#         J = J_c

#     ## Construct KKT matrix
#     K = bmat([[H, J.T],[J, None]], format='csc')

#     # if self.inertia_correction:
#     K = inertia_correction(K)

#     return K, J, H

# def _KKT_kaug(size): #nlp, mdict, use_cyipopt=True, index_list=None):
#     """Given the size of the variables and constraints, the Hessian and Jacobian
#     can be built using the output files from k_aug
    
#     :param tuple size: The m (con) and n (var) size of the Jacobian
    
#     :return: The Hessian and Jacobian as a tuple of sparse (coo) matrices
    
#     """
#     from pathlib import Path
#     m, n = size
#     kaug_files = Path('GJH')
    
#     hess_file = kaug_files.joinpath('H_print.txt')
#     hess = pd.read_csv(hess_file, delim_whitespace=True, header=None, skipinitialspace=True)
    
#     hess.columns = ['irow', 'jcol', 'vals']
#     hess['irow'] -= 1
#     hess['jcol'] -= 1
   
#     jac_file = kaug_files.joinpath('A_print.txt')
#     jac = pd.read_csv(jac_file, delim_whitespace=True, header=None, skipinitialspace=True)
#     jac.columns = ['irow', 'jcol', 'vals']
#     jac['irow'] -= 1
#     jac['jcol'] -= 1
    
#     # print(f'{jac.tail() = }')
#     # jac = jac[jac["irow"] <= m]
#     # jac = jac[jac["jcol"] <= n]
#     # #jac = jac[jac["irow"] != m + 1]
#     # print(f'{jac.tail() = }')
    
#     grad_file = kaug_files.joinpath('gradient_f_print.txt')
#     grad = pd.read_csv(grad_file, delim_whitespace=True, header=None, skipinitialspace=True)
#     grad.columns = ['gradient']
    
#     # print(f'{hess.shape = }')
#     # print(f'{jac.shape = }')
#     # print(f'{grad.shape = }')
#     # print(f'{max(jac["irow"]) = }')
#     # print(f'{max(jac["jcol"]) = }')
#     # print(f'{size = }')
    
#     J = coo_matrix((jac.vals, (jac.jcol, jac.irow)), shape=(m, n))
#     Hess_coo = coo_matrix((hess.vals, (hess.irow, hess.jcol)), shape=(n, n))
#     H = Hess_coo + triu(Hess_coo, 1).T
    
#     K = bmat([[H, J.T],[J, None]], format='csc')
#     #_kkt = csc_matrix(inertia_correction(K.todense()))
#     #K = csc_matrix(_kkt)
    
#     return K, H, J

# def evaluate_gradient(nlp, mdict, model_object_lists, use_cyipopt=False):
    
#     m = 0
#     global_param_index, _, dummy_con_index = generate_parameter_index_lists(nlp, model_object_lists)
#     duals = nlp.n_constraints()*[0.0]
    
#     if not use_cyipopt:
#         model = nlp.pyomo_model()
#         for index, con in zip(dummy_con_index, mdict['conList_dummy']):
#             duals[index] = model.dual[con]
#     else:
    
#         duals = nlp.get_duals()
#         # for index, con in zip(dummy_con_index, mdict['conList_dummy']):
#         #     duals[index] = nlp.get_duals()[index]
        
#     m += np.array(duals)
#     return m

        
        #%%
def matrix_inertia(matrix):
    
    zero_tolerance = 1e-14
    
    eigenvalues, eigenvectors = np.linalg.eigh(matrix)
    
    positive_eigenvalues = np.sum(np.where(eigenvalues > zero_tolerance, 1, 0))
    negative_eigenvalues = np.sum(np.where(eigenvalues < -1*zero_tolerance, 1, 0))
    null_eigenvalues = len(eigenvalues) - positive_eigenvalues - negative_eigenvalues
    
    inertia = (positive_eigenvalues, negative_eigenvalues, null_eigenvalues)
    print(f'{inertia = }')
    
    # M matrix condition check
    # if condition_check:
    #     p_count = 0
    #     n_count = 0
    #     z_count = 0
    #     e, v = np.linalg.eig(M)
    #     for i in range(len(e)):
    #         if e[i].real > 0:
    #             p_count += 1
    #         elif e[i].real < 0:
    #             n_count += 1
    #         else:
    #             z_count += 1

    print('M size:', matrix.shape, 'rank:',np.linalg.matrix_rank(matrix))
    
    print(f'Minimum eigenvalue = {min(abs(eigenvalues))}')
    
        # print('p:', p_count, 'n:', n_count, 'z:', z_count)
        

def inertia_correction(matrix):
    
    from scipy.sparse.csgraph import structural_rank
    No_of_pivot = structural_rank(matrix)
    modified_kkt = matrix

    if No_of_pivot == modified_kkt.shape[0]:
        print('--- Correct inertia without correction ---')
        IC_check = True
    else:
        print('--- Correct inertia with correction ---')
        IC_check = False

    if not IC_check:
        ## Set kappa_e and minimum and maximum e
        kappa_e = 10.0
        e_min = 1.0e-20
        e_max = 1.0
        ## Inertia correction
        e = 0.0
        counter = 0
        # Check the inertia of kkt matrix
        while True and counter < 2:
            if e > e_max:
                raise Exception('Inertia correction reaches the maximum limit')

            No_of_pivot = structural_rank(K)
            print(f'{No_of_pivot = }')
            if No_of_pivot == modified_kkt.shape[0]:
                print('--- Correct inertia with ( e=', e,') ---')
                break
            elif No_of_pivot < modified_kkt.shape[0]:

                counter += 1
                if counter < 2:
                    e = e_min
                else:
                    e *= kappa_e     
            
            print(f'{counter = }')
            print('Updated KKT')
            modified_kkt.setdiag(modified_kkt.diagonal() + e)

        print(f'{modified_kkt = }')
    
    # modified_kkt = matrix
    # ## Check whether the inertia correction is required or not
    # No_of_pivot = np.linalg.matrix_rank(modified_kkt)
    
    # if No_of_pivot == modified_kkt.shape[0]:
    #     print('--- Correct inertia without correction ---')
    #     IC_check = True
    # else:
    #     print('--- Correct inertia with correction ---')
    #     IC_check = False
    
    # if not IC_check:
    #     ## Set kappa_e and minimum and maximum e
    #     kappa_e = 10.0
    #     e_min = 1.0e-20
    #     e_max = 1.0
    #     ## Inertia correction
    #     e = 0.0
    #     counter = 0
    #     # Check the inertia of kkt matrix
    #     while True and counter < 2:
    #         if e > e_max:
    #             raise Exception('Inertia correction reaches the maximum limit')

    #         No_of_pivot = np.linalg.matrix_rank(modified_kkt)
    #         print(f'{No_of_pivot = }')
    #         if No_of_pivot == modified_kkt.shape[0]:
    #             print('--- Correct inertia with ( e=', e,') ---')
    #             break
    #         elif No_of_pivot < modified_kkt.shape[0]:

    #             counter += 1
    #             if counter < 2:
    #                 e = e_min
    #             else:
    #                 e *= kappa_e     
            
    #         print(f'{counter = }')
    #         print('Updated KKT')
    #         modified_kkt += e*np.identity(modified_kkt.shape[0])

    #     print(f'{modified_kkt = }')
    
    return modified_kkt




   
if __name__ == '__main__':
   
    pass
