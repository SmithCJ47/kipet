"""

This is a new NSD module for use with IPOPT

Goal: To be clean as possible with the implementation and to use as many of
      the existing code as possible - it is all in the reduced_hessian already.
"""
# Standard library imports
# import copy

# Third party imports
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.offline import plot
from pyomo.environ import (
    Objective,
    Suffix,
    )
from scipy.optimize import (
    Bounds,
    minimize,
    )

from scipy.sparse import coo_matrix

# KIPET library imports
# from kipet.library.common.parameter_handling import (
#     check_initial_parameter_values,
#     calculate_parameter_averages,
#     set_scaled_parameter_bounds,
#     )
# from kipet.library.common.parameter_ranking import (
#     parameter_ratios,
#     rank_parameters,
#     )
# from kipet.library.common.reduced_hessian import (
#     add_global_constraints,
#     calculate_reduced_hessian,
#     optimize_model,
#     calculate_duals,
#     )
# from kipet.library.post_model_build.scaling import (
#     remove_scaling,
#     scale_parameters,
#     update_expression,
#     )
from kipet.model_components.objectives import (
    conc_objective,
    comp_objective,
    )
from kipet.estimator_tools.parameter_estimator import ParameterEstimator
from kipet.calculation_tools.reduced_hessian import ReducedHessian
from kipet.estimator_tools.results_object import ResultsObject        

DEBUG = False
 

class NSD:

    """This handles all methods related to solving multiple scenarios with NSD strategies"""
    
    def __init__(self, 
                 models_dict,
                 strategy='ipopt',
                 init=None, 
                 global_parameters=None,
                 kwargs=None,
                 scaled=False,
                 ):
        
        kwargs = kwargs if kwargs is not None else {}
        parameter_name = kwargs.get('parameter_name', 'P')
        self.objective_multiplier = kwargs.get('objective_multiplier', 1)
        self.global_parameters = global_parameters
        self.isKipetModel = kwargs.get('kipet', True)
        self.scaled = scaled
        self.reduced_hessian_kwargs = {}
        self.reaction_models = models_dict
        pre_solve = False
        self._model_preparation(use_duals=True)
        self.model_dict = {name: r.p_model for name, r in models_dict.items()}
        avg_param = calculate_parameter_averages(self.model_dict)
        self.strategy = strategy
        self.d_init = {p: [v, -10, 10] for p, v in avg_param.items()}
        self.d_init_unscaled = None
        self.d_iter = []
        
        if init is not None:
            for k, v in init.items():
                self.d_init[k][0] = v
        
        print(self.d_init)
        
        self.method = 'trust-constr'
        
        all_parameters = []
        for model in self.model_list:
            for param in model.P.keys():
                if param not in all_parameters:
                    all_parameters.append(param)
        
        self.parameter_names = all_parameters
        
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
    
    def run_opt(self):
        
        print(f'# NSD method using {self.strategy} as the soving strategy')

        if self.strategy == 'ipopt':
            results = self.ipopt_method()
        
        elif self.strategy == 'trust-region':
            results = self.trust_region()
            
        elif self.strategy == 'newton-step':
            results = self.run_simple_newton_step(alpha=0.1, iterations=15) 
        
        return results
    
    
    def _model_preparation(self, use_duals=True):
        """Helper function that should prepare the models when called from the
        main function. Includes the experimental data, sets the objectives,
        simulates to warm start the models if no data is provided, sets up the
        reduced hessian model with "fake data", and discretizes all models

        """
        self.model_list = []
        
        for name, model in self.reaction_models.items():
            
            model.settings.parameter_estimator.covariance = None
            
            model._ve_set_up()
            model._pe_set_up(solve=False)
            
            if use_duals:
                model.p_model.dual = Suffix(direction=Suffix.IMPORT_EXPORT)
                model.p_model.ipopt_zL_out = Suffix(direction=Suffix.IMPORT)
                model.p_model.ipopt_zU_out = Suffix(direction=Suffix.IMPORT)
                model.p_model.ipopt_zL_in = Suffix(direction=Suffix.EXPORT)
                model.p_model.ipopt_zU_in = Suffix(direction=Suffix.EXPORT)
                # model.red_hessian = Suffix(direction=Suffix.EXPORT)
                # model.dof_v = Suffix(direction=Suffix.EXPORT)
                # model.rh_name = Suffix(direction=Suffix.IMPORT)
     
                # count_vars = 1
                # for k, v in model.P.items():
                #     model.dof_v[k] = count_vars
                #     count_vars += 1
                
            self.model_list.append(model.p_model)
                # model.npdp = Suffix(direction=Suffix.EXPORT)
            
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
    
    @staticmethod
    def objective_function(x, scenarios, parameter_names):
        """Inner problem calculation for the NSD
        
        Args:
            x (np.array): array of parameter values
            scenarios (list): list of reaction models
            parameter_names (list): list of global parameters
            
        Returns:
            
            objective_value (float): sum of sub-problem objectives
            
        """
        if DEBUG:
            stuck = '*'*50
            print(stuck)
            print('\nCalculating objective')
            
        
        for i, p in enumerate(parameter_names):
            print(f'{p} = {x[i]:0.12f}')
        
        objective_value = 0
        for i, model in enumerate(scenarios):
            
            rh = ReducedHessian(model, file_number=i)
            rh.parameter_set = parameter_names
            rh.optimize_model(d=x)
            objective_value += model.objective.expr()
        
        if DEBUG:
            print(f'Obj: {objective_value}')
            print(stuck)
            
        return objective_value
    
    @staticmethod
    def calculate_m(x, scenarios, parameter_names):
        """Calculate the vector of duals for the NSD
        
        Args:
            x (np.array): array of parameter values
            
            scenarios (list): list of reaction models
            
            parameter_names (list): list of global parameters
            
        Returns:
            
            m (np.array): vector of duals
            
        """
        if DEBUG:
            stuck = '*'*50
            print(stuck)
            print('\nCalculating m')
        
        m = pd.DataFrame(np.zeros((len(parameter_names), 1)), index=parameter_names, columns=['dual'])
        
        kwargs = {
            'param_con_method': 'global',
            'kkt_method': 'k_aug',
            'set_param_bounds': False,
            'param_set_name': 'parameter_names',
            }
        
        for i, model_opt in enumerate(scenarios):
            
            rh = ReducedHessian(model_opt, file_number=i, **kwargs)
            rh.parameter_set = parameter_names
            
            if not hasattr(model_opt, 'd'):
                rh.optimize_model(d=x)
            
            duals = rh.calculate_duals() 
            for param in m.index:
                if param in duals.keys():
                    m.loc[param] = m.loc[param] + duals[param]

            rh.delete_sol_files()

        m = m.values.flatten()
        
        if DEBUG:
            print(f'm: {m}')
            print(stuck)
        
        return m
    
    @staticmethod
    def calculate_M(x, scenarios, parameter_names):
        """Calculate the sum of reduced Hessians for the NSD
        
        Args:
            x (np.array): array of parameter values
            
            scenarios (list): list of reaction models
            
            parameter_names (list): list of global parameters
            
        Returns:
            
            M (np.array): sum of reduced Hessians
        """
        if DEBUG:
            stuck = '*'*50
            print(stuck)
            print('\nCalculating M')
            
        M_size = len(parameter_names)
        M = pd.DataFrame(np.zeros((M_size, M_size)), index=parameter_names, columns=parameter_names)
            
        for i, model in enumerate(scenarios):
            
            rh = ReducedHessian(model, file_number=i)
            rh.parameter_set = parameter_names
            reduced_hessian = rh.calculate_reduced_hessian()
            
            M = M.add(reduced_hessian).combine_first(M)
            M = M[parameter_names]
            M = M.reindex(parameter_names)
        
        M = M.values# + np.eye(M_size)*0.1
        
        if DEBUG:
            print(f'M: {M}')
            print(f'Det:  {np.linalg.det(M):0.4f}')
            print(f'Rank: {np.linalg.matrix_rank(M)}')
            print(f'EigVals: {np.linalg.eigh(M)[0]}')
            print(stuck)
        
        return M
    
    @staticmethod
    def calculate_grad(x, scenarios, parameter_names):
        """Calculate the average of the gradients for the NSD
        
        Args:
            x (np.array): array of parameter values
            
            scenarios (list): list of reaction models
            
            parameter_names (list): list of global parameters
            
        Returns:
            
            M (np.array): sum of reduced Hessians
        """
        return np.zeros((len(x), 1))
    
    
    def parameter_initialization(self):
        
        """Sets the initial parameter values in each scenario to d_init"""
        
        for model in self.model_list:
            for param, model_param in model.P.items():
                model_param.value = self.d_init[param][0]
                
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
        
        
    # def update_model_parameters(self):
        
    #     # Newton Step
    #     # Update the correct, final parameters taking scaling into account
    #     for m, reaction in enumerate(self.reaction_models.values()):
    #         for k, v in reaction.p_model.P.items():
    #             if scaled:
    #                 reaction.p_model.P[k].set_value(self.model_list[m].K[k].value*self.model_list[m].P[k].value)
    #             else:
    #                 reaction.p_model.P[k].set_value(self.model_list[m].P[k].value)
                    
    #     # Ipopt
    #     for name, model in self.reaction_models.items():
    #         for param, model_param in model.model.P.items():
    #             model_param.value = self.parameters_opt[param]
                
        # TR
        
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
        print(bounds)
        
        import ipopt as cyipopt
        nlp = cyipopt.problem(n = len(d_vals),
                              m = 0,
                              problem_obj = problem_object,
                              lb = bounds.lb,
                              ub = bounds.ub,
                              cl = [],
                              cu = [],
                              )
    
        options = {'tol': 1e-8, 
                 #  'bound_relax_factor': 1.0e-8, 
                   'max_iter': 100,
                   'print_user_options': 'yes', 
                   'nlp_scaling_method': 'none',
                   #'corrector_type': 'primal-dual',
                   #'alpha_for_y': 'full',
               #    'accept_every_trial_step': 'yes',
                  # 'linear_solver': 'ma57'
                   }

        if options: 
            for key, value in options.items():
                nlp.addOption(key, value)
        
        x, results = nlp.solve(d_vals)
        
        # Prepare parameter results
        # print(d_init_unscaled)
        # if scaled:
        #     s_factor = {k: self.d_init_unscaled[k] for k in self.d_init.keys()}
        # else:
        #     s_factor = {k: 1 for k in self.d_init.keys()}
        
        # self.parameters_opt = {k: results['x'][i]*s_factor[k] for i, k in enumerate(self.d_init.keys())}
        
        self.parameter_scaling_conversion(results['x'])
        
        for name, model in self.reaction_models.items():
            for param, model_param in model.model.P.items():
                model_param.value = self.parameters_opt[param]
    
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
            #'xtol': 1e-6,
            }
        
        results = minimize(
            self.objective_function, 
            d_vals,
            args=(self.model_list, self.parameter_names), 
            method=self.method,
            jac=self.calculate_m,
            hess=self.calculate_M,
            callback=self.callback,
            bounds=self._generate_bounds_object(),
            options=tr_options,
        )
            
        # End internal methods
        self.parameter_scaling_conversion(results.x)
        results_kipet = self.get_results()
        return results_kipet
    
    def run_simple_newton_step(self, debug=False, alpha=1, iterations=3, opt_tol=1e-8):
        """Performs NSD using simple Newton Steps
        
        This is primarily for testing purposes
        
        """
        d_vals = self.parameter_initialization()
        
        # Start internal methods   
        self.callback(d_vals)

        for i in range(iterations):
            
            self.objective_function(
                d_vals,
                self.model_list, 
                self.parameter_names,
            )
           
            # Get the M matrices to determine search direction
            M = self.calculate_M(d_vals, self.model_list, self.parameter_names)
            m = self.calculate_m(d_vals, self.model_list, self.parameter_names)
            
            # Calculate the search direction
            d = np.linalg.inv(M) @ -(m)
            d_vals = d*alpha + d_vals
            self.callback(d_vals)
            # self.d_iter.append(d_vals)
            
            print(f'Current Parameters in Iteration {i}: {d_vals}')
            # Update model parameters - This is not set-up properly - repeats calc above
            for model in self.model_list:
                for j, param in enumerate(self.parameter_names):
                    model.P[param].set_value(d[j]*alpha + model.P[param].value)
                    
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
            
        # End internal methods
        self.parameter_scaling_conversion(d_vals)
        results_kipet = self.get_results()
        return results_kipet
    
    def callback(self, x, *args):
        """Method to record the parameters in each iteration"""
        self.d_iter.append(x)
    
    def get_results(self):
        
        solver_results = {}
        for name, model in self.reaction_models.items():
            
            model.p_estimator._get_results()
            solver_results[name] = ResultsObject()
            solver_results[name].load_from_pyomo_model(model.p_model)
            model.results = solver_results[name]
            
        self.results = solver_results 
            
        return solver_results
    
    def plot_paths(self, filename=''):
        """Plot the parameter paths through parameter space during the NSD
        algorithm. For diagnostic purposes.
        
        """
        x_data = list(range(1, len(self.d_iter) + 1))
        y_data = np.r_[self.d_iter]
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

    def objective(self, x):
        
        scenarios = self.kwargs.get('scenarios', None)
        parameter_names = self.kwargs.get('parameter_names', None)
        
        return self.fun(x, scenarios, parameter_names)
    
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


def calculate_parameter_averages(model_dict):
    
    p_dict = {}
    c_dict = {}
    
    for key, model in model_dict.items():
        for param, obj in getattr(model, 'P').items():
            if param not in p_dict:
                p_dict[param] = obj.value
                c_dict[param] = 1
            else:
                p_dict[param] += obj.value
                c_dict[param] += 1
                
    avg_param = {param: p_dict[param]/c_dict[param] for param in p_dict.keys()}
    
    return avg_param

if __name__ == '__main__':
   
    pass
    # from kipet.new_examples.Ex_18_NSD import generate_models, generate_models_cstr
    
    # # Generate the ReactionModels (at some point KipetModel)
    # models = generate_models()

    # # Create the NSD object using the ReactionModels list
    # kwargs = {'kipet': True,
    #           'objective_multiplier': 1
    #           }
    # # The objective_multiplier is helpful when the objective is already small
    
    # # Choose the method used to optimize the outer problem
    # strategy = 'ipopt'
    # #strategy = 'newton-step'
    # #strategy = 'trust-region'
    
    # nsd = NSD(models, kwargs=kwargs)
    
    # # Set the initial values
    # nsd.set_initial_value({'k1' : 0.6,
    #                        'k2' : 1.2}
    #                       )
    
    # # nsd.set_initial_value({'Cfa' : 1.500000,
    # #                         'rho' : 1.600000,
    # #                         'ER' : 1.100000,
    # #                         'k' : 1.100000,
    # #                         'Tfc' : 1.100000,
    # #                         }
    # #                       )
    
    # print(nsd.d_init)
    
    # if strategy == 'ipopt':
    #     # Runs the IPOPT Method
    #     results = nsd.ipopt_method(scaled=True)
    
    # elif strategy == 'trust-region':
    #     # Runs the Trust-Region Method
    #     results = nsd.trust_region(scaled=False)
    #     # Plot the parameter value paths (Trust-Region only)
    #     nsd.plot_paths()
        
    # elif strategy == 'newton-step':
    #     # Runs the NSD using simple newton steps
    #     nsd.run_simple_newton_step(alpha=0.1, iterations=15)  
    
    # # Plot the results using ReactionModel format
    # nsd.plot_results()
