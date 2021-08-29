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

# Third party imports
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.offline import plot
from pyomo.environ import (
    Suffix,
    )
from scipy.optimize import (
    Bounds,
    minimize,
    )
from scipy.sparse import coo_matrix

# KIPET library imports
from kipet.estimator_tools.multiprocessing_kipet import Multiprocess
from kipet.estimator_tools.reduced_hessian_methods import calculate_reduced_hessian
from kipet.estimator_tools.results_object import ResultsObject        
from kipet.model_tools.pyomo_model_tools import get_vars

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
                 ):
        
        kwargs = kwargs if kwargs is not None else {}
        parameter_name = kwargs.get('parameter_name', 'P')
        self.objective_multiplier = kwargs.get('objective_multiplier', 1)
        
        self.isKipetModel = kwargs.get('kipet', True)
        self.scaled = scaled
        self.reduced_hessian_kwargs = {}
        self.reaction_models = models_dict
        pre_solve = False
        self._model_preparation(use_duals=True)
        self.model_dict = {name: r.p_model for name, r in models_dict.items()}
        
        avg_param = calculate_parameter_averages(self.model_dict)
        self.strategy = strategy
        
        self.method = 'trust-constr'
        
        all_parameters = []
        for model in self.model_list:
            for param in model.P.keys():
                if param not in all_parameters:
                    all_parameters.append(param)
        
        self.parameter_names = all_parameters
        self.parameter_global = [v for v in all_parameters if v in global_parameters]
        
        print(f'{self.parameter_names = }')
        print(f'{self.parameter_global = }')
        self.d_init = {p: v for p, v in avg_param.items() if p in global_parameters}
        self.d_init_unscaled = None
        self.d_iter = []
        self.parallel = parallel
        
        if init is not None:
            for k, v in init.items():
                if k in global_parameters:
                    self.d_init[k][0] = v
        
        print(self.d_init)
        
        self.x = [0 for k, v in self.d_init.items()]
        self.d = [0 for i in range(len(self.reaction_models))]
        self.M = pd.DataFrame(np.zeros((len(self.parameter_global), len(self.parameter_global))), index=self.parameter_global, columns=self.parameter_global)
        self.m = pd.DataFrame(np.zeros((len(self.parameter_global), 1)), index=self.parameter_global, columns=['dual'])
        self.duals = [0 for i in range(len(self.reaction_models))]
        self.rh = [0 for i in range(len(self.reaction_models))]
        self.obj = [0 for i in range(len(self.reaction_models))]
        self.stub = [None for v in self.reaction_models.values()]
        self.final_param = {}
        
       
        print(f'{self.x = }')
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
    
    def run_opt(self):
        
        print(f'# NSD method using {self.strategy} as the soving strategy')

        if self.strategy == 'ipopt':
            results = self.ipopt_method()
        
        elif self.strategy == 'trust-region':
            results = self.trust_region()
            
        elif self.strategy == 'newton-step':
            results = self.run_simple_newton_step(alpha=0.15, iterations=80) 
        else:
            pass
        
        self.final_param = dict(zip(self.parameter_names, self.d_iter[-1]))
        
        if self.parallel:
            self.solve_mp_simulation()
        
        return None
    
    
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

    
    def _update_x_impl(self, x):
        
        """Yet another wrapper for the objective function"""
        
        print(f'{self.parameter_global = }')
        objective_value = 0
        
        if self.parallel:
            objective_value = self.solve_mp('objective', [x, self.parameter_names, {}])
        else:
            for i, model in enumerate(self.model_list):
                objective_value += self.solve_element('obj', model, x, i, True)
                
        self.objective = objective_value
        self.x = x
        
        return None
    
    
    def _update_m_impl(self, x, optimize=False):
        
        m = pd.DataFrame(np.zeros((len(self.parameter_global), 1)), index=self.parameter_global, columns=['dual'])
        for i, model in enumerate(self.model_list):
            duals = self.solve_element('duals', model, x, i, optimize)
            for param in m.index:
                if param in duals.keys():
                    m.loc[param] = m.loc[param] + duals[param]

        self.m = m
        
    
    def _update_M_impl(self, x, optimize=False):
        
        M = pd.DataFrame(np.zeros((len(self.parameter_global), len(self.parameter_global))), index=self.parameter_global, columns=self.parameter_global)
        for i, model in enumerate(self.model_list):  
            reduced_hessian = self.solve_element('rh', model, x, i, optimize)
            M = M.add(reduced_hessian).combine_first(M)
            M = M[self.parameter_global]
            M = M.reindex(self.parameter_global)
            print(f'This is M for {i}')
            print(M)
                
        self.M = M
    
    
    def objective_function(self, x, scenarios, parameter_names):
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
        
        print('# Objective Function:')
        print(f'# {x = }')
        print(f'# {self.x = }')
        
        if not np.array_equal(x, self.x):
            print('# The values do not match, updating objective function')
            self._update_x_impl(x)
        else:
            print('# The values match, no optimization of the model')
        
        #for i, p in enumerate(parameter_names):
        #    print(f'{p} = {x[i]:0.12f}')
                
        if DEBUG:
            print(f'Obj: {self.objective}')
            print(stuck)
            
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
        if DEBUG:
            stuck = '*'*50
            print(stuck)
            print('\nCalculating m')
            
        optimize = False
        if not np.array_equal(x, self.x):
            print('# The values do not match, updating objective function')
            optimize = True
        else:
            print('# The values match, no optimization of the model')
            
        self._update_m_impl(x, optimize=optimize)

        m = self.m.values.flatten()
        
        if DEBUG:
            print(f'{m = }')
            print(stuck)
        
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
        if DEBUG:
            stuck = '*'*50
            print(stuck)
            print('\nCalculating M')
            
        optimize = False
        if not np.array_equal(x, self.x):
            print('# The values do not match, updating objective function')
            optimize = True
        else:
            print('# The values match, no optimization of the model')
            
        self._update_M_impl(x, optimize=optimize)
    
        
        print(type(self.M.values))
        M = self.M.values# + np.eye(M_size)*0.1
        
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
                if param in self.parameter_global:
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
            'xtol': 1e-4,
            }
    
        print('# Starting Trust-Region algorithm')
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
            
        print('# Finished with Trust-Region algorithm')
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
            M = self.calculate_M(d_vals)
            m = self.calculate_m(d_vals)
            
            print('NS data')
            print(f'{M = }')
            print(f'{m = }')
            
            
            # Calculate the search direction
            d = np.linalg.inv(M) @ -(m)
            d_vals = d*alpha + d_vals
            self.callback(d_vals)
            # self.d_iter.append(d_vals)
            
            print(f'{d = }')
            print(f'{d_vals = }')
            
            print(f'Current Parameters in Iteration {i}: {d_vals}')
            # Update model parameters - This is not set-up properly - repeats calc above
            for model in self.model_list:
                for j, param in enumerate(self.parameter_global):
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
        #if not [x] in self.d_iter:
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
    
    
    def general_update(self, model, x, file_number):
        """Updates the model using the reduced hessian method using global,
        fixed parameters. This may need to be updated to handle parameters that
        are not fixed, such as local effects.
        
        :param ConcreteModel model: The current model of the system used in optimization
        :param np.ndarra x: The current parameter array
        :param int file_number: The index of the model
        
        :return: None
        
        """
        print(f'{self.parameter_global = }')
        
        rh, stub, duals = calculate_reduced_hessian(
            model, 
            d=x, 
            optimize=True,
            parameter_set=self.parameter_names,
            fix_method='global', 
            rho=10, 
            scaled=False,
            stub=None,
            return_duals=True,
            global_set=self.parameter_global)
        
        self.rh[file_number] = rh
        self.stub[file_number] = stub
        self.duals[file_number] = duals
        self.obj[file_number] = model.objective.expr()

    
    def solve_element(self, element, model, x, number, optimize=True):
        
        if optimize:
            self.general_update(model, x, number)
        values = getattr(self, element)[number]
        print(f'{element} = {values}')
        
        return values
    
    
    def solve_model_objective_mp(self, model, x, parameter_names, file_number, kwargs={}):
        """Wrapper for obtaining the objective function value
        
        """
        rh, stub, duals, param_values = self.general_update_mp(model, x, file_number)
        
        return model.objective.expr(), rh, stub, duals, param_values
    
    
    def general_update_mp(self, model, x, file_number):
        
        rh, stub, duals = calculate_reduced_hessian(
            model, 
            d=x, 
            optimize=True,
            parameter_set=self.parameter_names,
            fix_method='global', 
            rho=10, 
            scaled=False,
            stub=None,
            return_duals=True,
            global_set=self.parameter_global)
        
        param_values = {k: v.value for k, v in model.P.items()}
        
        return rh, stub, duals, param_values
    
    
    # def m_func(self, q, i, args):
    #     """This takes the input and passes it to the target function
        
    #     """
    #     print('# Starting Multiprocessing Procedure')
    #     print(f'# Process ID: {os.getpid()}')
        
    #     x = args[0]
    #     parameter_names = args[1]
    #     kwargs = args[2]
        
    #     model_to_solve = self.model_list[i - 1]
        
    #     data = self.solve_element('duals', model_to_solve, x, parameter_names, i, kwargs)
    #     q.put(data)
        
    #     print('# Finished Multiprocessing Procedure')
    
    
    # def M_func(self, q, i, args):
    #     """This takes the input and passes it to the target function
        
    #     """
    #     print('# Starting Multiprocessing Procedure')
    #     print(f'# Process ID: {os.getpid()}')
        
    #     x = args[0]
    #     parameter_names = args[1]
    #     kwargs = args[2]
        
    #     model_to_solve = self.model_list[i - 1]
        
    #     data = self.solve_element('rh', model_to_solve, x, parameter_names, i, kwargs)
    #     q.put(data)
        
    #     print('# Finished Multiprocessing Procedure')


    def obj_func(self, q, i, args):
        """This takes the input and passes it to the target function
        
        """
        print('# Starting Multiprocessing Procedure')
        print(f'# Process ID: {os.getpid()}')
        
        x = args[0]
        parameter_names = args[1]
        kwargs = args[2]
        
        model_to_solve = self.model_list[i - 1]
        
        data = self.solve_model_objective_mp(model_to_solve, x, parameter_names, i, kwargs)
        q.put(data)
        
        print('# Finished Multiprocessing Procedure')

    
    def solve_mp(self, func, args):

        # if func == 'objective':
        mp = Multiprocess(self.obj_func)
        data = mp(args, num_processes = min(len(self.model_list), cpu_count()))
        obj = 0 
 
        for i, d in enumerate(data.values()):    
            obj += d[0]
            self.rh[i] = d[1]
            self.stub[i] = d[2]
            self.duals[i] = d[3]
            self.d[i] = d[4]
        
        return obj
            
        # elif func == 'M':
        #     mp = Multiprocess(self.M_func)
        #     data = mp(args, num_processes = min(len(self.reaction_models), cpu_count()))
            
        #     return data
            
            
        # elif func == 'm':
        #     mp = Multiprocess(self.m_func)
        #     data = mp(args, num_processes = min(len(self.reaction_models), cpu_count()))
            
        #     results = {}
            
        #     for key, value in data.items():
        #         for k, v in value.items():
        #             if k not in results:
        #                 results[k] = v
        #             else:
        #                 results[k] += v
                    
        #     return results
            
        # else:
        #     raise ValueError('You have made a big coding mistake')
            
        #     return None
        
        
    def func_simulation(self, q, i):
        print('starting')
        print('process_id', os.getpid())
        
        model_to_solve = list(self.reaction_models.values())[i - 1]
        model_to_solve.parameters.update('value', self.d[i - 1])
        # for param, p_obj in model_to_solve._P.items():
        #     if param in self.d[i-1]:
        #         p_obj.set_value(self.d[i-1][param])
        
        data = self.solve_simulation(model_to_solve)
        q.put(data)
        print('all done')

    def solve_simulation(self, model):
        """Uses the ReactionModel framework to calculate parameters instead of
        repeating this in the MEE
    
        # Add mp here
    
        """
        model.simulate(self.final_param)
        print('Model has been simulated')

        attr_list = ['name', 'results_dict']
    
        model_dict = {}
        for attr in attr_list:
            model_dict[attr] = getattr(model, attr)
    
        return model_dict['results_dict']
    
    def solve_mp_simulation(self):
       
        mp = Multiprocess(self.func_simulation)
        data = mp(num_processes = min(len(self.reaction_models), cpu_count()))
        
        self.mp_results = data
        simulator = 'simulator'
        estimator = 'p_estimator'
        
        print(f'{self.d = }')
        
        for i, model in enumerate(self.reaction_models.values()):
            
            model._pe_set_up(solve=False)
            
            setattr(model, 'results_dict', self.mp_results[i + 1])
            setattr(model, 'results', self.mp_results[i + 1][simulator])
            print(model.models)
            vars_to_init = get_vars(model.p_model)
            
            for var in vars_to_init:
                if hasattr(model.results_dict[simulator], var) and var != 'S':
                    getattr(model, estimator).initialize_from_trajectory(var, getattr(model.results_dict[simulator], var))
                elif var == 'S' and hasattr(model.results_dict[simulator], 'S'):
                    getattr(model, estimator).initialize_from_trajectory(var, getattr(model.results_dict[simulator], var))
                else:
                    print(f'Variable: {var} is not updated')
            
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

if __name__ == '__main__':
   
    pass
   