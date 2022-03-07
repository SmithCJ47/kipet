"""
This module implements the reduced Hessian parameter selection method outlined
in Chen and Biegler (AIChE 2020).

"""
# Standard library imports
from collections import namedtuple
import copy
from string import Template

# Third party imports
import numpy as np
from pyomo.environ import (Objective, SolverFactory,
                           Suffix)

# KIPET library imports
from kipet.model_components.objectives import comp_objective, conc_objective
from kipet.estimability_tools.parameter_handling import (check_initial_parameter_values,
                                                       set_scaled_parameter_bounds)
from kipet.estimability_tools.parameter_ranking import parameter_ratios, rank_parameters
from kipet.estimator_tools.parameter_estimator import ParameterEstimator
from kipet.estimator_tools.results_object import ResultsObject
from kipet.model_tools.scaling import (scale_parameters,
                                            update_expression)

from kipet.estimator_tools.reduced_hessian_methods import (
    prepare_pseudo_fixed_global_variables,
    generate_parameter_index_lists,
    build_matrices, 
    calculate_reduced_hessian,
    optimize_model,
    reduced_hessian,
    reduced_hessian_single_model,
    kkt_kaug,
    )
from kipet.general_settings.variable_names import VariableNames
from kipet.general_settings.solver_settings import solver_path

import kipet.estimator_tools.reduced_hessian_methods as rhm
# from pyomo.contrib.pynumero.interfaces.pyomo_nlp import PyomoNLP
# from pyomo.contrib.pynumero.interfaces.nlp_projections import ProjectedNLP
# from pyomo.contrib.pynumero.algorithms.solvers.cyipopt_solver import CyIpoptSolver, CyIpoptNLP

__author__ = 'Kevin McBride'  #: April 2020
    
class EstimationPotential():
    """This class is for estimability analysis. The algorithm here is the one
    presented by Chen and Biegler (accepted AIChE 2020) using the reduced 
    hessian to select the estimable parameters. 

    :Attributes:
    
        - model_builder (pyomo ConcreteModel): The pyomo model
    
        - simulation_data (pandas.DataFrame): Optional simulation data to use for
            warm starting (Needs testing!)
        
        - options (dict): Various options for the esimability algorithm:
        
            - nfe (int): The number of finite elements to use in the collocation.
            
            - ncp (int): The number of collocation points per finite element.
            
            - bound_approach (float): The accepted relative difference for
                determining whether or not a bound is considered active (Step 6).
                
            - rho (float): Factor used to determine the lower and upper bounds for
                each parameter in fitting (Step 5).
                
            - epsilon (float): The minimum value for parameter values
            
            - eta (float): Predetermined cut-off value for accepted std/parameter
                ratios.
                
            - max_iter_limit (int): Iteration limits for the estimability algorithm.
            
            - verbose (bool): Defaults to False, option to display the progress of
                the algorihm.
                
            - debug (bool): Defaults to False, option to ask for user input to
                proceed during the algorithm.
                
            - simulate_start (bool): Option to simulate using the model to
                warm start the optimization
        
    """

    def __init__(self, r_model, simulation_data=None, options=None,
                 method='k_aug', solver_opts={}, scaled=True,
                 use_bounds=False, use_duals=False, calc_method='fixed'):

        """Initialization of the EstimationPotential class

        :param ConcreteModel model: The Pyomo model to perform analysis on
        :param pandas.DataFrame simulation_data: Initialization data for the model
        :param dict options: Dictionary of options to use in the RHPS method
        :param str method: The method to use in calculating sensitivities
        :param dict solver_opts: Options for the solver, if any
        :param bool scaled: Optional scaling
        :param bool use_bounds: Optional use of parameter bounds (dev opt)
        :param bool use_duals: Use the dual option in sensitiivty calculations (dev opt)
        :param str calc_method: Method to calculate the reduced Hessian (dev opt)
        """
        # Options handling
        self.options = {} if options is None else options.copy()
        self._options = options.copy()
        
        print(self._options)
        
        self.debug = self._options.pop('debug', False)
        self.verbose = self._options.pop('verbose', False)
        self.nfe = self._options.pop('nfe', 50)
        self.ncp = self._options.pop('ncp', 3)
        self.bound_approach = self._options.pop('bound_approach', 1e-2)
        self.rho = self._options.pop('rho', 10)
        self.epsilon = self._options.pop('epsilon', 1e-16)
        self.eta = self._options.pop('eta', 0.1)
        self.max_iter_limit = self._options.pop('max_iter_limit', 20)
        self.simulate_start = self._options.pop('sim_start', False)
        self.method = method
        self.solver_opts = solver_opts
        self.scaled = scaled
        self.use_bounds = use_bounds
        self.use_duals = use_duals
        self.rh_method = calc_method
        self.r_model = r_model
        self.__var = VariableNames()
        
        # Copy the model
        self.model = copy.deepcopy(r_model.p_model)
        
        self.model.ipopt_zL_out = Suffix(direction=Suffix.IMPORT)
        self.model.ipopt_zU_out = Suffix(direction=Suffix.IMPORT)
        # Ipopt bound multipliers (sent to solver)
        self.model.ipopt_zL_in = Suffix(direction=Suffix.EXPORT)
        self.model.ipopt_zU_in = Suffix(direction=Suffix.EXPORT)
        # Obtain dual solutions from first solve and send to warm start
        self.model.dual = Suffix(direction=Suffix.IMPORT_EXPORT)
        
        self.simulation_data = simulation_data
        
        self.orig_bounds = None
        
        # This needs to be updated too
        if not self.scaled and self.use_bounds:
            self.orig_bounds = {k: (v.lb, v.ub) for k, v in self.model.P.items()}
        
        self.debug = False
        self.verbose = True
        
        self.params_init = self._build_var_dict(init=True)
        
        self.H_old = []
        self.H_new = []
        #self.rh = []
        
    def __repr__(self):
        
        repr_str = (f'EstimationPotential({self.model},'
                    f' simulation_data={"Provided" if self.simulation_data is not None else "None"},'
                    f' options={self.options})')
        
        return repr_str
        
    
    def _build_var_dict(self, params=[], init=False):
        
        print(f'{params = }')
        P_var = namedtuple('P_var', ('name', 'full_name', 'model_var', 'pyomo_var'))
        var_objs = {}

        if init:

            for opt_var in self.__var.optimization_variables:
                if hasattr(self.model, opt_var):
                    for k, v in getattr(self.model, opt_var).items():
                        #print(f'{k = }')
                        if v.fixed or v.stale or k in params:
                            # print(f'In set: {k in params}')
                            # print(f'Is fix: {v.fixed}')
                            # print(f'Is old: {v.stale}')
                                  
                            continue
                        else:
                            #print('adding to vars')
                            p_var = P_var(k, f'{opt_var}[{k}]', opt_var, v)
                            var_objs[p_var.name] = p_var
    
        else:
            
            for var, p_var in self.params_init.items():
                
                if var in params:
                    # print(f'In set: {k in params}')
                    # print(f'Is fix: {v.fixed}')
                    # print(f'Is old: {v.stale}')
                    continue
                else:
                    # print('adding to vars')
                    var_objs[p_var.name] = p_var
    
        return var_objs
    
    @staticmethod
    def _move_parameter(Se, Sf, up=True):
        
        if up:
            first_value = next(iter(Sf.items()))[0]
            param_to_move = {first_value: Sf[first_value]}
            Sf.pop(first_value)
            Se.update(param_to_move)
        else:
            param_to_move = Se.popitem()
            Sf_new = {param_to_move[0]: param_to_move[1]}
            Sf_new.update(Sf)
            Sf = Sf_new
            
        return Se, Sf
    
    
    def estimate(self):
        """This performs the estimability analysis based on the method
        developed in Chen and Biegler 2020 AIChE...

        :return: None

        """
        bound_check = True     

        flag = False
        step = Template('\n' + '*' * 20 + ' Step $number ' + '*' * 20)
        
        self._model_preparation()
        
        # Step 1
        if self.verbose:
            print(step.substitute(number=1))
            print('Initializing N_pre and N_curr\n')
        
        self.p_list = self._build_var_dict()
        #print(f'{self.p_list = }')
        
        #parameter_ref_dict = {}
        Se = self.p_list
        # Se_init = list(Se.keys())[0]
        # print(f'{Se_init = }')
        # Se = Se.fromkeys([Se_init], Se[Se_init])
        Sf = self._build_var_dict([p.name for p in Se.values()])
        # for var in self.__var.optimization_variables:
        #     if hasattr(self.model, var):
        #         for parameter, obj in getattr(self.model, var).items():
        #             #print('In the loop')
        #             #print(obj.pprint())
        #             if obj.fixed or obj.stale:
        #                 continue
        #             Se.append(parameter)
        #             parameter_ref_dict[parameter] = var
                
        #print(f'{parameter_ref_dict = }')
        
        N_pre = len(Se)
        N_curr = len(Se)
        
        # Step 2
        if self.verbose:
            print(step.substitute(number=2))
            print('Initialize Se and Sf\n')
        
        
        # parameter_ref_dict = {}
        # Se = []
        # for var in self.__var.optimization_variables:
        #     if hasattr(self.model, var):
        #         Se.extend([parameter for parameter in getattr(self.model, var).keys()])
        #         parameter_ref_dict.update({parameter: var for parameter in getattr(self.model, var).keys()})
                
        #print(f'{parameter_ref_dict = }')
        
        #Se = [parameter for parameter in self.model.P.keys()]
        #Sf = []
        
        if self.verbose:
            print(f'Se: {[p for p in Se]}')
            print(f'Sf: {[p for p in Sf]}')
        
        # Step 3 - Calculate the reduced hessian for the initial stage
        if self.verbose:
            print(step.substitute(number=3))
            print('Calculating the Reduced Hessian for the initial parameter set\n')
        
        rh_fun = '_calculate_reduced_hessian'
        
        #reduced_hessian_new = self._calculate_reduced_hessian_new(Se)
        #print(f'{reduced_hessian_new = }')
        reduced_hessian = getattr(self, rh_fun)(Se) #_calculate_reduced_hessian_new(Se)
        print(f'{reduced_hessian = }')
        
        print(f'{reduced_hessian.shape = }')
        if self.debug:
            
            input("Press Enter to continue...")
    
        # Step 4 - Rank the parameters using Gauss-Jordan Elimination
        if self.verbose:
            print(step.substitute(number=4))
            print('Ranking parameters for estimability and moving to Step 5')
        
        self.rh = reduced_hessian
        Se, Sf = rank_parameters(self.model, reduced_hessian, Se, epsilon=self.epsilon, eta=self.eta)
        
        Se = self._build_var_dict(Sf)
        Sf = self._build_var_dict([p.name for p in Se.values()])
        
        if len(Se) >= N_curr:
            number_of_parameters_to_move = len(Se) - N_curr + 1
            for i in range(number_of_parameters_to_move):
                
                Se, Sf = self._move_parameter(Se, Sf, up=False)
                
                # param_to_move = Se.popitem()
                # Sf_new = {param_to_move[0]: param_to_move[1]}
                # Sf_new.update(Sf)
                # Sf = Sf_new
            
            N_pre = len(Se)
            N_curr = len(Se)
       
        if self.verbose:
            print(f'\nThe updated parameter sets are:\nSe: {[p for p in Se]}\nSf: {[p for p in Sf]}')
        
        if self.debug:
            input("Press Enter to continue...")

        # Step 5 - Optimize the estimable parameters
        outer_iteration_counter = 0
        params_counter = 0
        self.saved_parameters_K = {}
        
        while True:
        
            if outer_iteration_counter > self.max_iter_limit:
                print('Maximum iteration limit reached - check the model!')
                break
            
            inner_iteration_counter = 0
            
            while True:
            
                if inner_iteration_counter > self.max_iter_limit:
                    print('Maximum iteration limit reached - check the model!')
                    break
                
                if self.verbose:
                    print(step.substitute(number=5))
                    print('Optimizing the estimable parameters\n')
                    
                print([p for p in Se])
                for free_param in Se:
                    
                    #
                    #print([p for p in free_param])
                    
                    Se[free_param].pyomo_var.unfix()
                    
                    #self.model.P[free_param].unfix()
                    
                for fixed_param in Sf:
                    if self.scaled:
                        
                        Sf[fixed_param].pyomo_var.fix(1)
                        
                        #getattr(self.model, parameter_ref_dict[fixed_param])[fixed_param].fix(1)
                        #self.model.P[fixed_param].fix(1) # changed from (1) to ()
                    else:
                        Sf[fixed_param].pyomo_var.fix()
                        #getattr(self.model, parameter_ref_dict[fixed_param])[fixed_param].fix()
                        #self.model.P[fixed_param].fix() # changed from (1) to ()
                        
                ipopt = SolverFactory(solver_path('ipopt'))
                ipopt.options['linear_solver'] = 'ma57'
                #ipopt.options['mu_init'] = 1e-6
                #ipopt.options['mu_strategy'] = 'adaptive'
                ipopt.solve(self.model, tee=True)#self.verbose)
                
                #if self.verbose:
                    
                    # for var in self.p_list.values():
                    #     print(var.pyomo_var.pprint())
                    
                    # for var in self.__var.optimization_variables:
                    #     if hasattr(self.model, var):
                    #         print(getattr(self.model, var).display())
                    
                # Step 6 - Check for active bounds
                number_of_active_bounds = 0
                
                if self.verbose:
                    print(step.substitute(number=6))
                    print('Checking for active bounds\n')
                else:
                    None
                
                if bound_check:
                    
                    # for var in self.__var.optimization_variables:
                        # if hasattr(self.model, var):
                    # var = self.__var.model_parameter
                    # for key, param in getattr(self.model, var).items():
                    #     if (param.value-param.lb + 1e-12)/(param.lb + 1e-12) <= self.bound_approach or (param.ub - param.value)/param.value <= self.bound_approach:
                    #         number_of_active_bounds += 1
                    #         if self.verbose:
                    #             print('There is at least one active bound - updating parameters and optimizing again\n')
                    #         break
                               
                
                
                
                    # for var in set(self.__var.optimization_variables): #.discard(self.__var.model_parameter):
                    #     if hasattr(self.model, var):
                    #         for key, param in getattr(self.model, var).items():
                    for param_name in Se:
                        param = Se[param_name].pyomo_var
                        if (param.value-param.lb)/(param.lb) <= self.bound_approach or (param.ub - param.value)/param.value <= self.bound_approach:
                            number_of_active_bounds += 1
                            if self.verbose:
                                print('There is at least one active bound - updating parameters and optimizing again\n')
                            break
                
                else:
                    None
                
                print(self.model.K.display())
                if self.scaled and hasattr(self.model, 'K'):
                    for k, v in self.model.K.items(): # apply K to the full set....
                        print(k, v.value)
                        self.model.K[k] = self.model.K[k] * self.params_init[k].pyomo_var.value
                        self.params_init[k].pyomo_var.set_value(1)
                        
                    print(self.model.K.display())
                    print(self.model.P.display())
                        
                else:
                    set_scaled_parameter_bounds(self.model,
                                                parameter_set=Se,
                                                parameter_ref_dict=parameter_ref_dict,
                                                rho=self.rho,
                                                scaled=self.scaled,
                                                original_bounds=self.orig_bounds)
                    
                # if hasattr(self.model, 'K'):
                #     param_val_save = 'K'
                # else:
                #     param_val_save = 'P'
                    
                for var in self.params_init:
                    self.saved_parameters_K[params_counter] = {k: self.params_init[k].pyomo_var.value for k in self.params_init}
                
                # saved_parameters_K[params_counter] = {}
                # for var in self.__var.optimization_variables:
                #     saved_parameters_K[params_counter].update({k: v.value for k, v in getattr(self.model, 'K').items()})
                
                
                
                params_counter += 1
                
                if bound_check:
                    if number_of_active_bounds == 0:
                        if self.verbose:
                            print('There are no active bounds, moving to Step 7')
                        break
                        if self.debug:    
                            input("Press Enter to continue...")
                else:
                    None
                        
                inner_iteration_counter += 1
                if self.debug:
                    input("Press Enter to continue...")
                
            if self.debug:
                self.model.P.display()
                self.model.K.display()
                
            reduced_hessian = getattr(self, rh_fun)(Se)
            
            # Step 7 - Check the ratios of the parameter std to value
            if self.verbose:
                print(reduced_hessian)
                print(step.substitute(number=7))
                print('Checking the ratios of each parameter in Se')
            
            ratios, eigvals = parameter_ratios(self.model, reduced_hessian, Se, epsilon=self.epsilon)
            
            if self.verbose:
                 print('Ratios:')
                 print(ratios)
            
            ratios_satisfied = max(ratios) < self.eta
        
            if ratios_satisfied:
                if self.verbose:
                    print(f'Step 7 passed, all paramater ratios are less than provided tolerance {self.eta}, moving to Step 10')
                    if self.debug:
                        input("Press Enter to continue...")

                    # Step 10 - Check the current number of free parameters
                    print(step.substitute(number=10))
                    print(f'N_curr = {N_curr}, N_pre = {N_pre}, N_param = {len(self.p_list)}')
                
                if (N_curr == (N_pre - 1)) or (N_curr == len(self.p_list)):
                    if self.verbose:
                        print('Step 10 passed, moving to Step 11, the procedure is finished')
                        print(f'Se: {[p for p in Se]}')
                        print(f'Sf: {[p for p in Sf]}')
                    break
                else:
                    if self.verbose:
                        print('Step 10 failed, moving first parameter from Sf to Se and moving to Step 5')
                    # first_value = next(iter(Sf.items()))[0]
                    # param_to_move = {first_value: Sf[first_value]}
                    # Sf.pop(first_value)
                    # Se.update(param_to_move)
                    
                    Se, Sf = self._move_parameter(Se, Sf, up=True)
                    
                    print(f'Se: {[p for p in Se]}')
                    print(f'Sf: {[p for p in Sf]}')
                    
                    N_pre = N_curr
                    N_curr = N_curr + 1
                    if self.debug:
                        input("Press Enter to continue...")
                
            else:
                # Step 8 - Compare number of current estimable parameter with previous iteration
                if self.verbose:
                    print('Step 7 failed, moving on to Step 8')
                    print(step.substitute(number=8))
                    print('Comparing the current number of parameters in Se with the previous number')
                    print(f'N_curr = {N_curr}, N_pre = {N_pre}, N_param = {len(self.p_list)}')
                if N_curr == (N_pre + 1):
                    # param_to_move = Se.popitem()
                    # Sf_new = {param_to_move[0]: param_to_move[1]}
                    # Sf_new.update(Sf)
                    # Sf = Sf_new
                    
                    Se, Sf = self._move_parameter(Se, Sf, up=False)
                    
                    print(f'Se: {[p for p in Se]}')
                    print(f'Sf: {[p for p in Sf]}')
                    
                    if self.scaled and hasattr(self.model, 'K'):
                        for k, v in self.model.K.items():
                            self.model.K[k] = self.saved_parameters_K[params_counter-2][k]
                            self.params_init[k].pyomo_var.set_value(1)
                            # self.model.P[k].set_value(1)
                    else:
                        for k, v in self.model.P.items():
                            self.model.P[k].set_value(self.saved_parameters_K[params_counter-1][k])
                        
                    if self.verbose:
                        print('Step 8 passed, moving to Step 11, reloading last Se the procedure is finished')
                        print(f'Se: {[p for p in Se]}')
                        print(f'Sf: {[p for p in Sf]}')
                    break
                else:
                    # Step 9 - Check the inequality condition given by Eq. 27
                    if self.verbose:
                        print('Step 8 failed, moving to Step 9\n')
                        print(step.substitute(number=9))
                        print('Calculating the inequality from Eq. 27 in Chen and Biegler 2020')
                        
                        
                    if sum(1.0/eigvals) < sum((np.array([var.pyomo_var.value for var in Se.values()]))**2)*(self.eta**2) and flag == True:
                        # param_to_move = Se.popitem()
                        # Sf_new = {param_to_move[0]: param_to_move[1]}
                        # Sf_new.update(Sf)
                        # Sf = Sf_new
                        
                        Se, Sf = self._move_parameter(Se, Sf, up=False)
                        
                        if self.verbose:
                            print('Step 9 passed, moving last parameter from Se into Sf and moving to Step 5')
                            print(f'Se: {[p for p in Se]}')
                            print(f'Sf: {[p for p in Sf]}')
                        N_pre = N_curr
                        N_curr = N_curr - 1

                        if self.debug:
                            input("Press Enter to continue...")
                    else:
                        # Step 2a - Reset the parameter vectors (all in Se)
                        if self.verbose:
                            print('Step 9 failed, moving to Step 2\n')
                            print(step.substitute(number=2))
                            print('Reseting the parameter vectors')
                        flag = True
                        
                        Se = self._build_var_dict()
                        Sf = self._build_var_dict([p.name for p in Se.values()])
                        print(f'Se: {[p for p in Se]}')
                        print(f'Sf: {[p for p in Sf]}')
                    
                        if self.debug:
                            input("Press Enter to continue...")
                        
                        if self.verbose:
                            # Step 3a - Recalculate the reduced hessian
                            print(step.substitute(number=3))
                            print('Recalculating the reduced hessian')
                            if self.debug:
                                print(f'Input model:\n')
                                self.model.P.display()
                  
                        reduced_hessian = getattr(self, rh_fun)(Se)
                        
                        if self.debug:
                            print(reduced_hessian)
                            input("Press Enter to continue...")
                        if self.verbose:
                            # Step 4 - Rank the updated parameters using Gauss-Jordan elimination
                            print(step.substitute(number=4))
                            print('Ranking the parameters (limited by N_curr)')
                        
                        Se, Sf = rank_parameters(self.model, reduced_hessian, Se, epsilon=self.epsilon, eta=self.eta)
                        
                        Se = self._build_var_dict(Sf)
                        Sf = self._build_var_dict([p.name for p in Se.values()])
                        
                        if len(Se) >= N_curr:
                            number_of_parameters_to_move = len(Se) - N_curr + 1
                            for i in range(number_of_parameters_to_move):
                                # param_to_move = Se.popitem()
                                # Sf_new = {param_to_move[0]: param_to_move[1]}
                                # Sf_new.update(Sf)
                                # Sf = Sf_new
                                
                                Se, Sf = self._move_parameter(Se, Sf, up=False)
                            
                        N_pre = len(Se)
                        N_curr = len(Se)
                           
                        if self.verbose:
                            print(f'The parameter sets are:\nSe: {Se}\nSf: {Sf}\nN_pre: {N_pre}\nN_curr: {N_curr}')
                        if self.debug:
                            input("Press Enter to continue...")

            outer_iteration_counter += 1                
        
        print(step.substitute(number='Finished'))
        
        self.model.K_vals = self.saved_parameters_K
        
        print(f'The estimable parameters are......: [{", ".join(Se)}]')
        print(f'The non-estimable parameters are..: [{", ".join(Sf)}]')
        #print('\nThe final parameter values are:\n')
        
        # if hasattr(self.model, 'K') and self.model.K is not None:
        #     self.model.K.pprint()
        # else:
        #     self.model.P.pprint()
            
        results = self._get_results(Se)
            
        return results, self.model, Se, Sf
    
    def _get_results(self, Se):
        """Arranges the results into a ResultsObject

        :param list Se: The list of estimable parameters

        :return: The results from the parameter estimation process
        :rtype: ResultsObject

        """
        scaled_parameter_var = 'K'
        results = ResultsObject()
        results.estimable_parameters = Se
        results.load_from_pyomo_model(self.model)

        # if hasattr(self.model, scaled_parameter_var): 
        #     results.P = {name: self.model.P[name].value*getattr(self.model, scaled_parameter_var)[name].value for name in self.model.parameter_names}
        # else:
        #     results.P = {name: self.model.P[name].value for name in self.model.parameter_names}

        return results
    
    def _model_preparation(self):
        """Helper function that should prepare the models when called from the
        main function. Includes the experimental data, sets the objectives,
        simulates to warm start the models if no data is provided, sets up the
        reduced hessian model with "fake data", and discretizes all models

        :return: None

        """
        if not hasattr(self.model, 'objective'):
            self.model.objective = self._rule_objective(self.model)
        
        # The model needs to be discretized
        model_pe = ParameterEstimator(self.r_model, 'p_model')
        model_pe.apply_discretization('dae.collocation',
                                      ncp=self.ncp,
                                      nfe=self.nfe,
                                      scheme='LAGRANGE-RADAU')
        
        # Here is where the parameters are scaled
        if self.scaled:
            scale_parameters(self.model, params=self.params_init)
            set_scaled_parameter_bounds(self.model, rho=self.rho)
        
        else:
            check_initial_parameter_values(self.model)
        
        if self.use_duals:
            self.model.dual = Suffix(direction=Suffix.IMPORT_EXPORT)
        
        return None
    
    def _rule_objective(self, model):
        """This function defines the objective function for the estimability
        
        This is equation 5 from Chen and Biegler 2020. It has the following
        form:
            
        .. math::
            \min J = \frac{1}{2}(\mathbf{w}_m - \mathbf{w})^T V_{\mathbf{w}}^{-1}(\mathbf{w}_m - \mathbf{w})
            
        Originally KIPET was designed to only consider concentration data in
        the estimability, but this version now includes complementary states
        such as reactor and cooling temperatures. If complementary state data
        is included in the model, it is detected and included in the objective
        function.
        
        :param ConcreteModel model: This is the pyomo model instance for the estimability problem.
                
        :return: This returns the objective function
        :rtype: expression

        """
        obj = 0
        obj += 0.5*conc_objective(model) 
        obj += 0.5*comp_objective(model)
    
        return Objective(expr=obj)

    def _calculate_reduced_hessian(self, Se):
        """This function solves an optimization with very restrictive bounds
        on the paramters in order to get the reduced hessian at fixed 
        conditions.

        :param list Se: The current list of estimable parameters.

        :return numpy.ndarray reduced_hessian: The reduced Hessian

        """
        rh_model = copy.deepcopy(self.model)
        self.rh_model = rh_model
        print(f'{Se = }')
        
        Se = [p.name for p in Se.values()]
        global_param_objs = []
        for var in self.__var.optimization_variables:
            if hasattr(rh_model, var):
                global_param_objs.extend([v for k, v in getattr(rh_model, var).items() if k in Se])
        
        model_object_lists = [global_param_objs, [], []]
        rhm.prepare_pseudo_fixed_global_variables(rh_model, Se, model_vars=self.__var.optimization_variables, delta=1e-20)
        nlp_object, _, _ = rhm.optimize_model(rh_model)
        index_list = rhm.generate_parameter_index_lists(nlp_object, model_object_lists)
        print([v.name for v in global_param_objs])
        
        H, J, K = rhm.kkt_kaug(self.rh_model, pyomo_model=True, nlp=nlp_object)
        rh = rhm.calculate_reduced_hessian(H, J, index_list, model_object_lists[0], as_df=True)
        
        #rh = reduced_hessian_single_model(self.r_model)
        #rh = reduced_hessian(nlp_object, model_object_lists, use_cyipopt=False, as_df=True)
        print(f'{rh = }')
        # rh_2 = reduced_hessian_single_model(self.r_model)
        
        #self.H_old.append(H)
        
        return rh

#     @staticmethod
#     def add_global_constraints(model, parameter_set, as_var=True):
#         """This adds the dummy constraints to the model forcing the local
#         parameters to equal the current global parameter values

#         :return: None
        
#         """
#         #%%
#         # model = r1.est_model.model
#         # parameter_set = r1.est_model.p_list
#         # as_var = True
        
#         from pyomo.environ import Constraint, Param, Set, Var
        
#         current_set = 'current_p_set'
#         global_param_name='d'
#         global_constraint_name='fix_params_to_global'
#         param_set_name='parameter_names'
        
#         # parameter_set is the self.p_list dict
        
#         global_param_init = {}
        
#         # The names to be used as the index in the dummy variables
#         model_parameter_set = [p for p in parameter_set]
        
#         print(f'{model_parameter_set = }')
#         #%%
#         for k, p in parameter_set.items():
#             #if p in getattr(model, variable_name):
#             global_param_init[k] = getattr(model, p.model_var)[k].value

#         for comp in [global_constraint_name, global_param_name, current_set]:
#             if hasattr(model, comp):
#                 model.del_component(comp)

#         print(f'{global_param_init = }')
#         setattr(model, current_set, Set(initialize=model_parameter_set))
        
#         if as_var:
            
#             setattr(model, global_param_name,
#                           Var(getattr(model, current_set),
#                           initialize=global_param_init,
#                           ))
#             # for d_var, d_obj in getattr(model, global_param_name).items():
#             #     d_obj.fix()
            
#         else:

#             setattr(model, global_param_name,
#                           Param(getattr(model, param_set_name),
#                           initialize=global_param_init,
#                           mutable=True,
#                           ))

#         def rule_fix_global_parameters(m, k):
#             return getattr(m, parameter_set[k].model_var)[k] - getattr(m, global_param_name)[k] == 0

#         setattr(model, global_constraint_name,
#                 Constraint(getattr(model, current_set),
#                            rule=rule_fix_global_parameters)
#                 )
# #%%
#         return None


#     def add_global_parameter_constraints(self, model):
#         """Adds the global constraints to the p_models instead of in each
#         iteration
        
#         This is not important for k_aug, but critical for pynumero's start
        
#         """
#         # self.conNames = {}
#         # self.varNames = {}
#         # self.allVarNames = {}
#         # self.projected_nlp_list = []
#         # self.nlp_list = []
#         # self.varList = []
        
#         # self.col_dict = {}
#         # self.con_dict = {}
        
#         # counter = 0
#         #%%
#         #self = r1.est_model
#         # for key, model in self.model_dict.items():
#         #model = self.model  
#         # key = self.r_model.name
        
        
#         #params = [p.name for p in self.all_params.parameters if p.identity in self.parameter_global]# in ['global', key]]
#         #print(params)
#         params = [p.name for p in self.p_list.values()]
#         print(params)
    
#         self.add_global_constraints(model, self.p_list, as_var=True)
#         print(model.d.display())
        
#         # nlp = self.r_model._nlp
#         from pyomo.contrib.pynumero.interfaces.pyomo_nlp import PyomoNLP
#         nlp = PyomoNLP(model)
#         self.varNames = nlp.primals_names()
        
        
#         pyomo_vars = dict(zip(nlp.primals_names(), nlp.get_pyomo_variables()))
#         #var_name = 'P'
#         #p_vars = getattr(model, var_name)
#         # self.col_dict = {}
#         # for p, obj in self.p_list.items():
#         #     print(p, obj)
#         #     self.col_dict[obj.full_name] = nlp.get_primal_indices([pyomo_vars[obj.full_name]])[0]
        
#         self.col_dict = {f'{obj.model_var}[{obj.name}]': nlp.get_primal_indices([pyomo_vars[obj.full_name]])[0] for p, obj in self.p_list.items() if not obj.pyomo_var.fixed}
        
#         global_param_name = 'd'
#         global_constraint_name = 'fix_params_to_global'
#         d_vars = getattr(model, global_constraint_name)
#         self.con_dict = [nlp.get_constraint_indices([obj])[0] for d, obj in d_vars.items()]
        
        
#         # all_var_names = list(self.varNames[counter])
#         # for i, (key, obj) in enumerate(getattr(model, global_param_name).items()):
#         #     var_name = f'{global_param_name}[{key}]'
#         #     if var_name in all_var_names:
#         #         all_var_names.pop(all_var_names.index(var_name))
                
#         #self.allVarNames[counter] = all_var_names
#         self.varList = nlp.get_pyomo_variables()[:-len(getattr(model, global_param_name))]
        
        

        
#         projected_nlp = ProjectedNLP(nlp, self.varNames[:-len(getattr(model, global_param_name))])
#         self.nlp_list = nlp
#         self.projected_nlp_list = projected_nlp
            
#         self._has_global_constraints = True
#         #%%
#         return None

#     def _calculate_reduced_hessian_new(self, Se):
#         """Updates the model using the reduced hessian method using global,
#         fixed parameters. This may need to be updated to handle parameters that
#         are not fixed, such as local effects.
        
#         :param ConcreteModel model: The current model of the system used in optimization
#         :param np.ndarra x: The current parameter array
#         :param int file_number: The index of the model
        
#         :return: None
        
#         """
#         #Needed x and r_model
        
#         """
#         Take the NLP of the problem and create a projected_nlp of it based
#         on the Se.
        
#         Calculate the RH based on the results.
        
#         """
#         #%%
#         from kipet.estimator_tools.reduced_hessian_methods import var_con_data, _build_reduced_hessian, free_variables, define_free_parameters, delete_from_csr, SparseRowIndexer, _reduced_hessian_matrix
        
        
        
#         file_number = 0
#         rh_model = copy.deepcopy(self.model)
#         #self = r1.est_model
#         # model_name = self.r_model.name
#         model = rh_model #self.r_model.p_model
#         self.add_global_parameter_constraints(model) # possibly pass Se here
        
#         # global_parameters = self.global_parameters[file_number]
#         nlp = PyomoNLP(model) #self.nlp_list
#         projected_nlp = ProjectedNLP(nlp, self.varNames[:-len(getattr(rh_model, 'd'))])
#         #projected_nlp = #self.projected_nlp_list
#         con_ind_dict = self.con_dict
#         varList = self.varList
#         col_ind_dict = self.col_dict
#         col_ind_list = [v for v in col_ind_dict.values()]
        
#         glv_indx = nlp.get_primal_indices([getattr(nlp.pyomo_model(), 'd')])
        
#         current_primals = nlp.get_primals()
        
#         print(f'{col_ind_dict = }')
        
        
#         #if self.local_as_global:
#         # x = self._arrange_model_specific_x(r_model, x)
        
#         # current_primals[col_ind_list] = x
#         # current_primals[glv_indx] = x
#         # nlp.set_primals(current_primals)
       
#         cy_nlp = CyIpoptNLP(projected_nlp)
#         csolve = CyIpoptSolver(cy_nlp, 
#                                 options = {'linear_solver': 'ma57',
#                                           'hsllib': 'libcoinhsl.dylib'})
#         r = csolve.solve(tee=True)
#         #%%
#         print('In the model')
#         print(rh_model.P.display())
#         print(rh_model.K.display())
        
#         orig_primals = nlp.get_primals()
#         #orig_primals[col_ind_list] = r[1]['x'][col_ind_list]
#         nlp.set_primals(orig_primals)
        
#         orig_duals = nlp.get_duals()
#         #orig_duals[con_ind_dict] = r[1]['mult_g'][con_ind_dict]
#         nlp.set_duals(orig_duals)
#         projected_nlp.set_duals(r[1]['mult_g'])
#         print('NLP vals')
#         print(nlp.get_primals()[col_ind_list])
        
#         print('PNLP vals')
#         print(projected_nlp.get_primals()[col_ind_list])
        
    
        
#         #parameter_set_full = define_free_parameters(model, global_params=global_parameters, kind='full')
#         # size = (nlp.n_constraints(), len(varList))
        
#         J = projected_nlp.evaluate_jacobian()
#         # grad = projected_nlp.evaluate_grad_objective()
#         # obj_val = nlp.evaluate_objective()
        
#         # There is something that is causing this to be non-zero
#         H = nlp.extract_submatrix_hessian_lag(varList, varList)
#         # from kipet.estimator_tools.reduced_hessian_methods import bounds_modification_nlp
#         # H = bounds_modification_nlp(projected_nlp, H)
        
#         #h_raw[col_ind_list, :] = 0
#         #h_raw[:, col_ind_list] = 0
#         # H = h_raw.tocoo()
        
#         self.H_new.append(H)
#         #%%
        
#         #duals = {self.labels[file_number][key]: -1*nlp.get_duals()[con_ind_dict[i]] for i, (key, val) in enumerate(getattr(model, 'fix_params_to_global').items())}
#         global_set = [p for p in Se]
#         local_set = list(set(self.p_list).difference(set(global_set)))
        
#         # Won't work with C and S!
#         col_ind_local = [v for k, v in col_ind_dict.items() if k.lstrip('P[').rstrip(']') in local_set]
#         col_ind_global = [v for k, v in col_ind_dict.items() if k.lstrip('P[').rstrip(']') not in local_set]

#         print(f'{col_ind_global = }')

#         J_c = delete_from_csr(J.tocsr(), row_indices=con_ind_dict).tocsc()
#         row_indexer = SparseRowIndexer(J_c.T)
#         J_f = row_indexer[col_ind_global].T
#         J_l = delete_from_csr(J_c.tocsr(), col_indices=col_ind_local + col_ind_global)
        
#         print(f'{H.shape = }')
#         print(f'{J_c.shape = }')
#         print(f'{J_f.shape = }')
#         print(f'{J_l.shape = }')
        
#         n_free = len(global_set)
#         rh, _z_mat = _reduced_hessian_matrix(J_f, J_l, H, col_ind_dict, n_free)
        
#         return rh
        
#         #%%
        
#         #self.reaction_models[model_name]._nlp = nlp
#         #self.nlp_list[file_number] = nlp

#         # Won't work with C and S!
#         #grad = nlp.evaluate_grad_objective()
#         #grad = {key: grad[col_ind_dict[f'P[{key}]']] for
#         #              key, val in getattr(model, 'd').items()}
        
#         # Won't work with C and S!
#         #param_values = {k: v.value for k, v in model.P.items()}
#         #new_labels = [self.labels[file_number][v] for v in rh.columns]
#         #rh.columns = new_labels
#         #rh.index = new_labels
        
#         #return rh, duals, grad, param_values, obj_val, model_name
    

def rhps_method(r_model, **kwargs):
    """Reduces a single model using the reduced hessian parameter selection
    method. It takes a pyomo ConcreteModel using P as the parameters to be fit
    and K as the scaled parameter values.

    :param ConcreteModel model: The full model to be reduced
    :param dict kwargs: The keyword args passed to EstimaitonPotential

    :return results: The results from the parameter selection and optimization
    :return reduced_model: returns the reduced model with full parameter set
    :rtype: tuple(ResultsObject, ConcreteModel)

    """
    simulation_data = kwargs.get('simulation_data', None)
    replace = kwargs.get('replace', True)
    no_scaling = kwargs.get('no_scaling', True)
    method = kwargs.get('method', 'k_aug')
    solver_opts = kwargs.get('solver_opts', {})
    scaled = kwargs.get('scaled', True)
    use_bounds = kwargs.get('use_bounds', False)
    use_duals = kwargs.get('use_duals', False)
    calc_method = kwargs.get('calc_method', 'fixed')
    
    r_model._pe_set_up(solve=False)
    options = kwargs#options if options is not None else dict()
    orig_bounds = {k: v.bounds for k, v in r_model.p_model.P.items()}
    est_param = EstimationPotential(r_model,
                                    simulation_data=None,
                                    options=options,
                                    method=method,
                                    solver_opts=solver_opts,
                                    scaled=scaled,
                                    use_bounds=use_bounds,
                                    use_duals=use_duals,
                                    calc_method=calc_method)
    
    return est_param


def replace_non_estimable_parameters(model, set_of_est_params):
    """Takes a model and a set of estimable parameters and removes the 
    unestimable parameters from the model by fixing them to their current 
    values in the model
    
    :param ConcreteModel model: The full model to be reduced
    :param set set_of_est_params: Parameters found to be estimable

    :return model: The model with parameters replaced
    :rtype: ConcreteModel

    """
    all_model_params = set([k for k in model.P.keys()])
    params_to_change = all_model_params.difference(set_of_est_params)
    
    for param in params_to_change:   
        if param in model.P.keys():
            if hasattr(model, 'K'):
                if param in model.K:
                    change_value = model.K[param].value
            else:
                change_value = model.P[param].value
        
            for k, v in model.odes.items():
                ep_updated_expr = update_expression(v.body, model.P[param], change_value)
                if hasattr(model, 'K'):
                    ep_updated_expr = update_expression(ep_updated_expr, model.K[param], 1)
                model.odes[k] = ep_updated_expr == 0
    
            model.parameter_names.remove(param)
            del model.P[param]
            if hasattr(model, 'K'):
                del model.K[param]

    return model


#%%

# m = r1.p_model

# from kipet.general_settings.variable_names import VariableNames

# var = VariableNames()

# var_objs = {}

# for opt_var in var.optimization_variables:
#     if hasattr(m, opt_var):
#         for k, v in getattr(m, opt_var).items():
            
#             if v.fixed or v.stale:
#                 continue
#             else:
#                 var_objs[(opt_var, k)] = v

# #%%

# Se = {1: 'a', 2: 'b', 3: 'c'}
# Sf = {4:'d', 5:'e'}


# first_value = next(iter(Sf.items()))[0]
# param_to_move = {first_value: Sf[first_value]}
# Sf.pop(first_value)
# Se.update(param_to_move)
# # Sf_new.update(Sf)
# Sf = Sf_new








