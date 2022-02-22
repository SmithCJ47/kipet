"""Multiple Experiment Estimator"""

# Standard library imports
import copy

# Third party imports
import numpy as np
import pandas as pd
from pyomo.environ import (Block, ConcreteModel, Constraint, minimize, 
                           Objective, SolverFactory, Var, Set, Param)

# KIPET library imports
from kipet.model_components.objectives import (absorption_objective, comp_objective,
                                               conc_objective)
from kipet.estimator_tools.reduced_hessian_methods import define_free_parameters
from kipet.estimator_tools.results_object import ResultsObject
from kipet.input_output.kipet_io import print_margin
from kipet.mixins.parameter_estimator_mixins import PEMixins
from kipet.general_settings.variable_names import VariableNames
from kipet.general_settings.settings import solver_path
from kipet.model_tools.pyomo_model_tools import to_dict

class MultipleExperimentsEstimator(PEMixins, object):
    """This class is for Estimation of Variances and parameters when we have multiple experimental datasets.
    This class relies heavily on the Pyomo block class as we put each experimental class into its own block.
    This blocks are first run individually in order to find good initializations and then they are linked and
    run together in a large optimization problem.

    :param list reaction_models: The full model TemplateBuilder problem needs to be fed into the
         MultipleExperimentsEstimator. This pyomo model will form the basis for all the optimization tasks

    """
    def __init__(self, reaction_models):
        
        self.reaction_models = reaction_models
        
        self.experiments = list(self.reaction_models.keys())
        self.variances = {name: model.variances for name, model in self.reaction_models.items()}
        self.global_params = None
        self.parameter_means = False
        self.rm_cov = None
        self.free_params = None
        self.__var = VariableNames()

    
    def _display_covariance(self):
        """Displays the covariance results to the console

        :param dict variances_p: The dict of parameter variances to display

        :return: None

        """
        import scipy.stats as st
        
        if not hasattr(self, 'confidence') or self.confidence is None:
            confidence = 0.95 # just because
        
        number_of_stds = st.norm.ppf(1 - (1 - confidence)/2)
        margin = 15
        variances_p = self.p_variances
                        
        print()
        print_margin('MEE Results', sub_phrase=f'Parameter Values (Confidence: {int(confidence*100)}%)')
        for exp in self.experiments:
            print(f'Experiment - {exp}:\n')
            for i, (k, p) in enumerate(self.model.experiment[exp].P.items()):
                if p.is_fixed():
                    continue
                is_global = '(local)'
                if k in self.global_params:
                    is_global = '(global)'
                
                key = f'P[{k}]'
                variance = self.rm_variances[exp][k]
                #if k in self.global_params:
                print(f'{k.rjust(margin)} = {p.value:0.4e} +/- {number_of_stds*(variance**0.5):0.4e}    {is_global}')
                
            # These need to be set equal to the local values
            if hasattr(self.model.experiment[exp], 'Pinit'):
                for i, k in enumerate(self.model.experiment[exp].Pinit.keys()):
                    self.model.experiment[exp].Pinit[k] = self.model.experiment[exp].init_conditions[k].value
                    key = f'Pinit[{k}]'
                    value = self.model.experiment[exp].Pinit[k].value
                    variance = self.rm_variances[exp][k]
                    print_name = f'{k} (init)'
                    print(f'{print_name.rjust(margin)} = {value:0.4e} +/- {number_of_stds*(variance)**0.5:0.4e}    (local)')
                        
            # These need to be set equal to the local values
            if hasattr(self.model.experiment[exp], 'time_step_change'):
                for i, k in enumerate(self.model.experiment[exp].time_step_change.keys()):
                    self.model.experiment[exp].time_step_change[k] = self.model.experiment[exp].time_step_change[k].value
                    key = f'time_step_change[{k}]'
                    value = self.model.experiment[exp].time_step_change[k].value
                    variance = self.rm_variances[exp][k]
                    print(f'{k.rjust(margin)} = {value:0.4e} +/- {number_of_stds*(variance)**0.5:0.4e}    (local)')
                          
        return None
    
    
    def covariance(self, solver, solver_factory):
        """Solves for the Hessian regardless of data source - only uses ipopt_sens

        :param str solver: The name of the solver

        :Keyword Args:
            sigma_sq (dict): variances

            optimizer (SolverFactory): Pyomo Solver factory object

            tee (bool,optional): flag to tell the optimizer whether to stream output
            to the terminal or not

        :return numpy.ndarray hessian: The hessian matrix for covariance calculations
            
        """
        components = self.reaction_models[self.experiments[0]].p_estimator.comps['unknown_absorbance']
        parameters = self.param_names_full
        models_dict = {k: v.p_model for k, v in self.reaction_models.items()}
        
        if solver == 'ipopt_sens':
            from kipet.estimator_tools.reduced_hessian_methods import covariance_sipopt
            covariance_matrix, covariance_matrix_reduced = covariance_sipopt(
                models_dict, 
                solver_factory,
                components, 
                parameters, 
                mee_obj=self.model,
            )
        
        elif solver == 'k_aug':
            from kipet.estimator_tools.reduced_hessian_methods import covariance_k_aug
            covariance_matrix, covariance_matrix_reduced = covariance_k_aug(
                models_dict, 
                None,
                components,
                parameters, 
                mee_obj=self.model,
            )
            
        else:
            from kipet.estimator_tools.reduced_hessian_methods import covariance_pynumero
            covariance_matrix, covariance_matrix_reduced = covariance_pynumero(
                models_dict, 
                None,
                components,
                parameters, 
                mee_obj=self.model,
            )
            
            #%%
            
        # self = lab.mee
        index = []
        for name in self.param_names_full:
            
            sep = name.split('[', 1)[1].split(']')
            exp = sep[0]
            kind = sep[1][1:].split('[')[0]
            var = sep[1][1:].split('[')[1]
            
            if var not in self.global_params:
                index.append(f'{kind}[{var}] {exp}')
            else:
                index.append(f'{kind}[{var}] {exp}')
            
        H = covariance_matrix_reduced
        if not hasattr(self.reaction_models[self.experiments[0]].p_model, 'C'):
            self.cov = pd.DataFrame(H, index=index, columns=index)  
        else:  
            from kipet.estimator_tools.reduced_hessian_methods import compute_covariance
            models_dict = {k: v.p_model for k, v in self.reaction_models.items()}
            free_params = len(self.param_names)
            all_variances = self.variances
            V_theta = compute_covariance(models_dict, H, free_params, all_variances)
            self.cov = pd.DataFrame(V_theta, index=index, columns=index)  
        
        #%%
        self.rm_variances = {}
        self.rm_cov = {}
        for name, rm in self.reaction_models.items():
            self.rm_variances[name] = {}
            
            col = []
            for i in index:
                if i.split(' ')[1] == name or i.split(' ')[0] in self.global_params_full:
                    
                    col.append(i)
                    param_name = i.split('[', 1)[1].split(']')[0]
                    self.rm_variances[name][param_name] = self.cov.loc[i, i]
            
            rm_cov = self.cov.loc[col, col]
            rm_cov.columns = [c.split('[', 1)[1].split(']')[0] for c in col]
          
            c = self.cov
            c = c.loc[col, col]
            new_cols = [c.split('[', 1)[1].split(']')[0] for c in col]
            
            c.columns = new_cols
            c.index = new_cols
            
            self.rm_cov[name] = c
        
        #%%
        
        self.p_variances = np.diag(self.cov.values)
        self._display_covariance()
        
        return None
        
    def _scale_variances(self,):
        """Option to scale the variances for MEE

        :return: None

        """
        var_scaled = dict()
        for s,t in self.variances.items():
            maxx = max(list(t.values()))
            ind_var = dict()
            for i,j in t.items():
                ind_var[i] = j/maxx
            var_scaled[s] = ind_var
        self.variances = var_scaled
        self.variance_scale = maxx
        
        return None


    def solve_consolidated_model(self, 
                                 global_params=None,
                                 **kwargs):
        """This function consolidates the individual models into a single
        optimization problem that links the parameters and spectra (if able)
        from each experiment.

        :param list global_params: This is the list of global parameters to be linked in the MEE
        :param dict kwargs: The dictionary of options passed from ReactionSet
        
        """
        solver_opts = kwargs.get('solver_opts', {'linear_solver': 'ma57'})
        tee = kwargs.get('tee', True)
        scaled_variance = kwargs.get('scaled_variance', False)
        shared_spectra = kwargs.get('shared_spectra', True)
        # solver = kwargs.get('solver', 'ipopt')
        solver = solver_path('ipopt')
        parameter_means = kwargs.get('mean_start', True)
        
        covariance = kwargs.get('covariance', None)
        
        from kipet import __version__ as version_number
        
        print_margin('Multiple Experiments Estimator (MEE)')
        
        print("# MEE: Starting parameter estimation \n")
       
        combined_model = ConcreteModel()
        
        self.variance_scale = 1
        if scaled_variance == True:
            self._scale_variances()

        if global_params is None:
            # This needs to be a global attr
            self.global_params = self.all_params
        else:
            self.global_params = global_params
            
        for model in self.reaction_models.values():
            for var, obj in model.p_model.P.items():
                print(f'{var}: {obj.value}')
            
        # Parameter name list
        self.global_params_full = [f'P[{p}]' for p in self.global_params]
    
        
        def build_individual_blocks(m, exp):
            """This function forms the rule for the construction of the individual blocks 
            for multiple experiments, referenced in run_parameter_estimation. This function 
            is not meant to be used by users directly.

            :param ConcreteModel m: The concrete model that we are adding the block to
            :param list exp: A list containing the experiments
                
            :return ConcreteModel m: Pyomo model inside the block (after modification)

            """
            list_components = self.reaction_models[exp].components.names
            with_d_vars= True
            m = copy.copy(self.reaction_models[exp].p_model)
            
            # Quick fix - I don't know what is causing this
            if hasattr(m, 'objective'):
                m.del_component('objective')
            if hasattr(m, 'alltime_domain'):
                m.del_component('alltime_domain')
            if hasattr(m, 'huplctime_domain'):
                m.del_component('huplctime_domain')
            
            if with_d_vars and hasattr(m, 'D'):
              
                m.D_bar = Var(m.times_spectral,
                              m.meas_lambdas)
    
                def rule_D_bar(m, t, l):   
                    return m.D_bar[t, l] == sum(getattr(m, self.__var.concentration_spectra)[t, k] * getattr(m, self.__var.spectra_species)[l, k] for k in self.reaction_models[exp].p_estimator.comps['unknown_absorbance'])
    
                m.D_bar_constraint = Constraint(m.times_spectral,
                                                m.meas_lambdas,
                                                rule=rule_D_bar)
            
            m.error = Var(bounds = (0, None))
                
            def rule_objective(m):
                
                expr = 0
                spectral_term = 0
                concentration_term = 0
                measured_concentration_term = 0
                complementary_state_term = 0
                weights = [1, 1, 1, 1]
                obj_variances = self.variances
                
                if hasattr(m, self.__var.spectra_data):
                    spectral_term = absorption_objective(
                        m, 
                        device_variance=obj_variances[exp]['device'],
                        g_option=self.reaction_models[exp]._G_data['G_contribution'],
                        with_d_vars=with_d_vars,
                        shared_spectra=shared_spectra,
                        species_list=list_components
                    )
                    concentration_term = conc_objective(m, variance=obj_variances[exp], source='spectra')
                
                if hasattr(m, self.__var.concentration_measured):
                    measured_concentration_term = conc_objective(m, variance=obj_variances[exp])
                
                if hasattr(m, self.__var.state):
                    complementary_state_term = comp_objective(m, variance=obj_variances[exp])
                    
                expr = weights[0]*spectral_term + \
                       weights[1]*concentration_term + \
                       weights[2]*measured_concentration_term + \
                       weights[3]*complementary_state_term
    
                return m.error == expr
    
            m.obj_const = Constraint(rule=rule_objective)
                
            return m  
        
        combined_model.experiment = Block(self.experiments, rule=build_individual_blocks)
        combined_model.map_exp_to_count = dict(enumerate(self.experiments))
        
        
        initial_P = {}
        
        model = 's_model'
        
        for i, exp in enumerate(self.experiments):
            if hasattr(self.reaction_models[exp], 'p_model'):
                param_model = 'p_model'
            elif hasattr(self.reaction_models[exp], 's_model'):
                param_model = 's_model'
                
            P_exp = to_dict(getattr(self.reaction_models[exp], param_model).P)
            if i == 0:
                initial_P = P_exp
                continue
            
            for k, v in P_exp.items():
                initial_P[k] += v
            
        for k, v in initial_P.items():
            initial_P[k] = v / len(self.experiments)
        
        print(f'{initial_P = }')
        
        global_param_name='d'
        setattr(combined_model, 'current_p_set', Set(initialize=self.global_params))
        # Better initializations needed
        setattr(
            combined_model,
            global_param_name,
            Var(
                getattr(combined_model, 'current_p_set'),
                initialize=initial_P,
            )
        )
        combined_model.d['k1'].setlb(0)
        combined_model.d['k2'].setlb(0)

        def rule_fix_global_parameters(m, exp, param):
            
            for key, val in combined_model.map_exp_to_count.items():
                if param in self.global_params and param in getattr(combined_model.experiment[exp], self.__var.model_parameter):
                    return getattr(combined_model.experiment[exp], self.__var.model_parameter)[param] - getattr(combined_model, global_param_name)[param] == 0
                else:
                    return Constraint.Skip

        set_fixed_params=set()
        for exp in self.experiments:
            for param, param_obj in getattr(combined_model.experiment[exp], self.__var.model_parameter).items():
                if param_obj.is_fixed():
                    set_fixed_params.add(param)
            
        if len(set_fixed_params) > 0:
            print(f'# MEE: The fixed parameters are:\n{set_fixed_params}')
        
        set_params_across_blocks = self.all_params.difference(set_fixed_params)
        combined_model.parameter_linking = Constraint(self.experiments, set_params_across_blocks, rule = rule_fix_global_parameters)
        
        
        if self.spectra_problem:
            
            global_S_name='S_global'
            
            # Initialize the S from the variance estimation results
            initial_S = {}
            for i, exp in enumerate(self.experiments):
                S_exp = to_dict(self.reaction_models[exp].v_model.S)
                if i == 0:
                    initial_S = S_exp
                    continue
        
                for k, v in S_exp.items():
                    initial_S[k] += v
                
            for k, v in initial_S.items():
                initial_S[k] = v / len(self.experiments)
            
            # initial_S = to_dict(self.reaction_models['reaction-1'].v_model.S)
            
            setattr(
                combined_model,
                global_S_name,
                Var(
                    list(self.all_wavelengths), list(self.all_species),
                    initialize=initial_S,
                    bounds=(0, None),
                )
            )
            #print(f'{initial_S = }')
            
            def rule_link_wavelengths(m, exp, wave, comp):
             
                for key, val in combined_model.map_exp_to_count.items():
                    #if wave in self.all_wavelengths and comp in combined_model.experiment[exp].mixture_components:
                    if (wave, comp) in getattr(combined_model.experiment[exp], self.__var.spectra_species):
                        return getattr(combined_model.experiment[exp], self.__var.spectra_species)[wave, comp] - getattr(combined_model, global_S_name)[wave, comp] == 0
                    else:
                        return Constraint.Skip
                    # else:
                    #     return Constraint.Skip
    
            if shared_spectra == True:
                print('shared = True')
                
                combined_model.spectra_linking = Constraint(self.experiments, self.all_wavelengths, self.all_species, rule = rule_link_wavelengths)
        
        # Add in experimental weights
        combined_model.objective = Objective(sense=minimize, expr=sum(b.error for b in combined_model.experiment[:]))
        
        from kipet.estimator_tools.reduced_hessian_methods import add_warm_start_suffixes
        
        #add_warm_start_suffixes(combined_model)
        
        self.simulation_initialization = False
        if self.simulation_initialization:
            
            parameters_to_unfix = {}
            
            global_parameters_fixed = []
            
            # You only need to fix the free_params - no more
            # Go through list and fix those with a model == reaction name
            # If a parameter is global, check if a model has it and fix it - only once!
            for fp in self.free_params:
                
                #print(fp.identity)
                
                if not fp.is_global:
                    model = combined_model.experiment[fp.model]
                    model.P[fp.name].fix()
                    global_parameters_fixed.append((fp.name, fp.model))
                else:
                    if fp.identity in global_parameters_fixed:
                        continue
                    for key, value in combined_model.map_exp_to_count.items():
                        model = combined_model.experiment[value]
                        if fp.name in model.P:
                            model.P[fp.name].fix()
                            global_parameters_fixed.append((fp.name, value))
                            break
                     
                        
            # print(f'{global_parameters_fixed = }')
                    
            
                    
            # print(f'{parameters_to_unfix = }')
            simulator = SolverFactory(solver_path('ipopt'))
            simulator.solve(combined_model, options=solver_opts, tee=True)
            
            for param_tuple in global_parameters_fixed:
                param, model_name = param_tuple
                model = combined_model.experiment[model_name]
                model.P[param].unfix()
                        
        self.model = combined_model
        
        models = {k: v.p_model for k, v in self.reaction_models.items()}
        
        self.param_names = define_free_parameters(models, self.global_params_full, kind='variable')
        self.param_names_full = define_free_parameters(models, self.global_params_full, kind='full')
        
        if covariance in ['k_aug', 'ipopt_sens']:
            
            tee = kwargs.pop('tee', True)
            solver_opts = kwargs.pop('solver_opts', dict())
            optimizer = None
            
            # At the moment k_aug is not working for this
            if covariance == 'k_aug':
                model_cov = solver_path('ipopt_sens')
                covariance = 'ipopt_sens'
            
            if covariance == 'ipopt_sens':
                model_cov = solver_path('ipopt_sens')
                if not 'compute_red_hessian' in solver_opts.keys():
                    solver_opts['compute_red_hessian'] = 'yes'
                    
                # Create the optimizer
                optimizer = SolverFactory(model_cov)
                for key, val in solver_opts.items():
                    optimizer.options[key] = val

            self.covariance(covariance, optimizer)
            
        else:
            # optimizer = SolverFactory('ipopt')
            optimizer = SolverFactory(solver_path('ipopt'))
            print('MEE: Starting optimization')
            optimizer.solve(combined_model, options=solver_opts, tee=True)  
            #self.covariance(covariance, None)
        
        solver_results = {}
        
        for i in combined_model.experiment:
            solver_results[i] = ResultsObject()
            solver_results[i].load_from_pyomo_model(combined_model.experiment[i])
            if self.rm_cov is not None:
                solver_results[i].parameter_covariance = self.rm_cov[i]
            
            #setattr(solver_results[i], 'variances', self.rm_variances[i])
            
        self.results = solver_results           
        print()          
        print_margin('MEE: Parameter estimation finished!')
                
        return solver_results
    
    @property
    def all_params(self):
        """Method to return all of the parameters in the ReactionModels

        :return set set_of_all_model_params: The set of all parameters

        """
        set_of_all_model_params = set()
        for name, model in self.reaction_models.items():
            set_of_all_model_params = set_of_all_model_params.union(model.parameters.names)
        return set_of_all_model_params
    
    
    @property
    def all_wavelengths(self):
        """Method to return all of the wavelengths in the ReactionModels

        :return set set_of_all_wavelengths: The set of all wavelengths

        """
        set_of_all_wavelengths = set()
        for name, model in self.reaction_models.items():
            set_of_all_wavelengths = set_of_all_wavelengths.union(list(model.p_model.meas_lambdas))
        return set_of_all_wavelengths
    
    @property
    def all_species(self):
        """Method to return all of the components in the ReactionModels

        :return set set_of_all_species: The set of all components

        """
        set_of_all_species = set()
        for name, model in self.reaction_models.items():
            set_of_all_species = set_of_all_species.union(model.components.names)
        return set_of_all_species