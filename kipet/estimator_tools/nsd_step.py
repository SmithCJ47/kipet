"""
NSD class for models - testing class
"""
# Standard library imports


# Third party imports
import matplotlib.pyplot as plt
from operator import index
import numpy as np
import pandas as pd
from pathlib import Path

from pyomo.environ import SolverFactory, Suffix, ConstraintList, VarList
from scipy.sparse.linalg import spsolve
from scipy.sparse import coo_matrix, csr_matrix, csc_matrix, triu, lil_matrix, bmat

# Kipet library imports
from kipet.general_settings.variable_names import VariableNames
from kipet.model_tools.pyomo_model_tools import convert
from kipet.general_settings.solver_settings import solver_path
import kipet.estimator_tools.reduced_hessian_methods as rhm
from kipet.estimator_tools.results_object import ResultsObject
    
__var = VariableNames()


class NSD_Model():
    """
    This will allow for a model to be updated using the NSD methods
    
    Parameters:
        :model ConcreteModel: The model from the ParameterEstimator
        
    """
    def __init__(self, 
                 reaction_model, 
                 use_cyipopt=False, 
                 use_full_method=False,
                 use_k_aug=False, 
                 debug=False, 
                 iter_count=1, 
                 model_vars=None,
                 ):
        
        self.r_model = reaction_model
        self.model = reaction_model.p_model #.clone(
        self.use_cyipopt = use_cyipopt
        self.use_full_method = use_full_method
        self.use_k_aug = use_k_aug
        self.debug = debug
        self.iter_count = iter_count

        if model_vars is None:
            model_vars = ['P', 'Pinit']
        self.model_vars = model_vars
        
        self.global_parameter_set = None
        self.local_parameter_set = None
        self.model_object_lists = None
        self.nlp_object = None
        
        self.updates = {}
        self.rh = {}
        self.primals = {}
        self.duals = {}
        self.obj_val = {}
        self.del_d = {}
        self.index_list = None
        self.mdict = None
        
        self._initialize_model_values()
        
        self.projected_nlp = None
        #self.nlp = None

    def _initialize_model_values(self):
        """Set up the parameters and the constraints in the models
       
        """
        self.r_model.start_parameter_manager(self.model_vars)
        self.global_parameter_set, self.local_parameter_set = self.r_model.globals_locals
        self.constraint_map = rhm.prepare_global_constraints(self.model, self.global_parameter_set, self.local_parameter_set, use_cyipopt=self.use_cyipopt)
        self.reverse_constraint_map = {v: k for k, v in self.constraint_map.items()}
        self.r_model._create_pynumero_model_object()
        self.model_object_lists = rhm.generate_parameter_object_lists(self.r_model)
        
        self.nlp = self.r_model._nlp
        
        return None
    
    
    def solve(self):
        """Control method for multiple updates
        
        """
        for i in range(self.iter_count):
            self.iterate(i)
        self.aggregate_iteration_data()
        
        return None
    
    
    def update_primals(self, x):
        """Update the primals in the model
        
        """
        # Update the x values in the model
          # nlp = self.nlp_list[model_name]
          # projected_nlp = self.projected_nlp_list[model_name]
          # index_list = generate_parameter_index_lists(nlp, model_object_lists)
         
        current_primals = self.nlp.get_primals()
        current_primals[self.index_list[0]] = x
        current_primals[-len(self.index_list[0]):] = x
        self.nlp.set_primals(current_primals)
        self.nlp.load_state_into_pyomo()
        
        #if not self.use_cyipopt:
        for i, param in enumerate(self.global_parameter_set.values()):        
            #param.pyomo_var.set_value(x[i])
            getattr(self.model, param.model_var)[param.index].set_value(x[i])
            self.model.d[i + 1].set_value(x[i])
            
            #print(getattr(self.model, param.model_var)[param.index].value)
        
        return None
    
    
    def optimize(self, initialize=False):
        """Optimize the model at the current set of parameters
        
        """
        # Optimize the model using SolverFactory or CyIpopt
        if self.use_cyipopt:
            # This will only make the projected NLP object in the first iteration
            if self.projected_nlp is None:
                self.nlp = PyomoNLP(self.model)
                self.projected_nlp = ProjectedNLP(self.nlp, self.nlp.primals_names()[:-len(self.model_object_lists[0])])
            self.nlp_object, info = rhm.optimize_model_cyipopt(self.projected_nlp)
        
        else:
            self.nlp, solver, opt_model = rhm.optimize_model(self.model, files=False, debug=self.debug)
            self.nlp_object = self.nlp
            
        if self.index_list is None:
            self.index_list = rhm.generate_parameter_index_lists(self.nlp_object, self.model_object_lists)
    
        return None
    
    
    def update_results(self):
        """Updates the dictionaries with the results from each iteration
        
        """
        if self.use_k_aug:
            rh = self.build_reduced_hessian_kaug()
        else:
            rh = self.build_reduced_hessian()
        
        labels = [p.name for p in self.model_object_lists[0]]
        duals = self.nlp.get_duals()[self.index_list[2]]
        duals = pd.DataFrame(duals, index=labels, columns=['value'])
        primals = self.nlp.get_primals()[self.index_list[0]]
        primals = pd.DataFrame(primals, index=labels, columns=['value'])
        
        i = self.iter_last + 1 if self.iter_last is not None else 0
        
        self.rh[i] = rh
        self.duals[i] = duals
        self.primals[i] = primals
        self.obj_val[i] = self.nlp_object.evaluate_objective()
        
        return None
    
    
    def build_reduced_hessian(self):
        """Return the reduced Hessian of the current model
        
        """
        # Use full KKT as in the NSD paper
        if self.use_full_method:
            
            # working for cyipopt and not cyipopt
            if self.mdict is None:
                self.mdict = rhm.generate_model_data(self.model, self.nlp_object)
            
            if self.use_cyipopt:
                H, J, K = rhm.kkt_pynumero_projected_nlp(self.projected_nlp, self.index_list)
            else:
                H, J, K = rhm.kkt_pynumero_nlp(self.nlp, self.mdict)
            
            labels = [p.name for p in self.model_object_lists[0]]
            E = rhm.build_E_matrix(self.nlp, self.mdict)
            rh = rhm.reduced_hessian_from_full_kkt(K, E, labels=labels)
            #duals = rhm.evaluate_gradient(self.nlp, self.mdict, self.index_list, self.use_cyipopt)[self.index_list[2]]
            #print(f'{duals = }')
            
        # Use the direct calculation of the reduced Hessian
        else:
            H, J, grad = rhm.build_matrices(self.nlp_object, self.index_list[0], use_cyipopt=self.use_cyipopt)
            rh = rhm.reduced_hessian(self.nlp_object, self.model_object_lists, use_cyipopt=self.use_cyipopt, as_df=True, debug=self.debug)

        return rh
    
    def build_reduced_hessian_kaug(self):
        
        if self.use_full_method:
            
            if self.mdict is None:
                self.mdict = rhm.generate_model_data(self.model, self.nlp_object)
            
            H, J, K = rhm.kkt_kaug(self.model, pyomo_model=True, nlp=self.nlp)      
            labels = [p.name for p in self.model_object_lists[0]]
            E = rhm.build_E_matrix(self.nlp, self.mdict)
            rh = rhm.reduced_hessian_from_full_kkt(K, E, labels=labels)
        else:
            rh = rhm.reduced_hessian_single_model(self.r_model, use_k_aug=True)
        
        return rh
            
        #     #%% remove the C
        #     global_labels = [v.name for v in self.global_parameter_set.values()]# if v.model_var == 'P']
        #     rh = rh.loc[global_labels, :]
    
    def newton_step_update(self, rh, duals):
        """Update the models and objects after an optimization
        
        """
        i = list(self.primals.keys())[-1]
        d_vals = self.primals[i]
        alpha = 0.2
        #duals = nlp.get_duals()[dummy_con_index]
        dual_modifier = 1 - 2*(not self.use_cyipopt)
        d = np.linalg.inv(rh) @ (dual_modifier*duals.values).reshape(d_vals.shape)
        self.del_d[i] = d
        
        d_vals_new = d*alpha + d_vals
        d_vals_new = np.asarray(d_vals_new).flatten()
        
        if self.debug:
            print(f'{d = }')
            print(f'{d_vals = }')
        
        current_primals = self.nlp.get_primals()
        current_primals[self.index_list[0]] = d_vals_new
        current_primals[-len(self.model_object_lists[0]):] = d_vals_new
        self.nlp.set_primals(current_primals)
        self.nlp.load_state_into_pyomo()
        
        print(f'Current Parameters in Iteration {i}: {d_vals.values.flatten()[:2]}')
        
        for i, param in enumerate(self.global_parameter_set.values()):
            param.pyomo_var.set_value(d_vals_new[i])
            self.model.d[i + 1].set_value(d_vals_new[i])
        
    
    def iterate(self, i):
        """Perform a single iteration using the NSD approach
        """
        self.optimize(initialize=i==0)
        self.update_results()
        
        # if self.use_k_aug:
        #     rh = self.build_reduced_hessian_kaug()
        # else:
        #     rh = self.build_reduced_hessian()
        
        # labels = [p.name for p in self.model_object_lists[0]]
        # duals = self.nlp.get_duals()[self.index_list[2]]
        # duals = pd.DataFrame(duals, index=labels, columns=['value'])
        # primals = self.nlp.get_primals()[self.index_list[0]]
        # primals = pd.DataFrame(primals, index=labels, columns=['value'])
        
        # Testing the modification of H using B and Vd
        #rh = self.BV_cov(rh)
        
        # if self.debug:
        #     print(f'rh(O) = {rh}')
        #     print(f'duals(reg) = {duals}')
        
        # self.rh[i] = rh
        # self.duals[i] = duals
        # self.primals[i] = primals
        #duals = duals.reshape(1, -1)[2:]
        
        ### Solve the NS update
        self.newton_step_update(self.rh[i], self.duals[i])


    def BV_cov(self, rh):
        
        inv_rh = np.linalg.inv(rh)
        from kipet.estimator_tools.reduced_hessian_methods import compute_covariance
        models_dict = {'reaction_model': self.model}
        free_params = 2 #len(self.global_parameter_set)
        variances = {'reaction_model': self.r_model.p_estimator.sigma_sq}
        print(f'{variances = }')
        covariance_modified = rhm.compute_covariance(models_dict, inv_rh, free_params, variances)
        
        return np.linalg.inv(inv_rh)
        
        
    def aggregate_iteration_data(self):
        
        self.iteration_data = {}
        self.iteration_data['primals'] = pd.concat(self.primals, axis=1, ignore_index=True).T
        self.iteration_data['duals'] = pd.concat(self.duals, axis=1, ignore_index=True).T
        
    def plot_duals(self, params=None, zone=None):
        
        df = self.iteration_data['duals']
        if params is None:
            df.plot()
        else:
            #if param in df:
            df.loc[:, params].plot()
                
        return None
    
        
    def plot_primals(self, params=None, zone=None):
        
        df = self.iteration_data['primals']
        if params is None:
            df.plot()
        else:
            #if params in df:
            df.loc[:, params].plot()
        
        return None
    
        
    def plot_model_var(self, model_var, iteration=None):
        
        if iteration is None:
            iteration = self.iter_count - 1
        if iteration > self.iter_count:
            iteration = self.iter_count - 1
        
        df = convert(getattr(self.model, model_var))
        df.plot()
        
        return None
    
    
    def _get_results(self, display=True):
        """Removed results unit from function

        :return: The formatted results
        :rtype: ResultsObject

        """
        results = ResultsObject()
        
        #if hasattr(self, 'objective_value'):
        results.objective = self.model.objective.expr()
        #else:
        #    results.objective = None
        
        #results.parameter_covariance = self.cov
        results.load_from_pyomo_model(self.model)
        
        #if display:
        #    results.show_parameters(self.confidence)

        # if hasattr(self.model, '_spectra_given:
        #     from kipet.calculation_tools.beer_lambert import D_from_SC
        #     D_from_SC(self.model, results)

        # if hasattr(self.model, self.__var.model_parameter_scaled): 
        #     setattr(results, self.__var.model_parameter, {name: getattr(self.model, self.__var.model_parameter)[name].value*getattr(self.model, self.__var.model_parameter_scaled)[name].value for name in self.model.parameter_names})
        # else:
        #     setattr(results, self.__var.model_parameter, {name: getattr(self.model, self.__var.model_parameter)[name].value for name in self.model.parameter_names})

        #if self.termination_condition != None and self.termination_condition != TerminationCondition.optimal:
        #    raise Exception("The current iteration was unsuccessful.")
        #else:
        #    if self._estimability == True:
        #        return self.hessian, results
        #    else:
        print(results)
            
        return results
        

    @property
    def iter_last(self):
        if 0 not in self.primals:
            return None
        else:
            return list(self.primals.keys())[-1]
        
        
        
        
        
        
        
        
        