"""
This module contains a convenience class to handle anything dealing with
parameters. It was primarily created to handle global and local variables and
provide methods to find models attached to parameters.
"""

class ModelParameter:
    
    """Class to handle parameters with a simple interface to needed attributes
    
    """
    
    def __init__(self, p_name, m_name, is_global, parameter_object):
        
        self.name = p_name
        self.model = m_name #if not is_global else 'global'
        self.is_global = is_global
        self.parameter_object = parameter_object
        
    def __str__(self):
        
        return f'Parameter: {self.name}, {self.model}, {self.is_global}'
    
    def __repr__(self):
        
        return f'ModelParameter(name={self.name}, model={self.model}, is_global={self.is_global}'
    
    @property
    def identity(self):
        
        return f'{self.name}-{self.model}'
    

class ParameterManager:
    
    """Class containing method related to handling of global and local
    parameters for ReactionModels
    
    """
    
    def __init__(self, reaction_models, global_parameters=None, model_vars=None):
        
        if isinstance(reaction_models, list):
            
            self.reaction_model_list = reaction_models
            self.reaction_model_dict = {r_model.name: r_model for r_model in self.reaction_model_list}
            
        elif isinstance(reaction_models, dict):
            
            self.reaction_model_dict = reaction_models
            self.reaction_model_list = list(self.reaction_model_dict.values())
        
        else:
            raise TypeError('Reaction Models must be in a dict or list')
        
        if model_vars is None:
            model_vars = ['P', 'Pinit', 'time_step_change']
            for r_model in self.reaction_model_list:
                if r_model.spectra is not None:
                    model_vars.append('S')
                    break
        
        if 'P' not in model_vars:
            model_vars.append('P', 'Pinit', 'time_step_change')
        
        self.model_vars = model_vars
        
        # Specify the correct flobal parameters
        if global_parameters is None:
            set_of_all_model_params = set()
            for name, model in self.reaction_model_dict.items():
                set_of_all_model_params = set_of_all_model_params.union(
                    model.parameters.get_match('fixed', False)
                )
            self.global_parameters = list(set_of_all_model_params)
        else:
            self.global_parameters = global_parameters
        
        self.parameters = self.build_parameter_list()
        self.parameter_dict = self._build_var_dict()
        
        
    def build_parameter_list(self):
        
        parameter_list = []
        for i, (name, model) in enumerate(self.reaction_model_dict.items()):
            for param in model.parameters:
                if param.fixed:
                    continue
                
                p_obj = ModelParameter(
                    param.name, 
                    model.name, 
                    param.name in self.global_parameters,
                    param)
                parameter_list.append(p_obj)
                
            # if hasattr(model.p_model, 'S'):
            #     parameter_list += self.add_S_params_to_global(model)
                
        return parameter_list
    
    
    @staticmethod
    def add_S_params_to_global(r_model):
        """Adds the S variables to the global list for use with NSD
        
        """
        from itertools import product
        
        spectra_list = []
        #for i, (name, r_model) in enumerate(reaction_set_object.reaction_models.items()):
        
        abs_comps = r_model.p_estimator.comps['absorbing']
        wavelengths = r_model.spectra.data.columns
        S_param_index = product(wavelengths, abs_comps)
        
        for S_param in S_param_index:
            s_obj = ModelParameter(
                ','.join(list(map(str, S_param))),
                r_model.name, 
                True,
                getattr(r_model.p_model, 'S')[S_param])
        
            spectra_list.append(s_obj)
            
        return spectra_list
    
    
    def _build_var_dict(self):
        
        #%%
        from kipet.general_settings.variable_names import VariableNames
        from collections import namedtuple
        #r_model = r1
        __var = VariableNames()
        
        P_var = namedtuple('P_var', ('fullname', 'name', 'index', 'model_var', 'pyomo_var', 'reaction', 'is_global', 'identity'))
        var_objs = {}
        params = []
        #reaction_set_object = lab
        
        for i, (name, r_model) in enumerate(self.reaction_model_dict.items()): 
        
            for opt_var in __var.optimization_variables:
                if hasattr(r_model.p_model, opt_var):
                    for k, v in getattr(r_model.p_model, opt_var).items():
                        #print(f'{k = }')
                        if v.fixed or v.stale or k in params:
                            # print(f'In set: {k in params}')
                            # print(f'Is fix: {v.fixed}')
                            # print(f'Is old: {v.stale}')
                                  
                            continue
                        else:
                            #print('adding to vars')
                            p_var = P_var(f'{name}.{v.name}', v.name, k, opt_var, v, name, k in self.global_parameters, f'{v.name}-{name}')
                            var_objs[p_var.fullname] = p_var
                            
            if hasattr(r_model.p_model, 'S'):
                for k, v in getattr(r_model.p_model, 'S').items():
                    #print(f'{k = }')
                    if v.fixed or v.stale or k in params:
                        # print(f'In set: {k in params}')
                        # print(f'Is fix: {v.fixed}')
                        # print(f'Is old: {v.stale}')
                              
                        continue
                    else:
                        #print('adding to vars')
                        p_var = P_var(f'{name}.{v.name}', v.name, k, 'S', v, name, 'S' in self.model_vars, f'{v.name}-{name}')
                        var_objs[p_var.fullname] = p_var
                        
            if hasattr(r_model.p_model, 'C'):
                for k, v in getattr(r_model.p_model, 'C').items():
                    #print(f'{k = }')
                    if v.fixed or v.stale or k in params:
                        # print(f'In set: {k in params}')
                        # print(f'Is fix: {v.fixed}')
                        # print(f'Is old: {v.stale}')
                              
                        continue
                    else:
                        #print('adding to vars')
                        p_var = P_var(f'{name}.{v.name}', v.name, k, 'C', v, name, 'C' in self.model_vars, f'{v.name}-{name}')
                        var_objs[p_var.fullname] = p_var
                        
        #print(var_objs)
        return var_objs
    
    #%%
        #%%
    
    # @property
    def parameters_model(self, model_name, query='all'):
        
        parameters_in_model = {}
        
        if query == 'all':
            for fullname, parameter in self.parameter_dict.items():
                if parameter.reaction == model_name:
                    parameters_in_model[fullname] = parameter
                    
        elif query == 'global':
            for fullname, parameter in self.parameter_dict.items():
                if parameter.is_global and parameter.reaction == model_name:
                    parameters_in_model[fullname] = parameter
                    
        elif query == 'local':
            for fullname, parameter in self.parameter_dict.items():
                if not parameter.is_global and parameter.reaction == model_name:
                    parameters_in_model[fullname] = parameter
                    
        return parameters_in_model
    
    
    # @property
    # def parameters_model(self, model_name, query='all'):
        
    #     parameters_in_model = []
        
    #     if query == 'all':
    #         for parameter in self.parameters:
    #             if parameter.model == model_name:
    #                 parameters_in_model.append(parameter)
                    
    #     elif query == 'global':
    #         for parameter in self.parameters:
    #             if parameter.is_global and parameter.model == model_name:
    #                 parameters_in_model.append(parameter)
                    
    #     elif query == 'local':
    #         for parameter in self.parameters:
    #             if not parameter.is_global and parameter.model == model_name:
    #                 parameters_in_model.append(parameter)
                    
    #     return parameters_in_model
    

    def unique_globals(self, include_locals=False, model_name=None, return_locals=False):
        
        global_set = {}
        global_accounted = {}
        
        for fullname, parameter in self.parameter_dict.items():
            
            if model_name is not None:
                if parameter.reaction != model_name:
                    continue
            
            if parameter.pyomo_var.fixed:
                continue
            
            if parameter.is_global and parameter.name in global_accounted:
                continue
            
            if not parameter.is_global and not include_locals:
                continue
            
            else:
                global_set[fullname] = parameter
                global_accounted[parameter.name] = True
            
        return global_set
    
    
    def globals_locals(self, model_name):
        
        global_parameter_set = self.parameters_model(model_name, query='global')
        local_parameter_set = self.parameters_model(model_name, query='local')
    
        return global_parameter_set, local_parameter_set
    
    def all_globals(self, model_name):
        
        return self.unique_globals(True, model_name), dict()
    
    # def unique_globals_dict(self, include_locals=False, model_name=None, return_locals=False):
        
    #     global_set = []
    #     global_accounted = {}
        
    #     for p_obj in self.parameters:
            
    #         if model_name is not None:
    #             if p_obj.model != model_name:
    #                 continue
            
    #         if p_obj.parameter_object.fixed:
    #             continue
            
    #         if p_obj.is_global and p_obj.name in global_accounted:
    #             continue
            
    #         if not p_obj.is_global and not include_locals:
    #             continue
            
    #         else:
    #             global_set.append(p_obj)
    #             global_accounted[p_obj.name] = True
            
    #     return global_set
    
    
    def free_parameters(self, model, include_locals=False):
        
        global_set = set()
        global_accounted = {}
        
        for p_obj in self.parameters:
            
            if p_obj.parameter_object.fixed:
                continue
            
            if p_obj.is_global and p_obj.name in global_accounted:
                continue
            
            if not p_obj.is_global and not include_locals:
                continue
            
            else:
                global_set.add(p_obj)
                global_accounted[p_obj.name] = True
            
        return global_set
            
    
    def find_parameter(self, parameter_name):
        
        
        #%%
        # self = lab.param_manager    
        # parameter_name = 'P[k1]'
        
            
        models = tuple(param.reaction for param in self.parameter_dict.values() if param.name == parameter_name)
        
        # print(models)
        #%%
        return models