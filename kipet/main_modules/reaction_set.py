"""
The ReactionSet class is a container for ReactionModels when multiple datasets
and model variations are available. This allows the user to perform parameter
fitting with the various models simultaneously.
"""
# Standard library imports
import copy
import inspect
# import multiprocessing as mp
# from multiprocessing import Process, Queue
import os
import pathlib
import time

# Third party imports
import pandas as pd

# Kipet library imports
# import kipet.core_methods.data_tools as data_tools
from kipet.estimator_tools.multiple_experiments import MultipleExperimentsEstimator
from kipet.main_modules.reaction_model import ReactionModel
from kipet.general_settings.settings import Settings
from kipet.general_settings.solver_settings import SolverSettings
from kipet.general_settings.unit_base import UnitBase
from kipet.estimator_tools.multiprocessing_kipet import Multiprocess
from kipet.input_output.kipet_io import print_margin

from kipet.model_tools.pyomo_model_tools import get_vars
from kipet.model_tools.parameter_manager import ParameterManager, ModelParameter
from kipet.general_settings.variable_names import VariableNames

class ReactionSet:
    
    """One of two highest level objects in KIPET. This is used to arrange
    multiple models/datasets for simultaneous parameter fitting.
    
    ReactionSet offers many avenues to setting up the problem. 
    
    * You can first generate several separate ReactionModel instances and add
      them to ReactionSet. This can be done using a single module or by importing
      several models from different modules
    
    * You can use the ReactionSet object to create ReactionModel instances and
      then perform parameter fittings with no additional effort
    
    * You can mix both of these methods, if you wish.
    
    Using ReactionSet makes it very simple to generate new ReactionModel
    instances that share many commonalities. This reduces redundant code
    writing if the models are very similar (such as when the only difference 
    is the data and a couple of states).
    
    :var dict reaction_models: The dictionary containing ReactionModel instances
    :var Settings settings: A Settings instance holding all modeling options
    :var dict results: A dictionary of ResultsObjects for each ReactionModel instance
    :var list global_parameters: A list of global parameters
    :var UnitBase ub: UnitBase object used to define the base units of the project
    
    :Methods:
    
    - :func:`add_reaction`
    - :func:`add_reaction_list`
    - :func:`remove_reaction`
    - :func:`new_reaction`
    - :func:`run_opt`
    - :func:`read_data_file`
    - :func:`write_data_file`
    - :func:`add_noise_to_data`
    
    :Properties:
    
    - :func:`all_params`
    
    :Example:
        >>> import kipet
        >>> reaction_lab = kipet.ReactionSet()
    
    """
    def __init__(self):
        """Initialize the ReactionSet instance.
        
        The ReactionSet object does not require any attributes at initialization.
        
        """
        self.reaction_models = {}
        self.settings = Settings()
        self.results = {}
        self.global_parameters = None
        self.ub = UnitBase()
        
        self.scale_variance = False
        self.file = pathlib.Path(inspect.stack()[1].filename)
        
        t = time.localtime()
        file_add = f'{t.tm_year}-{t.tm_mon:02}-{t.tm_mday:02}-{t.tm_hour:02}-{t.tm_min:02}-{t.tm_sec:02}'
        self.timestamp = f'{file_add}'
        
        self.solver_settings = SolverSettings()
        if self.solver_settings.custom_solvers.ipopt is None:
            if self.solver_settings.custom_solvers_lib['hsllib'] is not None:
                self.settings.solver.update(**self.solver_settings.use_hsllib())
            else:
                print('No HSL libraries found for Ipopt used in CyIpopt')    
        

    def __str__(self):
        
        block_str = "ReactionSet\n\n"

        for name, model in self.reaction_models.items():
            block_str += f"{name}\tDatasets: {len(model.datasets)}\n"

        return block_str

    def __repr__(self):
        return self.__str__()

    def __getitem__(self, value):
        return self.reaction_models[value]

    def __len__(self):
        return len(self.reaction_models)

    # def reset_base_units(self):
    #     """Sets the unit base values to the units declared in the settings
        
    #     :return: None
    #     """
    #     self.ub.TIME_BASE = self.settings.units.time
    #     self.ub.VOLUME_BASE = self.settings.units.volume
        
    #     return None


    def add_reaction_list(self, model_list):
        """Method to add a list of ReactionModel instances to the ReactionSet
        
        :parameter list model_list: A list of ReactionModel instances
        
        :return: None
        """
        
        for model in model_list:
            self.add_reaction(model)

        return None


    def remove_reaction(self, model):
        """Remove a model instance from the ReactionSet model list
        
        :parameter str/ReactionModel model: The name or instance of the reaction model to be removed
    
        :return: None
        """
        if isinstance(model, str):
            if model in self.reaction_models:
                self.reaction_models.pop(model)
        elif isinstance(model, ReactionModel):
            self.reaction_models.pop(model.name)
        else:
            print("ReactionSet does not have specified model")
        return None


    def new_reaction(self, name, model=None, ignore=None):

        """Declare new reactions to the ReactionSet using this function

        :parameter str name: The name of the model/experiment used in all references
                         made to it in KIPET, especially in python dicts
        :parameter ReactionModel model: Existing ReactionModel to initialize new ReactionModel with
        :parameter list[str] ignore: This is a list of strings for the various
                attributes in the ReactionModel that should not be copied to the new ReactionModel
                
        - **For example**, if you have a ReactionModel instance (r1) and want to create 
          a new instance (r2) with the same settings except for the data, you could use
          the following::

            r2 = reaction_lab.new_reaction('r2', model=r1, ignore=['datasets'])


        :return: A new ReactionModel instance
            
        """
        ignore = ignore if ignore is not None else []

        if model is None:

            self.reaction_models[name] = ReactionModel(name=name, unit_base=self.ub)
            self.reaction_models[name].file = self.file
            self.reaction_models[name].timestamp = self.timestamp
            
            self.reaction_models[name].settings = self.settings
            
            
            # assign_list = [
            #     "components",
            #     "parameters",
            #     "constants",
            #     "algebraics",
            #     "states",
            #     "ub",
            #     "c",
            # ]

            # for item in assign_list:
            #     if item not in ignore and hasattr(self, item):
            #         setattr(self.reaction_models[name], item, copy.deepcopy(getattr(self, item)))

        else:
            if isinstance(model, ReactionModel):
                
                new_kipet_model = copy.deepcopy(model)
        
                # name = kwargs.get('name', self.name + '_copy')
                # copy_model = kwargs.get('model', True)
                # copy_builder = kwargs.get('builder', True)
                # copy_components = kwargs.get('components', True)   
                # copy_parameters = kwargs.get('parameters', True)
                # copy_datasets = kwargs.get('datasets', True)
                # copy_constants = kwargs.get('constants', True)
                # copy_settings = kwargs.get('settings', True)
                # copy_algebraic_variables = kwargs.get('alg_vars', True)
                # copy_odes = kwargs.get('odes', True)
                # copy_algs = kwargs.get('algs', True)
                
                # Reset the datasets
                
                 #     # Reset the datasets
        
                new_kipet_model.name = name
                
                # if not copy_model:
                #     new_kipet_model.model = None
                
                # if not copy_builder:
                from kipet.model_tools.template_builder import TemplateBuilder
                new_kipet_model._builder = TemplateBuilder(name=new_kipet_model.name)
                    
                # if not copy_components:
                #     new_kipet_model.components = ComponentBlock()
                
                # if not copy_parameters:
                #     new_kipet_model.parameters = ParameterBlock()
                    
                #if not copy_datasets:
                del new_kipet_model.datasets
                from kipet.model_components.data_component import DataBlock
                new_kipet_model.datasets = DataBlock()
                    
                # if not copy_constants:
                #     new_kipet_model.constants = None
                    
                # if not copy_algebraic_variables:
                #     new_kipet_model.algebraic_variables = []
                    
                # if not copy_settings:
                #     new_kipet_model.settings = Settings()
                    
                # if not copy_odes:
                #     new_kipet_model.odes = None
                    
                # if not copy_algs:
                #     new_kipet_model.algs = None
                
                # list_of_attr_to_delete = ['p_model', 'v_model', 'p_estimator',
                #                           'v_estimator', 'simulator']
                
                # for attr in list_of_attr_to_delete:
                #     if hasattr(new_kipet_model, attr):
                #         setattr(new_kipet_model, attr, None)
                
                # new_kipet_model.results_dict = {}
                
                new_kipet_model.name = name
                self.reaction_models[name] = new_kipet_model
                
            #     kwargs = {}
            #     kwargs["name"] = name
            #     self.reaction_models[name] = ReactionModel(name=name, unit_base=self.ub)
            #     self.reaction_models[name].file = self.file
            #     self.reaction_models[name].timestamp = self.timestamp

            #     assign_list = [
            #         "components",
            #         "parameters",
            #         "constants",
            #         "algebraics",
            #         "states",
            #         "unit_base",
            #         "settings",
            #         "c",
            #         "odes_dict",
            #         "algs_dict",
            #     ]

            #     for item in assign_list:
            #         if item not in ignore and hasattr(model, item):
            #             setattr(self.reaction_models[name], item, copy.deepcopy(getattr(model, item)))

            # else:
            #     raise ValueError("ReactionSet can only add ReactionModel instances.")

        return self.reaction_models[name]
    

    def add_reaction(self, model):
        """Adds a ReactionModel instance to the ReactionSet instance
        
        :parameter ReactionModel model: ReactionModel instance
          
        :return: None
        """
        if isinstance(model, ReactionModel):
            model.unit_base = self.ub
            self.reaction_models[model.name] = model
            model.timestamp = self.timestamp
        else:
            raise ValueError("ReactionSet can only add ReactionModel instances.")

        return None
    
    def run_opt(self, method='mee', strategy='trust-region', parallel=False,
                pre_solve=False, rh_method='pynumero', outer_locals=False,
                variance_only=False):
        """This method will perform parameter fitting for all ReactionModels in
        the ReactionSet models attribute. If more than one ReactionModel instance
        is present, the MultipleExperimentEstiamtor is used to solve for the
        global parameter set.
        
        :parameter str method: Define the method used to solve the multiple
          dataset problem: either **mee** (solves one combined problem simultaneously)
          or **nsd** which solves the problem using a Nested Schur decomposition approach.
        
        .. warning::

            The NSD method is not implemented at this time!
        
        :return: None
        
        """
        from kipet import __version__ as version_number
        print_margin(f'KIPET v. {version_number}')
        
        print(f'# Date: {list(self.reaction_models.values())[0].timestamp}')
        print(f'# File: {list(self.reaction_models.values())[0].file.stem}')
        print(f'# ReactionModel instances: {", ".join(list(self.reaction_models.keys()))}')
        print()
        
        if hasattr(self, "mee"):
            del self.mee

        if self.global_parameters is None:
            self.global_parameters = list(self.all_params)

        from kipet.input_output.kipet_io import Tee
        results_dir = pathlib.Path.cwd().joinpath(self.file.parent, 'results', f'{self.file.stem.lstrip("<").rstrip(">")}-{self.timestamp}')
        if not results_dir.is_dir():
            results_dir.mkdir(parents=True)
        filename = results_dir.joinpath('log.txt') 

        #if len(self.reaction_models) > 1:
        if method == "mee":
            print('# ReactionSet: Starting multiple experiment estimator\n')
            
            if parallel:
                #self.solve_mpire()
                self.solve_mp()
            else:
                self._calculate_parameters(pre_solve=pre_solve, variance_only=variance_only)
                
            self._create_multiple_experiments_estimator()
            with Tee(filename): 
                self._run_full_model()
        
        elif method == 'nsd':
            print('# ReactionSet: Starting nested Schur decomposition estimator\n')
           
            with Tee(filename):  
                self._mee_nsd(strategy=strategy, parallel=parallel, rh_method=rh_method, pre_solve=pre_solve, local_as_global=outer_locals)
        
        else:
            raise ValueError("Not a valid method for optimization")
                
            print_margin('# KIPET procedure finished!')


        # else:
        #     reaction_model = self.reaction_models[list(self.reaction_models.keys())[0]]
        #     results = reaction_model.run_opt()
        #     self.results[reaction_model.name] = results

        return None

    def _create_multiple_experiments_estimator(self):
        """A quick wrapper for MEE without big changes
        
        """
        self.mee = MultipleExperimentsEstimator(self.reaction_models)
        self.mee.confidence_interval = self.settings.parameter_estimator.confidence
        self.mee.free_params = self.define_free_parameters

        if "spectral" in self.data_types:
            self.settings.general.spectra_problem = True
        else:
            self.settings.general.spectra_problem = False

        self.mee.spectra_problem = self.settings.general.spectra_problem

    def _calculate_parameters(self, pre_solve=False, variance_only=False):
        """Uses the ReactionModel framework to calculate parameters instead of
        repeating this in the MEE
        
        """
        from kipet.estimator_tools.reduced_hessian_methods import add_warm_start_suffixes, update_warm_start
        
        for name, model in self.reaction_models.items():
            if not model._optimized and not pre_solve:
                
                print_margin('ReactionSet - Model Preparation', sub_phrase=f'{model.name}')
                
                simulator = 'simulator'
                estimator = 'p_estimator'
                #model.simulate()
                model._ve_set_up()
                model._pe_set_up(solve=pre_solve)
                vars_to_init = get_vars(model.p_model)
                
                #add_warm_start_suffixes(model.p_model)
                model.p_model.ipopt_zL_in.update(model.p_model.ipopt_zL_out)
                model.p_model.ipopt_zU_in.update(model.p_model.ipopt_zU_out)
                
                # for var in vars_to_init:
                #     if hasattr(model.results_dict[simulator], var) and var != 'S':
                #         getattr(model, estimator).initialize_from_trajectory(var, getattr(model.results_dict[simulator], var))
                #     elif var == 'S' and hasattr(model.results_dict[simulator], 'S'):
                #         getattr(model, estimator).initialize_from_trajectory(var, getattr(model.results_dict[simulator], var))
                #     else:
                #         pass
                        #print(f'Variable: {var} is not updated')
            
            elif not model._optimized and pre_solve:
                print(f'you are here for some reason: {pre_solve}')
                model.run_opt(print_header=False)
            else:
                print(f"Model {model.name} has already been optimized or prepared")
                
            # for param, obj in model.p_model.P.items():
            #     if model.parameters[param].fixed:
            #         obj.fix()
            
        for name, model in self.reaction_models.items():
            model.min_variance = self.min_variance

        return None


    def _run_full_model(self):
        """Runs the MEE after everything has been set up"""
        
        consolidated_settings = self.settings.general
        consolidated_settings.solver_opts = self.settings.solver
        consolidated_settings.covariance = self.settings.parameter_estimator.covariance

        results = self.mee.solve_consolidated_model(
            self.global_parameters, **consolidated_settings
        )

        self.results = results
        for key, results_obj in self.results.items():
            self.reaction_models[key].results = results[key]

        return results

    def _mee_nsd(self, strategy='ipopt', parallel=False, rh_method='pynumero', pre_solve=False, local_as_global=False):
        """Performs the NSD on the multiple ReactionModel instances

        :parameter str strategy: Method used to control the outer problem
                ipopt, trust-region, newton-step

        :return results: A dictionary of ResultsObject instances
        :rtype: dict

        """
        from kipet.estimator_tools.nested_schur_decomposition import NSD
        kwargs = {'kipet': True,
                  'objective_multiplier': 1,
                  }
        
        print(self.settings)
        
        if self.scale_variance:
            for model in self.reaction_models.values():
                model.settings.parameter_estimator.scale_variance = True
                #if self.global_variance:
                model.min_variance_global = self.min_variance
                
        self._calculate_parameters(pre_solve=pre_solve)

        self.param_manager = ParameterManager(self.reaction_models)

        self.nsd = NSD(self.reaction_models,
                       strategy=strategy,
                       global_parameters=self.global_parameters,
                       parallel=parallel,
                       kwargs=kwargs,
                       rh_method=rh_method,
                       rs_min_variance=self.min_variance,
                       all_params=self.param_manager,
                       #all_params=self.rs_params,
                       free_params=self.define_free_parameters,
                       local_as_global=local_as_global)

        #print(self.nsd.d_init)
        results = self.nsd.run_opt()

        self.results = results
        # for key, results_obj in self.results.items():
        #     results_obj.file_dir = self.settings.general.charts_directory

        return results
    

    @property
    def rs_params(self):
        
        parameter_list = []
        for i, (name, model) in enumerate(self.reaction_models.items()):
            for param in model.parameters:
                if param.fixed:
                    continue
                
                p_obj = ModelParameter(
                    param.name, 
                    model.name, 
                    param.name in self.global_parameters,
                    param)
                parameter_list.append(p_obj)
                
        return parameter_list
        

    @property
    def all_params(self):
        """
        Returns the set of all parameters across all ReactionModel instances.
        
        :return: The set of all unique parameters in all models.
        :rtype: set
        """
        
        set_of_all_model_params = set()
        for name, model in self.reaction_models.items():
            set_of_all_model_params = set_of_all_model_params.union(
                model.parameters.get_match('fixed', False)
            )

        return set_of_all_model_params


    @property
    def data_types(self):
        """Returns a set of all data types represented in the contained ReactionModels
        
        :return: A set of data_types as strings
        :rtype: set
        
        """
        data_types = set()
        for name, kipet_model in self.reaction_models.items():
            for dataset_name, dataset in kipet_model.datasets.datasets.items():
                data_types.add(dataset.category)
                
            if kipet_model.spectra is not None and kipet_model.spectra.data.size > 0:
                data_types.add('spectral')
                
        return data_types
    
    
    @property
    def show_parameters(self):
        """Shows the resulting parameters for each of the ReactionModel instances.
        Naturally, global parameters will have the same values in each model.
         
        :return: DataFrame of the parameters in each reaction
        :rtype: pandas.DataFrame

        """
        df_param = pd.DataFrame(data=None, index=self.all_params, columns=self.reaction_models.keys())
            
        for reaction, model in self.reaction_models.items():
            for param in model.parameters.names:
                df_param.loc[param, reaction] = model.results.P[param]
            
        return df_param
    
    @property
    def min_variance(self):
        """Finds the lowest variance in the set of ReactionModel instances
        to better scale the objective function
        
        TODO: Take into account the scale of the numerator (the concentrations)
        
        """
        var = 1
        
        if self.scale_variance:
            for i, model in self.reaction_models.items():
                if not model.spectra:
                    var = min(min([v for v in model.components.variances.values()]), var)
                else:
                    var = min(min([v for v in model.results_dict['v_estimator'].sigma_sq.values()]), var)
    
        return var
    
    def plot(self, var=None):
        """Plot method for ReactionSet
        
        :param str var: The variable to plot (same as in ReactionModel)
        
        :return: None:
            
        """
        for name, model in self.reaction_models.items():
            
            if var is None:
                model.plot()
            else:
                model.plot(var)
        
    def report(self):
        """Plot method for ReactionSet
        
        :param str var: The variable to plot (same as in ReactionModel)
        
        :return: None:
            
        """
        from kipet.visuals.reports import Report
        for name, reaction in self.reaction_models.items():
            print(f'# ReactionSet: Making plots for {name}')
            reaction.plot()
            
        self.report_object = Report(list(self.reaction_models.values()))
        self.report_object.generate_report()
        
    def func(self, q, i):
        print('starting')
        print('process_id', os.getpid())
        
        model_to_solve = list(self.reaction_models.values())[i - 1]
        data = self.solve(model_to_solve)
        q.put(data)
        print('all done')

    def solve(self, model):
        """Uses the ReactionModel framework to calculate parameters instead of
        repeating this in the MEE
    
        # Add mp here
    
        """
        if not model._optimized:
            model.run_opt()
            print('Model has been optimized')
        else:
            print(f"Model {model.name} has already been optimized")
    
    
        attr_list = ['name', 'results_dict']
    
        model_dict = {}
        for attr in attr_list:
            model_dict[attr] = getattr(model, attr)
    
        return model_dict['results_dict']
    
    def solve_mp(self):
    
        from multiprocessing import set_start_method, cpu_count
        from kipet.model_tools.pyomo_model_tools import get_vars
        import platform
        
        if platform.system() == 'Darwin':
            try:
                set_start_method('fork')
                print('# Changing Multiprocessing start method to "fork"')
            except:
                print('# Multiprocessing start method already fixed')   
    
        mp = Multiprocess(self.func)
        data = mp(num_processes = len(self.reaction_models)) #cpu_count()))
        
        self.mp_results = data
        estimator = 'p_estimator'
        
        for i, model in enumerate(self.reaction_models.values()):
            model._pe_set_up(solve=False)
            
            setattr(model, 'results_dict', self.mp_results[i + 1])
            setattr(model, 'results', self.mp_results[i + 1][estimator])
            vars_to_init = get_vars(model.p_model)
            #print(model.results_dict[estimator])
            #print(vars_to_init)
            
            for var in vars_to_init:
                if hasattr(model.results_dict[estimator], var) and var != 'S':
                    getattr(model, estimator).initialize_from_trajectory(var, getattr(model.results_dict[estimator], var))
                elif var == 'S' and hasattr(model.results_dict[estimator], 'S'):
                    getattr(model, estimator).initialize_from_trajectory(var, getattr(model.results_dict[estimator], var))
                else:
                    pass
                    #print(f'Variable: {var} is not updated')
            
        return None
    
    @property
    def define_free_parameters(self):
        """Identifies the free variables used in MEE and NSD methods
           
        :return: list of free variable names
        
        """
        free_parameters = []
        global_accounted = {}
        
        for p_obj in self.rs_params:
            
            if p_obj.parameter_object.fixed:
                continue
            
            if p_obj.is_global and p_obj.name in global_accounted:
                continue
            
            else:
                free_parameters.append(p_obj)
                global_accounted[p_obj.name] = True
        
        return free_parameters
