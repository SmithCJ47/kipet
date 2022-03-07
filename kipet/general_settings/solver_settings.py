"""
General KIPET settings - separate from ReactionModel settings
"""

"""
Settings for KIPET
"""
# Standard library imports
import ast
from pathlib import Path
import platform

# Third party imports
from pyomo.environ import SolverFactory
import yaml

# Kipet library imports
from kipet.calculation_tools.helper import AttrDict


class SolverSettings:
    """This is a special settings class for the solver information
    
    :Methods:
    
    - :func:`updated_settings`
    
    """

    def __init__(self):
        """Settings object initialization that begins each time a
        ReactionModel instance is created.
        
        """
        self._load_settings()
        self._reset_model()

    def __str__(self):

        m = 25

        settings = '\nKIPET Solver Settings\n'
                
        if hasattr(self, 'custom_solvers'):
            settings += '\nUser Defined Solvers (custom_solvers):\n'
            for k, v in self.custom_solvers.items():
                settings += f'{str(k).rjust(m)} : {v}\n'
                
        if hasattr(self, 'custom_solvers_lib'):
            settings += '\nLibraries for User Defined Solvers (custom_solvers_lib):\n'
            for k, v in self.custom_solvers_lib.items():
                settings += f'{str(k).rjust(m)} : {v}\n'

        return settings

    def __repr__(self):
        return self.__str__()

    @staticmethod
    def _get_file_path():
        """Finds the path to the settings.yml file by referencing the directory
        of the current file
        
        """
        current_dir = Path(__file__).parent
        settings_file = (current_dir / '../general_settings/settings.yml').resolve()

        return settings_file

    def _load_settings(self):
        """Loads the settings and places the data into the correct dict
        structure
        
        """
        settings_file = self._get_file_path()

        with open(settings_file) as file:
            self.cfg = yaml.load(file, Loader=yaml.FullLoader)

            sub_dict = self.cfg['custom_solvers']
            for key, value in sub_dict.items():
                if isinstance(value, bool):
                    continue
                elif value in ['True', 'False', 'None']:
                    sub_dict[key] = ast.literal_eval(value)
                elif _is_number(value):
                    if float(value) == int(float(value)):
                        sub_dict[key] = int(float(value))
                    else:
                        sub_dict[key] = float(value)
                else:
                    None

        return None

    def _reset_model(self):
        """Initializes the settings dicts to their default values
        
        """
        self.custom_solvers = AttrDict(self.cfg['custom_solvers'])
        self.custom_solvers_lib = AttrDict(self.cfg['custom_solvers_lib'])

        return None

    def update_settings(self, category, item, value):
        """Sets the default settings to some new value
        
        This allows the user to make permanent changes to the settings file.
        
        .. warning::
            
            Careful! This may result in KIPET not working properly if you make
            a change that is incompatible!
        
        :param str category: The category containing the value to change
        :param str item: The name of the setting to change
        :param str value: The new value for the setting
        
        :return: None
        
        """
        settings_file = self._get_file_path()
        solver_categories = list(self.cfg.keys())
        if category not in solver_categories:
            raise ValueError('Solver Settings category must be in {", ".join(solver_categories)}')

        self.cfg[category][item] = value
        with open(settings_file, 'w') as yaml_file:
            yaml_file.write(yaml.dump(self.cfg, default_flow_style=False))
        print(f'Updated default settings:\n\n{category}:\n\t{item}: {value}')
        return None

    @property
    def as_dicts(self):
        
        full_name = {
            'custom_solvers': 'Used Defined Solvers',
            'custom_solvers_lib': 'Libraries for User Defined Solvers'
        }
        
        keys = list(self.cfg.keys())
        nested_dict = {}
        for key in keys:
            nested_dict[key] = (full_name[key], getattr(self, key))
        return nested_dict
    
    
    def use_hsllib(self):
        
        if self.custom_solvers.ipopt is None:
            return self.custom_solvers_lib
    

def solver_path(solver='ipopt', full_path=True):
    """This ensures that strange things don't happen when the paths are not
    found for the solvers. You can update this if not needed
    
    This might not work on windows OS
    TODO: Move this to part of the class yet keep it generic
    """
    import os
    import platform
    
    settings = SolverSettings()
    
    if not full_path:
        return solver
    else:
        if platform.system() in ['Darwin', 'Linux']:
            command = 'which'
        else:
            command = 'where'
            
        # This does not work in Spyder
        if settings.custom_solvers[solver] is None:
            print(f'# No user provided solver found for "{solver}"')
            os_command = f'{command} {solver}'
            full_path_to_solver = os.popen(os_command).read().rstrip('\n')
            if full_path_to_solver is None or full_path_to_solver == '':
                print(f'# Solver "{solver}" not found in path, using local reference.\n')
                return solver
            else:
                print(f'# Solver "{solver}" found at {full_path_to_solver}\n')
                return full_path_to_solver
        else:
            print(f'# Using custom solver "{solver}" found at {settings.custom_solvers[solver]}\n')
            return settings.custom_solvers[solver]
        
        
        
def check_libs(solver_factory_obj):
    """Checks if the provided ipopt solver has the hsllib option
    
    """
    solver_settings = SolverSettings()
    
    if Path(solver_factory_obj.executable()).stem != 'ipopt':
        return solver_factory_obj
    elif solver_factory_obj.executable() == solver_settings.custom_solvers['ipopt']:
        return solver_factory_obj
    else:
        if solver_settings.custom_solvers_lib['hsllib'] is not None:
            solver_factory_obj.options['hsllib'] = solver_settings.custom_solvers_lib['hsllib']
        return solver_factory_obj
        
        
def _is_number(s):
    """ Returns True if the input is a float (number), else False

    :param str s: String that will be checked as to whether it is a number
    
    :return: True or False, depending on whether a number is detected.
    :rtype: bool
    
    """
    try:
        float(s)
        return True
    finally:
        return False

