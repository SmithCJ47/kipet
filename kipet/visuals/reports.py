"""
Report generation class
"""

# Standard library imports
from pathlib import Path
import os
import re
import warnings 
import webbrowser
from zipfile import ZipFile

# Third party imports 
from jinja2 import Environment, FileSystemLoader
import pandas as pd
from pytexit import py2tex


def generate_template_file(templates_dir):
    """
    This method generates the template file if it does not already exist
    """
    template = templates_dir.joinpath('index.html')
    if not template.is_file():
        from kipet.visuals.template import template_string
        template_file = open(template, 'w+')
        template_file.write(template_string)
        template_file.close()

    template_js = templates_dir.joinpath('prism.js')
    if not template_js.is_file():
        from kipet.visuals.template_js import template_js_string
        template_file = open(template_js, 'w+')
        template_file.write(template_js_string)
        template_file.close()

    template_css = templates_dir.joinpath('report.css')
    if not template_css.is_file():
        from kipet.visuals.template_css import template_css_report
        template_file = open(template_css, 'w+')
        template_file.write(template_css_report)
        template_file.close()

    template_css = templates_dir.joinpath('prism.css')
    if not template_css.is_file():
        from kipet.visuals.template_css import template_css_prism
        template_file = open(template_css, 'w+')
        template_file.write(template_css_prism)
        template_file.close()

    return None


class Report:
    
    """ This class contains the methods used to prepre model data for inclusion in the HTML report"""
    
    def __init__(self, model_list, is_simulation=False):
        
        self.reactions = model_list # ReactionModels
        # define the results object here
        self.simulation = is_simulation

    @staticmethod
    def model_context(reaction_model):
        """Prepares a dictionary of high-level model attributes
        
        :param ReactionModel reaction_model: A solved/simulated ReactionModel instance
        
        :return model: Dictionary of attributes
        :rtype: dict
        
        """
        model = {}
        model['name'] = reaction_model.name
        model['n_comps'] = reaction_model.components
        model['n_states'] = reaction_model.states
        
        return model
    
    @staticmethod
    def component_context(reaction_model):
        """Prepares a dictionary of component model attributes
        
        :param ReactionModel reaction_model: A solved/simulated ReactionModel instance
        
        :return comps: Dictionary of attributes
        :rtype: dict
        
        """
        comps = []
        
        for comp in reaction_model.components:
            comp_data = {}
            comp_data['name'] = comp.name
            comp_data['units'] = dimensionless_check(comp.units)
            comp_data['value'] = comp.value
            comp_data['variance'] = comp.variance
            comp_data['known'] = 'Yes' if comp.known else 'No'
            comp_data['absorbing'] = 'Yes' if comp.absorbing else 'No'
            comp_data['description'] = 'Not provided' if comp.description is None else comp.description
            comps.append(comp_data)
            
        return comps
    
    @staticmethod
    def state_context(reaction_model):
        """Prepares a dictionary of state model attributes
        
        :param ReactionModel reaction_model: A solved/simulated ReactionModel instance
        
        :return comps: Dictionary of attributes
        :rtype: dict
        
        """
        comps = []
        
        for comp in reaction_model.states:
            comp_data = {}
            comp_data['name'] = comp.name
            comp_data['units'] = dimensionless_check(comp.units)
            comp_data['value'] = comp.value
            comp_data['variance'] = comp.variance
            comp_data['known'] = 'Yes' if comp.known else 'No'
            comp_data['description'] = 'Not provided' if comp.description is None else comp.description
            comps.append(comp_data)
            
        return comps
    
    def parameter_context(self, reaction_model):
        """Prepares a dictionary of parameter model attributes
        
        :param ReactionModel reaction_model: A solved/simulated ReactionModel instance
        
        :return params: Dictionary of attributes
        :rtype: dict
        
        """
        params = []

        if not self.simulation:
            results_obj = reaction_model.results
        else:
            results_obj = reaction_model.results_dict['simulator']

        for param in reaction_model.parameters:
            param_data = {}
            param_data['name'] = param.name
            param_data['units'] = dimensionless_check(param.units)
            param_data['initial'] = param.value
            param_data['value'] = results_obj.P[param.name]
            param_data['lb'] = param.bounds[0]
            param_data['ub'] = param.bounds[1]
            param_data['description'] = 'Not provided' if param.description is None else param.description
            param_data['fixed'] = param.fixed
            params.append(param_data)
            
            if hasattr(reaction_model, 'p_model'):
                param_data['fixed'] = reaction_model.p_model.P[param.name].fixed
            
        if hasattr(results_obj, 'time_step_change'):
            for indx, param in pd.DataFrame(results_obj.time_step_change).iterrows():
               
                param_data = {}
                param_data['name'] = indx
                param_data['units'] = reaction_model.unit_base.time
                param_data['value'] = param.loc[0]
                if hasattr(reaction_model, 's_model') and reaction_model.s_model is not None:
                    param_data['initial'] = reaction_model.s_model.time_step_change[indx].value
                    param_data['lb'] = reaction_model.s_model.time_step_change[indx].bounds[0]
                    param_data['ub'] = reaction_model.s_model.time_step_change[indx].bounds[1]
                else:
                    param_data['initial'] = reaction_model.p_model.time_step_change[indx].value
                    param_data['lb'] = reaction_model.p_model.time_step_change[indx].bounds[0]
                    param_data['ub'] = reaction_model.p_model.time_step_change[indx].bounds[1]
                param_data['description'] = 'Binary variable'
                
                if hasattr(reaction_model, 'p_model'):
                    param_data['fixed'] = reaction_model.p_model.time_step_change[indx].fixed
    
                params.append(param_data)
                
        if hasattr(results_obj, 'Pinit'):
            for indx, param in pd.DataFrame(results_obj.Pinit).iterrows():
               
                param_data = {}
                param_data['name'] = f'{indx}'
                param_data['units'] = dimensionless_check(reaction_model.components[indx].units)
                param_data['value'] = param.loc[0]
                param_data['lb'] = reaction_model.components[indx].bounds[0]
                param_data['ub'] = reaction_model.components[indx].bounds[1]
                param_data['description'] = 'Initial value'
    
                params.append(param_data)
            
        return params

    @staticmethod
    def constant_context(reaction_model):
        """Prepares a dictionary of constant model attributes
        
        :param ReactionModel reaction_model: A solved/simulated ReactionModel instance
        
        :return params: Dictionary of attributes
        :rtype: dict
        
        """
        params = []
        
        for param in reaction_model.constants:
            param_data = {}
            param_data['name'] = param.name
            param_data['units'] = dimensionless_check(param.units)
            param_data['value'] = param.value
            param_data['description'] = 'Not provided' if param.description is None else param.description
            params.append(param_data)
            
        return params

    @staticmethod
    def remove(text):
        """Removes brackets from strings
        
        :param str text: Text to be cleaned
        
        :return: Cleaned text
        :rtype: str
        
        """
        return re.sub("[\[].*?[\]]", "", text)
    

    def readable_odes(self, reaction_model, kind):
        """Converts the ODEs to LaTeX
        
        :param ReactionModel reaction_model: A solved/simulated ReactionModel instance
        :param str kind: odes or daes
        
        :return odes_new: Dictionary of equations as latex strings
        :rtype: dict
        
        """
        numeric_const_pattern = '[-+]? (?: (?: \d* \. \d+ ) | (?: \d+ \.? ) )(?: [Ee] [+-]? \d+ ) ?'
        rx = re.compile(numeric_const_pattern, re.VERBOSE)
        
        odes = {k: v.expression for k, v in getattr(reaction_model, kind).items()}
        odes_new = {}
        
        for k, v in odes.items():
            
            if kind == 'odes':
                try:
                    odes_new[f'{k}'] = self.remove(v.to_string())
                except:
                    odes_new[f'{k}'] = str(v)
            else:
                try:
                    odes_new[f'{k}'] = self.remove(v.to_string())
                except:
                    odes_new[f'{k}'] = str(v)
                    
            # floats = rx.findall(odes_new[k])

            # text = odes_new[k]
            # #print(text)
            # for num in floats:
            #     find_it = num
            #     if int(float(num)) == float(num):
            #         continue
            #     elif len(num) < 5:
            #         continue
            #     else:
            #         repl_it = f'{float(find_it):.4f}'
            #         text = text.replace(find_it, repl_it)

            odes_new[k] = py2tex(odes_new[k], print_latex=False, print_formula=False)
             
        ode_list = []
        for k, v in odes_new.items():
            ode_list.append(f'{k} = {v}')
            
        return odes_new
    
    
    @staticmethod
    def get_chart_file_dir(abs_name):
        """Shortens the absolute paths to relative paths
        
        :param str abs_name: The absolute path name
        
        :return: The relative path
        :rtype: pathlib.Path
        
        """        
        file = Path(abs_name)
        file_dir = file.parents[0].stem
        new_path = Path('charts').joinpath(file_dir, file.name)
        
        return new_path.parent.joinpath(new_path.stem)
        
        # return new_path
    
    def create_zip_file(self, save_dir, charts_dir):
        """This generates a ZIP object to hold the report and charts
        
        :param pathlib.Path save_dir: The current directory where results are saved
        :param pathlib.Path chart_dir: The path to the charts
        
        :return: None
        
        """
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            folder = save_dir
            filename = 'report'
            chart_files = retrieve_file_paths(charts_dir)
            output_file_name = folder.joinpath(f'{filename}.zip')
            zip_obj = ZipFile(output_file_name, 'w')
            zip_obj.write((save_dir / 'report.html').resolve(), 'report.html')
            for file in chart_files:
                file_path = self.get_chart_file_dir(file)
                zip_obj.write(file, file_path)
            zip_obj.close()
        
        return None

        
    def generate_report(self):
        """Generates a full HTML report for the KIPET model.
        
        This provides a more useful interface for representing results than using individual charts.
        Also included are tables with useful diagnostics and other results.
        
        :return: None
        
        """
        from kipet import __version__ as version
        model_dict = {}
        all_names = [rm.name for rm in self.reactions]

        suffix = '.html' # '.svg'
    
        user_file = self.reactions[0].file
        user_stem = user_file.stem.lstrip('<').rstrip('>')
        time = self.reactions[0].timestamp
        current_dir = Path(__file__).parent
        templates_dir = (current_dir / 'templates').resolve()
        
        if not templates_dir.is_dir():
            templates_dir.mkdir()
        
        generate_template_file(templates_dir)

        style_file = (templates_dir / 'report.css').resolve()
        prism_style_file = (templates_dir / 'prism.css').resolve()
        prism_js_file = (templates_dir / 'prism.js').resolve()
        
        style_file_text = ''
        with open(style_file, 'r+') as f:
            style_file_text = f.read()
        
        prism_css = ''
        with open(prism_style_file, 'r+') as f:
            prism_css = f.read()
        
        prism_js = ''
        with open(prism_js_file, 'r+') as f:
            prism_js = f.read()
        
        source_code = ''
        if user_file.is_file():
            with open(user_file, 'r+') as f:
                source_code = f.read()
        else:
            source_code = 'Not found or from a Jupyter Notebook'
        
        user_dir = Path(user_file).parent
        final_dir = (user_dir / 'results' / f'{user_stem}-{time}').resolve()
        filename = (final_dir / 'report.html').resolve()
        
        env = Environment( loader = FileSystemLoader(templates_dir) )

        template = env.get_template('index.html')

        problem_text = "This is an automatically generated report presenting the results found using KIPET."
             
        log_orig = (user_dir / f'log-{user_stem}-{time}.txt').resolve()
        log_file = (final_dir / 'log.txt').resolve()
        
        if log_orig.is_file():
            log_orig.rename(log_file)
        
        with open(log_file, 'r') as f:
            log = f.read()
        
        for reaction_model in self.reactions:
            
            name = reaction_model.name
            model_dict[name] = {}
            model_dict[name]['time'] = reaction_model.timestamp
            charts_dir = (final_dir / 'charts' / f'{reaction_model.name}').resolve()
            
            # Concentration profiles
            model_dict[name]['chart_C_files'] = [self.get_chart_file_dir(x) for x in charts_dir.glob(f'all-concentration-profiles{suffix}') if x.is_file()]
            model_dict[name]['chart_S_files'] = [self.get_chart_file_dir(x) for x in charts_dir.glob(f'absorbance-spectra-all{suffix}') if x.is_file()]
            model_dict[name]['chart_U_files'] = sorted([self.get_chart_file_dir(x) for x in charts_dir.glob(f'*state-profile{suffix}') if x.is_file()])
            model_dict[name]['chart_Y_files'] = sorted([self.get_chart_file_dir(x) for x in charts_dir.glob(f'*profile{suffix}') if x.is_file() and x not in model_dict[name]['chart_U_files']])

            data_chart_files = None
            spectral_info = None
            spectra_file = None
            
            #print(f'{model_dict[name]["chart_U_files"] = }')
            #print(f'{model_dict[name]["chart_Y_files"] = }')
            
            
            if reaction_model.spectra is not None:
                spectra_file = reaction_model.spectra.file if reaction_model.spectra.file is not None else 'Not provided or custom'
                data_chart_files = reaction_model._plot_object._plot_input_D_data()
                data_chart_files = f'{data_chart_files}'#'{suffix}'
                
                sd = reaction_model.spectra
                spectral_info = []
                for method in ['_sg', '_snv', '_msc', '_base', '_decreased_wavelengths', '_decreased_times']:
                    if getattr(sd, method) is not None:
                        spectral_info.append(getattr(sd, method))
                        
            model_dict[name]['data_chart_files'] = data_chart_files
            model_dict[name]['spectral_info'] = spectral_info
            model_dict[name]['spectra_file'] = spectra_file
                        
            abs_data = None
            S_data = []
            for comp in reaction_model.components:
                if comp.S is not None:
                    reaction_model._plot_object._plot_S(comp.name, orig=True)
                    S_data.append(comp)
                    abs_data = 1
                    
            model_dict[name]['abs_data'] = abs_data
                    
            chart_abs_files = None
            if len(S_data) > 0:
                chart_abs_files = sorted([x for x in charts_dir.glob('*absorbance-spectra{suffix}') if x.is_file()])
            
            model_dict[name]['chart_abs_files'] = chart_abs_files
            model_dict[name]['comp_data'] = self.component_context(reaction_model)
            model_dict[name]['param_data'] = self.parameter_context(reaction_model)
            model_dict[name]['const_data'] = self.constant_context(reaction_model)
            model_dict[name]['state_data'] = self.state_context(reaction_model)
            model_dict[name]['bounds'] = reaction_model._builder._prof_bounds
            model_dict[name]['variances'] = reaction_model.variances


            if not self.simulation:
                results_obj = reaction_model.results
            else:
                results_obj = reaction_model.results_dict['simulator']

            model_dict[name]['confidence'] = results_obj.deviations()
            model_dict[name]['covariance'] = results_obj.parameter_covariance
            
            
            # Figuring out what was done
            final_estimator = 'simulation'
            if reaction_model.models['p_model']:
                final_estimator = 'parameter estimation'
            elif reaction_model.models['v_model']:
                final_estimator = 'variance estimation'
                
            model_dict[name]['final_estimator'] = final_estimator
            
            used_variance = False
            if reaction_model.models['v_model']:
                used_variance = reaction_model.models['v_model']
            
            opt_data = {
                'var_method': reaction_model.settings.variance_estimator.method,
                'variance_used': used_variance,
                'final_method': final_estimator,
                'is_simulation': final_estimator == 'simulation',
                'var_add': '',
                }
            
            model_dict[name]['opt_data'] = opt_data
            
            if reaction_model.settings.variance_estimator.fixed_device_variance is not None:
                model_dict[name]['opt_data']['var_add'] = '- using fixed device variance'
        
        
            if reaction_model.settings.variance_estimator.method == 'direct_sigmas':
                mod_direct_sigma_dict = {}
                for key, value in reaction_model.direct_sigma_dict.items():
                    mod_direct_sigma_dict[key] = {}
                    mod_direct_sigma_dict[key]['delta'] = reaction_model.direct_sigma_dict[key]['delta']
                    mod_direct_sigma_dict[key].update(**reaction_model.direct_sigma_dict[key]['simgas'])
                model_dict[name]['delta_results'] = mod_direct_sigma_dict
            else:
                model_dict[name]['delta_results'] = None
            
            
            # Settings - make a series of tables for the settings
            settings = reaction_model.settings.as_dicts
            if 'solver_opts' in settings['simulator'][1]:
                settings['simulator'][1].pop('solver_opts')
            if 'solver_opts' in settings['variance_estimator'][1]:
                settings['variance_estimator'][1].pop('solver_opts')
            if 'solver_opts' in settings['parameter_estimator'][1]:
                settings['parameter_estimator'][1].pop('solver_opts')
            
            model_dict[name]['settings'] = settings
            model_dict[name]['ode_data'] = self.readable_odes(reaction_model, 'odes')
            
            age_data = None
            if hasattr(reaction_model, 'algs') and isinstance(getattr(reaction_model, 'algs'), dict) and len(getattr(reaction_model, 'algs')) > 0:
                age_data = {}
                all_algs = self.readable_odes(reaction_model, 'algs')
                all_rxns = reaction_model.algebraics.get_match('is_reaction', True)
                
                for k, eq in all_algs.items():
                    if k in all_rxns:
                        age_data[k] = ('Yes', eq)
                    else:
                        age_data[k] = ('No', eq)
            
            model_dict[name]['age_data'] = age_data
            
            from kipet.model_tools.diagnostics import model_fit

            diagnostics = None
            if hasattr(reaction_model, 'p_estimator'):

                diags = model_fit(reaction_model.p_estimator)
                diagnostics = []
                if final_estimator == 'parameter estimation':
                    for k, v in diags.items():
                        stat = {}
                        stat['name'] = k
                        stat['value'] = v[0]
                        stat['description'] = v[1]
                        diagnostics.append(stat)
                    
            model_dict[name]['diagnostics'] = diagnostics
                
            res_chart_files = None
            par_chart_files = None
            
            if reaction_model._builder._concentration_data is not None:
                res_chart_files = reaction_model._plot_object._plot_Z_residuals()
                par_chart_files = reaction_model._plot_object._plot_Z_parity()
                
            if reaction_model._builder._spectral_data is not None and not self.simulation:
                res_chart_files = reaction_model._plot_object._plot_D_residuals()
                par_chart_files = reaction_model._plot_object._plot_D_parity()
                
            model_dict[name]['res_chart'] = f'{res_chart_files}'#'{suffix}'
            model_dict[name]['par_chart'] = f'{par_chart_files}'#'{suffix}'
            
            feeds = None
            dosing_dict = reaction_model._dosing_points
            if dosing_dict is not None:
                feeds = [x.as_list for x in dosing_dict['Z']]
                
            model_dict[name]['feeds'] = feeds
            
            model_dict[name]['g_contrib'] = None
            if hasattr(reaction_model, '_G_contribution') and reaction_model._G_contribution is not None:
                model_dict[name]['g_contrib'] = reaction_model._G_contribution
            
            self.model_dict = model_dict
        
        with open(filename, 'w') as fh:
            fh.write(template.render(
                user_file = user_file,
                file_stem = user_stem,
                default = all_names[0],
                version = version,
                models = model_dict,
                filename = filename,
                description = problem_text,
                log_file = log_file,
                log = log,
                base_text = style_file_text,
                prism_text = prism_css,
                prism_js = prism_js,
                source_code = source_code,
                time = time,
            ))
        
        self.create_zip_file(final_dir, charts_dir)
        webbrowser.open('file://' + os.path.realpath(filename))
            
        return None


def dimensionless_check(units):
    """Converts dimensionless to a hyphen
    
    :param units: The model component units
    
    :return: The formatted unit string
    :rtype: str
    
    """
    if units.u == '':
        return '-'
    else:
        return units.u
    

def retrieve_file_paths(dir_name):
 
  # setup file paths variable
  filePaths = []
   
  # Read all directory, subdirectories and file lists
  for root, directories, files in os.walk(dir_name):
    for filename in files:
        # Create the full filepath by using os module.
        filePath = os.path.join(root, filename)
        filePaths.append(filePath)
         
  # return all paths
  return filePaths
    
    
if __name__ == '__main__':
    pass
