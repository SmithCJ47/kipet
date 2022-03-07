"""
Reduced Hessian methods renewed.

This module is to hold generic methods for parameter estimator and mee for determining the
reduced hessian. This will eventually replace the reduced hessian module.
"""
# Third party imports
from operator import index
import numpy as np
import pandas as pd
from pathlib import Path
from pyomo.environ import SolverFactory, Suffix, ConstraintList, VarList
from scipy.sparse.linalg import spsolve
from scipy.sparse import coo_matrix, csr_matrix, csc_matrix, triu, lil_matrix, bmat, vstack

# Kipet library imports
from kipet.general_settings.variable_names import VariableNames
from kipet.model_tools.pyomo_model_tools import convert
from kipet.general_settings.solver_settings import solver_path, SolverSettings, check_libs
from kipet.general_settings.settings import Settings

# Conditional Dependencies
try:
    from pyomo.contrib.pynumero.interfaces.pyomo_nlp import PyomoNLP
    from pyomo.contrib.pynumero.interfaces.nlp_projections import ProjectedNLP
    from pyomo.contrib.pynumero.algorithms.solvers.cyipopt_solver import CyIpoptNLP, CyIpoptSolver
    __pynumero_libs_installed = True
except:
    __pynumero_libs_installed = False
    pynumero_readme = 'https://github.com/Pyomo/pyomo/tree/master/pyomo/contrib/pynumero'
    raise ModuleNotFoundError(f'Pynumero libraries have not been found. Please got to {pynumero_readme} for installation instructions.')
    
__var = VariableNames()
__settings = Settings()


# TODO: NLP
def generate_model_data(model, nlp):
    """Generates a dictionary of model data used in reduced Hessian methods
    
    """
    if isinstance(nlp, ProjectedNLP):
        nlp = nlp._original_nlp
        
    dummy_constraints = [v for v in getattr(model, 'fix_params_to_global').values()]
    dummy_vars = [id(v) for v in getattr(model, 'd').values()]
    attr_dict = {}
    attr_dict['varList'] = [v for v in nlp.get_pyomo_variables() if id(v) not in dummy_vars]
    attr_dict['conList'] = nlp.get_pyomo_constraints()
    attr_dict['zLList'] = [v for v in attr_dict['varList'] if v.lb is not None]
    attr_dict['zUList'] = [v for v in attr_dict['varList'] if v.ub is not None]
    attr_dict['conList_dummy'] = [v for v in attr_dict['conList'] if v in dummy_constraints]
    attr_dict['conList_red'] = [v for v in attr_dict['conList'] if v not in dummy_constraints]
    attr_dict['var_index_names'] = [v.name for v in attr_dict['varList']]
    attr_dict['init_data'] = {v.name : v.value for v in attr_dict['varList']}
        
    return attr_dict


def define_free_parameters(models_dict, global_params=None, kind='full'):
    """Identifies the variables in the model used to define the reduced hessian  
       
    :param str kind: full, variable, simple
    
    :return: None
    
    .. todo::
        
        This works with PE too, just supply the r1.p_model
        In MEE, you provide a dict of p_models:
            {k: v.p_model for k, v in self.reaction_models.items()}

    """
    global_params = [] if global_params is None else global_params
    param_names = []
    global_accounted = {}
    
    methods = {'variable': 'getname',
               'simple': 'simple',
               'full': 'to_string'}
    
    if kind not in methods:
        print('Wrong variable naming convention')
    
    def name(method, p, v):
        
        if method in ['to_string', 'getname']:
            return getattr(v, method)()
        else:
            return p
    
    for i, (exp, model) in enumerate(models_dict.items()):
        for parameter_kind in __var.optimization_variables:
            if hasattr(model, parameter_kind):
                for p, v in getattr(model, parameter_kind).items():
                    
                    # print(f'The name in loop: {v}, {v.getname()}, {v.to_string()}')
                    # print(f'{v.fixed = }')
                    # print(v.pprint())
                    if v.fixed or v.stale:
                        # print(f'Not using {p}')
                        # print(f'{v.fixed = }')
                        # print(f'{v.stale = }')
                        continue
                    
                    if i > 0 and v.getname() in global_params and v.getname() in global_accounted:
                        # print(f'Not using {p}')
                        continue
                    
                    else:
                        param_names.append(name(methods[kind], p, v))
                        global_accounted[v.getname()] = True
        
    # print(f'{len(param_names) = }')
    
    return param_names


def spectral_hess_reorder(reaction_model):
    """Order of the variables for covariance with spectral problems. This is
    to make the calculations work for the pynumero and k_aug (ncp > 1).
    """
    index_to_variable = []
    model = reaction_model.p_model
    component_set = reaction_model.p_estimator.comps['unknown_absorbance']
    parameters = reaction_model.p_estimator.param_names_full
    
    if hasattr(model, 'C'):
        var = 'C'
        for index in getattr(model, var).index_set():
            if index[1] in component_set:
                v = getattr(model, var)[index]
                index_to_variable.append(v.name)

    if hasattr(model, 'S'):
        var = 'S'
        for index in getattr(model, var).index_set():
            if index[1] in component_set:
                v = getattr(model, var)[index]
                index_to_variable.append(v.name)
                
    for parameter_kind in __var.optimization_variables:
        if hasattr(model, parameter_kind):
            for v in getattr(model, parameter_kind).values():
                if v.to_string() in parameters and not v.fixed:
                    index_to_variable.append(v.name)
                        
    return index_to_variable


def define_reduce_hess_order(models_dict, component_set, param_names_full, k_aug):
    """
    This sets up the suffixes of the reduced hessian for use with SIpopt
    
    :param dict models_dict: The dict of parameter estimator models
    :param component_set list: The list of components used
    :param param_names_full: The global parameters as a list (var name)

    :return index_to_variable: Mapping of an index to the model variable
    :rtype: dict

    """
    index_to_variable = dict()
    count_vars = 1
    exp_count = True
    
    for exp, model in models_dict.items():
        if hasattr(model, 'C'):
            var = 'C'
            for index in getattr(model, var).index_set():
                if index[1] in component_set:
                    v = getattr(model, var)[index]
                    index_to_variable[count_vars] = v
                    if k_aug:
                        v.set_suffix_value(model.dof_v, count_vars)
                    count_vars += 1
                
    for exp, model in models_dict.items():
        if hasattr(model, 'S') and exp_count:
            var = 'S'
            for index in getattr(model, var).index_set():
                if index[1] in component_set:
                    v = getattr(model, var)[index]
                    index_to_variable[count_vars] = v
                    if k_aug:
                        v.set_suffix_value(model.dof_v, count_vars)
                    count_vars += 1
       # exp_count = False
        
    for exp, model in models_dict.items():
        
        for parameter_kind in __var.optimization_variables:
            if hasattr(model, parameter_kind):
                for v in getattr(model, parameter_kind).values():
                    #print(v.to_string, v.fixed)
                    if v.to_string() in param_names_full and not v.fixed:
                        index_to_variable[count_vars] = v
                        if k_aug:
                            v.set_suffix_value(model.dof_v, count_vars)
                        count_vars += 1

    return index_to_variable


def compute_covariance(models_dict, hessian, free_params, all_variances):
    """This method calculates the covariance matrix for spectral problems
    
    :param dict models_dict: The models in dict form
    :param numpy.ndarray hessian: The reduced hessian
    :param int free_params: The number of free parameters
    :param dict all_variances: The dict of variances (as dict)
    
    :return V_theta: The parameter covariance matrix
    :rtype: scipy.sparse.coo_matrix
    
    """
    H = hessian
    if isinstance(H, pd.DataFrame):
        H = H.values
    B = make_B_matrix(models_dict, free_params, all_variances)
    Vd = make_Vd_matrix(models_dict, all_variances)
    
    R = B.T @ H.T
    A = Vd @ R
    L = H @ B
    V_theta = 4*(A.T @ L.T).T 
    
    return V_theta
        
    
def make_B_matrix(models_dict, free_params, all_variances):
    """This method generates a B matrix for the covariance calculations in both
    single and multiple experiments.

    :param dict models_dict: The models in dict form
    :param int free_params: The number of free parameters
    
    :return B_matrix: The B matrix
    :rtype: scipy.sparse.coo_matrix
    
    """
    S_part = merge_blocks('S', models_dict, all_variances)
    C_part = merge_blocks('C', models_dict, all_variances)
    
    sy, sx = S_part.shape
    M = lil_matrix((C_part.size + S_part.size + free_params, C_part.shape[1]*S_part.shape[1]))
    c_index_start = C_part.size
    
    for i in range(C_part.shape[1]):
    
        M[i*sy:(i + 1)*sy, i*sx:(i + 1)*sx] = S_part
        Ct = C_part[:, i].reshape(-1, 1)
        cy, cx = Ct.shape

        for ci in range(sx):
            M[c_index_start + ci*cy : c_index_start + (ci + 1)*cy, cx*ci + i*sx : cx*ci + i*sx + cx] = Ct
    
    B_matrix = M.tocoo()
    
    return B_matrix


def merge_blocks(var, models_dict, all_variances):
    """Merges the defined data types into single blocks
    
    :param str var: The model variable
    
    :return: s_mat: the array of model data
    :rtype: numpy.ndarray
    
    """
    # TODO: Make sure blocks of different sizes (components) can be handled
    s_mat = None
    for e, exp in models_dict.items():
        
        comp_vars = [k for k, v in all_variances[e].items() if k != 'device']    
        
        if s_mat is None:
            s_mat = convert(getattr(exp, var)).loc[:, comp_vars].T.values / all_variances[e]['device']
        else:
            s_mat_append = convert(getattr(exp, var)).loc[:, comp_vars].T.values / all_variances[e]['device']
            s_mat = np.hstack((s_mat, s_mat_append))

    return s_mat


def make_Vd_matrix(models_dict, all_variances):
    """Builds d covariance matrix

    This method is not intended to be used by users directly

    :param dict models_dict: Either a pyomo ConcreteModel or a dict of ReactionModels
    :param dict all_variances: variances

    :return: The Vd matrix as a sparse matrix

    """
    from kipet.model_tools.pyomo_model_tools import convert
    
    Vd_dict = {}
    M_dict = {}
    total_shape = 0
    n_models = len(models_dict)
    
    for name, model in models_dict.items():

        variances = all_variances[name]
        times = model.allmeas_times.ordered_data()
        waves = model.meas_lambdas.ordered_data()
        n_waves = len(waves)
        n_times = len(times)
        
        Vd = lil_matrix((n_models * n_times * n_waves, n_models * n_times * n_waves))
        S = convert(model.S)
        comp_vars = [k for k, v in variances.items() if k != 'device']
        S = S.loc[:, comp_vars]
        
        device_variance = variances['device']
        M = np.array([v for k, v in variances.items() if k != 'device']) * S.values @ S.values.T
        M_diag = np.einsum('ii->i', M)
        M_diag += device_variance
        M_dict[name] = M

        for t in range(n_models*n_times):
            Vd[t*n_waves: (t+1)*n_waves, t*n_waves: (t+1)*n_waves] = M
            
        total_shape += Vd.shape[0]
        Vd_dict[name] = Vd
        
    if n_models > 1:
        
        Vd_combined = lil_matrix((total_shape, total_shape))
        start_index = 0
        for model, Vd in Vd_dict.items():        
            Vd_combined[start_index:Vd.shape[0]+start_index, start_index:Vd.shape[1]+start_index] = Vd
            start_index = Vd.shape[0]
        
        return Vd_combined.tocoo()
    
    return Vd.tocoo()


def index_variable_mapping(model_dict, components, parameter_names, mee_obj=None, k_aug=False):
    """This adds the needed suffixes for the reduced hessian to the model object
    used in covariance predictions
    
    :param ConcreteModel model_dict: The model from the parameter estimator or the MEE
    :param list components: The list of absorbing components
    :param list parameter_names: The list of parameter names
    :param mee_obj: An MEE instance, default None
    
    :return: index_to_variables
    :rtype: dict
    
    """
    if mee_obj is not None:
        model_obj = mee_obj
    else:
        model_obj = model_dict
        
    if not hasattr(model_obj, 'red_hessian'):
        model_obj.red_hessian = Suffix(direction=Suffix.IMPORT_EXPORT)
    index_to_variables = define_reduce_hess_order(model_dict, components, parameter_names, k_aug)
    
    for k, v in index_to_variables.items():
        model_obj.red_hessian[v] = k
        
    return index_to_variables


def read_covariance_matrix_sipopt(temp_file, parameter_number):
    """Generalize the covariance optimization with IPOPT Sens

    :param ConcreteModel model: The Pyomo model used in parameter fitting
    :param SolverFactory optimizer: The SolverFactory currently being used
    :param bool tee: Display option
    :param dict all_sigma_specified: The provided variances
    :param mee_obj: An MEE instance, default None

    :return hessian: The covariance matrix
    :rtype: numpy.ndarray

    """
    from kipet.input_output.read_hessian import split_sipopt_string, read_reduce_hessian
   
    with open(temp_file, 'r') as f:
        output_string = f.read()
    
    ipopt_output, hessian_output = split_sipopt_string(output_string)
    covariance_matrix = read_reduce_hessian(hessian_output, parameter_number)
    
    return covariance_matrix


def covariance_sipopt(model_obj, solver_factory, components, parameters, mee_obj=None):
    """Generalize the covariance optimization with IPOPT Sens

    :param ConcreteModel model: The Pyomo model used in parameter fitting
    :param SolverFactory optimizer: The SolverFactory currently being used
    :param bool tee: Display option
    :param dict all_sigma_specified: The provided variances
    :param mee_obj: An MEE instance, default None

    :return hessian: The covariance matrix
    :rtype: numpy.ndarray

    """
    _tmpfile = "sipopt_red_hess"
    index_to_variable = index_variable_mapping(model_obj, components, parameters, mee_obj)
    optimization_model = model_obj if mee_obj is None else mee_obj
    
    solver_results = solver_factory.solve(
        optimization_model,
        tee=False,
        logfile=_tmpfile,
        report_timing=True,
        symbolic_solver_labels=True,
        keepfiles=True)
    
    covariance_matrix = read_covariance_matrix_sipopt(_tmpfile, len(index_to_variable))
    
    labels = [v.name for v in index_to_variable.values()]
    covariance_matrix = pd.DataFrame(covariance_matrix, columns=labels, index=labels)
    #print(f'{covariance_matrix = }')
    covariance_matrix_reduced = covariance_matrix.loc[parameters, :]
    #print(f'{covariance_matrix_reduced = }')
    
    return covariance_matrix, covariance_matrix_reduced


def covariance_kaug_direct(reaction_model, as_df=True):
    """This returns the reduced Hessian via k_aug if 1 ncp is used
    
    """
    # Set up the variables
    model = reaction_model.p_model
    ncp = reaction_model.settings.collocation.ncp
    
    if ncp != 1:
        raise AttributeError('This method requires ncp = 1')
    
    k_aug = SolverFactory(solver_path('k_aug'))
    ipopt = SolverFactory(solver_path('ipopt'))
    components = reaction_model.p_estimator.comps['unknown_absorbance']
    parameters = reaction_model.p_estimator.param_names_full
    _tmpfile = "k_aug_hess"
    
    # Model preparation
    add_warm_start_suffixes(model, use_k_aug=True) 
    index_to_variable = index_variable_mapping(model, components, parameters, None, k_aug=True)
    
    # print(f'{components = }')
    # print(f'{parameters = }')
    # print(f'{index_to_variable = }')
    
    # Solve using IPOPT
    solver_results = ipopt.solve(
        model,
        tee=False,
        logfile=_tmpfile,
        report_timing=True,
        symbolic_solver_labels=True,
        keepfiles=True,
        )

    update_warm_start(model)

    # Finish using k_aug
    k_aug.options["compute_inv"] = ""
    #k_aug.options["print_kkt"] = ""
    k_aug.solve(model, tee=False)
    kaug_files = Path('kaug_debug')
    covariance_file = kaug_files.joinpath('result_red_hess.txt')
    unordered_hessian = np.loadtxt(covariance_file)
    var_loc = model.rh_name
    
    for v in index_to_variable.values():
        try:
            var_loc[v]
        except:
            var_loc[v] = 0

    n_vars = len(index_to_variable)
    hessian = np.zeros((n_vars, n_vars))
    
    for i, vi in enumerate(index_to_variable.values()):
        for j, vj in enumerate(index_to_variable.values()):
            if n_vars == 1:
                h = unordered_hessian
                hessian[i, j] = h
            else:
                h = unordered_hessian[(var_loc[vi]), (var_loc[vj])]
                hessian[i, j] = h
    
    covariance_matrix = hessian  
    labels = [v.name for v in index_to_variable.values()]
    covariance_matrix = pd.DataFrame(covariance_matrix, columns=labels, index=labels)
    print(f'{covariance_matrix = }')
    covariance_matrix_reduced = covariance_matrix.loc[parameters, :]
    print(f'{covariance_matrix_reduced = }')
    
    return covariance_matrix, covariance_matrix_reduced
    
    
def kkt_kaug(reaction_model, pyomo_model=False, nlp=None):
    """Gets the KKT information using k_aug for a model solved using the 
    Parameter Estimator
    
    """   
    if pyomo_model:
        model = reaction_model
    else:
        model = reaction_model.p_model
    
    k_aug = SolverFactory(solver_path('k_aug'))
    k_aug.options["print_kkt"] = ""
    k_aug.solve(model, tee=False)
    
    if nlp is None:
        nlp = reaction_model._nlp
        
    size = (nlp.n_constraints(), nlp.n_primals())
    H, J, G = _build_raw_J_and_H(size)
    K = bmat([[H, J.T],[J, None]], format='csc')
    
    return H, J, K


def reorder_rh_cov_matrix(reaction_model, df):
    """Reorders the covariance or reduced Hessian matrix to have the parameters
    in order from C to S to P
    """
    columns = spectral_hess_reorder(reaction_model)
    df = df.reindex(columns, columns=columns)
    
    return df
    
def reduced_hessian_kaug(reaction_model):
    """Convenience method for generating the reduced Hessian from k_aug
    
    """
    ncp = reaction_model.settings.collocation.ncp
    
    if ncp > 1:
        # This doesn't work for ncp > 1 at the moment
        rh = reduced_hessian_single_model(reaction_model, use_k_aug=False)
    else:
        cov, cov_reduced = covariance_kaug_direct(reaction_model)
        rh = np.linalg.inv(cov)
        rh = pd.DataFrame(rh, labels=cov.labels, index=cov.index)
        
    return rh


def covariance_k_aug(reaction_model):
    """Returns the covariance method using k_aug after a model has been solved
    
    """
    ncp = reaction_model.settings.collocation.ncp
    
    if ncp > 1:
        rh = reduced_hessian_single_model(reaction_model, use_k_aug=False)
        rh = reorder_rh_cov_matrix(reaction_model, rh)
        cov = np.linalg.inv(rh)
        covariance_matrix = pd.DataFrame(cov, columns=rh.columns, index=rh.index)
        col_ind_P, col_ind_param_hr = new_free_variables(reaction_model)
        covariance_matrix_reduced = covariance_matrix.loc[list(col_ind_P.keys()), :]
    else:
        covariance_matrix, covariance_matrix_reduced = covariance_kaug_direct(reaction_model)
        
    return covariance_matrix, covariance_matrix_reduced


def build_E_matrix(nlp, mdict):
    """Builds the E matrix (see NSD theory) that places a one at aligned with
    the parameter and the dummy constraint in the full KKT matrix
    
    """
    _J_dum = nlp.extract_submatrix_jacobian(pyomo_variables=mdict['varList'], pyomo_constraints=mdict['conList_dummy'])
    _z = coo_matrix((len(mdict['conList_red']), len(mdict['conList_dummy'])))
    E = vstack([_J_dum.T, _z], format='csc')
    
    return E


def kkt_pynumero_nlp(nlp, mdict, use_cyipopt=False, index_list=None):
    """Builds the KKT matrix using PyomoNLP object
    
    """
    J = nlp.extract_submatrix_jacobian(pyomo_variables=mdict['varList'], pyomo_constraints=mdict['conList_red'])
    H = nlp.extract_submatrix_hessian_lag(pyomo_variables_rows=mdict['varList'], pyomo_variables_cols=mdict['varList'])
    K = bmat([[H, J.T],[J, None]], format='csc')

    # if use_inertia_correction:
        # K = inertia_correction(K)

    return H, J, K

    
def kkt_pynumero_projected_nlp(nlp, index_list=None):
    """Builds the KKT matrix from a ProjectedNLP object
    
    """
    use_cyipopt=True
    H, J, grad = build_matrices(nlp, index_list[0], use_cyipopt=use_cyipopt)
    J_c = delete_from_csr(J.tocsr(), row_indices=index_list[2]).tocsc()
    J = J_c
    K = bmat([[H, J.T],[J, None]], format='csc')
    
    # if use_inertia_correction:
        # K = inertia_correction(K)

    return H, J, K


def reduced_hessian_from_full_kkt(K, E, labels=None):
    """Solves the system of equations for the matrix S
    
    """
    lhs = bmat([[K, E],[E.T, None]], format='csc')
    G = np.eye(E.shape[1])
    _z = csc_matrix((K.shape[1], G.shape[1]))
    rhs = vstack([_z, -G], format='csc') 
    QS = spsolve(lhs, rhs).tolil()
    S = QS[-E.shape[1]:, :]
    rh = np.asarray(S.todense())
    
    if labels is not None:
        rh = pd.DataFrame(rh, columns=labels, index=labels)

    return rh


def evaluate_gradient(nlp, mdict, index_list, use_cyipopt=False):
    """Return the multipliers for the dummy constraints (m in NSD theory)
    
    """
    m = 0
    global_param_index, _, dummy_con_index = index_list
    duals = nlp.n_constraints()*[0.0]
    
    if not use_cyipopt:
        model = nlp.pyomo_model()
        for index, con in zip(dummy_con_index, mdict['conList_dummy']):
            duals[index] = model.dual[con]
    else:
    
        duals = nlp.get_duals()
        # for index, con in zip(dummy_con_index, mdict['conList_dummy']):
        #     duals[index] = nlp.get_duals()[index]
        
    m += np.array(duals)
    return m


def new_free_variables(reaction_model):
    
    """
    This uses pynumero instead to find the index of the vars instead of needing to provide a bunch of info
    """
    nlp = reaction_model._nlp
    var_dict = {n.name: n for n in nlp.get_pyomo_variables()}
    var_index = {n.name : nlp.get_primal_indices([n])[0] for n in var_dict.values()}
    
    spectral_vars = []
    if hasattr(nlp.pyomo_model(), 'C') and hasattr(nlp.pyomo_model(), 'S'):
        for var in ['C', 'S']:
            spectral_vars += [v.local_name for k, v in var_dict.items() if v.parent_component().name == var and v.index()[1] in reaction_model.components.get_match('absorbing', True)]
    
    from kipet.general_settings.variable_names import VariableNames
    __var = VariableNames()
    parameters = [v.local_name for k, v in var_dict.items() if v.parent_component().name in __var.optimization_variables]
    
    col_ind = [var_index[v] for v in spectral_vars + parameters]
    col_ind_P = {v: var_index[v] for v in parameters}
    col_ind_param_hr = [col_ind.index(p) for p in col_ind_P.values()]

    return col_ind_P, col_ind_param_hr


def _build_raw_J_and_H(size):
    """Given the size of the variables and constraints, the Hessian and Jacobian
    can be built using the output files from k_aug
    
    :param tuple size: The m (con) and n (var) size of the Jacobian
    
    :return: The Hessian and Jacobian as a tuple of sparse (coo) matrices
    
    """
    m, n = size
    kaug_files = Path('GJH')
    
    kkt_file = kaug_files.joinpath('kkt_print.txt')
    kkt = pd.read_csv(kkt_file, delim_whitespace=True, header=None, skipinitialspace=True)
    kkt.columns = ['irow', 'jcol', 'vals']
    kkt['irow'] -= 1
    kkt['jcol'] -= 1
    
    hess_file = kaug_files.joinpath('H_print.txt')
    hess = pd.read_csv(hess_file, delim_whitespace=True, header=None, skipinitialspace=True)
    hess.columns = ['irow', 'jcol', 'vals']
    hess['irow'] -= 1
    hess['jcol'] -= 1

    jac_file = kaug_files.joinpath('A_print.txt')
    jac = pd.read_csv(jac_file, delim_whitespace=True, header=None, skipinitialspace=True)
    jac.columns = ['irow', 'jcol', 'vals']
    jac['irow'] -= 1
    jac['jcol'] -= 1

    grad_file = kaug_files.joinpath('gradient_f_print.txt')
    grad = pd.read_csv(grad_file, delim_whitespace=True, header=None, skipinitialspace=True)
    grad.columns = ['gradient']
    
    J = coo_matrix((jac.vals, (jac.jcol, jac.irow)), shape=(m, n))
    Hess_coo = coo_matrix((hess.vals, (hess.irow, hess.jcol)), shape=(n, n))
    H = Hess_coo + triu(Hess_coo, 1).T
    #K = coo_matrix((kkt.vals, (kkt.jcol, kkt.irow)), shape=(m + n, m + n))
    
    return H, J, grad
        
    
def get_duals_k_aug(model_object, constraint_name, param_name):
    
    duals = {key: model_object.dual[getattr(model_object, constraint_name)[key]] for
                  key, val in getattr(model_object, param_name).items()}
    
    return duals
    
def add_warm_start_suffixes(model, use_k_aug=False):
    """Adds suffixed variables to problem

    :param ConcreteModel model: A Pyomo model
    :param bool use_k_aug: Indicates if k_aug solver is being used

    :return: None

    """
    if use_k_aug:
        if not hasattr(model, 'dof_v'):
            model.dof_v = Suffix(direction=Suffix.EXPORT)
        if not hasattr(model, 'rh_name'):
            model.rh_name = Suffix(direction=Suffix.IMPORT)
    else:
        # Ipopt bound multipliers (obtained from solution)
        model.ipopt_zL_out = Suffix(direction=Suffix.IMPORT)
        model.ipopt_zU_out = Suffix(direction=Suffix.IMPORT)
        # Ipopt bound multipliers (sent to solver)
        model.ipopt_zL_in = Suffix(direction=Suffix.EXPORT)
        model.ipopt_zU_in = Suffix(direction=Suffix.EXPORT)
        # Obtain dual solutions from first solve and send to warm start
        model.dual = Suffix(direction=Suffix.IMPORT_EXPORT)
    
    return None

        
def update_warm_start(model):
    """Updates the suffixed variables for a warmstart

    :param ConcreteModel model: A Pyomo model

    :return: None

    """
    model.ipopt_zL_in.update(model.ipopt_zL_out)
    model.ipopt_zU_in.update(model.ipopt_zU_out)
    
    return None

def build_matrices(nlp_object, global_param_index, use_cyipopt=True, use_sigma=True):
    """Method to build the H, J, and G matrices/vectors
    
    Unfortunately, the H calculated using the cyipopt method is different
    """
    if use_cyipopt:
    
        # print(f'{nlp_object.get_obj_scaling() = }')
        # print(f'{nlp_object.get_obj_factor() = }') 
    
        h_raw = nlp_object.evaluate_hessian_lag().tolil()
        # if nlp_object.get_obj_factor() != 1:
        #     factor = nlp_object.get_obj_factor()
        #     if factor == 0:
        #         factor = 1
        #     h_raw.setdiag(h_raw.diagonal()/factor)

        # # These need to be checked for S variables! (after they work with SolverFactory)
        h_raw[global_param_index, :] *= -1
        h_raw[:, global_param_index] *= -1
        H = h_raw.tocoo()
        #H = nlp_object.evaluate_hessian_lag()
        
    else:
        H = nlp_object.evaluate_hessian_lag()
        
    # if use_sigma:
    #     H = bounds_modification_nlp(nlp_object, H, global_param_index)
        
    J = nlp_object.evaluate_jacobian()
    G = nlp_object.evaluate_grad_objective()

    return H, J, G


def bounds_modification_nlp(nlp_object, H, global_param_index):

    bound_relax_factor = 1e-8 
    eps = np.finfo(float).eps
    #print('Adding Sigmas')
    
    if isinstance(nlp_object, PyomoNLP):
        model = nlp_object.pyomo_model()
        varList = nlp_object.get_pyomo_variables()
    elif isinstance(nlp_object, ProjectedNLP):
        model = nlp_object._original_nlp.pyomo_model()
        varList = nlp_object._original_nlp.get_pyomo_variables()[:nlp_object.n_primals()]
    else:
        raise ValueError('Incorrect NLP object')
        
    direction = 'out'
    suffix_U = f'ipopt_zU_{direction}'
    suffix_L = f'ipopt_zL_{direction}'
    #print(getattr(model, suffix_U).display())
    #print(getattr(model, suffix_L).display())
    #print(f'{global_param_index = }')
    sigma_L = np.zeros((len(varList)))
    sigma_U = np.zeros((len(varList)))

    if hasattr(model, suffix_L):
        if len(getattr(model, suffix_L)) == 0:
            #print('Nothing to get here...moving on')
            return H

    # print(f'{len(model.ipopt_zU_in)}')
    # print(f'{len(model.ipopt_zU_out)}')
    # print(f'{len(model.ipopt_zL_in)}')
    # print(f'{len(model.ipopt_zL_out)}')
        
    for i, indx in enumerate(global_param_index):
        
        v = varList[indx]
        sig_L = 0
        if v.lb is not None:
            #print(f'{v.value = }\n{v.lb = }\n{v.ub = }')
            zL = getattr(model, suffix_L)[v]
            #print(f'{zL = }')
            sL = v.value - v.lb
            xlb_max = np.maximum(1, abs(v.lb))
            sL += bound_relax_factor*xlb_max
            if sL <  1e-40:
                sL += 10*eps*xlb_max
                
            sig_L = zL/sL
                
        sigma_L[indx] = (sig_L)
        
        sig_U = 0
        if v.ub is not None:
            # print(f'{v.value = }\n{v.lb = }\n{v.ub = }')
            zU = abs(getattr(model, suffix_U)[v])
            #print(f'{zU = }')
            sU = v.ub - v.value
            xub_max = np.maximum(1, abs(v.ub))
            sU += bound_relax_factor*xub_max
            if sU <  1e-40:
                sU += 10*eps*xub_max
                
            sig_U = zU/sU
            
        sigma_U[indx] = (sig_U)
        #print(f'{sig_L = }; {sig_U = }')
        
    SL = np.diag(sigma_L)
    SU = np.diag(sigma_U)
    _SL = coo_matrix(SL)
    _SU = coo_matrix(SU)

    _H = H + _SL + _SU
    
    return _H


def calculate_reduced_hessian(H, J, index_list, global_param_objs, as_df=True, debug=False):
    """Final method for calculating the reduced Hessian
    
    """
    global_param_index, local_param_index, dummy_con_index = index_list
    all_param_index = global_param_index + local_param_index
    
    J_c = delete_from_csr(J.tocsr(), row_indices=dummy_con_index).tocsc()
    row_indexer = SparseRowIndexer(J_c.T)
    J_f = row_indexer[global_param_index].T
    J_l = delete_from_csr(J_c.tocsr(), col_indices=all_param_index) 
    
    if debug:
        print(f'{J.shape = }')
        print(f'{H.shape = }')
        print(f'{J_c.shape = }')
        print(f'{J_f.shape = }')
        print(f'{J_l.shape = }')
        print(f'{J_f = }')
    
    n_free = len(global_param_index)
    n = H.shape[0]
    X = spsolve(J_l.tocsc(), -J_f.tocsc())
    constrained_var_index = list(set(range(n)).difference(set(all_param_index)))
    constrained_var_index.sort()
    Z = lil_matrix(np.zeros([n, n_free]))
    Z[global_param_index, :] = np.eye(n_free)
    Z[constrained_var_index, :] = X
    reduced_hessian = Z.T * H * Z
    rh = reduced_hessian.todense()
    
    if as_df:
        labels = [p.name for p in global_param_objs]
        rh = pd.DataFrame(rh, columns=labels, index=labels)
    
    return rh


def prepare_pseudo_fixed_global_variables(model, global_params, model_vars=['P'], delta=1e-20):
    """Sets tight bounds for the global variables (RHPS)
    
    """
    for model_var in model_vars:
        if hasattr(model, model_var):    
            for k, v in getattr(model, model_var).items():
                if k in global_params:
                    ub = v.value
                    lb = v.value - delta
                    v.setlb(lb)
                    v.setub(ub)
                    v.unfix()
                else:
                    v.fix()

    return None


def prepare_global_constraints(model, global_params, local_params, use_cyipopt=True):
    """Finds the global and local parameters and adds the dummy constraints
    to the model
    
    """
    global_param_name='d'
    global_constraint_name='fix_params_to_global'
    fix_d_vars=not use_cyipopt
    global_param_init = {}
    global_constraints = []
    constraint_map = {}
    
    for comp in [global_constraint_name, global_param_name]:
        if hasattr(model, comp):
            model.del_component(comp)
        if hasattr(model, f'{comp}_index'):
            model.del_component(f'{comp}_index')
    
    setattr(model, global_constraint_name, ConstraintList())
    setattr(model, global_param_name, VarList())
    
    for i, (k, v) in enumerate(global_params.items()):
        
        getattr(model, global_param_name).add()
        getattr(model, global_param_name)[i + 1] = getattr(model, v.model_var)[v.index].value
        getattr(model, global_constraint_name).add(getattr(model, v.model_var)[v.index] - getattr(model, global_param_name)[i + 1] == 0)
        constraint_map[v.name] = i + 1
        
        if fix_d_vars:
            getattr(model, global_param_name)[i + 1].fix()
        
    return constraint_map
    

def generate_parameter_object_lists(r_model, single=False):
    
    """Parameter object lists are used to generate the RH
    
    """
    model = r_model.p_model
    if not single:
        global_params_dict, local_params_dict = r_model.globals_locals
    else:
        global_params_dict, local_params_dict = r_model.all_globals
    global_param_objs = [v.pyomo_var for v in global_params_dict.values()]
    local_param_objs = [v.pyomo_var for v in local_params_dict.values()]
    dummy_con_objs = []
    global_constraint_name = 'fix_params_to_global'
    
    if hasattr(model, global_constraint_name):
        dummy_con_objs = [v for v in getattr(model, global_constraint_name).values()]
    
    return global_param_objs, local_param_objs, dummy_con_objs


def generate_parameter_index_lists(nlp, model_object_lists):
    """Makes lists of parameter indeces
    
    """
    if isinstance(nlp, ProjectedNLP):
        nlp = nlp._original_nlp
    
    global_param_objs, local_param_objs, dummy_con_objs = model_object_lists
    global_param_index = nlp.get_primal_indices(global_param_objs)
    local_param_index = nlp.get_primal_indices(local_param_objs)
    dummy_con_index = nlp.get_constraint_indices(dummy_con_objs)
    
    return global_param_index, local_param_index, dummy_con_index


def optimize_model(model, solver_options=None, nlp_object=None, return_nlp=True, files=False, use_sipopt=False, debug=False):
    """Generic method to optimize a ConcreteModel using the SolverFactory.
    
    """
    solver = SolverFactory(solver_path('ipopt'))
    #solver = check_libs(solver)
    
    if solver_options is None:
        solver_options = {'linear_solver': 'ma57'}
    
    if isinstance(solver_options, dict):
        for k, v in solver_options.items():
            solver.options[k] = v
    
    solver_kwargs = {'tee': True}
    if files:
        
        solver_kwargs['logfile'] = "opt_model_k_aug"
        solver_kwargs['report_timing'] = True
        solver_kwargs['symbolic_solver_labels'] = True
        solver_kwargs['keepfiles'] = True
    
    solver.solve(model, **solver_kwargs)
    update_warm_start(model)
    
    nlp_object = None
    if return_nlp:
        nlp_object = PyomoNLP(model)
    
    return nlp_object, solver, model


def optimize_model_cyipopt(nlp_object, solver_options=None):
    """Generic method to optimize a ConcreteModel using a projected NLP object
    using CyIpopt.
    
    """    
    if nlp_object is None:
        raise ValueError('The ProjectedNLP object needs to be provided')
        
    solver_settings = SolverSettings()
    if solver_options is None:
        solver_options = {
            'linear_solver': 'ma57',
            }
        
    if solver_settings.custom_solvers_lib['hsllib'] is not None:
        solver_options['hsllib'] = solver_settings.custom_solvers_lib['hsllib']
        
    cy_nlp = CyIpoptNLP(nlp_object)
    csolve = CyIpoptSolver(cy_nlp, 
                           options = solver_options,
                           )
    
    x, info = csolve.solve(tee=True)
    nlp_object._original_nlp.set_duals(info['x'])
    nlp_object._original_nlp.set_duals(info['mult_g'])
    nlp_object._original_nlp.load_state_into_pyomo(
                bound_multipliers=(info['mult_x_L'], info['mult_x_U']))

    return nlp_object, info


def covariance_matrix_single_model(reaction_model, use_sigma=True, use_k_aug=False):
    """A simple method to get the covariance matrix without much effort from the user
    
    :param ReactionModel reaction_model: The ReactionModel object with a solved p_model
    
    :return pd.DataFrame covariance_matrix: The covariance matrix of the p_model
    
    """
    rh = reduced_hessian_single_model(reaction_model, use_sigma=use_sigma, use_k_aug=use_k_aug)
    covariance_matrix = np.linalg.inv(rh)
    covariance_matrix = pd.DataFrame(covariance_matrix, columns=rh.columns, index=rh.index)
    covariance_matrix = reorder_rh_cov_matrix(reaction_model, covariance_matrix)
    col_ind_P, col_ind_param_hr = new_free_variables(reaction_model)
    covariance_matrix_reduced = covariance_matrix.loc[list(col_ind_P.keys()), :]

    return covariance_matrix, covariance_matrix_reduced


def reduced_hessian_single_model(reaction_model, use_sigma=True, use_k_aug=False):
    """A simple method to get the reduced Hessian without much effort from the user
    
    :param ReactionModel reaction_model: The ReactionModel object with a solved p_model
    
    :return pd.DataFrame rh: The reduced Hessian of the p_model
    
    """
    if use_k_aug:
        use_sigma = False
        
    nlp_object = reaction_model._nlp
    model_object_lists = generate_parameter_object_lists(reaction_model, single=True)
    index_list = generate_parameter_index_lists(nlp_object, model_object_lists)
    
    if use_k_aug:
        H, J, K = kkt_kaug(reaction_model)
    else:
        H, J, grad = build_matrices(nlp_object, index_list[0], use_cyipopt=False, use_sigma=use_sigma)
    
    rh = calculate_reduced_hessian(H, J, index_list, model_object_lists[0], as_df=True, debug=True)
    
    return rh


def reduced_hessian(nlp_object, model_object_lists, use_cyipopt=False, as_df=True, debug=False, use_k_aug=False):
    """Consolidates some methods together
    
    """    
    index_list = generate_parameter_index_lists(nlp_object, model_object_lists)
    
    if not use_k_aug:
        H, J, grad = build_matrices(nlp_object, index_list[0], use_cyipopt=use_cyipopt)
        rh = calculate_reduced_hessian(H, J, index_list, model_object_lists[0], as_df=as_df, debug=debug)
    else:
        pass
    
        
    
    return rh


def inertia_correction(m, kappa_e = 10.0, e_min =1.0e-20, e_max = 1.0):
    """Compute the inertia correction for a given matrix m
    Arg:
        m: ndarray

        kappa_e: increment of the correction value

        e_min: initial value of the correction value

        e_max: maximum value of the correction value
    
    Retuen:
        m : ndarray corrected m matrix
    """
    # Check whether the inertia correction is required or not
    is_IC_correct = False
    n_pivot = np.linalg.matrix_rank(m)
    if n_pivot == m.shape[0]:
        print('--- Correct inertia without correction ---')
        is_IC_correct = True

    if not is_IC_correct:
        # Initialize correction value and counter
        e = e_min
        counter = 0 
        
        # Check the inertial of matrix m
        while True:

            if e > e_max:
                raise Exception('Inertia correction reaches the maximum limit')

            m = m + e*np.indentity(m.shape[0])
            n_pivot = np.linalg.matrix_rank(m)

            if n_pivot == m.shape[0]:
                print(f'--- Correct inertia with e = {e} ---')
                break
            elif n_pivot < m.shape[0]:
                counter += 1
                e *= kappa_e

    return m


def delete_from_csr(mat, row_indices=[], col_indices=[]):
    """
    Remove the rows (denoted by ``row_indices``) and columns (denoted by 
    ``col_indices``) from the CSR sparse matrix ``mat``.
    WARNING: Indices of altered axes are reset in the returned matrix

    :param csr_matrix mat: Sparse matrix to delete rows and cols from
    :param list row_indicies: rows to delete
    :param list col_indicies: cols to delete

    :return csr_matrix mat: The sparse matrix with the rows and cols removed

    """
    if not isinstance(mat, csr_matrix):
        raise ValueError("works only for CSR format -- use .tocsr() first")

    rows = []
    cols = []
    if row_indices:
        rows = list(row_indices)
    if col_indices:
        cols = list(col_indices)

    if len(rows) > 0 and len(cols) > 0:
        row_mask = np.ones(mat.shape[0], dtype=bool)
        row_mask[rows] = False
        col_mask = np.ones(mat.shape[1], dtype=bool)
        col_mask[cols] = False
        return mat[row_mask][:, col_mask]
    elif len(rows) > 0:
        mask = np.ones(mat.shape[0], dtype=bool)
        mask[rows] = False
        return mat[mask]
    elif len(cols) > 0:
        mask = np.ones(mat.shape[1], dtype=bool)
        mask[cols] = False
        return mat[:, mask]
    else:
        return mat


class SparseRowIndexer:
    """Class used to splice sparse matrices"""

    def __init__(self, matrix):
        data = []
        indices = []
        indptr = []

        _class = 'csr'
        if isinstance(matrix, csc_matrix):
            _class = 'csc'

        self._class = _class
        # Iterating over the rows this way is significantly more efficient
        # than matrix[row_index,:] and matrix.getrow(row_index)
        for row_start, row_end in zip(matrix.indptr[:-1], matrix.indptr[1:]):
            data.append(matrix.data[row_start:row_end])
            indices.append(matrix.indices[row_start:row_end])
            indptr.append(row_end - row_start)  # nnz of the row

        self.data = np.array(data, dtype=object)
        self.indices = np.array(indices, dtype=object)
        self.indptr = np.array(indptr)
        self.n_columns = matrix.shape[1]

    def __getitem__(self, row_selector):
        data = np.concatenate(self.data[row_selector])
        indices = np.concatenate(self.indices[row_selector])
        indptr = np.append(0, np.cumsum(self.indptr[row_selector]))

        shape = [indptr.shape[0] - 1, self.n_columns]

        if self._class == 'csr':
            return csr_matrix((data, indices, indptr), shape=shape)
        else:
            return csr_matrix((data, indices, indptr), shape=shape).tocsc()