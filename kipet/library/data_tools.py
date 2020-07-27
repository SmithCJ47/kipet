from contextlib import contextmanager
import scipy
import six
import sys

import re
import matplotlib as cm
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pathlib import Path

#=============================================================================
#-----------------------DATA READING AND WRITING TOOLS------------------------
#=============================================================================

def read_file(filename):       
    """ Reads data from a csv or txt file and converts it to a DataFrame
    
        Args:
            filename (str): name of input file
          
        Returns:
            DataFrame

    """
    filename = Path(filename)
    data_dict = {}
    if filename.suffix == '.txt':
    
        with open(filename, 'r') as f:
            for line in f:
                if line not in ['','\n','\t','\t\n']:
                    l = line.split()
                    if is_float_re(l[1]):
                        l[1] = float(l[1])
                    data_dict[float(l[0]), l[1]] = float(l[2])
        
        df_data = dict_to_df(data_dict)
        df_data.sort_index(ascending=True, inplace=True)
        return df_data

    elif filename.suffix == '.csv':
        
        df_data = pd.read_csv(filename, index_col=0)
        return df_data   

    else:
        raise ValueError(f'The file extension {filename.suffix} is currently not supported')
        return None
#%%
def write_file(filename, dataframe, filetype='csv'):
    """ Write data to file.
    
        Args:
            filename (str): name of output file
          
            dataframe (DataFrame): pandas DataFrame
        
            filetype (str): choice of output (csv, txt)
        
        Returns:
            None

    """
    if filetype not in ['csv', 'txt']:
        print('Savings as CSV - invalid file extension given')
        filetype = 'csv'
    
    suffix = '.' + filetype
    
    filename = Path(filename)
    print(filename.suffix)
    if filename.suffix == '':
        filename = filename.with_suffix(suffix)
    else:
        suffix = filename.suffix
        if suffix not in ['.txt', '.csv']:
            print('Savings as CSV - invalid file extension given')
            filename = Path(filename.stem).with_suffix('.csv')
    
    print(filename)
    
    if filename.suffix == '.csv':
        dataframe.to_csv(filename)

    elif filename.suffix == 'txt':    
        with open(filename, 'w') as f:
            for i in dataframe.index:
                for j in dataframe.columns:
                    if not np.isnan(dataframe[j][i]):
                        f.write("{0} {1} {2}\n".format(i,j,dataframe[j][i]))
                        
    print(f'Data successfully saved as {filename}')
    return None
#%%
def read_spectral_data_from_csv(filename, instrument = False, negatives_to_zero = False):
    """ Reads csv with spectral data
    
        Args:
            filename (str): name of input file
            instrument (bool): if data is direct from instrument
            negatives_to_zero (bool): if data contains negatives and baseline shift is not
                                        done then this forces negative values to zero.

        Returns:
            DataFrame

    """
    data = pd.read_csv(filename,index_col=0)
    if instrument:
        #this means we probably have a date/timestamp on the columns
        data = pd.read_csv(filename,index_col=0, parse_dates = True)
        data = data.T
        for n in data.index:
            h,m,s = n.split(':')
            sec = (float(h)*60+float(m))*60+float(s)
            data.rename(index={n:sec}, inplace=True)
        data.index = [float(n) for n in data.index]
    else:
        data.columns = [float(n) for n in data.columns]

    #If we have negative values then this makes them equal to zero
    if negatives_to_zero:
        for t in (data.index):
            for l in data.columns:
                if data.loc[t,l] < 0:
                    data.loc[t,l] = 0.0

    return data

# for redirecting stdout to files
@contextmanager
def stdout_redirector(stream):
    old_stdout = sys.stdout
    sys.stdout = stream
    try:
        yield
    finally:
        sys.stdout = old_stdout

#=============================================================================
#---------------------------- DATA CONVERSION TOOLS --------------------------
#=============================================================================

def dict_to_df(data_dict):

    """Takes a dictionary of typical pyomo data and converts it to a dataframe
    
    """    
    dfs_stacked = pd.Series(index=data_dict.keys(), data=list(data_dict.values()))
    dfs = dfs_stacked.unstack()
    return dfs

def is_float_re(str):
    """Checks if a value is a float or not"""
    _float_regexp = re.compile(r"^[-+]?(?:\b[0-9]+(?:\.[0-9]*)?|\.[0-9]+\b)(?:[eE][-+]?[0-9]+\b)?$").match
    return True if _float_regexp(str) else False

def df_from_pyomo_data(varobject):

    val = []
    ix = []
    for index in varobject:
        ix.append(index)
        val.append(varobject[index].value)
    
    a = pd.Series(index=ix, data=val)
    dfs = pd.DataFrame(a)
    index = pd.MultiIndex.from_tuples(dfs.index)
   
    dfs = dfs.reindex(index)
    dfs = dfs.unstack()
    dfs.columns = [v[1] for v in dfs.columns]

    return dfs

#=============================================================================
#--------------------------- DIAGNOSTIC TOOLS ------------------------
#=============================================================================
def rank(A, eps=1e-10):
    """ obtains the rank of a matrix based on SVD
    
        Args:
            eps (optional, float): the value of the singular values that corresponds to 0 
                            when smaller than eps. Default = 1e-10

        Returns:
            rank (int): The rank of the matrix

    """
    print(type(A))  
    if isinstance(A, np.matrix):
        u, s, vh = np.linalg.svd(A)
        return len([x for x in s if abs(x) > eps]) 
    elif isinstance(A, pd.core.frame.DataFrame): 
        A = np.array(A)
        U, s, V = np.linalg.svd(A, full_matrices=True)
        return len([x for x in s if abs(x) > eps]) 
    else:
        raise RuntimeError("Must provide A as either numpy matrix or pandas dataframe")

def nullspace(A, atol=1e-13, rtol=0):
    """ obtains the nullspace of a matrix based on SVD. Taken from the SciPy cookbook
    
        Args:
            atol : float
        The absolute tolerance for a zero singular value.  Singular values
        smaller than `atol` are considered to be zero.
            rtol : float
        The relative tolerance.  Singular values less than rtol*smax are
        considered to be zero, where smax is the largest singular value.

        Returns:
           ns : ndarray
        If `A` is an array with shape (m, k), then `ns` will be an array
        with shape (k, n), where n is the estimated dimension of the
        nullspace of `A`.  The columns of `ns` are a basis for the
        nullspace; each element in numpy.dot(A, ns) will be approximately
        zero.

    """
    A = np.atleast_2d(A)
    u, s, vh = scipy.linalg.svd(A)
    tol = max(atol, rtol * s[0])
    nnz = (s >= tol).sum()
    ns = vh[nnz:].conj().T
    return ns
       
def basic_pca(dataFrame,n=None,with_plots=False):
    """ Runs basic component analysis based on SVD
    
        Args:
            dataFrame (DataFrame): spectral data
            
            n (int): number of largest singular-values
            to plot
            
            with_plots (boolean): argument for files with plots due to testing

        Returns:
            None

    """
            
    times = np.array(dataFrame.index)
    lambdas = np.array(dataFrame.columns)
    D = np.array(dataFrame)
    #print("D shape: ", D.shape)
    U, s, V = np.linalg.svd(D, full_matrices=True)
    #print("U shape: ", U.shape)
    #print("s shape: ", s.shape)
    #print("V shape: ", V.shape)
    #print("sigma/singular values", s)
    if n == None:
        print("WARNING: since no number of components is specified, all components are printed")
        print("It is advised to select the number of components for n")
        n_shape = s.shape
        n = n_shape[0]
        
    u_shape = U.shape
    #print("u_shape[0]",u_shape[0])
    n_l_vector = n if u_shape[0]>=n else u_shape[0]
    n_singular = n if len(s)>=n else len(s)
    idxs = range(n_singular)
    vals = [s[i] for i in idxs]
    v_shape = V.shape
    n_r_vector = n if v_shape[0]>=n else v_shape[0]
    
    if with_plots:
        for i in range(n_l_vector):
            plt.plot(times,U[:,i])
        plt.xlabel("time")
        plt.ylabel("Components U[:,i]")
        plt.show()
        
        plt.semilogy(idxs,vals,'o')
        plt.xlabel("i")
        plt.ylabel("singular values")
        plt.show()
        
        for i in range(n_r_vector):
            plt.plot(lambdas,V[i,:])
        plt.xlabel("wavelength")
        plt.ylabel("Components V[i,:]")
        plt.show()

def perform_data_analysis(dataFrame, pseudo_equiv_matrix, rank_data):  
    """ Runs the analysis by Chen, et al, 2018, based upon the pseudo-equivalency
    matrix. User provides the data and the pseudo-equivalency matrix and the analysis
    provides suggested number of absorbing components as well as whether there are
    likely to be unwanted spectral contributions.
    
        Args:
            dataFrame (DataFrame): spectral data
            
            pseudo_equiv_matrix (list of lists): list containing the rows of the pseudo-equivalency
                                matrix.
            
            rank_data (int): rank of the data matrix, as determined from SVD (number of coloured species)
                
            with_plots (boolean): argument for files with plots due to testing

        Returns:
            None

    """  
    if not isinstance(dataFrame, pd.DataFrame):
        raise TypeError("data must be inputted as a pandas DataFrame, try using read_spectral_data_from_txt or similar function first")
    
    if not isinstance(pseudo_equiv_matrix, list):
        raise TypeError("The Pseudo-equivalency matrix must be inputted as a list containing lists with each row of the pseudo-equivalency matrix")
    PEM = np.matrix(pseudo_equiv_matrix)
    rkp = rank(PEM)
    print("Rank of pseudo-equivalency matrix is ", rkp)
    
    ns = nullspace(PEM)
    print("Nullspace/kernel of pseudo-equivalency matrix is ", ns)
    if ns.size == 0:
        print("Null space of pseudo-equivalency matrix is null")
        rankns = 0
    else:
        rankns = rank(ns)
    
    print("the rank of the nullspace/kernel of pseudo-equivalency matrix is ", rankns)
    
    num_components = PEM.shape[1]
    if rankns > 0:
        ncr = num_components - rankns
        print("Choose the following number of absorbing species:", ncr)
    else:
        ncr = num_components
    ncms = rank_data
    
    if ncr == ncms:
        print("Solve standard problem assuming no unwanted contributions")
    elif ncr == ncms - 1:
        print("Solve with unwanted contributions")
    else:
        print("There may be uncounted for species in the model, or multiple sources of unknown contributions")
    
#=============================================================================
#---------------------------PROBLEM GENERATION TOOLS------------------------
#============================================================================= 
    
def gaussian_single_peak(wl,alpha,beta,gamma):
    """
    helper function to generate absorption data based on 
    lorentzian parameters
    """
    return alpha*np.exp(-(wl-beta)**2/gamma)

def absorbance(wl,alphas,betas,gammas):
    """
    helper function to generate absorption data based on 
    lorentzian parameters
    """
    return sum(gaussian_single_peak(wl,alphas[i],betas[i],gammas[i]) for i in range(len(alphas)))

def generate_absorbance_data(wl_span,parameters_dict):
    """
    helper function to generate absorption data based on 
    lorentzian parameters
    """
    components = parameters_dict.keys()
    n_components = len(components)
    n_lambdas = len(wl_span)
    array = np.zeros((n_lambdas,n_components))
    for i,l in enumerate(wl_span):
        j = 0
        for k,p in six.iteritems(parameters_dict):
            alphas = p['alphas']
            betas  = p['betas']
            gammas = p['gammas']
            array[i,j] = absorbance(l,alphas,betas,gammas)
            j+=1

    data_frame = pd.DataFrame(data=array,
                              columns = components,
                              index=wl_span)
    return data_frame


def generate_random_absorbance_data(wl_span,component_peaks,component_widths=None,seed=None):

    np.random.seed(seed)
    parameters_dict = dict()
    min_l = min(wl_span)
    max_l = max(wl_span)
    #mean=1000.0
    #sigma=1.5*mean
    for k,n_peaks in component_peaks.items():
        params = dict()
        if component_widths:
            width = component_widths[k]
        else:
            width = 1000.0
        params['alphas'] = np.random.uniform(0.1,1.0,n_peaks)
        params['betas'] = np.random.uniform(min_l,max_l,n_peaks)
        params['gammas'] = np.random.uniform(1.0,width,n_peaks)
        parameters_dict[k] = params

    return generate_absorbance_data(wl_span,parameters_dict)

def add_noise_to_signal(signal, size):
    """
    Adds a random normally distributed noise to a clean signal. Used mostly in Kipet
    To noise absorbances or concentration profiles obtained from simulations. All
    values that are negative after the noise is added are set to zero
    Args:
        signal (data): the Z or S matrix to have noise added to it
        size (scalar): sigma (or size of distribution)
    Returns:
        pandas dataframe
    """
    clean_sig = signal    
    noise = np.random.normal(0,size,clean_sig.shape)
    sig = clean_sig+noise    
    df= pd.DataFrame(data=sig)
    df[df<0]=0
    return df

#=============================================================================
#---------------------------PRE-PROCESSING TOOLS------------------------
#=============================================================================
    
def savitzky_golay(dataFrame, window_size, orderPoly, orderDeriv=0):
    """
    Implementation of the Savitzky-Golay filter for Kipet. Used for smoothing data, with
    the option to also differentiate the data. Can be used to remove high-frequency noise.
    Creates a least-squares fit of data within each time window with a high order polynomial centered
    centered at the middle of the window of points.
    
    Args:
        dataFrame (DataFrame): the data to be smoothed (either concentration or spectral data)
        window_size (int): the length of the window. Must be an odd integer number
        orderPoly (int): order of the polynoial used in the filter. Should be less than window_size-1
        orderDeriv (int) (optional): the order of the derivative to compute (default = 0 means only smoothing)
        
    Returns:
        DataFrame containing the smoothed data
    
    References:
        This code is an amalgamation of those developed in the scipy.org cookbook and that employed in Matlab 
        by WeiFeng Chen.
        Original paper: A. Savitzky, M. J. E. Golay, Smoothing and Differentiation of Data by 
        Simplified Least Squares Procedures. Analytical Chemistry, 1964, 36 (8), pp 1627-1639.
    """
    # data checks
    try:
        window_size = np.abs(np.int(window_size))
        orderPoly = np.abs(np.int(orderPoly))
    except ValueError:
        raise ValueError("window_size and order have to be of type int")
    if window_size % 2 != 1 or window_size < 1:
        raise TypeError("window_size size must be a positive odd number")
    if window_size < orderPoly + 2:
        raise TypeError("window_size is too small for the polynomials order")    
    if orderPoly >= window_size:
        raise ValueError("polyorder must be less than window_length.")

    if not isinstance(dataFrame, pd.DataFrame):
        raise TypeError("data must be inputted as a pandas DataFrame, try using read_spectral_data_from_txt or similar function first")
    print("Applying the Savitzky-Golay filter")
    
    order_range = range(orderPoly+1)
    half_window = (window_size -1) // 2
    # precompute coefficients
    b = np.mat([[k**i for i in order_range] for k in range(-half_window, half_window+1)])
    m = np.linalg.pinv(b).A[orderDeriv]
    #rate = 1
    #m = np.linalg.pinv(b).A[orderDeriv] * rate**orderDeriv * factorial(orderDeriv)
    D = np.array(dataFrame)
    no_noise = np.array(dataFrame)
    # pad the signal at the extremes with values taken from the signal itself
    for t in range(len(dataFrame.index)):
        row = list()
        for l in range(len(dataFrame.columns)):
            row.append(D[t,l])
        firstvals = row[0] - np.abs( row[1:half_window+1][::-1] - row[0] )
        lastvals = row[-1] + np.abs(row[-half_window-1:-1][::-1] - row[-1])
        y = np.concatenate((firstvals, row, lastvals))
        new_row = np.convolve( m, y, mode='valid')
        no_noise[t]=new_row
        
    if orderDeriv == 0:
        for t in range(len(dataFrame.index)):
            for l in range(len(dataFrame.columns)):
                if no_noise[t,l] < 0:
                    no_noise[t,l] = 0
    
    data_frame = pd.DataFrame(data=no_noise,
                              columns = dataFrame.columns,
                              index=dataFrame.index)
    
    return data_frame

def snv(dataFrame, offset=0):
    """
    Implementation of the Standard Normal Variate (SNV) filter for Kipet which is a weighted normalization
    method that is commonly used to remove scatter effects in spectroscopic data, this pre-processing 
    step can be applied before the SG filter or used on its own. SNV can be sensitive to noisy entries 
    in the spectra and can increase nonlinear behaviour between S and C as it is not a linear transformation.
    
    
    Args:
        dataFrame (DataFrame): the data to be processed (either concentration or spectral data)
        offset (float): user-defined offset which can be used to avoid over-normalization for samples
                        with near-zero standard deviation. Guide for choosing this value is for something 
                        near the expected noise level to be specified. Default value is zero.
        
    Returns:
        DataFrame containing pre-processed data
    
    References:

    """
    # data checks
    if not isinstance(dataFrame, pd.DataFrame):
        raise TypeError("data must be inputted as a pandas DataFrame, try using read_spectral_data_from_txt or similar function first")
    print("Applying the SNV pre-processing")    

    D = np.array(dataFrame)
    snv_proc = np.array(dataFrame)
    for t in range(len(dataFrame.index)):
        row = list()
        sum_spectra = 0
        for l in range(len(dataFrame.columns)):
            row.append(D[t,l])
            sum_spectra += D[t,l]
        mean_spectra = sum_spectra/(len(dataFrame.columns))
        std = 0
        for l in range(len(dataFrame.columns)):
            std += (mean_spectra-D[t,l])**2
        new_row = list()
        for l in range(len(dataFrame.columns)):
            if offset ==0:
                w = (D[t,l]-mean_spectra)*(std/(len(dataFrame.columns)-1))**0.5
            else:
                w = (D[t,l]-mean_spectra)*(std/(len(dataFrame.columns)-1))**0.5 + 1/offset
            new_row.append(w)
                
        snv_proc[t]=new_row

    data_frame = pd.DataFrame(data=snv_proc,
                              columns = dataFrame.columns,
                              index=dataFrame.index)
    return data_frame

def msc(dataFrame, reference_spectra=None):
    """
    Implementation of the Multiplicative Scatter Correction (MSC) filter for Kipet which is simple pre-processing
    method that attempts to remove scaling effects and offset effects in spectroscopic data. This pre-processing 
    step can be applied before the SG filter or used on its own. This approach requires a reference spectrum which
    must be determined beforehand. In this implementation, the default reference spectrum is the average spectrum 
    of the dataset provided, however an optional argument exists for user-defined reference spectra to be provided.    
    
    Args:
        dataFrame (DataFrame):          the data to be processed (either concentration or spectral data)
        reference_spectra (DataFrame):  optional user-provided reference spectra argument. Default is to automatically
                                        determine this using the average spectra values.
        
    Returns:
        DataFrame pre-processed data
    
    References:

    """
    # data checks
    if not isinstance(dataFrame, pd.DataFrame):
        raise TypeError("data must be inputted as a pandas DataFrame, try using read_spectral_data_from_txt or similar function first")
    print("Applying the MSC pre-processing")  
    
    #Want to make it possible to include user-defined reference spectra
    #this is not great as we could provide the data with some conditioning 
    #in order to construct references based on different user inputs
    if reference_spectra != None:
        if not isinstance(reference_spectra, pd.DataFrame):
            raise TypeError("data must be inputted as a pandas DataFrame, try using read_spectral_data_from_txt or similar function first")
        
        if len(dataFrame.columns) != len(reference_spectra.columns) and len(dataFrame.rows) != len(reference_spectra.rows):
            raise NotImplementedError("the reference spectra must have the same number of entries as the data")
    
    D = np.array(dataFrame)
    ref = np.array(dataFrame)
    msc_proc = np.array(dataFrame)
    
    # the average spectrum is calculated as reference spectra for MSC when none is given by user
    if reference_spectra == None:
        sum_spectra = 0
        
        for t in range(len(dataFrame.index)):
            sum_spectra = 0
            for l in range(len(dataFrame.columns)):
                sum_spectra += D[t,l]
            mean_spectra = sum_spectra/(len(dataFrame.columns))
            for l in range(len(dataFrame.columns)):
                ref[t,l] = mean_spectra 
    else:
        #should add in some checks and additional ways to formulate these depending on what input the user provides
        #need to find out the type of data usually inputted here in order to do this
        ref = reference_spectra
    for t in range(len(dataFrame.index)):
        row = list()
        fit = np.polyfit(ref[t,:],D[t,:],1, full=True)
        row[:] = (D[t,:] - fit[0][1]) / fit[0][0]
        msc_proc[t,:]=row  

    data_frame = pd.DataFrame(data=msc_proc,
                              columns = dataFrame.columns,
                              index=dataFrame.index)
    return data_frame

def baseline_shift(dataFrame, shift=None):
    """
    Implementation of basic baseline shift. 2 modes are avaliable: 1. Automatic mode that requires no
    user arguments. The method identifies the lowest value (NOTE THAT THIS ONLY WORKS IF LOWEST VALUE
    IS NEGATIVE) and shifts the spectra up until this value is at zero. 2. Baseline shift provided by
    user. User provides the number that is added to every wavelength value in the full spectral dataset.
    
    
    Args:
        dataFrame (DataFrame): the data to be processed (spectral data)
        shift (float, optional): user-defined baseline shift
        
    Returns:
        DataFrame containing pre-processed data
    
    References:

    """
    # data checks
    if not isinstance(dataFrame, pd.DataFrame):
        raise TypeError("data must be inputted as a pandas DataFrame, try using read_spectral_data_from_txt or similar function first")
    print("Applying the baseline shift pre-processing") 
    if shift == None:
        shift = float(dataFrame.min().min())*(-1)
    
    print("shifting dataset by: ", shift)    
    D = np.array(dataFrame)
    for t in range(len(dataFrame.index)):
        for l in range(len(dataFrame.columns)):
            D[t,l] = D[t,l]+shift
    
    data_frame = pd.DataFrame(data=D, columns = dataFrame.columns, index = dataFrame.index)
    return data_frame

def decrease_wavelengths(original_dataset, A_set = 2, specific_subset = None):
    '''
    Takes in the original, full dataset and removes specific wavelengths, or only keeps every
    multipl of A_set. Returns a new, smaller dataset that should be easier to solve
    
    Args:
        original_dataset (DataFrame):   the data to be processed
        A_set (float, optional):  optional user-provided multiple of wavelengths to keep. i.e. if
                                    3, every third value is kept. Default is 2.
        specific_subset (list or dict, optional): If the user already knows which wavelengths they would like to
                                    remove, then a list containing these can be included.
        
    Returns:
        DataFrame with the smaller dataset
    
    '''
    if specific_subset != None:
        if not isinstance(specific_subset, (list, dict)):
            raise RuntimeError("subset must be of type list or dict!")
             
        if isinstance(specific_subset, dict):
            lists1 = sorted(specific_subset.items())
            x1, y1 = zip(*lists1)
            specific_subset = list(x1)
            
        new_D = pd.DataFrame(np.nan,index=original_dataset.index, columns = specific_subset)
        for t in original_dataset.index:
            for l in original_dataset.columns.values:
                if l in subset:
                    new_D.at[t,l] = self.model.D[t,l]           
    else:
        count=0
        for l in original_dataset.columns.values:
            remcount = count%A_set
            if remcount==0:
                original_dataset.drop(columns=[l],axis = 1)
            count+=1
        new_D = original_dataset[original_dataset.columns[::A_set]]     
    return new_D