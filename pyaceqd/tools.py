import numpy as np
import itertools
import configparser 


def _merge_intervals(intervals):
    """
    assumes intervals sorted by their respective start-value.
    merges them, also merging if the bounds of one interval lie on the next, i.e.,
    [[0,1],[1,2]] -> [[0,2]]
    """
    if len(intervals) > 1:
        for i in range(len(intervals)-1):
            if intervals[i][1] >= intervals[i+1][0]:
                # catch the case, that the whole interval i+1 is contained in i
                _max_t = np.max([intervals[i][1],intervals[i+1][1]])
                intervals[i][1] = _max_t
                del intervals[i+1]
                # not super efficient, because it always loops over every interval
                # instead of using something like i_start = i-1
                _merge_intervals(intervals)
                break
    return intervals

def get_gaussian_t(t0,tend,*pulses,dt_max=1.0,dt_min=0.01,interval_per_step=0.05):
    t_array = [t0]
    t_test = np.arange(t0,tend,dt_min)  # from this it implicitly follows that also the output has this dt_min
    counter = 0
    n_max = int(dt_max/dt_min)
    interval_contains = 0
    intfunc = lambda t: np.sum([p.get_integral(t) for p in pulses])
    for i in range(1,len(t_test)):
        t_now = t_test[i]
        t_prev = t_test[i-1]
        interval_contains += intfunc(t_now) - intfunc(t_prev)
        counter += 1
        if interval_contains >= interval_per_step or counter == n_max:
            t_array.append(t_now)
            counter = 0
            interval_contains = 0
    return np.array(t_array)

def construct_t(t0, tend, dt_small=0.1, dt_big=1.0, *pulses, factor_tau=4, simple_exp=False, gaussian_t=False):
    """
    constructs t-axis that has dt_small discretization during the pulses and dt_big otherwise.
    standard factor is 4, i.e., -4*tau_pulse,..,4*tau_pulse
    """
    # put t0 and tau in arrays to sort them
    t0s = []
    taus = []
    for _p in pulses:
        if _p.t0 < tend and _p.t0 > t0:
            t0s.append(_p.t0)
            taus.append(_p.tau)
        else:
            if _p.t0 > tend:
                print("WARNING: tend is smaller than the end of a pulse")
            if _p.t0 < t0:
                print("WARNING: t0 is greater than the start of a pulse")
    # sort taus with respect to t0s
    t0s = np.array(t0s)
    taus = np.array(taus)
    start_v = t0s - factor_tau*taus  # values, where the intervals with small timestep start
    end_v = t0s + factor_tau*taus  # values, where the intervals with small timestep end
    _temp = list(sorted(zip(start_v,end_v)))  # sort them with respect to the start value
    start_v,end_v = zip(*_temp)
    start_v = list(start_v)
    end_v = list(end_v)
    intervals = []  # set up the intervals
    for _ts,_te in zip(start_v,end_v):
        intervals.append([_ts,_te])
    intervals = _merge_intervals(intervals)  # merges intervals if they overlap
    if intervals[0][0] < t0:
        print("WARNING: t0 is greater than the start of the first pulse")
    if intervals[-1][1] > tend:
        print("WARNING: tend is smaller than the end of the last pulse")
    ts = []  # array where the time-axes are stored
    # use, that arange:
    # 1) gives an empty array, if tstart=tend
    # 2) does not include the final value
    ts.append(np.arange(t0,intervals[0][0],dt_big))
    if simple_exp and len(intervals) == 1 and intervals[0][1] != 0:
        if gaussian_t: 
            # interval_per_step: "1/steps per pi pulse area"
            t_gaussian = get_gaussian_t(intervals[0][0],intervals[0][1],*pulses,dt_max=dt_big,dt_min=dt_small,interval_per_step=0.05)
            ts.append(t_gaussian)
        else:
            ts.append(np.arange(intervals[0][0],intervals[0][1],dt_small))
        _exp_part = np.exp(np.arange(np.log(intervals[0][1]),np.log(tend),dt_small))
        # make sure that there are no crazy numbers, round the exponentially spaced part to 2 decimals. somehow this seems to work best
        # even better than rounding to the step-size dt_small
        ts.append(np.round(_exp_part))
        ts.append(np.array([tend]))
        return np.concatenate(ts,axis=0)  # np.round(np.concatenate(ts,axis=0), decimals=2)  
    for i in range(len(intervals)):
        if i > 0:
            ts.append(np.arange(intervals[i-1][1],intervals[i][0],dt_big))
        ts.append(np.arange(intervals[i][0],intervals[i][1],dt_small))
    ts.append(np.arange(intervals[-1][1],tend,dt_big))
    ts.append(np.array([tend]))
    return np.concatenate(ts,axis=0)

def simple_t_gaussian(t0, texp, tend, dt_small=0.1, dt_big=1.0, *pulses, decimals=2, exp_part=True):
    """
    uses gaussian timespacing from t0,...,texp, then exponential timespacing from
    texp,...,tend
    """
    ts = []
    t_gaussian = get_gaussian_t(t0,texp,*pulses,dt_max=dt_big,dt_min=dt_small,interval_per_step=0.05)
    ts.append(t_gaussian)
    if exp_part:
        t_exp = np.exp(np.arange(np.log(texp-t0),np.log(tend-t0),dt_small))+t0
        ts.append(t_exp)
    else:
        ts.append(np.arange(texp,tend,10*dt_small))
    ts.append(np.array([tend]))
    return np.round(np.concatenate(ts,axis=0), decimals=decimals)  

def export_csv(filename, *arg, precision=4, delimit=',', verbose=False):
    """
    Exportiert Arrays als .csv Datei
    :param delimit: delimiter 
    :param filename: filename
    :param precision: number of decimal places after which the number is truncated
    :return: null
    """
    p = '%.{k}f'.format(k=precision)
    ps = []
    for arguments in arg:
        ps.append(p)
    try:
        np.savetxt(
            filename,
            np.c_[arg],
            fmt=ps,
            delimiter=delimit,
            newline='\n',
            # footer='end of file',
            # comments='# ',
            # header='X , MW'
        )
        if verbose:
            print("[i] csv saved to {}".format(filename))
    except TypeError:
        print("TypeError occured")
        for arguments in arg:
            print(arguments)

def concurrence(rho):
        T_matrix = np.flip(np.diag([-1.,1.,1.,-1.]),axis=1)  # antidiagonal matrix
        M_matrix = np.dot(rho,np.dot(T_matrix,np.dot(np.conjugate(rho),T_matrix)))
        _eigvals = np.real(np.linalg.eigvals(M_matrix))
        _eigvals = np.sqrt(np.sort(_eigvals))
        return np.max([0.0,_eigvals[-1]-np.sum(_eigvals[:-1])])

def serialize_dm(rho):
    """
    serializes a density matrix into a vector, splitting real and imag parts
    """
    return np.concatenate((np.real(rho).flatten(),np.imag(rho).flatten()))

def deserialize_dm(rho):
    """
    deserializes a density matrix from a vector
    """
    dim = int(np.sqrt(len(rho)/2))
    return rho[:dim**2].reshape((dim,dim)) + 1j*rho[dim**2:].reshape((dim,dim))


def compose_dm(outputs, dim=2):
    """
    composes a density matrix from the output of ACE, with every output-array being the time dynamics for the corresponding output operator
    """
    # dim is the dimension of the system
    rho = np.zeros((len(outputs[0]),dim,dim),dtype=np.complex128)
    n = 1  # start at 1, as the zeroth output is the time axis
    for j in range(dim):
        for k in range(j,dim):
            rho[:,j,k] = outputs[n]
            rho[:,k,j] = np.conjugate(outputs[n])
            n += 1
    t = np.real(outputs[0])
    return t, rho

def generate_basis_states(dim):
        basis_states = []
        indices_range = [range(d) for d in dim]
        # from itertools.product documentation:
        # Cartesian product of input iterables.
        # The nested loops cycle like an odometer with the rightmost element advancing on every iteration. 
        for indices in itertools.product(*indices_range):
            basis_states.append(indices)
        return basis_states

def basis_states(dim):
    # generates readable basis state representation, for use in plotting etc.
    # if dim is no list, make it one
    if not isinstance(dim, list):
        dim = [dim]
    basis_states = generate_basis_states(dim)
    _basis_states = []
    for basis_state in basis_states:
        basis_state_str = '|'
        for index in basis_state:
            basis_state_str += f'{index},'
        basis_state_str = basis_state_str.rstrip(',')
        basis_state_str += '⟩'
        _basis_states.append(basis_state_str)
    return _basis_states

def matrix_element_operators(basis_states, dim, readable=False):
        operators = []
        for i in range(len(basis_states)):
            bra_state = basis_states[i]
            for j in range(i,len(basis_states)):
                ket_state = basis_states[j]
                operator_str = ''
                for k, (bra_index, ket_index) in enumerate(zip(bra_state, ket_state)):
                    if readable:
                        operator_str += f'|{bra_index}⟩⟨{ket_index}|_{dim[k]} ⊗ '
                    else:
                        operator_str += f'|{bra_index}><{ket_index}|_{dim[k]} otimes '
                if readable:
                    operator_str = operator_str.rstrip(' ⊗ ')
                else:
                    operator_str = operator_str.rstrip('otimes ')
                operators.append(operator_str)
        return operators

def output_ops_dm(dim=[2,2], readable=False):
    """
    returns the output operators for a system with n1*n2*n3... levels
    to turn this into a density matrix, use:
    compose_dm(outputs, dim=np.prod(dim))
    can also be used instead of output_ops_dm
    """
    if not isinstance(dim, list) and not isinstance(dim, tuple):
        dim = [dim]
    basis_states = generate_basis_states(dim)
    return matrix_element_operators(basis_states, dim, readable=readable)

def read_calibration_file(calibration_file):

    # reads in experimentally aquired quantum dot parameters 
    config = configparser.ConfigParser()
    config.read(calibration_file)

    # read the calibration file
    central_wavelength = float(config['EMISSION']['exciton_wavelength']) #nm
    biexciton_wavelength = float(config['EMISSION']['biexciton_wavelength'])
    dark_wavelength = float(config['EMISSION']['dark_wavelength']) 

    fss_bright = float(config['SPLITTING']['fss_bright'])*1e-3 #meV
    fss_dark = float(config['SPLITTING']['fss_dark']) *1e-3 # meV 

    lifetime_exciton = float(config['LIFETIMES']['exciton']) #ps
    lifetime_biexciton = float(config['LIFETIMES']['biexciton'])
    #lifetime_dark = float(config['LIFETIMES']['dark']) 

    g_ex = float(config['G_FACTORS']['g_ex'])
    g_hx = float(config['G_FACTORS']['g_hx'])
    g_ez = float(config['G_FACTORS']['g_ez'])
    g_hz = float(config['G_FACTORS']['g_hz'])

    exciton_meV = 1239.8*1e3/central_wavelength #meV
    biexciton_meV = 1239.8*1e3/biexciton_wavelength
    dark_meV = 1239.8*1e3/dark_wavelength

    exciton_x_energy = fss_bright/2 #sign correct? 
    exciton_y_energy = -fss_bright/2
    binding_energy = -(exciton_meV - biexciton_meV) # negatively defined
    dark_energy = (dark_meV-exciton_meV)
    dark_x_energy = dark_energy + fss_dark/2
    dark_y_energy = dark_energy - fss_dark/2 

    gamma_e = 1/lifetime_exciton
    gamma_b = 1/(lifetime_biexciton*2)
    #gamma_d = 1/lifetime_dark

    return exciton_x_energy, exciton_y_energy, dark_x_energy, dark_y_energy, binding_energy, gamma_e, gamma_b, g_ex, g_hx, g_ez, g_hz

