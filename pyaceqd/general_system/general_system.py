import numpy as np
import os
import subprocess
from pyaceqd.tools import export_csv

hbar = 0.6582173  # meV*ps

def system_ace(t_start, t_end, *pulses, dt=0.1, phonons=False, generate_pt=False, t_mem=10, ae=3.0, temperature=1, verbose=False, temp_dir='/mnt/temp_data/', pt_file=None, suffix="", \
                  apply_op=None, apply_op_t=0, apply="", nintermediate=10, pulse_file_x=None, pulse_file_y=None, system_prefix="",threshold="7",
                  system_op=None, boson_op=None, initial=None, lindblad_ops=None, interaction_ops=None, output_ops=[], delete_param=True, prepare_only=False):
    
    duration = np.abs(t_end)+np.abs(t_start)  # time interval of simulation
    tmp_file = temp_dir + "{}_{}.param".format(system_prefix, suffix)  # parameter file
    out_file = temp_dir + "{}_{}.out".format(system_prefix, suffix)  # file ACE writes to
    # sanity checks
    if system_op is None:
        print("System operator not supplied, assuming TLS")
    if phonons and boson_op is None:
        print("using phonons, but boson operator not specified")
        exit(1)
    if initial is None:
        print("No initial state specified, assuming TLS ground state")
        initial = "|0><0|_2"
    # allow not using an interaction hamiltonian
    if verbose and interaction_ops is None:
        print("No interaction hamiltonian ")

    if pt_file is None:
        pt_file = "{}_{}ps_{}nm_{}k_th{}_tmem{}_dt{}.pt".format(system_prefix,duration,ae,temperature,threshold,t_mem,dt)
    if phonons:
        # try to detect pt_file, else calculate it
        if not os.path.exists(pt_file):
            print("{} not found. Calculating...".format(pt_file))
            generate_pt = True  # if pt_file is not present, set to verbose and calculate it
            verbose = True
    # check, if during propagation an operator is applied from left/right to the density matrix
    multitime = False
    if apply_op is not None:
        multitime = True
    # pulse file generation
    t = np.arange(1.1*t_start,1.1*t_end,step=dt/(10*nintermediate))
    # if a specific pulse file is supplied, do not delete it after the calculation.
    # this allows re-using the pulse file, for example for multi-time correlation functions
    # where the pulse is not changed for many calculations
    _remove_pulse_file = False
    if pulse_file_x is None:
        _remove_pulse_file = True
        pulse_file_x = temp_dir + "{}_pulse_x_{}.dat".format(system_prefix, suffix)
        pulse_file_y = temp_dir + "{}_pulse_y_{}.dat".format(system_prefix, suffix)
        pulse_x = np.zeros_like(t, dtype=complex)
        pulse_y = np.zeros_like(t, dtype=complex)
        for _p in pulses:
            pulse_x = pulse_x + _p.polar_x * _p.get_total(t)
            pulse_y = pulse_y + _p.polar_y * _p.get_total(t)
        # this exports to a format that is readable by ACE.
        # not the precision
        export_csv(pulse_file_x, t, pulse_x.real, pulse_x.imag, precision=8, delimit=' ')
        export_csv(pulse_file_y, t, pulse_y.real, pulse_y.imag, precision=8, delimit=' ')
    try:
        with open(tmp_file,'w') as f:
            f.write("ta    {}\n".format(t_start))
            f.write("te    {}\n".format(t_end))
            f.write("dt    {}\n".format(dt))
            f.write("Nintermediate    {}\n".format(nintermediate))
            f.write("use_symmetric_Trotter true\n")
            if generate_pt:
                f.write("t_mem    {}\n".format(t_mem))
                f.write("threshold 1e-{}\n".format(threshold))
                f.write("use_Gaussian true\n")
                f.write("Boson_SysOp    {{ {} }}\n".format(boson_op))
                f.write("Boson_J_type         QDPhonon\n")
                f.write("Boson_J_a_e    {}\n".format(ae))
                f.write("Boson_temperature    {}\n".format(temperature))
                f.write("Boson_subtract_polaron_shift       true\n")
            if phonons and not generate_pt:
                # process tensor path has to be given or in current dir!
                f.write("read_PT    {}\n".format(pt_file))
                f.write("Boson_subtract_polaron_shift       true\n")
            f.write("initial    {{ {} }}\n".format(initial))
            if system_op is not None:
                for _op in system_op:
                    f.write("add_Hamiltonian {{ {} }}\n".format(_op))
            if lindblad_ops is not None:
                for _op in lindblad_ops:
                    # assume lindblad_ops contains tuples of (operator, rate), ex:("|0><1|_2",1/100)
                    f.write("add_Lindblad {:.5f}  {{ {} }}\n".format(_op[1],_op[0]))  
            # pulse
            if interaction_ops is not None:
                for _op in interaction_ops:
                    # distinguish different polarizations
                    # standard is x
                    p_file = pulse_file_x
                    # op has to be tuple of ("operator","polarization")
                    if _op[1]=="y":
                        p_file = pulse_file_y
                        if pulse_file_y is None:
                            print("Pulse file y not given")
                            exit(1)
                    f.write("add_Pulse file {}  {{ -0.5*pi*hbar*({}) }}\n".format(p_file,_op[0]))
            if multitime:
                # apply_Operator 20 {|0><1|_2} would apply the operator |0><1|_2 at t=20 from the left and the h.c. on the right on the density matrix
                # note the Operator is applied at time t, i.e., in this example at t=20, so its effect is only visible at t=20+dt
                if apply == "left":
                    f.write("apply_Operator_left {} {{ {} }}\n".format(apply_op_t, apply_op))
                elif apply == "right":
                    f.write("apply_Operator_right {} {{ {} }}\n".format(apply_op_t, apply_op))
                else:
                    f.write("apply_Operator {} {{ {} }}\n".format(apply_op_t, apply_op))
            # output 
            for _op in output_ops:
                f.write("add_Output {{ {} }}\n".format(_op))
            if generate_pt:
                f.write("write_PT {}\n".format(pt_file))
            f.write("outfile {}\n".format(out_file))
        # param file is now written, start ACE
        if prepare_only:
            delete_param = False
            _remove_pulse_file = False
            print("prepared file {}, exiting.".format(tmp_file))
            return 0
        if not verbose:
            subprocess.check_output(["ACE",tmp_file])
        else:
            subprocess.check_call(["ACE",tmp_file])

        data = np.genfromtxt(out_file)

    finally:
        try:
            os.remove(out_file)
        except FileNotFoundError:
            pass
        if delete_param:
            os.remove(tmp_file)
        if _remove_pulse_file:
            os.remove(pulse_file_x)
            if pulse_file_y is not None:
                os.remove(pulse_file_y)
    return data