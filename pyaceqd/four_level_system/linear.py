import subprocess
import numpy as np
import os
from pyaceqd.tools import export_csv, read_calibration_file
import tqdm
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import wait
from pyaceqd.general_system.general_system import system_ace, system_ace_stream
from pyaceqd.general_system.general_dressed_states import dressed_states

hbar = 0.6582173  # meV*ps

def biexciton(t_start, t_end, *pulses, dt=0.5, delta_xy=0, delta_b=4, gamma_e=1/100, gamma_b=None, phonons=False, ae=3.0, temperature=4, verbose=False, lindblad=False, temp_dir='/mnt/temp_data/', pt_file=None, suffix="", \
               multitime_op=None, pulse_file_x=None, pulse_file_y=None, prepare_only=False, output_ops=["|0><0|_4","|1><1|_4","|2><2|_4","|3><3|_4"], initial="|0><0|_4", t_mem=20.48, dressedstates=False, rf=False, rf_file=None,
                calibration_file=None, firstonly=False):
    system_prefix = "b_linear"

    if calibration_file is not None:
        E_X, E_Y, _, _, E_B, gamma_e, gamma_b, _, _, _, _ = read_calibration_file(calibration_file)
        delta_xy = np.abs(E_X - E_Y)
        delta_b = -E_B
        system_op = ["-{}*|3><3|_4".format(delta_b),"{}*|1><1|_4".format(E_X),"{}*|2><2|_4".format(E_Y)]
        

    # |0> = G, |1> = X, |2> = Y, |3> = B
    else:
        system_op = ["-{}*|3><3|_4".format(delta_b),"-{}*|1><1|_4".format(delta_xy/2),"{}*|2><2|_4".format(delta_xy/2)]
    boson_op = "1*(|1><1|_4 + |2><2|_4) + 2*|3><3|_4"
    lindblad_ops = []
    if lindblad:
        if gamma_b is None:
            gamma_b = gamma_e
        lindblad_ops = [["|0><1|_4",gamma_e],["|0><2|_4",gamma_e],
                        ["|1><3|_4",gamma_b],["|2><3|_4",gamma_b]]
    interaction_ops = [["|1><0|_4+|3><1|_4","x"],["|2><0|_4+|3><2|_4","y"]]
    
    rf_op = None
    if rf:
        # 2*|3><3|_4 because B contains 2 excitons
        rf_op = "|1><1|_4 + |2><2|_4 + 2*|3><3|_4" 

    result = system_ace_stream(t_start, t_end, *pulses, dt=dt, phonons=phonons, t_mem=t_mem, ae=ae, temperature=temperature, verbose=verbose, temp_dir=temp_dir, pt_file=pt_file, suffix=suffix, \
                  multitime_op=multitime_op, system_prefix=system_prefix, threshold="10", threshold_ratio="0.3", buffer_blocksize="-1", dict_zero="16", precision="12", boson_e_max=7,
                  system_op=system_op, pulse_file_x=pulse_file_x, pulse_file_y=pulse_file_y, boson_op=boson_op, initial=initial, lindblad_ops=lindblad_ops, interaction_ops=interaction_ops, output_ops=output_ops, prepare_only=prepare_only,
                  dressedstates=dressedstates, rf_op=rf_op, rf_file=rf_file, firstonly=firstonly)
    return result

def biexciton_dressed_states(t_start, t_end, *pulses, plot=True, filename="biexciton_dressed", firstonly=False, **options):
    colors = ["#0000FF", "#00CC33", "#F9A627", "#FF0000"]
    dim = 4
    return dressed_states(biexciton, dim, t_start, t_end, *pulses, filename=filename, plot=plot, firstonly=firstonly, colors=colors, **options)

def biexciton_photons(t_start, t_end, *pulses, dt=0.5, delta_xy=0, delta_b=4, gamma_e=1/100, cav_coupl=0.06, cav_loss=0.12/hbar, delta_cx=-2, gamma_b=None, phonons=False, ae=3.0, temperature=4, verbose=False, lindblad=False, temp_dir='/mnt/temp_data/', pt_file=None, suffix="", \
               multitime_op=None, pulse_file_x=None, pulse_file_y=None, prepare_only=False, output_ops=["|0><0|_4 otimes Id_2 otimes Id_2","|1><1|_4 otimes Id_2 otimes Id_2","|2><2|_4 otimes Id_2 otimes Id_2","|3><3|_4 otimes Id_2 otimes Id_2"], initial="|0><0|_4 otimes |0><0|_2 otimes |0><0|_2", n_photon=1,
                t_mem=20.48, dressedstates=False, rf=False, rf_file=None, firstonly=False):
    n = n_photon + 1
    for i in range(len(output_ops)):
        output_ops[i] = output_ops[i].replace("_2","_{}".format(n))
    initial = initial.replace("_2","_{}".format(n))
    system_prefix = "b_linear_cavity"
    # |0> = G, |1> = X, |2> = Y, |3> = B
    system_op = ["-{}*|3><3|_4 otimes Id_2 otimes Id_2".format(delta_b).replace("_2","_{}".format(n)),
                 "-{}*|1><1|_4 otimes Id_2 otimes Id_2".format(delta_xy/2).replace("_2","_{}".format(n)),
                 "{}*|2><2|_4 otimes Id_2 otimes Id_2".format(delta_xy/2).replace("_2","_{}".format(n))]
    boson_op = "|1><1|_4 otimes Id_2 otimes Id_2 + |2><2|_4 otimes Id_2 otimes Id_2 + 2*|3><3|_4 otimes Id_2 otimes Id_2".replace("_2","_{}".format(n))
    lindblad_ops = []
    # QD decay outside of the cavity
    if lindblad:
        if gamma_b is None:
            gamma_b = gamma_e
        lindblad_ops = [["|0><1|_4 otimes Id_2 otimes Id_2".replace("_2","_{}".format(n)),gamma_e],
                        ["|0><2|_4 otimes Id_2 otimes Id_2".replace("_2","_{}".format(n)),gamma_e],
                        ["|1><3|_4 otimes Id_2 otimes Id_2".replace("_2","_{}".format(n)),gamma_b],
                        ["|2><3|_4 otimes Id_2 otimes Id_2".replace("_2","_{}".format(n)),gamma_b]]
    # interaction with laser
    interaction_ops = [["|1><0|_4 otimes Id_2 otimes Id_2 +|3><1|_4 otimes Id_2 otimes Id_2 ".replace("_2","_{}".format(n)),"x"],
                       ["|2><0|_4 otimes Id_2 otimes Id_2 +|3><2|_4 otimes Id_2 otimes Id_2 ".replace("_2","_{}".format(n)),"y"]]
    # cavity decay
    lindblad_ops.append(["Id_4 otimes b_2 otimes Id_2".replace("_2","_{}".format(n)),cav_loss])
    lindblad_ops.append(["Id_4 otimes Id_2 otimes b_2".replace("_2","_{}".format(n)),cav_loss])
    # cavity detuning
    system_op.append(" {} * (Id_4 otimes n_2 otimes Id_2)".format(delta_cx).replace("_2","_{}".format(n)))
    system_op.append(" {} * (Id_4 otimes Id_2 otimes n_2)".format(delta_cx).replace("_2","_{}".format(n)))
    # cavity-qd coupling
    # X-cavity
    system_op.append("{} * (|1><0|_4 otimes b_2 otimes Id_2 + |0><1|_4 otimes bdagger_2 otimes Id_2)".format(cav_coupl).replace("_2","_{}".format(n)))
    system_op.append("{} * (|3><1|_4 otimes b_2 otimes Id_2 + |1><3|_4 otimes bdagger_2 otimes Id_2)".format(cav_coupl).replace("_2","_{}".format(n)))
    # Y-cavity
    system_op.append("{} * (|2><0|_4 otimes Id_2 otimes b_2 + |0><2|_4 otimes Id_2 otimes bdagger_2)".format(cav_coupl).replace("_2","_{}".format(n)))
    system_op.append("{} * (|3><2|_4 otimes Id_2 otimes b_2 + |2><3|_4 otimes Id_2 otimes bdagger_2)".format(cav_coupl).replace("_2","_{}".format(n)))
    
    rf_op = None
    if rf:
        rf_op = "|1><1|_4 otimes Id_{} otimes Id_{}".format(n,n)
        rf_op = rf_op + " + Id_4 otimes n_{} otimes Id_{}".format(n,n)
        rf_op = rf_op + " + Id_4 otimes Id_{} otimes n_{}".format(n,n)
        if pulse_file_x is not None and rf_file is None:
            print("Error: pulse file is given, but no file for rotating frame")
            return 0

    result = system_ace_stream(t_start, t_end, *pulses, dt=dt, phonons=phonons, t_mem=t_mem, ae=ae, temperature=temperature, verbose=verbose, temp_dir=temp_dir, pt_file=pt_file, suffix=suffix, \
                  multitime_op=multitime_op, system_prefix=system_prefix, threshold="10", threshold_ratio="0.3", buffer_blocksize="-1", dict_zero="16", precision="12", boson_e_max=7,
                  system_op=system_op, pulse_file_x=pulse_file_x, pulse_file_y=pulse_file_y, boson_op=boson_op, initial=initial, lindblad_ops=lindblad_ops, interaction_ops=interaction_ops, output_ops=output_ops, prepare_only=prepare_only,
                  dressedstates=dressedstates, rf_op=rf_op, rf_file=rf_file, firstonly=firstonly)
    return result

def biexciton_photons_dressed_states(t_start, t_end, *pulses, plot=True, filename="tls_photons_dressed", firstonly=False, **options):
    # dim = 2 for TLS
    n = options["n_photon"] + 1
    dim = [2,n,n]
    return dressed_states(biexciton_photons, dim, t_start, t_end, *pulses, filename=filename, plot=plot, firstonly=firstonly, colors=None, **options)

def biexciton_photons_extended(t_start, t_end, *pulses, dt=0.5, delta_xy=0, delta_b=4, gamma_e=1/100, cav_coupl=0.06, cav_loss=0.12/hbar, delta_cx=-2, gamma_b=None, phonons=False, ae=3.0, temperature=4, verbose=False, lindblad=False, temp_dir='/mnt/temp_data/', pt_file=None, suffix="", \
               multitime_op=None, pulse_file_x=None, pulse_file_y=None, prepare_only=False, output_ops=["|0><0|_18 + |1><1|_18 + |2><2|_18 + |3><3|_18 + |4><4|_18 + |5><5|_18","|6><6|_18 + |7><7|_18 + |8><8|_18 + |9><9|_18","|10><10|_18 + |11><11|_18 + |12><12|_18 + |13><13|_18","|14><14|_18 + |15><15|_18 + |16><16|_18 + |17><17|_18"], initial="|0><0|_18",
               t_mem=20.48, dressedstates=False, rf=False, rf_file=None, firstonly=False):
    system_prefix = "b_linear_cavity_extended"
    # state mapping: |G,0,0> 0, |G,1,0> 1, |G,0,1> 2, |G,1,1> 3, |G,2,0> 4, |G,0,2> 5, |X,0,0> 6, |X,1,0> 7, |X,0,1> 8, |X,1,1> 9, |Y,0,0> 10, |Y,1,0> 11, |Y,0,1> 12, |Y,1,1> 13, |B,0,0> 14, |B,1,0> 15, |B,0,1> 16, |B,1,1> 17
    # this system accounts for two excitations in total, i.e., G,2,0 and G,0,2 are also considered.
    # it shows that this is enough in most cases.
    d_C = delta_cx
    d_0 = delta_xy
    d_B = delta_b
    system_op = ["{}*|1><1|_18".format(d_C), "{}*|2><2|_18".format(d_C), "{}*|3><3|_18".format(2*d_C), "{}*|4><4|_18".format(2*d_C), "{}*|5><5|_18".format(2*d_C), "{}*|6><6|_18".format(-d_0/2), "{}*|7><7|_18".format(-d_0/2 + d_C),
                 "{}*|8><8|_18".format(-d_0/2 + d_C), "{}*|9><9|_18".format(-d_0/2 + 2*d_C), "{}*|10><10|_18".format(d_0/2), "{}*|11><11|_18".format(d_0/2 + d_C), "{}*|12><12|_18".format(d_0/2 + d_C), "{}*|13><13|_18".format(d_0/2 + 2*d_C),
                 "{}*|14><14|_18".format(-d_B), "{}*|15><15|_18".format(-d_B + d_C), "{}*|16><16|_18".format(-d_B + d_C), "{}*|17><17|_18".format(-d_B + 2*d_C)]
    boson_op = "|6><6|_18 + |7><7|_18 + |8><8|_18 + |9><9|_18 + |10><10|_18 + |11><11|_18 + |12><12|_18 + |13><13|_18 + 2 * ( |14><14|_18 + |15><15|_18 + |16><16|_18 + |17><17|_18)"
    lindblad_ops = []
    # QD decay outside of the cavity
    if lindblad:
        if gamma_b is None:
            gamma_b = gamma_e
        lindblad_ops = [["|0><6|_18 + |1><7|_18 + |2><8|_18 + |3><9|_18",gamma_e],["|0><10|_18 + |1><11|_18 + |2><12|_18 + |3><13|_18",gamma_e],
                        ["|6><14|_18 + |7><15|_18 + |8><16|_18 + |9><17|_18",gamma_b],["|10><14|_18 + |11><15|_18 + |12><16|_18 + |13><17|_18",gamma_b]]
    # interaction with laser
    interaction_ops = [["|6><0|_18 + |7><1|_18 + |8><2|_18 + |9><3|_18 + |14><6|_18 + |15><7|_18 + |16><8|_18 + |17><9|_18","x"],
                       [" |10><0|_18 + |11><1|_18 + |12><2|_18 + |13><3|_18 + |14><10|_18 + |15><11|_18 + |16><12|_18 + |17><13|_18","y"]]
    # cavity decay
    # cavity loss x
    lindblad_ops.append(["|0><1|_18 + sqrt(2)*|1><4|_18 + |2><3|_18 + |6><7|_18 + |8><9|_18 + |10><11|_18 + |12><13|_18 + |14><15|_18 + |16><17|_18",cav_loss])
    # cavity loss y
    lindblad_ops.append(["|0><2|_18 + |1><3|_18 + sqrt(2)*|2><5|_18 + |6><8|_18 + |7><9|_18 + |10><12|_18 + |11><13|_18 + |14><16|_18 + |15><17|_18",cav_loss])
    # cavity-qd coupling
    # X-cavity
    system_op.append("{} * ( |1><6|_18 + |3><8|_18 + sqrt(2)*|4><7|_18 + |6><1|_18 + sqrt(2)*|7><4|_18 + |7><14|_18 + |8><3|_18 + |9><16|_18 + |14><7|_18 + |16><9|_18)".format(cav_coupl))
    # Y-cavity
    system_op.append("{} * ( |2><10|_18 + |3><11|_18 + sqrt(2)*|5><12|_18 + |10><2|_18 + |11><3|_18 + sqrt(2)*|12><5|_18 + |12><14|_18 + |13><15|_18 + |14><12|_18 + |15><13|_18)".format(cav_coupl))
    
    rf_op = None
    if rf:
        # the factors correpond to the number of excitations (QD&photons) in the respective state
        rf_op = "|1><1|_18 + |2><2|_18 + 2*|3><3|_18 + 2*|4><4|_18 + 2*|5><5|_18 + |6><6|_18 + 2*|7><7|_18 + 2*|8><8|_18 + 3*|9><9|_18 + |10><10|_18 + 2*|11><11|_18 + 2*|12><12|_18 + 3*|13><13|_18 + 2*|14><14|_18 + 3*|15><15|_18 + 3*|16><16|_18 + 4*|17><17|_18"

    result = system_ace_stream(t_start, t_end, *pulses, dt=dt, phonons=phonons, t_mem=t_mem, ae=ae, temperature=temperature, verbose=verbose, temp_dir=temp_dir, pt_file=pt_file, suffix=suffix, \
                  multitime_op=multitime_op, system_prefix=system_prefix, threshold="10", threshold_ratio="0.3", buffer_blocksize="-1", dict_zero="16", precision="12", boson_e_max=7,
                  system_op=system_op, pulse_file_x=pulse_file_x, pulse_file_y=pulse_file_y, boson_op=boson_op, initial=initial, lindblad_ops=lindblad_ops, interaction_ops=interaction_ops, output_ops=output_ops, prepare_only=prepare_only, 
                  dressedstates=dressedstates, rf_op=rf_op, rf_file=rf_file, firstonly=firstonly)
    return result

def biexciton_photons_extended_dressed_states(t_start, t_end, *pulses, plot=True, filename="biexciton_photons_extended_dressed", firstonly=False, **options):
    dim = 18
    return dressed_states(biexciton_photons_extended, dim, t_start, t_end, *pulses, filename=filename, plot=plot, firstonly=firstonly, colors=None, **options)

def biexciton_(t_start, t_end, *pulses, dt=0.5, delta_xy=0, delta_b=4, gamma_e=1/100, gamma_b=None, phonons=False, generate_pt=False, t_mem=10, threshold="7", ae=3.0, temperature=4, verbose=False, lindblad=False, temp_dir='/mnt/temp_data/', pt_file=None, suffix="", \
               multitime_op=None, pulse_file_x=None, pulse_file_y=None, ninterm=10, prepare_only=False, output_ops=["|0><0|_4","|1><1|_4","|2><2|_4","|3><3|_4"]):
    system_prefix = "b_linear"
    system_op = ["-{}*|3><3|_4".format(delta_b),"-{}*|2><2|_4".format(delta_xy)]
    boson_op = "1*(|1><1|_4 + |2><2|_4) + 2*|3><3|_4"
    initial = "|0><0|_4"
    lindblad_ops = []
    if lindblad:
        if gamma_b is None:
            gamma_b = gamma_e
        lindblad_ops = [["|0><1|_4",gamma_e],["|0><2|_4",gamma_e],
                        ["|1><3|_4",gamma_b],["|2><3|_4",gamma_b]]
    interaction_ops = [["|1><0|_4+|3><1|_4","x"],["|2><0|_4+|3><2|_4","y"]]
    
    result = system_ace(t_start, t_end, *pulses, dt=dt, generate_pt=generate_pt, t_mem=t_mem, ae=ae, temperature=temperature, verbose=verbose, temp_dir=temp_dir,phonons=phonons, pt_file=pt_file, suffix=suffix,\
                      multitime_op=multitime_op, nintermediate=ninterm, pulse_file_x=pulse_file_x, pulse_file_y=pulse_file_y, system_prefix=system_prefix, threshold=threshold,\
                      system_op=system_op, boson_op=boson_op, initial=initial, lindblad_ops=lindblad_ops, interaction_ops=interaction_ops, output_ops=output_ops, prepare_only=prepare_only)
    return result

def biexciton_ace(t_start, t_end, *pulses, dt=0.5, delta_xy=0, delta_b=4, gamma_e=1/100, gamma_b=1/100, phonons=False, generate_pt=False, t_mem=10, threshold="7",ae=3.0, temperature=4,verbose=False, lindblad=False, temp_dir='/mnt/temp_data/', pt_file=None, suffix="", \
                  apply_op_l=None, apply_op_t=0):
    tmp_file = temp_dir + "biex{}.param".format(suffix)
    out_file = temp_dir + "biex{}.out".format(suffix)
    duration = np.abs(t_end)+np.abs(t_start)
    if pt_file is None:
        pt_file = "biexciton_linear_{}ps_{}nm_{}k_th{}_tmem{}_dt{}.pt".format(duration,ae,temperature,threshold,t_mem,dt)
    pulse_file_x = temp_dir + "biexciton_linear_pulse_x{}.dat".format(suffix)
    pulse_file_y = temp_dir + "biexciton_linear_pulse_y{}.dat".format(suffix)
    t,g,x,y,b = 0,0,0,0,0
    gamma_b = gamma_b / 2  # both X and Y decay. the input gamma_b is 1/tau_b, where tau_b is the lifetime of the biexciton
    if phonons:
        if not os.path.exists(pt_file):
            print("{} not found. Calculating...".format(pt_file))
            generate_pt = True
            verbose = True
    multitime = False
    if apply_op_l is not None:
        multitime = True
    try:
        t = np.arange(1.1*t_start,1.1*t_end,step=0.1*dt)
        pulse_x = np.zeros_like(t, dtype=complex)
        pulse_y = np.zeros_like(t, dtype=complex)
        for _p in pulses:
            pulse_x = pulse_x + _p.polar_x * _p.get_total(t)
            pulse_y = pulse_y + _p.polar_y * _p.get_total(t)
        export_csv(pulse_file_x, t, pulse_x.real, pulse_x.imag, precision=8, delimit=' ')
        export_csv(pulse_file_y, t, pulse_y.real, pulse_y.imag, precision=8, delimit=' ')
        with open(tmp_file,'w') as f:
            f.write("ta    {}\n".format(t_start))
            f.write("te    {}\n".format(t_end))
            f.write("dt    {}\n".format(dt))
            f.write("Nintermediate    10\n")
            f.write("use_symmetric_Trotter true\n")            
            if generate_pt:
                f.write("t_mem    {}\n".format(t_mem))
                f.write("threshold 1e-{}\n".format(threshold))
                f.write("use_Gaussian true\n")
                f.write("Boson_SysOp    { 1*(|1><1|_4 + |2><2|_4) + 2*|3><3|_4}\n")
                f.write("Boson_J_type         QDPhonon\n")
                f.write("Boson_J_a_e    {}\n".format(ae))
                f.write("Boson_temperature    {}\n".format(temperature))
                f.write("Boson_subtract_polaron_shift       true\n")
            if phonons and not generate_pt:
                # process tensor path has to be given or in current dir!
                f.write("read_PT    {}\n".format(pt_file))
                f.write("Boson_subtract_polaron_shift       true\n")
            f.write("initial    {}\n".format("{|0><0|_4}"))
            if lindblad:
                f.write("add_Lindblad {:.5f}  {{|0><1|_4}}\n".format(gamma_e))  # x->g
                f.write("add_Lindblad {:.5f}  {{|0><2|_4}}\n".format(gamma_e))  # y->g
                f.write("add_Lindblad {:.5f}  {{|1><3|_4}}\n".format(gamma_b))  # b->x
                f.write("add_Lindblad {:.5f}  {{|2><3|_4}}\n".format(gamma_b))  # b->y
            # energies
            f.write("add_Hamiltonian  {{ -{}*|3><3|_4}}\n".format(delta_b))
            f.write("add_Hamiltonian  {{ -{}*|2><2|_4}}\n".format(delta_xy))
            # pulse
            f.write("add_Pulse file {}  {{-{}*(|1><0|_4+|3><1|_4)}}\n".format(pulse_file_x,np.pi*hbar/2))
            f.write("add_Pulse file {}  {{-{}*(|2><0|_4+|3><2|_4)}}\n".format(pulse_file_y,np.pi*hbar/2))
            if multitime:
                # apply_Operator 20 {|1><0|_2} would apply the operator |1><0|_2 at t=20 from the left and the h.c. on the right on the density matrix
                # note the Operator is applied at time t, i.e., in this example at t=20, so its effect is only visible at t=20+dt
                f.write("apply_Operator {} {{ {} }}\n".format(apply_op_t, apply_op_l))
            # output 
            f.write("add_Output {|0><0|_4}\n")
            f.write("add_Output {|1><1|_4}\n")
            f.write("add_Output {|2><2|_4}\n")
            f.write("add_Output {|3><3|_4}\n")
            f.write("add_Output {|0><3|_4}\n")
            # if multitime:
            #     f.write("add_Output {|0><1|_4}\n")
            #     f.write("add_Output {|0><2|_4}\n")
            #     f.write("add_Output {|0><3|_4}\n")
            #     f.write("add_Output {|1><2|_4}\n")
            #     f.write("add_Output {|1><3|_4}\n")
            #     f.write("add_Output {|2><3|_4}\n")

                # f.write("add_Output {|1><0|_4}\n")
                # f.write("add_Output {|2><0|_4}\n")
                # f.write("add_Output {|3><0|_4}\n")
                # f.write("add_Output {|2><1|_4}\n")
                # f.write("add_Output {|3><1|_4}\n")
                # f.write("add_Output {|3><2|_4}\n")
            if generate_pt:
                f.write("write_PT {}\n".format(pt_file))
            f.write("outfile {}\n".format(out_file))
        if not verbose:
            subprocess.check_output(["ACE",tmp_file])
        else:
            subprocess.check_call(["ACE",tmp_file])
        data = np.genfromtxt(out_file)
        t = data[:,0]
        g = data[:,1]
        x = data[:,3]
        y = data[:,5]
        b = data[:,7]
        pgb = data[:,9]+1j*data[:,10]
        # if multitime:
        #     p_gx = data[:,9] + 1j*data[:,10]
        #     p_gy = data[:,11] + 1j*data[:,12]
        #     p_gb = data[:,13] + 1j*data[:,14]
        #     p_xy = data[:,15] + 1j*data[:,16]
        #     p_xb = data[:,17] + 1j*data[:,18]
        #     p_yb = data[:,19] + 1j*data[:,20]

            # p_xg = data[:,21] + 1j*data[:,22]
            # p_yg = data[:,23] + 1j*data[:,24]
            # p_bg = data[:,25] + 1j*data[:,26]
            # p_yx = data[:,27] + 1j*data[:,28]
            # p_bx = data[:,29] + 1j*data[:,30]
            # p_by = data[:,31] + 1j*data[:,32]
    finally:
        try:
            os.remove(out_file)
        except FileNotFoundError:
            pass
        os.remove(tmp_file)
        os.remove(pulse_file_x)
        os.remove(pulse_file_y)
    # if multitime:
    #     return t,g,x,y,b,p_gx,p_gy,p_gb,p_xy,p_xb,p_yb
    return t,g,x,y,b,pgb


def G2(t0=0, tend=600, tau0=0, tauend=600, dt=0.1, *pulses, ae=5.0, delta_b=4, delta_xy=0, gamma_e=1/100, gamma_b=2/100, phonons=False, pt_file="g2_tensor.pt", thread=False, workers=15):
    """
    calculates G2 for the x->g emission
    for every t1 in t, propagate to t1, then
    apply sigma = |g><x| from left and sigma^dagger from the right to the density matrix
    propagate from t1 to t1+tau_max
    use results to calculate G2(t1,tau=0,..,tau_max) by applying sigma^dagger*sigma from the left to the density matrix
    and then taking the trace of the dens. matrix
    """
    t = np.linspace(t0, tend, int((tend-t0)/dt)+1)
    n_tau = int((tauend-tau0)/dt)
    tau = np.linspace(tau0, tauend, n_tau + 1)

    # calculate process tensor for longest time tend+tauend. this can then be re-used for every following phonon calculation
    if phonons:
        biexciton_ace(t0,tend+tauend,*pulses,dt=0.1,ae=ae,verbose=True,phonons=phonons, delta_b=4, pt_file=pt_file)

    # special case tau=0:
    # all 4 operators are applied at the same time.
    # G2(t,0) = Tr(sigma^dagger * sigma * sigma * rho(t) * sigma^dagger) = 0, as is sigma*sigma always zero.
    options = {"dt": dt, "ae": ae, "verbose": False, "phonons": phonons, "delta_b": delta_b, "delta_xy": delta_xy, "gamma_e": gamma_e, "gamma_e": gamma_b, "lindblad": True, "apply_op_l": "|0><1|_4", "pt_file": pt_file}
    _G2 = np.zeros([len(t),len(tau)])
    if thread:
        with tqdm.tqdm(total=len(t)) as tq:
            with ThreadPoolExecutor(max_workers=workers) as executor:
                futures = []
                for i in range(len(t)):
                    _e = executor.submit(biexciton_ace,t0,t[i] + tauend,*pulses,apply_op_t=t[i], suffix=i, **options)
                    _e.add_done_callback(lambda f: tq.update())
                    futures.append(_e)
                wait(futures)
            for i in range(len(futures)):
                futures[i] = futures[i].result()
            # futures now contains t,g,x,y,b for every i
            for i in range(len(t)):
                # futures[i][2] are the x values 
                _G2[i,1:] = futures[i][2][-n_tau:]
    else:
        for i in tqdm.trange(len(t)):
            _tend = t[i] + tauend
            t,g,x,y,b = biexciton_ace(t0,_tend,*pulses,apply_op_t=t[i], suffix=i, **options)
            # use, that Tr(sigma_x^dagger*sigma_x*rho) = x
            # for the last n_tau elements, not including tau=0, which stays zero
            _G2[i,1:] = x[-n_tau:]
    return _G2
