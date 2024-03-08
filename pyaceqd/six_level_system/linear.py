import subprocess
import numpy as np
import os
from pyaceqd.tools import export_csv, output_ops_dm, compose_dm, read_calibration_file
from pyaceqd.general_system.general_system import system_ace_stream
from pyaceqd.general_system.general_dressed_states import dressed_states

#d0 = 0.25  # meV default set 
d0 = 0.13  # meV , Rene's dot 
d1 = 0.12
d2 = 0.05
mu_b = 5.7882818012e-2   # meV/T
g_ex = -0.65  # in plane electron g factor
g_ez = -0.8  # out of plane electron g factor
g_hx = -0.35  # in plane hole g factor
g_hz = -2.2  # out of plane hole g factor
hbar = 0.6582173  # meV*ps

def energies_linear(d0=0.25, d1=0.12, d2=0.05, delta_B=4, delta_E=0.0):
    E_X = delta_E + (d0 + d1)/2.0 
    E_Y = delta_E + (d0 - d1)/2.0 
    E_S = delta_E - (d0 - d2)/2.0 
    E_F = delta_E - (d0 + d2)/2.0 
    E_B = 2.*delta_E - delta_B
    return E_X, E_Y, E_S, E_F, E_B

def sixls_linear(t_start, t_end, *pulses, dt=0.5, delta_b=4, gamma_e=1/100, gamma_b=None, gamma_d=0, bx=0, bz=0, phonons=False, ae=3.0, temperature=4, verbose=False, lindblad=False, temp_dir='/mnt/temp_data/', pt_file=None, suffix="", \
               multitime_op=None, pulse_file_x=None, pulse_file_y=None, prepare_only=False, output_ops=["|0><0|_6","|1><1|_6","|2><2|_6","|3><3|_6"], initial="|0><0|_6", t_mem=20.48, output_dm=False, dressedstates=False, rf=False, rf_file=None,
         firstonly=False,calibration_file=None):
    system_prefix = "sixls_linear"
    # |0> = G, |1> = X, |2> = Y, |3> = S = Dx, |4> = F = Dy, |5> = B
    if calibration_file is not None:
        E_X, E_Y, E_S, E_F, E_B, gamma_e, gamma_b, g_ex, g_hx, g_ez, g_hz = read_calibration_file(calibration_file)
    else:
        E_X, E_Y, E_S, E_F, E_B = energies_linear(delta_B=delta_b)
        g_ex = -0.65  # in plane electron g factor
        g_ez = -0.8  # out of plane electron g factor
        g_hx = -0.35  # in plane hole g factor
        g_hz = -2.2  # out of plane hole g factor
    system_op = ["{}*|1><1|_6 + {}*|2><2|_6 + {}*|3><3|_6 + {}*|4><4|_6 + {}*|5><5|_6".format(E_X,E_Y,E_S,E_F,E_B)]
    # bright-dark coupling depending on Bx
    if bx != 0:
        system_op.append("{}*(|1><3|_6 + |3><1|_6 )".format(-0.5*mu_b*bx*(g_ex+g_hx)))
        system_op.append("{}*(|2><4|_6 + |4><2|_6 )".format(-0.5*mu_b*bx*(g_ex-g_hx)))
    # bright-bright and dark-dark coupling depending on Bz
    if bz != 0.0:
        system_op.append("-i*{}*(|2><1|_6 - |1><2|_6 )".format(-0.5*mu_b*bz*(g_ez-3*g_hz)))
        system_op.append("-i*{}*(|4><3|_6 - |3><4|_6 )".format(+0.5*mu_b*bz*(g_ez+3*g_hz)))
    boson_op = "1*(|1><1|_6+|2><2|_6+|3><3|_6+|4><4|_6) + 2*|5><5|_6"
    lindblad_ops = []
    if lindblad:
        if gamma_b is None:
            gamma_b = gamma_e
        lindblad_ops = [["|0><1|_6",gamma_e],["|0><2|_6",gamma_e],
                        ["|1><5|_6",gamma_b],["|2><5|_6",gamma_b],
                        ["|0><3|_6",gamma_d],["|0><4|_6",gamma_d]] # Dark decay actually not needed
    interaction_ops = [["|1><0|_6+|5><1|_6","x"],["|2><0|_6+|5><2|_6","y"]]
    
    rf_op = None
    if rf:
        rf_op = "|1><1|_6+|2><2|_6+|3><3|_6+|4><4|_6+2*|5><5|_6"  # factor 2 for biexciton
    
    if output_dm:
        output_ops = output_ops_dm(dim=6)
    result = system_ace_stream(t_start, t_end, *pulses, dt=dt, phonons=phonons, t_mem=t_mem, ae=ae, temperature=temperature, verbose=verbose, temp_dir=temp_dir, pt_file=pt_file, suffix=suffix, \
                  multitime_op=multitime_op, system_prefix=system_prefix, threshold="10", threshold_ratio="0.3", buffer_blocksize="-1", dict_zero="16", precision="12", boson_e_max=7,
                  system_op=system_op, pulse_file_x=pulse_file_x, pulse_file_y=pulse_file_y, boson_op=boson_op, initial=initial, lindblad_ops=lindblad_ops, interaction_ops=interaction_ops, output_ops=output_ops, prepare_only=prepare_only, 
                  dressedstates=dressedstates, rf_op=rf_op, rf_file=rf_file, firstonly=firstonly)
    if output_dm:
        return compose_dm(result, dim=6)
    return result

def sixls_linear_dressed_states(t_start, t_end, *pulses, plot=True, filename="sixls_linear_dressed", firstonly=False,make_transparent = [], **options):
    colors = ["#0000cf", "#45b0ee", "#ff0022", "#9966cc", "#009e00", "#ffde39"]
    #paper colors 1 ,2, 5 (XX)
    colors = ["#4c9900", "#00ff00", "#00ff00", "#4c9900", "#4c9900", "#ffde39"]
    

    for i in make_transparent:
        colors[i] = colors[i] + "00"

    dim = 6
    return dressed_states(sixls_linear, dim, t_start, t_end, *pulses, filename=filename, plot=plot, firstonly=firstonly, colors=colors, **options)

