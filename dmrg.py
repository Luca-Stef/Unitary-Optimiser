"""Example illustrating the use of DMRG in tenpy.

The example functions in this class do the same as the ones in `toycodes/d_dmrg.py`,
but make use of the classes defined in tenpy.
"""
# Copyright 2018-2021 TeNPy Developers, GNU GPLv3

import numpy as np

from tenpy.networks.mps import MPS
from tenpy.models.tf_ising import TFIChain
from tenpy.models.spins import SpinModel
from tenpy.algorithms import dmrg


def example_DMRG_tf_ising_infinite(g, D=4, J=1):
    #print("infinite DMRG, transverse field Ising model")
    #print("g={g:.2f}".format(g=g))
    model_params = dict(L=2, J=J, g=g, bc_MPS='infinite', conserve=None)
    M = TFIChain(model_params)
    product_state = ["up"] * M.lat.N_sites
    psi = MPS.from_product_state(M.lat.mps_sites(), product_state, bc=M.lat.bc_MPS)
    dmrg_params = {
        'mixer': True,  # setting this to True helps to escape local minima
        'trunc_params': {
            'chi_max': D,
            'svd_min': 1.e-10
        },
        'max_E_err': 1.e-10,
    }
    # Sometimes, we want to call a 'DMRG engine' explicitly
    eng = dmrg.TwoSiteDMRGEngine(psi, M, dmrg_params)
    E, psi = eng.run()  # equivalent to dmrg.run() up to the return parameters.
    #print("E = {E:.13f}".format(E=E))
    #print("final bond dimensions: ", psi.chi)
    mag_x = np.mean(psi.expectation_value("Sigmax"))
    mag_z = np.mean(psi.expectation_value("Sigmaz"))
    #print("<sigma_x> = {mag_x:.5f}".format(mag_x=mag_x))
    #print("<sigma_z> = {mag_z:.5f}".format(mag_z=mag_z))
    #print("correlation length:", psi.correlation_length())
    # compare to exact result
    #from tfi_exact import infinite_gs_energy
    #E_exact = infinite_gs_energy(1., g)
    #print("Analytic result: E (per site) = {E:.13f}".format(E=E_exact))
    #print("relative error: ", abs((E - E_exact) / E_exact))
    return E, psi, M


def example_1site_DMRG_tf_ising_infinite(g, D=4, J=-1):
    #print("single-site infinite DMRG, transverse field Ising model")
    #print("g={g:.2f}".format(g=g))
    model_params = dict(L=2, J=J, g=g, bc_MPS='infinite', conserve=None)
    M = TFIChain(model_params)
    product_state = ["up"] * M.lat.N_sites
    psi = MPS.from_product_state(M.lat.mps_sites(), product_state, bc=M.lat.bc_MPS)
    dmrg_params = {
        'mixer': True,  # setting this to True is essential for the 1-site algorithm to work.
        'trunc_params': {
            'chi_max': D,
            'svd_min': 1.e-10
        },
        'max_E_err': 1.e-10,
        'combine': True
    }
    eng = dmrg.SingleSiteDMRGEngine(psi, M, dmrg_params)
    E, psi = eng.run()  # equivalent to dmrg.run() up to the return parameters.
    #print("E = {E:.13f}".format(E=E))
    #print("final bond dimensions: ", psi.chi)
    mag_x = np.mean(psi.expectation_value("Sigmax"))
    mag_z = np.mean(psi.expectation_value("Sigmaz"))
    #print("<sigma_x> = {mag_x:.5f}".format(mag_x=mag_x))
    #print("<sigma_z> = {mag_z:.5f}".format(mag_z=mag_z))
    #print("correlation length:", psi.correlation_length())
    # compare to exact result
    #from tfi_exact import infinite_gs_energy
    #E_exact = infinite_gs_energy(1., g)
    #print("Analytic result: E (per site) = {E:.13f}".format(E=E_exact))
    #print("relative error: ", abs((E - E_exact) / E_exact))
    return E, psi, M


if __name__ == "__main__":
    import logging
    #logging.basicConfig(level=logging.INFO)
    #example_DMRG_tf_ising_finite(L=10, g=1.)
    print("-" * 100)
    #example_1site_DMRG_tf_ising_finite(L=10, g=1.)
    print("-" * 100)
    example_DMRG_tf_ising_infinite(g=1)
    print("-" * 100)
    example_1site_DMRG_tf_ising_infinite(g=1)
    print("-" * 100)
    #example_DMRG_heisenberg_xxz_infinite(Jz=1.5)