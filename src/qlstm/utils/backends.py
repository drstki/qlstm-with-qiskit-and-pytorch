import numpy as np
from qiskit.providers.fake_provider import GenericBackendV2
from qiskit_ibm_runtime.fake_provider import FakeManilaV2, FakeSherbrooke
from qiskit_aer import AerSimulator

# creating fake devices from different QPU providers outside IBM/qiskit

# Quantware Contralto - 21+4 qubits: https://www.quantware.com/product/contralto
def qw_contralto():
    qw_contralto_backend = GenericBackendV2(
        21,
        basis_gates = ["id", "rx", "u1", "cz"], # need to be verfied
        coupling_map = [[0,1],[0,4],[1,2],[1,5],[2,6],[3,4],[3,8],[4,5],[4,9],[5,6],[5,10],[6,7],[6,11],[7,12],[8,9],[8,13],[9,10],[9,14],[10,11],[10,15],[11,12],[11,16],[12,17],[13,14],[14,15],[14,18],[15,16],[15,19],[16,17],[16,20],[18,19],[19,20]]
    )
    return qw_contralto_backend

# take im 5 qubit device
def ibm_5q():
    ibm_5q_backend = FakeManilaV2(
    )
    return ibm_5q_backend

#take a 127 qubit device
def ibm_127q():
    ibm_127q_backend = FakeSherbrooke(  
    )
    return ibm_127q_backend

# different Aer Simulator modules

# A dense statevector simulation that can sample measurement outcomes from ideal circuits with all measurements at end of the circuit. For noisy simulations each shot samples a randomly sampled noisy circuit from the noise model.
def aer_sv():
    aer_sv_sim = AerSimulator(
        method='statevector'
    )
    return aer_sv_sim
# A dense density matrix simulation that may sample measurement outcomes from noisy circuits with all measurements at end of the circuit.
def aer_dm():
    aer_dm_sim = AerSimulator(
        method='density_matrix'
    )
    return aer_dm_sim 

# A tensor-network statevector simulator that uses a Matrix Product State (MPS) representation for the state. This can be done either with or without truncation of the MPS bond dimensions depending on the simulator options. The default behaviour is no truncation.
def aer_mps():
    aer_mps_sim = AerSimulator(
        method='matrix_product_state'
    )
    return aer_mps_sim 