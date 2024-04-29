import numpy as np
from qiskit.providers.fake_provider import GenericBackendV2

# creating fake devices from different QPU providers outside IBM/qiskit

# Quantware Contralto - 21+4 qubits: https://www.quantware.com/product/contralto
def qw_contralto():
    qw_contralto_backend = GenericBackendV2(
        21,
        basis_gates = ["id", "rx", "u1", "cz"], # need to be verfied
        coupling_map = [[0,1],[0,4],[1,2],[1,5],[2,6],[3,4],[3,8],[4,5],[4,9],[5,6],[5,10],[6,7],[6,11],[7,12],[8,9],[8,13],[9,10],[9,14],[10,11],[10,15],[11,12],[11,16],[12,17],[13,14],[14,15],[14,18],[15,16],[15,19],[16,17],[16,20],[18,19],[19,20]]
    )
    return qw_contralto_backend

# Rigetti Novera - 9 qubits
def rigetti_novera():
    rigetti_novera_backend = GenericBackendV2(
        9,
        basis_gates = ["id", "rx", "rz", "iswap"], # need to be verfied
        coupling_map = [[0,1],[0,3],[1,2],[1,4],[2,5],[3,4],[3,6],[4,5],[4,7],[5,8],[6,7],[7,8]]
    )
    return rigetti_novera_backend