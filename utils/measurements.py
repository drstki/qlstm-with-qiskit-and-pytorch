from qiskit.quantum_info import SparsePauliOp

def create_pauli_ops(q):
    if not isinstance(q, int) or q <= 0:
        raise ValueError("q must be a positive integer")
    
    pauli_ops = []
    for i in range(q):
        pauli_str = "I" * i + "Z" + "I" * (q - i - 1)
        pauli_op = SparsePauliOp.from_list([(pauli_str,1)])
        pauli_ops.append(pauli_op)
    
    return pauli_ops