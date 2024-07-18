from __future__ import annotations

from qiskit import QuantumCircuit, QuantumRegister
from qiskit.circuit import ParameterVector
import math

# test 

class ghz_circuit(QuantumCircuit):
    
    def __init__(self, num_qubits: int, name ="ghz circuit"):
        super().__init__(num_qubits)
        """Returns a quantum circuit implementing the GHZ state.

        Keyword arguments:
        num_qubits -- number of qubits of the returned quantum circuit
        """

        self.h([-1])
        for i in range(1, num_qubits):
            self.cx([num_qubits - i], [num_qubits - i - 1])
        self.measure_all()
        


# Paper 1

class VariationalLayer_1(QuantumCircuit):
    """
    paper_1 - Chen2022 bibtex
    Variational Layer for Quantum LSTM from: 
    https://arxiv.org/pdf/2009.01783.pdf
    """
    def __init__(self, num_qubits: int, name="variational-layer"):
        super().__init__(num_qubits, name=name)
        # weight parameters
        alpha = ParameterVector('α', length=num_qubits)
        beta = ParameterVector('β', length=num_qubits)
        gamma = ParameterVector('γ', length=num_qubits)

        # entanglement
        self.cx(0,1)
        self.cx(1,2)
        self.cx(2,3)
        self.cx(3,0)
        self.cx(0,2)
        self.cx(1,3)
        self.cx(2,0)
        self.cx(3,1)

        # x,y,z rotation
        for i in range(num_qubits):
            self.rx(alpha[i], i)
            self.ry(beta[i], i)
            self.rz(gamma[i], i)

# Paper 2

class VariationalLayer_2(QuantumCircuit):
    """
    paper_2 - Yu2023 bibtex
    and paper_3 - Qi2021 bibtex
    """
    def __init__(self, num_qubits: int, name="variational-layer"):
        super().__init__(num_qubits, name=name)
        # weight parameters
        alpha = ParameterVector('α', length=num_qubits)
        beta = ParameterVector('β', length=num_qubits)
        gamma = ParameterVector('γ', length=num_qubits)

        # entanglement

        self.cx(0,1)
        self.cx(1,2)
        self.cx(2,3)
        self.cx(3,0)
    

        # x,y,z rotation
        for i in range(num_qubits):
            self.rx(alpha[i], i)
            self.ry(beta[i], i)
            self.rz(gamma[i], i)



# paper_4 - Sim2019 circuit 14


class VariationalLayer_4(QuantumCircuit):
    """
    paper_4 circuit 14 - Sim2019 bibtex
    """
    def __init__(self, num_qubits: int, name="variational-layer"):
        super().__init__(num_qubits, name=name)
        # weight parameters
        alpha = ParameterVector('α', length=num_qubits)
        beta = ParameterVector('β', length=num_qubits)
        gamma = ParameterVector('γ', length=num_qubits)
        delta = ParameterVector('δ', length=num_qubits)

        for i in range(num_qubits):
            self.rx(alpha[i], i)
            self.rz(beta[i], i)


        self.cx(3,2)
        self.cx(3,1)
        self.cx(3,0)
        self.cx(2,3)
        self.cx(2,1)
        self.cx(2,0)
        self.cx(1,3)
        self.cx(1,2)
        self.cx(1,0)
        self.cx(0,3)
        self.cx(0,2)
        self.cx(0,1)


        for i in range(num_qubits):
            self.rx(gamma[i], i)
            self.rz(delta[i], i)

class VariationalLayer_5(QuantumCircuit):
    """
    own vl
    """
    def __init__(self, num_qubits: int, name="variational-layer"):
        super().__init__(num_qubits, name=name)
        # weight parameters
        alpha = ParameterVector('α', length=num_qubits)
        beta = ParameterVector('β', length=num_qubits)
        gamma = ParameterVector('γ', length=num_qubits)
        delta = ParameterVector('δ', length=num_qubits)

        for i in range(num_qubits):
            self.rx(alpha[i], i)
            self.rz(beta[i], i)

        for i in range(1, num_qubits):
            self.cx(num_qubits - i, num_qubits - i - 1)

        for i in range(num_qubits):
            self.rx(gamma[i], i)
            self.rz(delta[i], i)
    