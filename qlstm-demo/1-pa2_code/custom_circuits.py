from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector
import math

# Paper 1
class FeatureMap_1(QuantumCircuit):
    """
    paper_1 - Chen2022 bibtex
    Feature Map for Quantum LSTM from: 
    https://arxiv.org/pdf/2009.01783.pdf

    also used for https://link.springer.com/article/10.1007/s42484-023-00115-2
    """
    def __init__(self, qbits=4, name="data-encoding"):
        super().__init__(qbits, name=name)
        # input parameters
        x_input = ParameterVector('x', length=4)
        
        # encoding
        for i in range(4):
            self.h(i)
            self.ry(x_input[i].arctan(), i)
            self.rz((x_input[i]*x_input[i]).arctan(), i)


class VariationalLayer_1(QuantumCircuit):
    """
    paper_1 - Chen2022 bibtex
    Variational Layer for Quantum LSTM from: 
    https://arxiv.org/pdf/2009.01783.pdf
    """
    def __init__(self, qbits=4, name="variational-layer"):
        super().__init__(qbits, name=name)
        # weight parameters
        alpha = ParameterVector('α', length=4)
        beta = ParameterVector('β', length=4)
        gamma = ParameterVector('γ', length=4)

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
        for i in range(4):
            self.rx(alpha[i], i)
            self.ry(beta[i], i)
            self.rz(gamma[i], i)

# Paper 2
class FeatureMap_2(QuantumCircuit):
    """
    paper_2 - Yu2023 bibtex
    """
    def __init__(self, qbits=4, name="data-encoding"):
        super().__init__(qbits, name=name)
        # input parameters
        x_input = ParameterVector('x', length=4)
        
        # encoding
        for i in range(4):
            self.h(i)
            self.rx(x_input[i], i)

class VariationalLayer_2(QuantumCircuit):
    """
    paper_2 - Yu2023 bibtex
    and paper_3 - Qi2021 bibtex
    """
    def __init__(self, qbits=4, name="variational-layer"):
        super().__init__(qbits, name=name)
        # weight parameters
        alpha = ParameterVector('α', length=4)
        beta = ParameterVector('β', length=4)
        gamma = ParameterVector('γ', length=4)

        # entanglement
        self.cx(0,1)
        self.cx(1,2)
        self.cx(2,3)
        self.cx(3,0)
    

        # x,y,z rotation
        for i in range(4):
            self.rx(alpha[i], i)
            self.ry(beta[i], i)
            self.rz(gamma[i], i)


# paper_3
class FeatureMap_3(QuantumCircuit):
    """
    paper_3 - Qi2021 bibtex
    """
    def __init__(self, qbits=4, name="data-encoding"):
        super().__init__(qbits, name=name)
        # input parameters
        x_input = ParameterVector('x', length=4)
        
        # encoding
        for i in range(4):
            self.h(i)
            self.ry(x_input[i]*math.pi, i)


# paper_4 - Sim2019 circuit 14

class FeatureMap_4(QuantumCircuit):
    """
    paper_1 - Chen2022 bibtex
    modified by myself
    """
    def __init__(self, qbits=4, name="data-encoding"):
        super().__init__(qbits, name=name)
        # input parameters
        x_input = ParameterVector('x', length=4)
        
        # encoding
        for i in range(4):
            self.h(i)
            self.ry(x_input[i]*math.pi, i)
            self.rz((x_input[i]*x_input[i])*math.pi, i)

class VariationalLayer_4(QuantumCircuit):
    """
    paper_4 circuit 14 - Sim2019 bibtex
    """
    def __init__(self, qbits=4, name="variational-layer"):
        super().__init__(qbits, name=name)
        # weight parameters
        alpha = ParameterVector('α', length=4)
        beta = ParameterVector('β', length=4)
        gamma = ParameterVector('γ', length=4)
        delta = ParameterVector('δ', length=4)

        for i in range(4):
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


        for i in range(4):
            self.rx(gamma[i], i)
            self.rz(delta[i], i)
