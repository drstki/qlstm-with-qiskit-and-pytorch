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
    def __init__(self, num_qubits: int, name="data-encoding"):
        super().__init__(num_qubits, name=name)
        # input parameters
        x_input = ParameterVector('x', length=num_qubits)
        
        # encoding
        for i in range(num_qubits):
            self.h(i)
            self.ry(x_input[i].arctan(), i)
            self.rz((x_input[i]*x_input[i]).arctan(), i)


# Paper 2
class FeatureMap_2(QuantumCircuit):
    """
    paper_2 - Yu2023 bibtex
    """
    def __init__(self, num_qubits: int, name="data-encoding"):
        super().__init__(num_qubits, name=name)
        # input parameters
        x_input = ParameterVector('x', length=num_qubits)
        
        # encoding
        for i in range(num_qubits):
            self.h(i)
            self.rx(x_input[i], i)


# paper_3
class FeatureMap_3(QuantumCircuit):
    """
    paper_3 - Qi2021 bibtex
    """
    def __init__(self, num_qubits: int, name="data-encoding"):
        super().__init__(num_qubits, name=name)
        # input parameters
        x_input = ParameterVector('x', length=num_qubits)
        
        # encoding
        for i in range(num_qubits):
            self.h(i)
            self.ry(x_input[i]*math.pi, i)


# paper_4 - Sim2019 circuit 14

class FeatureMap_4(QuantumCircuit):
    """
    paper_1 - Chen2022 bibtex
    modified by myself
    """
    def __init__(self, num_qubits: int, name="data-encoding"):
        super().__init__(num_qubits, name=name)
        # input parameters
        x_input = ParameterVector('x', length=num_qubits)
        
        # encoding
        for i in range(num_qubits):
            self.h(i)
            self.ry(x_input[i]*math.pi, i)
            self.rz((x_input[i]*x_input[i])*math.pi, i)


### dictionary of all available feature map layers
ALL_FEATURE_MAP_DICT = {
    "fm_1": FeatureMap_1,
    "fm_2": FeatureMap_2,
    "fm_3": FeatureMap_3,
    "fm_4": FeatureMap_4
}

def get_feature_map(feature_map, num_qubits: int) -> QuantumCircuit:
    """
    Returns a quantum circuit implementing the specified ansatz.

    Keyword arguments:
    feature_map - name of the feature map to be returned
    num_qubits - number of qubits of the returned quantum circuit
    """
    if isinstance(feature_map, QuantumCircuit):
        return feature_map
    elif not feature_map in ALL_FEATURE_MAP_DICT.keys():
        raise ValueError("Ansatz not available please choose:" + "".join(f" \'{key}\';" for key in ALL_FEATURE_MAP_DICT.keys()) + "\nAlternatively input custom feature map as Qiskit QuantumCircuit object.")
    else:
        return ALL_FEATURE_MAP_DICT[feature_map](num_qubits)