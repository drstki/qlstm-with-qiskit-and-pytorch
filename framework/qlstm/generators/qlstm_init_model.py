import torch
import torch.nn as nn

#import module_utils
import generators.ansatz as astz
import generators.feature_map as fm
import generators.backends as be
import generators.noise_models as nm

from qiskit.primitives import BackendEstimator as Estimator
from qiskit.quantum_info import SparsePauliOp
from qiskit_machine_learning.neural_networks import EstimatorQNN
from qiskit_machine_learning.connectors import TorchConnector 



class QuantumLongShortTermMemory(nn.Module):
    def __init__(self, vqc, num_qubits, feature_map, ansatz, input_size: int=4, hidden_size: int=1, seed: int=1):
        super().__init__()

        self.input_sz = input_size
        self.hidden_sz = hidden_size
        self.seed = seed

        # construct quantum layer
        self.VQC = nn.ModuleDict() # WICHTIG to connect 
        self.construct_VQC_layer(vqc, num_qubits, feature_map, ansatz)

        # classical layer
        self.input_layer = nn.Linear(self.input_sz + self.hidden_sz, self.input_sz)
        self.input_layer_2 = nn.Linear(1, self.input_sz)


    def construct_VQC_layer(self, vqc, num_qubits, feature_map, ansatz):
        # construct the 4 QNN layer
        for layer_name in ["1", "2", "3", "4", "5"]:
            # initialize the QNN layer
            obsv = SparsePauliOp(["Z"*num_qubits]) 
            estimator = Estimator(backend=backend, options={'NoiseModel': noise_model})
            qnn = EstimatorQNN(
                    circuit=vqc,
                    estimator=estimator,
                    observables=obsv,
                    input_params=feature_map.parameters,
                    weight_params=ansatz.parameters,
                    input_gradients=True
            )

            # WICHTIG connector
            self.VQC[layer_name] = TorchConnector(qnn)


    def forward(self, X: torch.Tensor, memory_states: tuple = None):
        if memory_states is None:
            # initialize memory states
            h_t, c_t = (torch.zeros(1, self.hidden_sz).to(X.device), 
                        torch.zeros(1, self.hidden_sz).to(X.device))
        else:
            h_t, c_t = memory_states 

        outputs = []
        for sample_x in X: 
            v_t = torch.cat([sample_x, h_t], dim=0)
            v_t_input = self.input_layer(v_t.reshape(1, -1)).reshape(-1)
            # QNN layer
            f_t = torch.sigmoid(self.VQC["1"](v_t_input))
            i_t = torch.sigmoid(self.VQC["2"](v_t_input))
            c_tilde = torch.tanh(self.VQC["3"](v_t_input))
            c_t = f_t * c_t + i_t * c_tilde
            o_t = torch.sigmoid(self.VQC["4"](v_t_input))
            h_t = self.VQC["5"]((self.input_layer_2(o_t * torch.tanh(c_t))))
            outputs.append(h_t.unsqueeze(0))
        
        outputs = torch.cat(outputs, dim=0)
        # reshape from shape (sequence, batch, feature) to (batch, sequence, feature)
        outputs = outputs.transpose(0, 1).contiguous()

        return outputs, (h_t, c_t)