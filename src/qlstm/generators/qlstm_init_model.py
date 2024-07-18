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

    def __init__(self, feature_map, ansatz, ansatz_reps, backend, noise_model, input_size, hidden_size, num_qubits):
        super().__init__()
        
        self.feature_map = feature_map
        self.ansatz = ansatz
        self.ansatz_reps = ansatz_reps
        self.backend =backend
        self.noise_model = noise_model
        self.hidden_sz = hidden_size
        self.input_sz = input_size
        self.num_qubits = num_qubits

        # check feature map and ansatz compatibility
        if self.feature_map.num_qubits != self.ansatz.num_qubits:
            raise ValueError(f"Mismatch in number of qubits: feature_map has {self.feature_map.num_qubits}, ansatz has {self.ansatz.num_qubits}.")

        # construct quantum layer
        self.VQC = nn.ModuleDict()
        self.construct_VQC_layer(ansatz_reps)

        # classical layer
        self.input_layer = nn.Linear(self.input_sz + self.hidden_sz, self.input_sz)
        self.input_layer_2 = nn.Linear(1, self.input_sz)


    def construct_VQC_layer(feature_map, ansatz, ansatz_reps, backend, noise_model):        
       
        for i in range(ansatz_reps):
            ansatz_full = ansatz.compose(ansatz)
        
        vqc = feature_map.compose(ansatz_full)

        # construct the QNN layer
        for layer_name in ["1", "2", "3", "4", "5"]:
            # initialize the QNN layer
            obsv = SparsePauliOp(["Z"*feature_map.num_qubits]) 
            estimator = Estimator(backend=backend, options={'NoiseModel': noise_model})
            qnn = EstimatorQNN(
                    circuit=vqc,
                    estimator=estimator,
                    observables=obsv,
                    input_params=feature_map.parameters,
                    weight_params=ansatz.parameters,
                    input_gradients=True
            )
            
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
    
    def get_model_info(self):
        return {
            "feature_map": self.feature_map,
            "ansatz": self.ansatz,
            "vqc": self.vqc,
            "backend": self.backend,
            "noise_model": self.noise_model,
            "hidden_size": self.hidden_sz
        }
    
class QModel(nn.Module):

    def __init__(self, 
                feature_map,
                ansatz,
                ansatz_reps,
                backend,
                input_size,
                hidden_dim,
                target_size,
                noise_model
                ):
         super(QModel, self).__init__()

        
    QLSTM = QuantumLongShortTermMemory()    
    self.lstm = QLSTM(feature_map, ansatz, ansatz_reps, backend, noise_model, input_size, hidden_size = hidden_dim)

    # The linear layer that maps from hidden state space to target space
    self.dense = nn.Linear(hidden_dim, target_size)

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        dense_out = self.dense(lstm_out)
        out_scores=dense_out
        # out_scores = F.log_softmax(dense_out, dim=1)
        
        return out_scores