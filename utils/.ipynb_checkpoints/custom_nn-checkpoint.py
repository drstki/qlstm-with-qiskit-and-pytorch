import torch
import torch.nn as nn
import math
from qiskit import QuantumCircuit
from qiskit_machine_learning.connectors import TorchConnector
from qiskit_machine_learning.neural_networks import EstimatorQNN


class LongShortTermMemory(nn.Module):
    """"""
    def __init__(self, input_size: int=1, hidden_size: int=4):
        """ Initializes custom build LSTM model.
        Parameters:
            input_size (int): The dimensionality of the input for the LSTM (input feature).
            hidden_sz (int): The number of hidden units in the LSTM (dimensionality of the hidden state).
        """
        super().__init__()

        self.input_sz = input_size
        self.hidden_sz = hidden_size

        # weight matrix W - input gate
        self.W = nn.Parameter(torch.Tensor(input_size, hidden_size * 4))
        # weight matrix U - forget gate
        self.U = nn.Parameter(torch.Tensor(hidden_size, hidden_size * 4))
        self.bias = nn.Parameter(torch.Tensor(hidden_size * 4))
        # call the init weights function
        self.init_weights()

        # output layer
        self.linear = nn.Linear(hidden_size, 1)
                
    def init_weights(self):
        """ initialize weights with random uniform distribution """
        stdv = 1.0 / math.sqrt(self.hidden_sz)
        for weight in self.parameters():
            weight.data.uniform_(-stdv, stdv)
         
    def forward(self, x: torch.Tensor, memory_states: tuple = None):
        """Perform a forward pass through the LSTM model.

        Parameters:
            x (torch.Tensor): Input data of shape (batch_size, sequence_window, feature).
            memory_states (tuple): Tuple containing the initial hidden and cell states (h_t, c_t).

        Returns:
            torch.Tensor: Output predictions of shape (batch_size, 1).
            tuple: Tuple containing the final hidden and cell states after processing the sequence (h_t, c_t).
        """
        bs, seq_sz, _ = x.size()
        if memory_states is None:
            # initialize memory states
            h_t, c_t = (torch.zeros(bs, self.hidden_sz).to(x.device), 
                        torch.zeros(bs, self.hidden_sz).to(x.device))
        else:
            h_t, c_t = memory_states
        
        HS = self.hidden_sz
        for t in range(seq_sz):
            x_t = x[:, t, :]
            # batch the computations into a single matrix multiplication
            gates = x_t @ self.W + h_t @ self.U + self.bias
            i_t, f_t, g_t, o_t = (
                torch.sigmoid(gates[:, :HS]), # input
                torch.sigmoid(gates[:, HS:HS*2]), # forget
                torch.tanh(gates[:, HS*2:HS*3]),
                torch.sigmoid(gates[:, HS*3:]), # output
            )
            c_t = f_t * c_t + i_t * g_t
            h_t = o_t * torch.tanh(c_t)

            y_t = self.linear(h_t)
        return  y_t, (h_t, c_t)
 

class QuantumLongShortTermMemory(nn.Module):
    def __init__(self, feature_map, ansatz, reps, input_size: int=4, hidden_size: int=1):
        super().__init__()
        num_qbits = 4

        self.input_sz = input_size
        self.hidden_sz = hidden_size

        # construct quantum layer
        self.qnn_layer = nn.ModuleDict()
        self.construct_VQC_layer(num_qbits, feature_map, ansatz, reps)

        # classical layer
        self.input_layer = nn.Linear(self.input_sz + self.hidden_sz, self.input_sz)
        self.input_layer_2 = nn.Linear(1, self.input_sz)


    def construct_VQC_layer(self, qbits, feature_map, ansatz, reps):
        # construct the 4 QNN layer
        for layer_name in ["1", "2", "3", "4", "5"]:
            # construct the quantum circuit
            qc = QuantumCircuit(qbits)
            # append the feature map and ansatz (with reps) to the circuit
            qc.append(feature_map, range(qbits))
            for _ in range(reps):
                qc.append(ansatz, range(qbits))

            # initialize the QNN layer
            vqc = EstimatorQNN(
                    circuit=qc,
                    input_params=feature_map.parameters,
                    weight_params=ansatz.parameters,
                    input_gradients=True
            )

            self.qnn_layer[layer_name] = TorchConnector(vqc)


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
            f_t = torch.sigmoid(self.qnn_layer["1"](v_t_input))
            i_t = torch.sigmoid(self.qnn_layer["2"](v_t_input))
            c_tilde = torch.tanh(self.qnn_layer["3"](v_t_input))
            c_t = f_t * c_t + i_t * c_tilde
            o_t = torch.sigmoid(self.qnn_layer["4"](v_t_input))
            h_t = self.qnn_layer["5"]((self.input_layer_2(o_t * torch.tanh(c_t))))
            outputs.append(h_t.unsqueeze(0))
        
        outputs = torch.cat(outputs, dim=0)
        # reshape from shape (sequence, batch, feature) to (batch, sequence, feature)
        outputs = outputs.transpose(0, 1).contiguous()

        return outputs, (h_t, c_t)
    