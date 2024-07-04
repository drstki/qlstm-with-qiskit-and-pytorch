from qiskit_aer import Aer
from qiskit_ibm_runtime import  Options
from qiskit_aer.noise import NoiseModel
from qiskit_ibm_provider import IBMProvider

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import ListedColormap
from matplotlib.collections import PatchCollection
from matplotlib.patches import Rectangle

from sklearn.preprocessing import MinMaxScaler

import torch
import torch.nn as nn
import torch.optim as optim
from torchmetrics import MeanAbsoluteError, MeanSquaredError

import sys
sys.path.append("../")


#import module_utils
import torchQLSTM.module_utils.ansatz as astz
import torchQLSTM.module_utils.feature_map as fm
import torchQLSTM.module_utils.backends as be
import torchQLSTM.module_utils.noise_models as nm

from qiskit.primitives import BackendEstimator as Estimator
from qiskit.quantum_info import SparsePauliOp
from qiskit_machine_learning.neural_networks import EstimatorQNN
from qiskit_machine_learning.connectors import TorchConnector 

#from utils.data_processing import time_window, time_window_batch
#from utils.measurements import create_pauli_ops
#from torchQLSTM.QLSTM import QuantumLongShortTermMemory


sns.set_theme()
objective_func_vals = []
plt.rcParams["figure.figsize"] = (12, 6)




class DATAPREP: # load and prepare data set for using quantum backend

    def data_prep(dataset,sep, header, x_axis, y_axis, idco, FIRST, LAST, FIRST_REFERENCE, LAST_REFERENCE, figsize, LIM, dataset_image_title, dataset_image_file, LAST_TRAIN, LAST_VALID, LAST_TEST, periods, dataset_curve_file):
        data_file = pd.read_csv(dataset, sep=sep, header=header, names=[x_axis, y_axis], index_col=[idco])
        
        anomaly = data_file.loc[FIRST:LAST, y_axis].dropna()
        
        reference = anomaly.loc[FIRST_REFERENCE:LAST_REFERENCE].mean()
    
        cmap = ListedColormap(
            ['#08306b', '#08519c', '#2171b5', '#4292c6','#6baed6', 
             '#9ecae1', '#c6dbef', '#deebf7','#fee0d2', '#fcbba1', 
             '#fc9272', '#fb6a4a', '#ef3b2c', '#cb181d', '#a50f15', '#67000d']
             )

        fig = plt.figure(figsize=figsize)

        ax = fig.add_axes([0, 0, 1, 1])
        ax.set_axis_off()

        col = PatchCollection([
            Rectangle((y, 0), 1, 1)
            for y in range(FIRST, LAST + 1)
        ])

        col.set_array(anomaly)
        col.set_cmap(cmap)
        col.set_clim(reference - LIM, reference + LIM)
        ax.add_collection(col)

        ax.set_ylim(0, 1)
        ax.set_xlim(FIRST, LAST + 1)
        ax.set_title(dataset_image_title)
        

        fig = plt.savefig(dataset_image_file)

        train_df = data_file.loc[FIRST:LAST_TRAIN, y_axis].dropna().reset_index().set_index(x_axis)
        valid_df = data_file.loc[LAST_TRAIN+1:LAST_VALID, y_axis].dropna().reset_index().set_index(x_axis)
        test_df = data_file.loc[LAST_VALID+1:LAST_TEST, y_axis].dropna().reset_index().set_index(x_axis)

        train_df = pd.concat([train_df,train_df[y_axis].shift(periods=periods)],axis=1)
        valid_df = pd.concat([valid_df,valid_df[y_axis].shift(periods=periods)],axis=1)
        test_df = pd.concat([test_df,test_df[y_axis].shift(periods=periods)],axis=1)
    
        train_scaler_amount=MinMaxScaler()
        transform_1 = y_axis
        transform_2 = '%s_periods[%d]' % (y_axis, 0)
        transform_3 = '%s_periods[%d]' % (y_axis, 1)
       
        #train_df = train_scaler_amount.fit_transform(train_df[[y_axis,'%s_periods[%d]' % (y_axis, periods[0]),'%s_periods[%d]' % (y_axis, periods[1])]])[periods[1]:,:]
        #valid_df = train_scaler_amount.transform(valid_df[[y_axis,'%s_periods[%d]' % (y_axis, periods[0]),'%s_periods[%d]' % (y_axis, periods[1])]])[periods[1]:,:]
        #test_df = train_scaler_amount.transform(test_df[[y_axis,'%s_periods[%d]' % (y_axis, periods[0]),'%s_periods[%d]' % (y_axis, periods[1])]])[periods[1]:,:]
        train_df=train_scaler_amount.fit_transform(train_df[['anomaly','%s_%d' % (y_axis, periods[0]),'%s_%d' % (y_axis, periods[1])]])[5:,:]
        valid_df=train_scaler_amount.transform(valid_df[['anomaly','%s_%d' % (y_axis, periods[0]),'%s_%d' % (y_axis, periods[1])]])[5:,:]
        test_df=train_scaler_amount.transform(test_df[['anomaly','%s_%d' % (y_axis, periods[0]),'%s_%d' % (y_axis, periods[1])]])[5:,:]
    
        plt.figure(figsize=figsize)
        plt.plot(train_df[:,0], color='blue', label='Train data')
        fig = plt.savefig(dataset_curve_file)

        return train_df, valid_df, test_df


class QLSTM(nn.Module):
    def __init__(self, feature_map, ansatz, ansatz_reps, backend, noise_model, input_size, hidden_size: int=1):
        super().__init__()
        
        self.hidden_sz = hidden_size
        self.input_sz = input_size

        # load predefined quantum hyperparameters
        self.backend =be.get_backend(backend)
        self.noise_model = nm.get_noise_model(noise_model)
        self.feature_map = fm.get_feature_map(feature_map, input_size)
        self.ansatz = astz.get_ansatz(ansatz, input_size)
        self.ansatz_reps = ansatz_reps

        # check feature map and ansatz compatibility
        if self.feature_map.num_qubits != self.ansatz.num_qubits:
            raise ValueError(f"Mismatch in number of qubits: feature_map has {self.feature_map.num_qubits}, ansatz has {self.ansatz.num_qubits}.")

        # construct quantum layer
        self.VQC = nn.ModuleDict()
        self.construct_VQC_layer(ansatz_reps)

        # classical layer
        self.input_layer = nn.Linear(self.input_sz + self.hidden_sz, self.input_sz)
        self.input_layer_2 = nn.Linear(1, self.input_sz)


    def construct_VQC_layer(self, ansatz_reps):        
        # construct the VQC
        self.vqc = self.feature_map.compose(self.ansatz, inplace=False)
        # TODO: add ansatz repetitions

        # construct the QNN layer
        for layer_name in ["1", "2", "3", "4", "5"]:
            # initialize the QNN layer
            obsv = SparsePauliOp(["Z"*self.feature_map.num_qubits]) 
            estimator = Estimator(backend=self.backend, options={'NoiseModel': self.noise_model})
            qnn = EstimatorQNN(
                    circuit=self.vqc,
                    estimator=estimator,
                    observables=obsv,
                    input_params=self.feature_map.parameters,
                    weight_params=self.ansatz.parameters,
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
                input_size,
                hidden_dim,
                target_size,
                noise_model
                 ):
        super(QModel, self).__init__()

        seed = 71
        np.random.seed = seed
        torch.manual_seed=seed
        
        self.lstm = QLSTM(feature_map="fm_1", ansatz="ghz", ansatz_reps=2, backend="aer_sv", noise_model=noise_model, input_size=input_size, hidden_size = hidden_dim)

        # The linear layer that maps from hidden state space to target space
        self.dense = nn.Linear(hidden_dim, target_size)

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        dense_out = self.dense(lstm_out)
        out_scores=dense_out
        # out_scores = F.log_softmax(dense_out, dim=1)
        return out_scores