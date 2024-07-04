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

from generators.qlstm_aer import DATAPREP

# Data Load and Prep
FIRST = 1850
LAST = 2018  # inclusive

# Reference period for the center of the color scale

FIRST_REFERENCE = 1971
LAST_REFERENCE = 2000
LIM = 0.7 # degrees

dataset = './data/MetOffice_HadCRUT4.csv'

sep=';'
header=0
x_axis = ' year'
y_axis = 'anomaly'
idco=0

LAST_TRAIN =2005
LAST_VALID =2018
LAST_TEST = 2023

dataset_image_file = './results/warming-stripes.png'
dataset_image_title = "GLOBAL WARMING STRIPES"
dataset_curve_file = './results/global_warming_curve.png'

periods=[2,5]

figsize=(15,5)

#data_load = DATAPREP.data_load(dataset, sep, header, x_axis, y_axis, idco, FIRST, LAST, FIRST_REFERENCE, LAST_REFERENCE, LIM, dataset_image_title, dataset_image_file)

data_prep = DATAPREP.data_prep(
    dataset,sep, header, x_axis, y_axis, idco, FIRST, LAST, FIRST_REFERENCE, LAST_REFERENCE, 
    figsize, LIM, dataset_image_title, dataset_image_file, LAST_TRAIN, LAST_VALID, LAST_TEST, 
    periods, dataset_curve_file
    )