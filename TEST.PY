from WLS import wls
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
# from matplotlib.pylab import rcParams
from caseTN2 import caseTN
from pypower.api import ppoption, runpf
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from keras.models import Sequential, load_model
from keras.layers import Dense, Activation
from keras import optimizers
from matplotlib.pylab import rcParams
import time

rcParams['figure.figsize'] = 15, 5
nbus=41

X = pd.read_excel(r'C:/Private/Subjects/NCKH/CodeAI_WLS/Tonghop/TN/InputTrainMatrixTN10000.xls', sheet_name='Sheet') 
y = pd.read_excel(r'C:/Private/Subjects/NCKH/CodeAI_WLS/Tonghop/TN/OutputVoltageTN10000.xls', sheet_name='Sheet1') 
yV=y.iloc[:,:41]
yA=y.iloc[:,42:83]
X_trainV, X_testV, yV_train, yV_test = train_test_split(X, yV, random_state=None, test_size=0.2)
X_trainA, X_testA, yA_train, yA_test = train_test_split(X, yA, random_state=None, test_size=0.2)

X_trainV, X_testV, yV_train, yV_test = train_test_split(X, yV, random_state=None, test_size=0.2)
X_trainA, X_testA, yA_train, yA_test = train_test_split(X, yA, random_state=None, test_size=0.2)
X_error = pd.read_excel(r'C:/Private/Subjects/NCKH/CodeAI_WLS/Tonghop/TN/InputTrainMatrixTN100_error.xls', sheet_name='Sheet')
y_error = pd.read_excel(r'C:/Private/Subjects/NCKH/CodeAI_WLS/Tonghop/TN/OutputVoltageTN100_error.xls', sheet_name='Sheet1')

X_testV2=
print("X_test", X_testV.iloc[:,4:4])
# print("yV", yV)
#V