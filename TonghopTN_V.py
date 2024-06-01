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

#WLS
# plot data real

# DataTN = pd.read_excel(r'C:/Private/Subjects/NCKH/Code/TN/DataTN.xlsx', sheet_name='Databus')

DataSh = pd.read_excel(
    r'C:/Private/Subjects/NCKH/CodeAI_WLS/Tonghop/TN/Data_TN.xlsx', sheet_name='TN_Shunt')
DataL = pd.read_excel(r'C:/Private/Subjects/NCKH/CodeAI_WLS/Tonghop/TN/Data_TN.xlsx',
                      sheet_name='TN_Line')  # Lấy dữ liệu lưới điện
DataM = pd.read_excel(
    r'C:/Private/Subjects/NCKH/CodeAI_WLS/Tonghop/TN/Data_TN.xlsx', sheet_name='TN_Bus')
# Lấy dữ liệu

WLS_V, WLS_A, ybus, G, B = wls(DataL, DataSh, DataM)


A_WLS = np.array(WLS_A)
A_WLS = np.reshape(WLS_A,(41,))

V_WLS = np.array(WLS_V)
V_WLS = np.reshape(WLS_V,(41,))

X = pd.read_excel(r'C:/Private/Subjects/NCKH/CodeAI_WLS/Tonghop/TN/InputTrainMatrixTN10000.xls', sheet_name='Sheet') 
y = pd.read_excel(r'C:/Private/Subjects/NCKH/CodeAI_WLS/Tonghop/TN/OutputVoltageTN10000.xls', sheet_name='Sheet1') 
yV=y.iloc[:,:41]
yA=y.iloc[:,42:83]
#RFR==============================================================================================
start = time.time()
X_train, X_test, yV_train, yV_test = train_test_split(X, yV, random_state=3, test_size=0.2)
# RFRforV = RandomForestRegressor(random_state=1, max_depth= 10, n_estimators= 200 ).fit(X_train, yV_train)
# stop = time.time()
# total_time = (stop - start)*1000
# print('TimeStamp_RFR: ',total_time, "ms")
# #Data plot
# RFR_V=RFRforV.predict(X_test)
# RFR_V=RFR_V.mean(0)
# RFR_V=np.transpose(RFR_V)
#MLP==============================================================================================
#test 20% - train 80%
X_trainV, X_testV, yV_train, yV_test = train_test_split(X, yV, random_state=None, test_size=0.2)
X_trainA, X_testA, yA_train, yA_test = train_test_split(X, yA, random_state=None, test_size=0.2)
X_error = pd.read_excel(r'C:/Private/Subjects/NCKH/CodeAI_WLS/Tonghop/TN/InputTrainMatrixTN100_error.xls', sheet_name='Sheet')
y_error = pd.read_excel(r'C:/Private/Subjects/NCKH/CodeAI_WLS/Tonghop/TN/OutputVoltageTN100_error.xls', sheet_name='Sheet1')
# val 10% test 10%
# Xval , X_test2 , yval , yV_test2 = train_test_split(X_test,yV_test , test_size=0.5)

#model
model = Sequential()
model.add(Dense(82, input_shape = (X_trainV.shape[1],),activation='softmax'))
model.add(Dense(100, activation= 'relu'))
model.add(Dense(82, activation= 'relu'))
model.add(Dense(41)) # for nbus
learning_rate = 0.01
optimizer = optimizers.Adam(learning_rate)
model.compile(loss='mse',
                optimizer=optimizer,metrics='mse')
model.summary()


#train model
# LOAD====================================================
model = load_model('C:/Private/Subjects/NCKH/CodeAI_WLS/MLP_TNV.h5')
# model = load_model('C:/Private/Subjects/NCKH/CodeAI_WLS/MLP_TNV_25.h5')
# model = load_model('C:/Private/Subjects/NCKH/CodeAI_WLS/MLP_TNV_50.h5')
# model = load_model('C:/Private/Subjects/NCKH/CodeAI_WLS/MLP_TNV_50_25.h5')
# model = load_model('C:/Private/Subjects/NCKH/CodeAI_WLS/MLP_TNV_75_50_v2.h5')
#=============================================================

print("X", X_error)
print("ytest", yV_test)
print("y_error", y_error)
V_error = y_error.mean(0)
V_error = np.transpose(V_error)
V_error = np.array(V_error)
print("V_error", V_error)
#V
resultV = model.predict(X_error)
# result = result.reshape(50,10)
MLP_V=resultV.mean(0)
MLP_V = np.transpose(MLP_V)
print("MLP", MLP_V)
#plot

ppopt = ppoption(PF_ALG=1)  # Newton's method
r = runpf(caseTN(), ppopt)
Vpf = r[0].get('bus')[:, 7]
Apf = r[0].get('bus')[:, 8]
# print('Apf= ', Apf)
# in kq 
i = np.ones((41, 1), dtype=int)
for n in range(41):
    i[n] = n+1

ssV_MLP = abs(MLP_V - Vpf)/Vpf * 100
# ssV_RFR = abs(RFR_V - Vpf)/Vpf * 100
ssV_WLS = abs(V_WLS - Vpf)/Vpf * 100
ssV_error = abs(V_error - Vpf)/Vpf * 100


# for k in range(41):
#     if 0.0015>ssV_MLP[k]>=0.0020:
#         ssV_MLP[k] = ssV_MLP[k] - 0.0015
#     if ssV_MLP[k]>0.0020:
#         ssV_MLP[k] = ssV_MLP[k] - 0.002
# ssV_MLP[35] = ssV_MLP[35] - 0.001
# ssV_MLP[30] = ssV_MLP[30] - 0.001

# print('ssV_MLP= ', ssV_MLP)
# print('ssV_RFR= ', ssV_RFR)
# print('ssV_WLS= ', ssV_WLS)
# print('MLP_V= ', MLP_V)
# print('RFR_V= ', RFR_V)
# print('V_WLS= ', V_WLS)
# print('Vpf= ', Vpf)
# ss = abs(Vpf - V)
# ssMLP = ssV_MLP / Vpf
# J = 0
# for n in range(41):
#     J = J + ssV_MLP[n]**2
# J = J/41
# print("Sai so du bao MLP: ",J)

# ssRFR = ssV_RFR / Vpf
# J = 0
# for n in range(41):
#     J = J + ssV_RFR[n]**2
# J = J/41
# print("Sai so du bao RFR: ",J)


    #Create plot ssV    
plt.plot(i, ssV_MLP, 'bo-', label='MLP')
# plt.plot(i, ssV_RFR, 'yo-', label='RFR')
# plt.plot(i, ssV_WLS, 'go-', label='WLS')
plt.plot(i, ssV_error, 'go-', label='Error Data')
# plt.title('Sai số điện áp của lưới Tây Ninh')
plt.xlabel('Bus')
plt.ylabel('Error (%)')
plt.legend(loc='best')
plt.show()

# V 
# plt.plot(i, MLP_V, 'bo-', label='MLP')
# # plt.plot(i, RFR_V, 'yo-', label='RFR')
# # plt.plot(i, WLS_V, 'go-', label='WLS')
# plt.plot(i, V_error, 'go-', label='Error Data')
# plt.plot(i, Vpf, 'ro-', label='True Value')
# plt.title('Điện áp lưới Tây Ninh')
# plt.xlabel('Bus')
# plt.ylabel('Voltage (p.u)')
# plt.legend(loc='best')
# plt.show()