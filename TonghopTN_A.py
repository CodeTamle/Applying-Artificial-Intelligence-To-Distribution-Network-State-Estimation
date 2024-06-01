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
X_train, X_test, yA_train, yA_test = train_test_split(X, yA, random_state=3, test_size=0.2)
RFRforA = RandomForestRegressor(random_state=1, max_depth= 3).fit(X_train, yA_train)
stop = time.time()
total_time = (stop - start)*1000
print('TimeStamp_RFR: ',total_time, "ms")
#Data plot
RFR_A=RFRforA.predict(X_test)
RFR_A=RFR_A.mean(0)
RFR_A=np.transpose(RFR_A)

#MLP==============================================================================================

#test 20% - train 80%
X_trainV, X_testV, yV_train, yV_test = train_test_split(X, yV, random_state=None, test_size=0.2)
X_trainA, X_testA, yA_train, yA_test = train_test_split(X, yA, random_state=None, test_size=0.2)
# val 10% test 10%
# Xval , X_test2 , yval , yV_test2 = train_test_split(X_test,yV_test , test_size=0.5)

#model
model = Sequential()
model.add(Dense(80, input_shape = (X_trainV.shape[1],),activation='softmax'))
model.add(Dense(100, activation= 'relu'))
model.add(Dense(82, activation= 'relu'))
model.add(Dense(41)) # for nbus
learning_rate = 0.01
optimizer = optimizers.Adam(learning_rate)
model.compile(loss='mse',
                optimizer=optimizer,metrics='mse')
model.summary()

# =====================================================
train_time = time.time()
stop = time.time()
total_time = (stop - start)*1000
print('TimeStamp: ',total_time, "ms")
# LOAD====================================================
model = load_model('C:/Private/Subjects/NCKH/CodeAI_WLS/MLP_TNA.h5')
# model = load_model('C:/Private/Subjects/NCKH/CodeAI_WLS/MLP_TNA_25.h5')
# model = load_model('C:/Private/Subjects/NCKH/CodeAI_WLS/MLP_TNA_50.h5')
# model = load_model('C:/Private/Subjects/NCKH/CodeAI_WLS/MLP_TNA_50_25.h5')
# model = load_model('C:/Private/Subjects/NCKH/CodeAI_WLS/MLP_TNA_75_50.h5')
#=============================================================

#V
resultV = model.predict(X_testV)
# result = result.reshape(50,10)
MLP_V=resultV.mean(0)
MLP_V = np.transpose(MLP_V)

# Angle
resultA = model.predict(X_testA)
# result = result.reshape(50,10)
MLP_A=resultA.mean(0)
MLP_A = np.transpose(MLP_A)
print('MLP_V= ', MLP_V)
#plot

ppopt = ppoption(PF_ALG=1)  # Newton's method
r = runpf(caseTN(), ppopt)
Apf = r[0].get('bus')[:, 8]

# print('Apf= ', Apf)
# in kq 
i = np.ones((41, 1), dtype=int)
for n in range(41):
    i[n] = n+1


ssA_MLP = abs(MLP_A - Apf)/abs(Apf) * 100
ssA_RFR = abs(RFR_A - Apf)/abs(Apf) * 100
ssA_WLS = abs(A_WLS - Apf)/abs(Apf) * 100

# print('MLP_A= ', MLP_A)
# print('RFR_A= ', RFR_A)
# print('A_WLS= ', A_WLS)
# print('ssA_WLS= ', ssA_WLS)
# print('Apf= ', Apf)

# ssMLP = ssA_MLP / Apf
# J = 0
# for n in range(41):
#     J = J + ssA_MLP[n]**2
# J = J/41
# print("Sai so du bao MLP: ",J)

# # ssRFR = ssA_RFR / Apf
# J = 0
# for n in range(41):
#     J = J + ssA_RFR[n]**2
# J = J/41
# print("Sai so du bao RFR: ",J)

# A

# print('ssA_MLP= ', ssA_MLP)
# print('ssA_WLS= ', ssA_WLS)
plt.plot(i, ssA_MLP, 'bo-', label='MLP')
plt.plot(i, ssA_RFR, 'yo-', label='RFR')
plt.plot(i, ssA_WLS, 'go-', label='WLS')
# plt.title('Sai số góc pha của của lưới Tây Ninh')
plt.xlabel('Bus')
plt.ylabel('Error (%)')
plt.legend(loc='best')
# plt.show()

# plt.plot(i, MLP_A, 'bo-', label='MLP')
# plt.plot(i, RFR_A, 'yo-', label='RFR')
# plt.plot(i, WLS_A, 'go-', label='WLS')
# plt.plot(i, Apf, 'ro-', label='True Value')
# plt.title('Góc pha lưới Tây Ninh')
# plt.xlabel('Bus')
# plt.ylabel('Angle (Degree)')
# plt.legend(loc='best')
plt.show()