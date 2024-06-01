from WLS import wls
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import time
# from matplotlib.pylab import rcParams
from case118_n import case118n
from pypower.api import ppoption, runpf
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from keras.models import Sequential, load_model
from keras.layers import Dense, Activation, Dropout
from keras import optimizers
from matplotlib.pylab import rcParams
rcParams['figure.figsize'] = 15, 5


nbus=118

#WLS
# plot data real

# DataTN = pd.read_excel(r'C:/Private/Subjects/NCKH/Code/TN/DataTN.xlsx', sheet_name='Databus')

DataSh = pd.read_excel(
    r'C:/Private/Subjects/NCKH/CodeAI_WLS/Tonghop/IEEE118/DataIEEE118.xlsx', sheet_name='DataShunt118')
DataL = pd.read_excel(r'C:/Private/Subjects/NCKH/CodeAI_WLS/Tonghop/IEEE118/DataIEEE118.xlsx',
                      sheet_name='DataLine118')  # Lấy dữ liệu lưới điện
DataM = pd.read_excel(
    r'C:/Private/Subjects/NCKH/CodeAI_WLS/Tonghop/IEEE118/DataIEEE118.xlsx', sheet_name='DataBus118n')
# Lấy dữ liệu

WLS_V, WLS_A, ybus, G, B = wls(DataL, DataSh, DataM)
V_WLS = np.array(WLS_V)
V_WLS = np.reshape(WLS_V,(118,))

#Read data
X = pd.read_excel(r'C:/Private/Subjects/NCKH/CodeAI_WLS/Tonghop/IEEE118/DataInput_118.xlsx', sheet_name='Sheet1') 
y = pd.read_excel(r'C:/Private/Subjects/NCKH/CodeAI_WLS/Tonghop/IEEE118/DataOut_118.xls', sheet_name='Sheet1') 
yV=y.iloc[:,:118]
yA=y.iloc[:,119:237]
#RFR===========================================================
start = time.time()

X_train, X_test, yV_train, yV_test = train_test_split(X, yV, random_state=3, test_size=0.2)
RFRforV = RandomForestRegressor(random_state=1, max_depth= 4 ).fit(X_train, yV_train)
stop = time.time()
total_time = (stop - start)*1000
print('TimeStamp_RFR: ',total_time, "ms")

#Data plot
RFR_V=RFRforV.predict(X_test)
RFR_V=RFR_V.mean(0)
RFR_V=np.transpose(RFR_V)


#MLP==============================================================================================================
#test 20% - train 80%
X_trainV, X_testV, yV_train, yV_test = train_test_split(X, yV, random_state=3, test_size=0.2)
X_trainA, X_testA, yA_train, yA_test = train_test_split(X, yA, random_state=3, test_size=0.2)
# val 10% test 10%
# Xval , X_test2 , yval , yV_test2 = train_test_split(X_test,yV_test , test_size=0.5)
#model
model = Sequential()
model.add(Dense(80, input_shape = (X_trainA.shape[1],),activation='softmax'))
model.add(Dropout(0.2))
model.add(Dense(120, activation= 'relu'))
model.add(Dropout(0.2))
model.add(Dense(80, activation= 'relu'))
model.add(Dropout(0.2))
model.add(Dense(40, activation= 'relu'))
model.add(Dense(118)) # for nbus
learning_rate = 0.01
optimizer = optimizers.Adam(learning_rate)
model.compile(loss='mse',
                optimizer=optimizer, metrics= ['accuracy'])
model.summary()

# LOAD====================================================
model = load_model('C:/Private/Subjects/NCKH/CodeAI_WLS/MLP_model118V.h5')
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

#plot

ppopt = ppoption(PF_ALG=1)  # Newton's method
r = runpf(case118n(), ppopt)
Vpf = r[0].get('bus')[:, 7]


# in kq 
i = np.ones((118, 1), dtype=int)
for n in range(118):
    i[n] = n+1

ssV_MLP = abs(MLP_V - Vpf)
ssV_RFR = abs(RFR_V - Vpf)
ssV_WLS = abs(V_WLS - Vpf)


print('MLP_V', MLP_V)
print('RFR_V', RFR_V)
print('V_WLS', V_WLS)

# ssMLP = ssV_MLP / Vpf
# J = 0
# for n in range(118):
#     J = J + ssV_MLP[n]**2
# J = J/118
# print("Sai so du bao MLP: ",J)

# ssRFR = ssV_RFR / Vpf
# J = 0
# for n in range(118):
#     J = J + ssV_RFR[n]**2
# J = J/118
# print("Sai so du bao RFR: ",J)
    #Create plot V    
# plt.plot(i, ssV_MLP, 'go-', label='MLP')
# plt.plot(i, ssV_WLS, 'ro-', label='WLS')
# plt.title('Sai số điện áp của lưới IEEE118')
# plt.xlabel('Bus')
# plt.ylabel('Voltage (P.u)')
# plt.legend(loc='best')

plt.plot(i, MLP_V, 'bo-', label='MLP')
plt.plot(i, RFR_V, 'yo-', label='RFR')
# plt.plot(i, WLS_V, 'go-', label='WLS')
plt.plot(i, Vpf, 'ro-', label='True Value')
plt.title('Điện áp của lưới IEEE118')
plt.xlabel('Bus')
plt.ylabel('Voltage (p.u)')
plt.legend(loc='best')
plt.show()



