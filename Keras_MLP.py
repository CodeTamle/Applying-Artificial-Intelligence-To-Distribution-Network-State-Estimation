import keras
from keras.models import Sequential
from keras.layers import Dense, Activation
from keras import optimizers
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pypower.api import ppoption, runpf
from caseTN2 import caseTN
import time
from matplotlib.pylab import rcParams
rcParams['figure.figsize'] = 15, 5
start1 = time.time()
# chia data
X = pd.read_excel(r'C:/Private/Subjects/NCKH/CodeAI_WLS/Tonghop/TN/InputTrainMatrixTN10000-50.xls', sheet_name='Sheet') 
y = pd.read_excel(r'C:/Private/Subjects/NCKH/CodeAI_WLS/Tonghop/TN/OutputVoltageTN10000.xls', sheet_name='Sheet1') 
yV=y.iloc[:,:41]
yA=y.iloc[:,42:83]
#test 20% - train 80%
X_trainV, X_testV, yV_train, yV_test = train_test_split(X, yV, random_state=3, test_size=0.2)
X_trainA, X_testA, yA_train, yA_test = train_test_split(X, yA, random_state=3, test_size=0.2)
# val 10% test 10%
# Xval , X_test2 , yval , yV_test2 = train_test_split(X_test,yV_test , test_size=0.5)

#model
model = Sequential()
model.add(Dense(82, input_shape = (X_trainV.shape[1],),activation='softmax'))
model.add(Dense(164, activation= 'relu'))
model.add(Dense(41, activation= 'relu'))
model.add(Dense(41)) # for nbus
learning_rate = 0.01
optimizer = optimizers.Adam(learning_rate)
model.compile(loss='mse',
                optimizer=optimizer,metrics='mse')
model.summary()

#train model
model.fit(
        X_trainV, 
        yV_train,
        batch_size = 32,
        epochs=200,
        verbose=0,
        shuffle=True,steps_per_epoch = int(X.shape[0] / 32))


stop1= time.time()
total_time1 = (stop1 - start1)*1000
print('TimeStamp_Train: ',total_time1, "ms")

start = time.time()
# in kq 
#V
resultV = model.predict(X_testV)
# result = result.reshape(50,10)
V=resultV.mean(0)
V = np.transpose(V)
# Angle
resultA = model.predict(X_testA)
# result = result.reshape(50,10)
A=resultA.mean(0)
A = np.transpose(A)

stop = time.time()
total_time = (stop - start)*1000
print('TimeStamp_Run: ',total_time, "ms")

ppopt = ppoption(PF_ALG=1)  # Newton's method
r = runpf(caseTN(), ppopt)
Vpf = r[0].get('bus')[:, 7]
Apf = r[0].get('bus')[:, 8]

bV=yV_test
bV=bV.mean(0)
bV=np.transpose(bV)

bA=yA_test
bA=bA.mean(0)
bA=np.transpose(bA)
i=np.ones( (41, 1), dtype=int )
for n in range(41):
    i[n]=n+1

#Create plot V    
plt.plot(i, V, 'go-', label='MLP')
# plt.plot(i, bV, 'ro-', label='Test Value')
# plt.plot(i, Vpf, 'bo-', label='TrueValue')
plt.title('So sánh điện áp giữa MLP vs TrueValue')
plt.xlabel('Bus')
plt.ylabel('Voltage (p.u)')
plt.legend(loc='best')
# plt.savefig('So sánh điện áp IEEETN.png')
plt.show()

# plt.plot(i, A, 'go-', label='MLP')
# # plt.plot(i, bA, 'ro-', label='Test Value')
# # plt.plot(i, Apf, 'bo-', label='TrueValue')
# plt.title('So sánh góc pha giữa MLP vs TrueValue')
# plt.xlabel('Bus')
# plt.ylabel('Angle (Degree)')
# plt.legend(loc='best')
# # plt.savefig('So sánh góc pha IEEETN.png')
# plt.show()

# ss = abs(Vpf - V)
# ss1 = ss / Vpf
# J = 0
# for n in range(41):
#     J = J + ss[n]**2
# J = J/41

# plt.plot(i, ss1, 'bo-', label='Sai số')
# plt.title('Sai số điện áp giữa MLP vs TrueValue')
# plt.xlabel('Bus')
# plt.ylabel('Sai số (%)')
# plt.legend(loc='best')
# plt.show()

# ss = abs(Apf - A)
# ss1 = ss / abs(Apf)
# J = 0
# for n in range(41):
#     J = J + ss[n]**2
# J = J/41

# plt.plot(i, ss1, 'bo-', label='Sai số')
# plt.title('Sai số góc pha giữa MLP vs TrueValue')
# plt.xlabel('Bus')
# plt.ylabel('Sai số (%)')
# plt.legend(loc='best')
# plt.show()

# print("Sai so du bao: ",J)

# print('Vpf\n',Vpf)
# print('V\n',V)
# print('SS\n',ss)

# print('Apf\n',Apf)
# print('A\n',A)
# print('J\n',J)
