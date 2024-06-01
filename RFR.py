import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
from case118_n import case118n
from pypower.api import ppoption, runpf
import time

#start timer
start_time = time.time()
#Read data
X = pd.read_excel(r'C:/Private/Subjects/NCKH/CodeAI_WLS/AI/IEEE118/DataInput_118.xlsx', sheet_name='Sheet1') 
y = pd.read_excel(r'C:/Private/Subjects/NCKH/CodeAI_WLS/AI/IEEE118/DataOut_118.xls', sheet_name='Sheet1') 
yV=y.iloc[:,:118]
yA=y.iloc[:,119:237]

X_train, X_test, yV_train, yV_test = train_test_split(X, yV, random_state=3, test_size=0.1)
RFRforV = RandomForestRegressor(random_state=1).fit(X_train, yV_train)

#training Angle
X_train, X_test, yA_train, yA_test = train_test_split(X, yA, random_state=3, test_size=0.1)
RFRforA = RandomForestRegressor(random_state=1).fit(X_train, yA_train)


#timer train and save model
train_time = time.time()

ppopt = ppoption(PF_ALG=1)  # Newton's method
r = runpf(case118n(), ppopt)
Vpf = r[0].get('bus')[:, 7]
Apf = r[0].get('bus')[:, 8]

#Data plot
V=RFRforV.predict(X_test)
V=V.mean(0)
V=np.transpose(V)
bV=yV_test
bV=bV.mean(0)
bV=np.transpose(bV)

A=RFRforA.predict(X_test)
A=A.mean(0)
A=np.transpose(A)
bA=yA_test
bA=bA.mean(0)
bA=np.transpose(bA)

#stt plot
i=np.ones( (118, 1), dtype=int )
for n in range(118):
    i[n]=n+1

#Create plot V    
# plt.plot(i, V, 'go-', label='MLP')
# plt.plot(i, bV, 'ro-', label='True Value')
# plt.title('So sánh điện áp giữa MLP vs PF')
# plt.xlabel('Bus')
# plt.ylabel('Voltage (p.u)')
# plt.legend(loc='best')
# plt.show()

# ss = abs(Vpf - V)
# ss1 = ss / Vpf
# J = 0
# for n in range(118):
#     J = J + ss[n]**2
# J = J/118
# print('Sai so du bao: ',J)

# plt.plot(i, ss1, 'bo-', label='Sai số')
# plt.title('Sai số điện áp giữa RFR vs TrueValue')
# plt.xlabel('Bus')
# plt.ylabel('Sai số (%)')


#Create plot A
# plt.plot(i, A, 'go-', label='MLP')
# plt.plot(i, Apf, 'ro-', label='True Value')
# plt.title('So sánh góc pha giữa MLP vs PF')
# plt.xlabel('Bus')
# plt.ylabel('Angle (degree)')

ss = abs(Apf - A)
ss1 = ss/ abs(Apf)
J = 0
for n in range(118):
    J = J + ss[n]**2
J = J/118
print('Sai so du bao: ',J)
print('Training time: %f ms' % ((train_time - start_time) * 1000))

plt.plot(i, ss1, 'bo-', label='Sai số')
plt.title('Sai số góc pha giữa RFR vs TrueValue')
plt.xlabel('Bus')
plt.ylabel('Sai so (%)')

plt.legend(loc='best')
plt.show()
