from WLS import wls
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.pylab import rcParams
from caseTN2 import caseTN
from pypower.api import ppoption, runpf
nbus=41

# plot data real

# DataTN = pd.read_excel(r'C:/Private/Subjects/NCKH/Code/TN/DataTN.xlsx', sheet_name='Databus')

DataSh = pd.read_excel(
    r'C:/Private/Subjects/NCKH/Code/TN/Data_TN2.xlsx', sheet_name='TN_Shunt')
DataL = pd.read_excel(r'C:/Private/Subjects/NCKH/Code/TN/Data_TN2.xlsx',
                      sheet_name='TN_Line')  # Lấy dữ liệu lưới điện
DataM = pd.read_excel(
    r'C:/Private/Subjects/NCKH/Code/TN/Data_TN2.xlsx', sheet_name='TN_Bus')
# Lấy dữ liệu

V, delta, ybus, G, B = wls(DataL, DataSh, DataM)


i = np.ones((nbus, 1), dtype=int)
for n in range(nbus):
    i[n] = n+1

ppopt = ppoption(PF_ALG=4)  # Gauss's method
r = runpf(caseTN(), ppopt)
Vpf = r[0].get('bus')[:, 7]
Apf = r[0].get('bus')[:, 8]
# U = DataTN['U(PU)']
# Deg = DataTN['Angle (Deg)']
print("U=\n",V)
print("Angle=\n",delta)

fig, ax = plt.subplots(nrows=2, ncols=1, figsize=(20, 5))
ax[0].plot(i, Vpf, 'g*-.', label='Value')
ax[0].plot(i, V, 'ro--', label='WLS')
ax[0].set_title('So sánh điện áp giữa WLS vs Lưới')
ax[0].set_xlabel('Bus')
ax[0].set_ylabel('Voltage (p.u)')
ax[0].get_legend()

ax[1].plot(i, Apf, 'b*-', label='Value')
ax[1].plot(i, delta, 'yo--', label='WLS')
ax[1].set_title('So sánh góc pha giữa WLS vs Lưới')
ax[1].set_xlabel('Bus')
ax[1].set_ylabel('Angle (degree)')
fig.legend()
plt.show()