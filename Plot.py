from WLS import wls
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pypower.api import case30, ppoption, runpf, case14, case118
from matplotlib.pylab import rcParams
from case118_n import case118n
import time
rcParams['figure.figsize'] = 10, 5
nbus = 118
start = time.time()
# DataSh = pd.read_excel(
#     r'C:/Private/Subjects/NCKH/Code/Code118/Data.xlsx', sheet_name='DataShunt118')
# DataL = pd.read_excel(r'C:/Private/Subjects/NCKH/Code/Code118/Data.xlsx',
#                       sheet_name='DataLine118')  # Lấy dữ liệu lưới điện
# DataM = pd.read_excel(
#     r'C:/Private/Subjects/NCKH/Code/Code118/Data.xlsx', sheet_name='DataBus118')
# Lấy dữ liệu
DataSh = pd.read_excel(
    r'C:/Private/Subjects/NCKH/CodeAI_WLS/WLS/IEEE118/DataIEEE118.xlsx', sheet_name='DataShunt118')
DataL = pd.read_excel(r'C:/Private/Subjects/NCKH/CodeAI_WLS/WLS/IEEE118/DataIEEE118.xlsx',
                      sheet_name='DataLine118')  # Lấy dữ liệu lưới điện
DataM = pd.read_excel(
    r'C:/Private/Subjects/NCKH/CodeAI_WLS/WLS/IEEE118/DataIEEE118.xlsx', sheet_name='DataBus118n')
# Lấy dữ liệu

V, delta, ybus, G, B = wls(DataL, DataSh, DataM)

stop = time.time()

total_time = (stop - start)*1000
print('TimeStamp: ',total_time, "ms")

# print('V:/n', V)
# print('Delta:/n', delta)

ppopt = ppoption(PF_ALG=1)  # Newton's method
r = runpf(case118n(), ppopt)
Vpf = r[0].get('bus')[:, 7]
Apf = r[0].get('bus')[:, 8]
i = np.ones((nbus, 1), dtype=int)
for n in range(nbus):
    i[n] = n+1

# fig, ax = plt.subplots(nrows=2, ncols=1, figsize=(20, 5))
# plt.plot(i, V, 'g*-.', label='WLS')
# plt.plot(i, Vpf, 'ro--', label='Value')
# plt.title('So sánh điện áp giữa WLS vs Lưới')
# plt.xlabel('Bus')
# plt.ylabel('Voltage (p.u)')
# plt.get_legend()

plt.plot(i, delta, 'b*-', label='WLS')
plt.plot(i, Apf, 'ro--', label='Value')
plt.title('So sánh góc pha giữa WLS vs Lưới')
plt.xlabel('Bus')
plt.ylabel('Angle (degree)')
plt.legend()
plt.show()

# print('ybus[28]', ybus[27])
# print("Re/n", G)
