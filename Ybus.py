import numpy as np


def ybus(dataL, dataS):
  # labels
    fb = dataL['from']
    tb = dataL['to']
    r = dataL['R']
    x = dataL['X']
    b = dataL['B']
    a = dataL['tap a']
  # cac dai luong tro
    z = r+1j*x
    y = 1/z
    b = 1j*b
    nbranch = len(fb)
    nbus = int(max(max(fb), max(tb)))
    ybus = np.zeros((nbus, nbus), dtype=complex)
    fb -= 1
    tb -= 1
# giua cac nhanh
    for k in range(nbranch):
        ybus[int(fb[k]), int(tb[k])] -= y[k]/a[k]
        ybus[int(tb[k]), int(fb[k])] = ybus[int(fb[k]), int(tb[k])]

# tinh duong cheo trong ma tran Ybus
    for m in range(nbus):
        for n in range(nbranch):
            if fb[n] == m:
                # print(fb[n],'  ',a[n],' ',m,' ',y[n])
                ybus[m, m] += y[n]/(a[n]**2) + b[n]/2
            elif tb[n] == m:
                ybus[m, m] += y[n] + b[n]/2
# suseptance of shunt
    Sus = dataS['Susceptance']
    P_shunt = (dataS['Bus']-1)
    for i in range(len(Sus)):
        ybus[int(P_shunt[i]), int(P_shunt[i])] += 1j*Sus[i]

    return ybus
# ??


def Bs(data):
    fb = data['from']
    tb = data['to']
    r = data['R']
    x = data['X']
    b = data['B']
    a = data['tap a']
    z = r+1j*x
    y = 1/z
    nbus = int(max(max(fb+1), max(tb+1)))  # no. of buses...
    nbranch = len(fb+1)  # no. of branches...
    bbus = np.zeros((nbus, nbus))
    for k in range(nbranch):
        if a[k] == 1:
            bbus[int(fb[k]), int(tb[k])] = b[k]
            bbus[int(tb[k]), int(fb[k])] = bbus[int(fb[k]), int(tb[k])]
        else:
            bbus[int(fb[k]), int(tb[k])] = 2*y[k].imag*(1-a[k])/a[k]**2
            bbus[int(tb[k]), int(fb[k])] = 2*y[k].imag*(a[k]-1)/a[k]
    return bbus
