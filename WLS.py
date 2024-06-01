from Ybus import ybus, Bs
import numpy as np
import math


def wls(DataL, DataSh, DataM):

    Ybus = ybus(DataL, DataSh)
    nbus = len(Ybus)
    G = Ybus.real
    B = Ybus.imag
    bs = Bs(DataL)
    fbus = DataM['From']  # From bus
    fbus = fbus.values
    tbus = DataM['To']  # To bus
    tbus = tbus.values
    Measure = DataM['Measurement']
    # z = h + e
    Val = DataM['Values'].values  # gia tri phep do
    z = np.array(Val).reshape(np.size(Measure), 1)
    Ri = np.diag(DataM['Rii'])  # sai so cua phep do

    # phan loai phep do : 1-V 2-Pi 3-Qi 4-Pij 5-Qij 6-Iij
    typeM = DataM['Type']
    # khoi tao cac gi tri ban dau
    V = np.ones((nbus, 1))
    delta = np.zeros((nbus, 1))
    # delta[0] = 10.682*math.pi/180  # 118
    delta[0] = 30*math.pi/180  # 118
    # delta[0] = 0  # 
    E = np.bmat([[delta[1:]], [V]])
    # phan loai phep do
    vi = np.where(typeM == 1)  # điện áp
    vi = np.transpose(vi)
    ppi = np.where(typeM == 2)  # công suất thực tại nút
    ppi = np.transpose(ppi)
    qi = np.where(typeM == 3)  # công suất phản kháng tại nút
    qi = np.transpose(qi)
    pf = np.where(typeM == 4)  # công suất thực trên đường dây
    pf = np.transpose(pf)
    qf = np.where(typeM == 5)  # công suất phản kháng trên đường dây
    qf = np.transpose(qf)

    # so phep do tai nut va duong day
    nvi = len(vi)
    npi = len(ppi)
    nqi = len(qi)
    npf = len(pf)
    nqf = len(qf)

    # cac gia tri lap ban dau
    iter = 1
    dEE = 1
    ss = 1

    while(ss > 1e-5 and iter <= 50):
        h1 = np.zeros((nvi, 1))
        # print('h1',h1)
        for i in range(nvi):
            m = int(fbus[vi[i]])-1
            h1[i] = V[m]
        # print('h1 before',h1)
        h1.resize((nvi, 1))
        # print('h1 after',h1)
        # h1 = V[int(fbus[vi])-1] #Hàm biểu diễn của Vi
        # h1.resize((nvi,1))
        h2 = np.zeros((npi, 1))  # Hàm biểu diễn của Pi
        h3 = np.zeros((nqi, 1))  # Hàm biểu diễn của Qi
        h4 = np.zeros((npf, 1))  # Hàm biểu diễn của Pij
        h5 = np.zeros((nqf, 1))  # Hàm biểu diễn của Qij
        # dao ham P theo goc pha va theo V tai nut
        H21 = np.zeros((npi, nbus-1))
        H22 = np.zeros((npi, nbus))
# dao ham Q theo goc pha va theo V tai nut
        H31 = np.zeros((nqi, nbus-1))
        H32 = np.zeros((nqi, nbus))
# dao ham P theo goc pha va theo V cac nhanh
        H41 = np.zeros((npf, nbus-1))
        H42 = np.zeros((npf, nbus))
# dao ham P theo goc pha va theo V cac nhanh
        H51 = np.zeros((nqf, nbus-1))
        H52 = np.zeros((nqf, nbus))

        for i in range(npi):
            m = int(fbus[ppi[i]])-1
            for k in range(nbus):
                h2[i] = h2[i] + V[m]*V[k]*(G[m, k]*math.cos(delta[m]-delta[k]) +
                                           B[m, k]*math.sin(delta[m]-delta[k]))

        for i in range(nqi):
            m = int(fbus[qi[i]])-1
            for k in range(nbus):
                h3[i] = h3[i] + V[m]*V[k]*(G[m, k]*math.sin(delta[m]-delta[k]) -
                                           B[m, k]*math.cos(delta[m]-delta[k]))

        for i in range(npf):
            m = int(fbus[pf[i]])-1
            n = int(tbus[pf[i]])-1
            h4[i] = (-V[m]**2)*G[m, n] - V[m]*V[n]*(-G[m, n]*math.cos(delta[m]-delta[n]) -
                                                    B[m, n]*math.sin(delta[m]-delta[n]))

        for i in range(nqf):
            m = int(fbus[qf[i]])-1
            n = int(tbus[qf[i]])-1
            h5[i] = (-V[m]**2)*(-B[m, n]+bs[m, n]/2) - V[m]*V[n]*(-G[m, n]*math.sin(delta[m]-delta[n]) +
                                                                  B[m, n]*math.cos(delta[m]-delta[n]))

        h = np.bmat([[h1], [h2], [h3], [h4], [h5]])
        e = z - h
        H11 = np.zeros((nvi, nbus-1))
        H12 = np.zeros((nvi, nbus))
        # print('e1 before',e)
# Đạo hàm của V trong ma trận Jacobian
        for k in range(nvi):
            for n in range(nbus):
                if n == fbus[k]-1:
                    H12[k, n] = 1

# Đạo hàm của P nút trong ma trận Jacobian
        for i in range(npi):
            m = int(fbus[ppi[i]])-1
            for k in range(nbus-1):
                if k+1 == m:
                    for n in range(nbus):
                        H21[i, k] = H21[i, k] + V[m] * V[n]*(-G[m, n]*math.sin(
                            delta[m]-delta[n]) + B[m, n]*math.cos(delta[m]-delta[n]))
                    H21[i, k] = H21[i, k] - (V[m]**2)*B[m, m]
                else:
                    H21[i, k] = V[m] * V[k+1]*(G[m, k+1]*math.sin(
                        delta[m]-delta[k+1]) - B[m, k+1]*math.cos(delta[m]-delta[k+1]))

        for i in range(npi):
            m = int(fbus[ppi[i]])-1
            for k in range(nbus):
                if k == m:
                    for n in range(nbus):
                        H22[i, k] = H22[i, k] + V[n] * \
                            (G[m, n]*math.cos(delta[m]-delta[n]) +
                             B[m, n]*math.sin(delta[m]-delta[n]))
                    H22[i, k] = H22[i, k] + V[m]*G[m, m]
                else:
                    H22[i, k] = V[m]*(G[m, k]*math.cos(delta[m]-delta[k]) +
                                      B[m, k]*math.sin(delta[m]-delta[k]))
# Đạo hàm của Q nút trong ma trận Jacobian
        for i in range(nqi):
            m = int(fbus[qi[i]])-1
            for k in range(nbus-1):
                if k+1 == m:
                    for n in range(nbus):
                        H31[i, k] = H31[i, k] + V[m]*V[n] * \
                            (G[m, n]*math.cos(delta[m]-delta[n]) +
                             B[m, n]*math.sin(delta[m]-delta[n]))
                    H31[i, k] = H31[i, k] - (V[m]**2)*G[m, m]
                else:
                    H31[i, k] = V[m] * V[k+1]*(-G[m, k+1]*math.cos(
                        delta[m]-delta[k+1]) - B[m, k+1]*math.sin(delta[m]-delta[k+1]))

        for i in range(nqi):
            m = int(fbus[qi[i]])-1
            for k in range(nbus):
                if k == m:
                    for n in range(nbus):
                        H32[i, k] = H32[i, k] + V[n] * \
                            (G[m, n]*math.sin(delta[m]-delta[n]) -
                             B[m, n]*math.cos(delta[m]-delta[n]))
                    H32[i, k] = H32[i, k] - V[m]*B[m, m]
                else:
                    H32[i, k] = V[m]*(G[m, k]*math.sin(delta[m]-delta[k]) -
                                      B[m, k]*math.cos(delta[m]-delta[k]))
# Đạo hàm của P nhánh trong ma trận Jacobian
        for i in range(npf):
            m = int(fbus[pf[i]])-1
            n = int(tbus[pf[i]])-1
            for k in range(nbus-1):
                if k+1 == m:
                    H41[i, k] = V[m] * V[n] * \
                        (-G[m, n]*math.sin(delta[m]-delta[n]) +
                         B[m, n]*math.cos(delta[m]-delta[n]))
                elif k+1 == n:
                    H41[i, k] = -V[m] * V[n] * \
                        (-G[m, n]*math.sin(delta[m]-delta[n]) +
                         B[m, n]*math.cos(delta[m]-delta[n]))
                else:
                    H41[i, k] = 0

        for i in range(npf):
            m = int(fbus[pf[i]])-1
            n = int(tbus[pf[i]])-1
            for k in range(nbus):
                if k == m:
                    H42[i, k] = -V[n]*(-G[m, n]*math.cos(delta[m]-delta[n]) -
                                       B[m, n]*math.sin(delta[m]-delta[n])) - 2*G[m, n]*V[m]
                elif k == n:
                    H42[i, k] = -V[m]*(-G[m, n]*math.cos(delta[m] -
                                       delta[n]) - B[m, n]*math.sin(delta[m]-delta[n]))
                else:
                    H42[i, k] = 0
# Đạo hàm của Q nhánh trong ma trận Jacobian
        for i in range(nqf):
            m = int(fbus[pf[i]])-1
            n = int(tbus[pf[i]])-1
            for k in range(nbus-1):
                if k+1 == m:
                    H51[i, k] = -V[m] * V[n] * \
                        (-G[m, n]*math.cos(delta[m]-delta[n]) -
                         B[m, n]*math.sin(delta[m]-delta[n]))
                elif k+1 == n:
                    H51[i, k] = V[m] * V[n] * \
                        (-G[m, n]*math.cos(delta[m]-delta[n]) -
                         B[m, n]*math.sin(delta[m]-delta[n]))
                else:
                    H51[i, k] = 0

        for i in range(nqf):
            m = int(fbus[pf[i]])-1
            n = int(tbus[pf[i]])-1
            for k in range(nbus):
                if k == m:
                    H52[i, k] = -V[n]*(-G[m, n]*math.sin(delta[m]-delta[n]) + B[m, n]
                                       * math.cos(delta[m]-delta[n])) - 2*V[m]*(-B[m, n] + bs[m, n]/2)
                elif k == n:
                    H52[i, k] = -V[m]*(-G[m, n]*math.sin(delta[m] -
                                       delta[n]) + B[m, n]*math.cos(delta[m]-delta[n]))
                else:
                    H52[i, k] = 0

        H = np.bmat([[H11, H12], [H21, H22], [
                    H31, H32], [H41, H42], [H51, H52]])
        Gm = H.transpose()*np.linalg.inv(Ri)*H  # G = H^T*Ri^-1*H
        # Sai số của các trọng số sau 1 lần lặp (G^-1 @ H^T @ Ri^-1 @ (z-h))*
        dE = np.linalg.pinv(Gm).dot(H.transpose()).dot(
            np.linalg.inv(Ri)).dot(e)
        # print('iter', iter)
        E = E + dE  # Cập nhật giá trị mới*
        delta[1:, ] = E[0:nbus-1, 0]
        iter = iter + 1  # Tăng số lần đếm vòng lặp
        ss = max(abs(dE[nbus-1:]))
        # print('ss', ss)
        V = np.array(E[nbus-1:])  # cập nhật giá trị cho điện áp
    delta = (180/math.pi)*delta
    print('V:\n', V)
    print('Delta:\n', delta)
    return V, delta, Ybus, G, B
