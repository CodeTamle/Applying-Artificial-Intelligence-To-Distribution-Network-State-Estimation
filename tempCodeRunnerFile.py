_V, 'bo-', label='MLP')
# plt.plot(i, RFR_V, 'yo-', label='RFR')
# plt.plot(i, WLS_V, 'go-', label='WLS')
plt.plot(i, V_error, 'go-', label='Error Data')
plt.plot(i, Vpf, 'ro-', label='True Value')
plt.title('Điện áp lưới Tây Ninh')
plt.xlabel('Bus')
plt.ylabel('Voltage (p.u)')
plt.legend(loc='best')
plt.show()