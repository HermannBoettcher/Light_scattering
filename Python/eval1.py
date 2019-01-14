import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
def exponenial_func(x, a, b, c):
    return a*np.exp(-b*x)+c

def linear_func(x, a, b):
    return a * x + b


data = np.loadtxt('tabular.csv')
data = np.transpose(data)


potentials = plt.figure(9, figsize=(5,5))
a=np.array([0, 1e-07])
plt.plot(data[4], data[5])
plt.plot(data[6], data[7])
plt.plot(data[8], data[9])
plt.plot(data[10], data[11])
plt.grid(True)
plt.xlabel('Distance in meters')
plt.ylabel('Light force potential in units of k_B*T')
plt.legend(('U_tweezers = 0,70  V', 'U_tweezers = 0,65 V', 'U_tweezers = 0,60 V', 'U_tweezers = 0,00 V'), loc='upper center')
plt.show()
potentials.savefig('potentials.pdf')

potentials12 = plt.figure(9, figsize=(5,5))
plt.plot(data[0], data[1])
plt.plot(data[2], data[3])
plt.grid(True)
plt.xlabel('Distance in meters')
plt.ylabel('Light force potential in units of k_B*T')
plt.legend(('d = 218 nm', 'd = 75 nm'), loc='lower center')
plt.show()
potentials12.savefig('potentials12.pdf')



min_index = np.argmin(data, axis=1)

exp_pot_70 = data[:, :min_index[5]]
lin_pot_70 = data[:, min_index[5]:]
#print(lin_pot_70[4])
stop_index = np.argwhere(lin_pot_70[4] == 9.87e-08)
#print(stop_index)
lin_pot_70 = lin_pot_70[:,:stop_index[0][0]]

popt700, pcov700 = curve_fit(linear_func, lin_pot_70[4], lin_pot_70[5], p0=(1, 1))

popt70, pcov70 = curve_fit(exponenial_func, exp_pot_70[4], exp_pot_70[5], p0=(2, 1, 0))

figure_70 = plt.figure(1, figsize=(5, 5))
plt.plot(data[4], data[5])
plt.plot(lin_pot_70[4], linear_func(lin_pot_70[4], *popt700))
plt.plot(exp_pot_70[4], exponenial_func(exp_pot_70[4], *popt70))
plt.xlabel('Distance in meters')
plt.ylabel('Potential in units of k_B*T')
plt.grid(True)
plt.legend(('V', 'V_lin', 'V_exp'), loc='upper right')
figure_70.savefig('70_potential.pdf')
#plt.show()


exp_pot_65 = data[:, :min_index[7]]
lin_pot_65 = data[:, min_index[7]:]
print(lin_pot_65[6])
stop_index = np.argwhere(lin_pot_65[6] == 4.13e-08)
lin_pot_65 = lin_pot_65[:,:stop_index[0][0]]

popt650, pcov650 = curve_fit(linear_func, lin_pot_65[6], lin_pot_65[7], p0=(1, 1))
popt65, pcov65 = curve_fit(exponenial_func, exp_pot_65[6], exp_pot_65[7], p0=(2, 1, 0))

figure_65 = plt.figure(2, figsize=(5, 5))
plt.plot(data[6], data[7])
plt.plot(lin_pot_65[6], linear_func(lin_pot_65[6], *popt650))
plt.plot(exp_pot_65[6], exponenial_func(exp_pot_65[6], *popt65))
plt.xlabel('Distance in meters')
plt.ylabel('Potential in units of k_B*T')
plt.grid(True)
plt.legend(('V', 'V_lin', 'V_exp'), loc='upper right')
figure_65.savefig('65_potential.pdf')
#plt.show()

exp_pot_60 = data[:, :min_index[9]]
lin_pot_60 = data[:, min_index[9]:]
#print(lin_pot_60[8])
stop_index = np.argwhere(lin_pot_60[8] == 8.88e-08)
lin_pot_60 = lin_pot_60[:,:stop_index[0][0]]

popt600, pcov600 = curve_fit(linear_func, lin_pot_60[8], lin_pot_60[9], p0=(1, 1))
popt60, pcov60 = curve_fit(exponenial_func, exp_pot_60[8], exp_pot_60[9], p0=(2, 1, 0))

figure_60 = plt.figure(3, figsize=(5, 5))
plt.plot(data[8], data[9])
plt.plot(lin_pot_60[8], linear_func(lin_pot_60[8], *popt600))
plt.plot(exp_pot_60[8], exponenial_func(exp_pot_60[8], *popt60))
plt.xlabel('Distance in meters')
plt.ylabel('Potential in units of k_B*T')
plt.grid(True)
plt.legend(('V', 'V_lin', 'V_exp'), loc='upper right')
figure_60.savefig('60_potential.pdf')
#plt.show()

exp_pot_75 = data[:, :min_index[3]]
lin_pot_75 = data[:, min_index[3]:]

popt750, pcov750 = curve_fit(linear_func, lin_pot_75[2], lin_pot_75[3], p0=(1, 1))

popt75, pcov75 = curve_fit(exponenial_func, exp_pot_75[2], exp_pot_75[3], p0=(2, 1, 0))

figure_75 = plt.figure(4, figsize=(5, 5))
plt.plot(data[2], data[3])
plt.plot(lin_pot_75[2], linear_func(lin_pot_75[2], *popt750))
plt.plot(exp_pot_75[2], exponenial_func(exp_pot_75[2], *popt75))
plt.xlabel('Distance in meters')
plt.ylabel('Potential in units of k_B*T')
plt.grid(True)
plt.legend(('V', 'V_lin', 'V_exp'), loc='upper right')
figure_75.savefig('75_potential.pdf')
#plt.show()


#exp_pot_76 = data[:, :min_index[1]]
#lin_pot_76 = data[:, min_index[1]:]

#lin_pars_76 = np.polyfit(lin_pot_76[0], lin_pot_76[1], 1)
#lin_func_76 = np.poly1d(lin_pars_76)

#popt76, pcov76 = curve_fit(exponenial_func, exp_pot_76[0], exp_pot_76[1], p0=(2, 1, 0))

#figure_76 = plt.figure(5, figsize=(10, 10))
#plt.plot(data[0], data[1])
#plt.plot(lin_pot_76[0], lin_func_76(lin_pot_76[0]))
#plt.plot(exp_pot_76[0], exponenial_func(exp_pot_76[0], *popt76))
#plt.xlabel('Distance in meters')
#plt.ylabel('Potential')
#figure_76.savefig('76_potential.pdf')
#plt.show()


exp_pot_0 = data[:, :min_index[11]]
lin_pot_0 = data[:, min_index[11]:]
#print(lin_pot_0[10])
stop_index = np.argwhere(lin_pot_0[10] == 5.33e-08)
lin_pot_0 = lin_pot_0[:,:stop_index[0][0]]

popt0, pcov0 = curve_fit(linear_func, lin_pot_0[10], lin_pot_0[11], p0=(1, 1))

popt00, pcov00 = curve_fit(exponenial_func, exp_pot_0[10], exp_pot_0[11], p0=(2, 1, 0))

print(popt0[0] * 1.38064852e-23 * 300)

figure_0 = plt.figure(6, figsize=(5, 5))
plt.plot(data[10], data[11])
plt.plot(lin_pot_0[10], linear_func(lin_pot_0[10], *popt0))
plt.plot(exp_pot_0[10], exponenial_func(exp_pot_0[10], *popt00))
plt.xlabel('Distance in meters')
plt.grid(True)
plt.ylabel('Potential in units of k_B*T')
plt.legend(('V', 'V_lin', 'V_exp'), loc='upper right')
figure_0.savefig('0_potential.pdf')
#plt.show()

#print(lin_pars_65)
#print(lin_pars_70)
#print(lin_pars_76)
#print(lin_pars_0)
#print(lin_func_0)

lin_pot_70[5] = lin_pot_70[5] - linear_func(lin_pot_70[4], *popt0)

figure_70_fl = plt.figure(7, figsize=(5,5))
plt.plot(lin_pot_70[4], linear_func(lin_pot_70[4], *popt700) - linear_func(lin_pot_70[4], *popt0))
plt.xlabel('Distance in meters')
plt.ylabel('Light force potential in units of k_B*T')
plt.grid(True)
plt.show()
figure_70_fl.savefig('70_light_force.pdf')

figure_fl = plt.figure(8, figsize=(5,5))
a=np.array([0, 1e-07])
plt.plot(a, a*popt700[0] - popt0[0] * a)
plt.plot(a, a * popt650[0] - a * popt0[0])
plt.plot(a, a * popt600[0] - a * popt0[0])
plt.grid(True)
plt.xlabel('Distance in meters')
plt.ylabel('Light force potential in arbitrary units')
plt.legend(('U_tweezers = 0,70  V', 'U_tweezers = 0,65 V', 'U_tweezers = 0,60 V'), loc='upper right')
plt.show()
figure_fl.savefig('light_forces.pdf')

print((popt700[0] - popt0[0])* 1.38064852e-23 * 300)
print((popt650[0] - popt0[0])* 1.38064852e-23 * 300)
print((popt600[0] - popt0[0])* 1.38064852e-23 * 300)