import numpy as np
import matplotlib.pyplot as plt
import math
from scipy.optimize import curve_fit
from scipy.stats import linregress

# Load data

# data without Glycerol
data_1a = np.genfromtxt('data/1a', delimiter=',', skip_header=22, usecols=(0, 1))
data_1b = np.genfromtxt('data/1b', delimiter=',', skip_header=22, usecols=(0, 1))
data_1c = np.genfromtxt('data/1c', delimiter=',', skip_header=22, usecols=(0, 1))
data_1d = np.genfromtxt('data/1d', delimiter=',', skip_header=22, usecols=(0, 1))
data_1e = np.genfromtxt('data/1e', delimiter=',', skip_header=22, usecols=(0, 1))

# data with Glycerol

data_2c = np.genfromtxt('data/2c', delimiter=',', skip_header=22, usecols=(0, 1))
data_3c = np.genfromtxt('data/3c', delimiter=',', skip_header=22, usecols=(0, 1))

# lifetime data

data_1a_lifetime = np.genfromtxt('data/1a_lifetime', delimiter=',', skip_header=10, usecols=(0, 1))
data_1e_lifetime = np.genfromtxt('data/1e_lifetime.txt', delimiter=',', skip_header=10, usecols=(0, 1))

# data from UV-Vis from Irina and Damaris

data_1a_UV_irina = np.genfromtxt('data/UVVis_1a.Sample.asc', skip_header=90, usecols=(0, 1))
data_1e_UV_irina = np.genfromtxt('data/UVVis_1e.Sample.asc', skip_header=90, usecols=(0, 1))
data_2c_UV_irina = np.genfromtxt('data/UVVis_2c.Sample.asc', skip_header=90, usecols=(0, 1))
data_3c_UV_irina = np.genfromtxt('data/UVVis_3c.Sample.asc', skip_header=90, usecols=(0, 1))




fig, ax1 = plt.subplots()
# Plot on the first axis (left)
ax1.plot(data_1a_UV_irina[:, 0], data_1a_UV_irina[:, 1], label='1a', color='blue')
ax1.plot(data_1e_UV_irina[:, 0], data_1e_UV_irina[:, 1], label='1e', color='green')
ax1.set_xlabel('Wavelength [nm]')
ax1.set_ylabel('UV/Vis Absorbance')
# Create a second y-axis on the right
ax2 = ax1.twinx()
# Plot on the second axis (right)
ax2.plot(data_1a[:, 0], data_1a[:, 1] / 200000, label='1a (scaled)', color='orange')
ax2.plot(data_1e[:, 0], data_1e[:, 1] / 200000, label='1e (scaled)', color='cyan')
ax2.set_ylabel('Scaled fluorescence intensity')
ax1.set_xlim(350, 650)
plt.savefig('plots/UVVis_Fluorescence_without_Glycerol.pdf')
plt.show()

fig, ax1 = plt.subplots()
# Plot on the first axis (left)
ax1.plot(data_2c_UV_irina[:, 0], data_2c_UV_irina[:, 1], label='2c', color='red')
ax1.plot(data_3c_UV_irina[:, 0], data_3c_UV_irina[:, 1], label='3c', color='purple')
ax1.set_xlabel('Wavelength [nm]')
ax1.set_ylabel('UV/Vis Absorbance')
# Create a second y-axis on the right
ax2 = ax1.twinx()
# Plot on the second axis (right)
ax2.plot(data_2c[:, 0], data_2c[:, 1] / 200000, label='2c (scaled)', color='magenta')
ax2.plot(data_3c[:, 0], data_3c[:, 1] / 200000, label='3c (scaled)', color='brown')
ax2.set_ylabel('Scaled fluorescence intensity')
ax1.set_xlim(350, 650)
plt.savefig('plots/UVVis_Fluorescence_with_glycerol.pdf')
plt.show()


# Calculate exact concentration of the quencher

m_flasks = [0, 30.31, 32.10, 31.01, 30.04]  # Mass of empty flasks (g)
m_flasks_full = [0, 32.31, 37.23, 38.129, 40.15]  # Mass of flasks with KI (g)
total_volume = 25  # Total volume of the solution (mL)

# Molar mass of KI
molar_mass_ki = 166.0028  # g/mol

# Calculate the mass of KI in each flask
mass_ki = [m_full - m_empty for m_empty, m_full in zip(m_flasks, m_flasks_full)]

# Calculate the total moles of KI in each flask
moles_ki = [mass / molar_mass_ki for mass in mass_ki]

# Convert the total volume from mL to L
total_volume_liter = total_volume / 1000  # 1 mL = 0.001 L

# Calculate the concentration of KI in each flask (in mol/L)
concentration_ki = [moles / total_volume_liter for moles in moles_ki]

# Print the concentrations
for i, conc in enumerate(concentration_ki, start=1):
    print(f'Flask {i}: Concentration of KI = {conc:.3f} mol/L')


# Stern-Volmer plot

# calculate highest intensity of 1a

i_0 = np.max(data_1a[:, 1])

# calculate highest intensities of 1b, 1c, 1d, 1e

i_q = []

for data in [data_1b, data_1c, data_1d, data_1e]:
    i_q.append(np.max(data[:, 1]))

print(i_q)

# calculate ratio of intensities

i_ratio = [(i_0 / i ) - 1 for i in i_q]



print(i_ratio)


plt.figure(5)
plt.scatter(concentration_ki[1:], i_ratio, c='k', s=20)

# Linear regression using linregress
slope_Volmer, intercept, r_value, p_value, std_err = linregress(concentration_ki[1:], i_ratio)

plt.plot(concentration_ki[1:], slope_Volmer * np.array(concentration_ki[1:]) + intercept, label='linear fit', c='r', linestyle='--')
plt.xlabel('Concentration of KI [mol/L]')
plt.ylabel('Ratio of intensities')
plt.savefig('plots/Stern_Volmer.pdf')
plt.show()

print(f'Slope of linear fit: {slope_Volmer:.3f} and y-intercept: {intercept:.3f}')
print(f'Standard error of the slope: {std_err:.3f}')

def calculate_k_diff(R, T, viscosity):
    return ((8*R*T)/(3*viscosity))/1000


k_diff = calculate_k_diff(8.314, 293.15, 0.001)

print(k_diff)


print(slope_Volmer)
print(std_err)

# Calculate fluorescence life time

lifetime = slope_Volmer/k_diff

print(lifetime)


def gauss_expo_convolution(data, a, mu, sigma, tau):
    res = []
    for x in data:
        expo = np.exp((sigma**2-2*tau*x+2*mu*tau)/(2*tau**2))
        ero = math.erfc((tau*(mu-x)+sigma**2)/(np.sqrt(2)*sigma*tau))
        res.append(expo * ero)
    return a * np.array(res)


#fitting lifetime of 1a and 1e


initial_guess_1e = [4000, 7.5, 1, 4.5]

params_1e, cov_1e = curve_fit(gauss_expo_convolution, data_1e_lifetime[:, 0], data_1e_lifetime[:, 1], p0=initial_guess_1e)


plt.figure(6)
plt.scatter(data_1e_lifetime[:, 0], data_1e_lifetime[:, 1], label='1e', s=0.5, c='k')
plt.plot(data_1e_lifetime[:, 0], gauss_expo_convolution(data_1e_lifetime[:, 0], *params_1e), label='fit', c='r', linestyle='--')
plt.xlabel('Time [ns]')
plt.ylabel('Fluorescence intensity [a.u.]')
#plt.legend()
plt.savefig('plots/Lifetime1e.pdf')
plt.show()

initial_guess_1a = [3750, 7.5, 1, 4.5]

params_1a, cov_1a = curve_fit(gauss_expo_convolution, data_1a_lifetime[:, 0], data_1a_lifetime[:, 1], p0=initial_guess_1a)

plt.figure(7)
plt.scatter(data_1a_lifetime[:, 0], data_1a_lifetime[:, 1], label='1a', s=0.5, c='k')
plt.plot(data_1a_lifetime[:, 0], gauss_expo_convolution(data_1a_lifetime[:, 0], *params_1a), label='fit', c='r', linestyle='--')
#plt.axvline(x=params_1a[1], color='r', linestyle='--', label='offset time')
plt.xlabel('Time [ns]')
plt.ylabel('Fluorescence intensity [a.u.]')
#plt.legend()
plt.savefig('plots/Lifetime1a.pdf')
plt.show()

print(f'Lifetime of 1a: {params_1a[3]:.3f} ns')
print(params_1a[3])
print(f'Lifetime of 1e: {params_1e[3]:.3f} ns')

#experimental uncertainties

error_tau_1a = np.sqrt(np.diag(cov_1a))[3]
error_tau_1e = np.sqrt(np.diag(cov_1e))[3]

print(f'Error of lifetime of 1a: {error_tau_1a:.3f} ns')
print(error_tau_1a)
print(f'Error of lifetime of 1e: {error_tau_1e:.3f} ns')

#residuals

residuals_1a = data_1a_lifetime[:, 1] - gauss_expo_convolution(data_1a_lifetime[:, 0], *params_1a)
residuals_1e = data_1e_lifetime[:, 1] - gauss_expo_convolution(data_1e_lifetime[:, 0], *params_1e)


sigma_1a = np.std(data_1a_lifetime[:, 1])
sigma_1e = np.std(data_1e_lifetime[:, 1])

y_error_1a = residuals_1a/sigma_1a
y_error_1e = residuals_1e/sigma_1e

plt.figure(8)
plt.plot(data_1a_lifetime[:, 0], y_error_1a, label='1a')
plt.xlabel('Time [ns]')
plt.ylabel('Residuals')
#plt.legend()
plt.savefig('plots/Residuals1a.pdf')
plt.show()

plt.figure(9)
plt.plot(data_1e_lifetime[:, 0], y_error_1e, label='1e')
plt.xlabel('Time [ns]')
plt.ylabel('Residuals')
#plt.legend()
plt.savefig('plots/Residuals1e.pdf')
plt.show()



i_0_for_3a = np.max(data_1a[:, 1])
i_0_for_3c = np.max(data_2c[:, 1])

i_q_for_3a = np.max(data_1c[:, 1])
i_q_for_3c = np.max(data_3c[:, 1])

i_ratio_for_3a = (i_0_for_3a/i_q_for_3a) - 1
i_ratio_for_3c = (i_0_for_3c/i_q_for_3c) - 1

lst_ratio = [0, i_ratio_for_3a, i_ratio_for_3c]

print(i_ratio_for_3a)
print(i_ratio_for_3c)

viscosity_q_for_3a = 1.005
viscosity_q_for_3c = 4.829

lst_viscosity_plot = [0, 1/viscosity_q_for_3a, 1/viscosity_q_for_3c]

plt.figure(10)
plt.scatter(lst_viscosity_plot[1:], lst_ratio[1:], c='k', s=20)

# Linear regression using linregress
slope, intercept, r_value, p_value, std_err = linregress(lst_viscosity_plot[1:], lst_ratio[1:])

plt.plot(lst_viscosity_plot, slope * np.array(lst_viscosity_plot) + intercept, label='linear fit', c='r', linestyle='--')
plt.xlabel('1/viscosity [1/cP]')
plt.ylabel('Ratio of intensities')
plt.savefig('plots/Viscosity.pdf')
plt.show()

print(f'Slope of linear fit: {slope:.3f} and y-intercept: {intercept:.3f}')
print(f'Standard error of the slope: {std_err:.3f}')

print(slope)
print(std_err)
lifetime_fluorescence_vis = (3 * slope) / (8 * 8.314 * 293.15 * concentration_ki[2])
print(lifetime_fluorescence_vis)

#calculate kq

kq = slope_Volmer/params_1e[3]
print(kq)




