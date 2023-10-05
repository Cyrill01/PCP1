import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from scipy.optimize import curve_fit

# Task 1

# Concentrations of the solutions

concentrations = [10**-4, 5*10**-5, 2.5*10**-5, 10**-5, 2*10**-6]

# Import all Data from the files and store them in a list

basic = ['NaOH_1.Sample.asc', 'NaOH_22.Sample.Raw.asc', 'NaOH_33.Sample.Raw.asc', 'NaOH_4.Sample.asc', 'NaOH_5.Sample.asc']
acidic = ['HCl_1.Sample.asc', 'HCl_2.Sample.asc', 'HCl_3.Sample.asc', 'HCl_4.Sample.asc', 'HCl_5.Sample.asc']
buffer = ['Buff_50.22.Sample.Raw.asc', 'Buff_52.22.Sample.Raw.asc']


basic_absorbance = []
basic_wavelength = []

for i in basic:
    basic_absorbance.append(np.genfromtxt(i, skip_header=90, skip_footer=100, usecols=1))
    basic_wavelength.append(np.genfromtxt(i, skip_header=90, skip_footer=100, usecols=0))


acidic_absorbance = []
acidic_wavelength = []

for i in acidic:
    acidic_absorbance.append(np.genfromtxt(i, skip_header=90, skip_footer=100, usecols=1))
    acidic_wavelength.append(np.genfromtxt(i, skip_header=90, skip_footer=100, usecols=0))

buffer_absorbance = []
buffer_wavelength = []

for i in buffer:
    buffer_absorbance.append(np.genfromtxt(i, skip_header=90, skip_footer=100, usecols=1))
    buffer_wavelength.append(np.genfromtxt(i, skip_header=90, skip_footer=100, usecols=0))

# Gauss  fit for 10^-4 M acidic solution

def gauss(x, a, b, c, d):
    return a*np.exp(-((x-b)**2)/(2*c**2)) + d


x_0 = acidic_wavelength[0]
y_0 = acidic_absorbance[0]
x_1 = acidic_wavelength[1]
y_1 = acidic_absorbance[1]
x_2 = acidic_wavelength[2]
y_2 = acidic_absorbance[2]
x_3 = acidic_wavelength[3]
y_3 = acidic_absorbance[3]
x_4 = acidic_wavelength[4]
y_4 = acidic_absorbance[4]


popt_0, pcov_0 = curve_fit(gauss, x_0, y_0, p0=[0.5, 518, 1, 0.5])
popt_1, pcov_1 = curve_fit(gauss, x_1, y_1, p0=[0.5, 518, 1, 0.5])
popt_2, pcov_2 = curve_fit(gauss, x_2, y_2, p0=[0.5, 518, 1, 0.5])
popt_3, pcov_3 = curve_fit(gauss, x_3, y_3, p0=[0.5, 518, 1, 0.5])
popt_4, pcov_4 = curve_fit(gauss, x_4, y_4, p0=[0.5, 518, 1, 0.5])




plt.figure(figsize=(10, 6))
plt.plot(x_0, y_0, 'bo', label='data')
plt.plot(x_0, gauss(x_0, *popt_0), 'r-', label='fit')
plt.legend()
#plt.title('Gauss fit for 10^-4 M acidic solution')
plt.xlabel('Wavelength / nm')
plt.ylabel('Absorbance')
plt.grid(True)
plt.savefig('gauss_fit.pdf')
#plt.show()


# Plotting the absorbance spectra

plt.figure(figsize=(10, 6))
plt.plot(basic_wavelength[0], basic_absorbance[0], label='100 μM')
plt.plot(basic_wavelength[1], basic_absorbance[1], label='50 μM')
plt.plot(basic_wavelength[2], basic_absorbance[2], label='25 μM')
plt.plot(basic_wavelength[3], basic_absorbance[3], label='10 μM')
plt.plot(basic_wavelength[4], basic_absorbance[4], label='2 μM')
plt.xlabel('Wavelength / nm')
plt.ylabel('Absorbance')
#plt.title('Absorbance spectra of basic solutions')
plt.grid(True)
plt.legend()
plt.savefig('basic.pdf')
#plt.show()

plt.figure(figsize=(10, 6))
plt.plot(acidic_wavelength[0], acidic_absorbance[0], label='100 μM')
plt.plot(acidic_wavelength[1], acidic_absorbance[1], label='50 μM')
plt.plot(acidic_wavelength[2], acidic_absorbance[2], label='25 μM')
plt.plot(acidic_wavelength[3], acidic_absorbance[3], label='10 μM')
plt.plot(acidic_wavelength[4], acidic_absorbance[4], label='2 μM')
plt.xlabel('Wavelength / nm')
plt.ylabel('Absorbance')
#plt.title('Absorbance spectra of acidic solutions')
plt.grid(True)
plt.legend()
plt.savefig('acidic.pdf')
plt.show()

plt.figure(figsize=(10, 6))
plt.plot(acidic_wavelength[0], acidic_absorbance[0], label='100 μM')
plt.plot(acidic_wavelength[1], acidic_absorbance[1], label='50 μM')
plt.plot(acidic_wavelength[2], acidic_absorbance[2], label='25 μM')
plt.plot(acidic_wavelength[3], acidic_absorbance[3], label='10 μM')
plt.plot(acidic_wavelength[4], acidic_absorbance[4], label='2 μM')
plt.plot(acidic_wavelength[0], gauss(acidic_wavelength[0], *popt_0), label='fit 100 μM')
plt.plot(acidic_wavelength[1], gauss(acidic_wavelength[1], *popt_1), label='fit 50 μM')
plt.plot(acidic_wavelength[2], gauss(acidic_wavelength[2], *popt_2), label='fit 25 μM')
plt.plot(acidic_wavelength[3], gauss(acidic_wavelength[3], *popt_3), label='fit 10 μM')
plt.plot(acidic_wavelength[4], gauss(acidic_wavelength[4], *popt_4), label='fit 2 μM')
plt.xlabel('Wavelength / nm')
plt.ylabel('Absorbance')
#plt.title('Absorbance spectra of acidic solutions with Gauss fit')
plt.grid(True)
plt.legend()
plt.savefig('acidic_gauss.pdf')
plt.show()

plt.figure(figsize=(10, 6))
plt.plot(buffer_wavelength[0], buffer_absorbance[0], label='pH 5.0')
plt.plot(buffer_wavelength[1], buffer_absorbance[1], label='pH 5.2')
plt.plot(acidic_wavelength[0], acidic_absorbance[0], label='acidic 100 μM')
plt.plot(basic_wavelength[0], basic_absorbance[0], label='basic 100 μM')
plt.xlabel('Wavelength / nm')
plt.ylabel('Absorbance')
#plt.title('Absorbance spectra of buffer solutions and 100 μM solutions')
plt.grid(True)
plt.legend()
plt.savefig('buffer_and_10e-4.pdf')
#plt.show()

# Beer-Lambert Plot

basic_max = []

for i in basic_absorbance:
    basic_max.append(max(i))


acidic_max = []

for i in acidic_absorbance:
    acidic_max.append(max(i))


plt.figure(figsize=(10, 6))

plt.scatter(concentrations, basic_max, label='Basic')
plt.scatter(concentrations, acidic_max, label='Acidic')


slope_basic, intercept_basic, r_value, p_value, std_err_basic = stats.linregress(concentrations, basic_max)
plt.plot(concentrations, slope_basic*np.array(concentrations) + intercept_basic, label=f'Basic fit: y = {slope_basic:.2f} * x + {intercept_basic:.2f}')

slope_acidic_2, intercept_acidic_2, r_value, p_value, std_err_acidic_2 = stats.linregress(concentrations[1:], acidic_max[1:])
plt.plot(concentrations[1:], slope_acidic_2*np.array(concentrations[1:]) + intercept_acidic_2, label=f'Acidic fit without 100 μM: y = {slope_acidic_2:.2f} * x + {intercept_acidic_2:.2f}')

slope_acidic, intercept_acidic, r_value, p_value, std_err_acidic = stats.linregress(concentrations, acidic_max)
plt.plot(concentrations, slope_acidic*np.array(concentrations) + intercept_acidic, label=f'Acidic fit: y = {slope_acidic:.2f} * x + {intercept_acidic:.2f}')


plt.xlabel('Concentration / M')
plt.ylabel('Absorbance')
#plt.title('Beer-Lambert Plot')
plt.grid(True)
plt.legend()
plt.savefig('beer_lambert.pdf')
#plt.show()

print('The slope of the basic fit is', slope_basic, 'with a standard error of', std_err_basic)
print('The slope of the acidic fit is', slope_acidic , 'with a standard error of', std_err_acidic)
print('The slope of the acidic fit without 10^-4 M is', slope_acidic_2 , 'with a standard error of', std_err_acidic_2)

epsilon_basic = slope_basic
epsilon_acidic = slope_acidic
epsilon_acidic_2 = slope_acidic_2

# calculate the wavelength of the maximum absorbance as a mean of the maximum absorbance of the 5 solutions
lst_max_wavelength_basic = []
lst_max_wavelength_acidic = []
lst_max_wavelength_buffer50 = []
lst_max_wavelength_buffer52 = []
for i in basic_absorbance:
    lst_max_wavelength_basic.append(basic_wavelength[0][np.argmax(i)])

# exclude the first solution of the acidic solutions
for i in acidic_absorbance[1:]:
    lst_max_wavelength_acidic.append(acidic_wavelength[0][np.argmax(i)])

for i in buffer_absorbance:
    lst_max_wavelength_buffer50.append(buffer_wavelength[0][np.argmax(i)])


max_wavelength_basic = np.mean(lst_max_wavelength_basic)
max_wavelength_acidic = np.mean(lst_max_wavelength_acidic)


print('\nThe wavelength of the maximum absorbance of the basic solution is', max_wavelength_basic, 'nm')
print('The wavelength of the maximum absorbance of the acidic solution is', max_wavelength_acidic, 'nm')


#absorbance in buffer at maximum wavelength

absorbance_buffer1acidic = buffer_absorbance[0][np.where(buffer_wavelength[0] == 518.0)]
print('\nThe absorbance of the buffer 5.0 solution in the acidic at 518 nm is', absorbance_buffer1acidic)
absorbance_buffer2acidic = buffer_absorbance[1][np.where(buffer_wavelength[1] == 518.0)]
print('The absorbance of the buffer 5.2 solution in the acidic at 518 nm is', absorbance_buffer2acidic)
absorbance_buffer1basic = buffer_absorbance[0][np.where(buffer_wavelength[0] == 435.0)]
print('The absorbance of the buffer 5.0 solution in the basic at 435 nm is', absorbance_buffer1basic)
absorbance_buffer2basic = buffer_absorbance[1][np.where(buffer_wavelength[1] == 435.0)]
print('The absorbance of the buffer 5.2 solution in the basic at 435 nm is', absorbance_buffer2basic)


def calculate_fwhm(wavelengths, absorbance):
    # Find the maximum absorbance and its corresponding wavelength
    max_absorbance = np.max(absorbance)
    max_index = np.argmax(absorbance)
    lambda_max = wavelengths[max_index]

    # Calculate the half-maximum absorbance value
    half_max_absorbance = max_absorbance / 2.0

    # Find the wavelengths where the absorbance is closest to half-maximum on both sides of lambda_max
    lambda_left = None
    lambda_right = None

    for i in range(max_index, 0, -1):
        if absorbance[i] <= half_max_absorbance:
            lambda_left = np.interp(half_max_absorbance, [absorbance[i], absorbance[i + 1]], [wavelengths[i], wavelengths[i + 1]])
            break

    for i in range(max_index, len(absorbance) - 1):
        if absorbance[i] <= half_max_absorbance:
            lambda_right = np.interp(half_max_absorbance, [absorbance[i - 1], absorbance[i]], [wavelengths[i - 1], wavelengths[i]])
            break

    # Calculate FWHM
    fwhm = lambda_right - lambda_left

    return abs(fwhm)

# Calculate FWHM for the basic and acidic spectra
lst_fwhm_basic = []
lst_fwhm_acidic = []
lst_fwhm_acidic_2 = []

for i in range(len(basic_absorbance)):
    lst_fwhm_basic.append(calculate_fwhm(basic_wavelength[i], basic_absorbance[i]))

for i in range(len(acidic_absorbance)):
    lst_fwhm_acidic.append(calculate_fwhm(acidic_wavelength[i], acidic_absorbance[i]))

for i in range(len(acidic_absorbance[1:])):
    lst_fwhm_acidic_2.append(calculate_fwhm(acidic_wavelength[i], acidic_absorbance[i]))


fwhm_basic = np.mean(lst_fwhm_basic)
fwhm_acidic = np.mean(lst_fwhm_acidic)
fwhm_acidic_2 = np.mean(lst_fwhm_acidic_2)

print('\nThe FWHM of the basic solution spectrum is', abs(fwhm_basic), 'nm')
print('The FWHM of the acidic solution spectrum is', abs(fwhm_acidic), 'nm')
print('The FWHM of the acidic solution spectrum without 10^-4 M is', abs(fwhm_acidic_2), 'nm')

# transition dipole moment

def calculate_transition_dipole(epsilon, fwhm, lambda_max):

    transition_dipole = np.sqrt(0.0092*epsilon*((fwhm)/(lambda_max)))

    return transition_dipole

transition_dipole_basic = calculate_transition_dipole(epsilon_basic, fwhm_basic, max_wavelength_basic)
transition_dipole_acidic = calculate_transition_dipole(epsilon_acidic_2, fwhm_acidic, max_wavelength_acidic)

print('\nThe transition dipole moment of the basic solution is', transition_dipole_basic, 'D')
print('The transition dipole moment of the acidic solution is', transition_dipole_acidic, 'D')

# pKa calculation

def calculate_pKa(pH, absorbance_Ind, absorbance_HInd, epsilon_a, epsilon_b):

    ind = absorbance_Ind / epsilon_b
    hind = absorbance_HInd / epsilon_a
    pKa = pH + np.log10(ind/hind)

    return pKa


pka1 = calculate_pKa(5.08, absorbance_buffer1basic, absorbance_buffer1acidic, epsilon_acidic_2, epsilon_basic)
pka2 = calculate_pKa(5.29, absorbance_buffer2basic, absorbance_buffer2acidic, epsilon_acidic_2, epsilon_basic)

print('\nThe pKa of the first buffer solution is', pka1)
print('The pKa of the second buffer solution is', pka2)

# Calculate Ka

def calculate_Ka(pKa):

        Ka = 10**(-pKa)

        return Ka

Ka1 = calculate_Ka(pka1)
Ka2 = calculate_Ka(pka2)

print('\nThe Ka of the first buffer solution is', Ka1)
print('The Ka of the second buffer solution is', Ka2)


# Error Calculation

error_epsilon_basic = std_err_basic
error_epsilon_acidic = std_err_acidic_2

error_mu_basic = (0.5 * np.sqrt(0.0092*(fwhm_basic/max_wavelength_basic)) * (1/np.sqrt(epsilon_basic))) * error_epsilon_basic
error_mu_acidic = (0.5 * np.sqrt(0.0092*(fwhm_acidic/max_wavelength_acidic)) * (1/np.sqrt(epsilon_acidic_2))) * error_epsilon_acidic

error_c_a1 = absorbance_buffer1acidic * -(1/(epsilon_acidic_2)**2) * error_epsilon_acidic
error_c_a2 = absorbance_buffer2acidic * -(1/(epsilon_acidic_2)**2) * error_epsilon_acidic
error_c_b1 = absorbance_buffer1basic * -(1/(epsilon_basic)**2) * error_epsilon_basic
error_c_b2 = absorbance_buffer2basic * -(1/(epsilon_basic)**2) * error_epsilon_basic

ind1 = absorbance_buffer1basic/epsilon_basic
ind2 = absorbance_buffer2basic/epsilon_basic
hind1 = absorbance_buffer1acidic/epsilon_acidic_2
hind2 = absorbance_buffer2acidic/epsilon_acidic_2

print('\nThe concentration of the acidic solution of the first buffer solution is', ind1)
print('The concentration of the acidic solution of the second buffer solution is', ind2)
print('The concentration of the basic solution of the first buffer solution is', hind1)
print('The concentration of the basic solution of the second buffer solution is', hind2)


error_pka1 = np.sqrt((((1/(np.log(10)*ind1))**2) * error_c_b1**2) + (((-1/(np.log(10)*hind1))**2) * error_c_a1**2))
error_pka2 = np.sqrt(((1/(np.log(10)*ind2))**2) * error_c_b2**2 + ((-1/(np.log(10)*hind2))**2) * error_c_a2**2)

error_ka1 = np.sqrt((((1/hind1)**2)*(error_c_b1**2)) + (((-ind1/(hind1**2))**2)*(error_c_a1**2)))
error_ka2 = np.sqrt((((1/hind2)**2)*(error_c_b2**2)) + (((-ind2/(hind2**2))**2)*(error_c_a2**2)))

print('\nThe error of the transition dipole moment of the basic solution is', error_mu_basic, 'D')
print('The error of the transition dipole moment of the acidic solution is', error_mu_acidic, 'D')
print('The error of the concentration of the acidic solution of the first buffer solution is', error_c_a1)
print('The error of the concentration of the acidic solution of the second buffer solution is', error_c_a2)
print('The error of the concentration of the basic solution of the first buffer solution is', error_c_b1)
print('The error of the concentration of the basic solution of the second buffer solution is', error_c_b2)
print('The error of the pKa of the first buffer solution is', error_pka1)
print('The error of the pKa of the second buffer solution is', error_pka2)
print('The error of the Ka of the first buffer solution is', error_ka1)
print('The error of the Ka of the second buffer solution is', error_ka2)
