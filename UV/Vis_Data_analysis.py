import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

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

# Plotting the absorbance spectra

plt.figure(figsize=(10, 6))
plt.plot(basic_wavelength[0], basic_absorbance[0], label='10^-4 M')
plt.plot(basic_wavelength[1], basic_absorbance[1], label='5*10^-5 M')
plt.plot(basic_wavelength[2], basic_absorbance[2], label='2.5*10^-5 M')
plt.plot(basic_wavelength[3], basic_absorbance[3], label='10^-5 M')
plt.plot(basic_wavelength[4], basic_absorbance[4], label='2*10^-6 M')
plt.xlabel('Wavelength (nm)')
plt.ylabel('Absorbance')
plt.title('Absorbance spectra of basic solutions')
plt.legend()
plt.show()

plt.figure(figsize=(10, 6))
plt.plot(acidic_wavelength[0], acidic_absorbance[0], label='10^-4 M')
plt.plot(acidic_wavelength[1], acidic_absorbance[1], label='5*10^-5 M')
plt.plot(acidic_wavelength[2], acidic_absorbance[2], label='2.5*10^-5 M')
plt.plot(acidic_wavelength[3], acidic_absorbance[3], label='10^-5 M')
plt.plot(acidic_wavelength[4], acidic_absorbance[4], label='2*10^-6 M')
plt.xlabel('Wavelength (nm)')
plt.ylabel('Absorbance')
plt.title('Absorbance spectra of acidic solutions')
plt.legend()
plt.show()

plt.figure(figsize=(10, 6))
plt.plot(buffer_wavelength[0], buffer_absorbance[0], label='pH 5.0')
plt.plot(buffer_wavelength[1], buffer_absorbance[1], label='pH 5.2')
plt.plot(acidic_wavelength[0], acidic_absorbance[0], label='10^-4 M')
plt.plot(basic_wavelength[0], basic_absorbance[0], label='10^-4 M')
plt.xlabel('Wavelength (nm)')
plt.ylabel('Absorbance')
plt.title('Absorbance spectra of buffer solutions and 10^-4 M solutions')
plt.legend()
plt.show()

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
plt.plot(concentrations, slope_basic*np.array(concentrations) + intercept_basic, label='Basic fit')

slope_acidic_2, intercept_acidic_2, r_value, p_value, std_err_acidic_2 = stats.linregress(concentrations[1:], acidic_max[1:])
plt.plot(concentrations[1:], slope_acidic_2*np.array(concentrations[1:]) + intercept_acidic_2, label='Acidic fit without 10^-4 M')

slope_acidic, intercept_acidic, r_value, p_value, std_err_acidic = stats.linregress(concentrations, acidic_max)
plt.plot(concentrations, slope_acidic*np.array(concentrations) + intercept_acidic, label='Acidic fit')


plt.xlabel('Concentration (M)')
plt.ylabel('Absorbance')
plt.title('Beer-Lambert Plot')
plt.legend()
plt.show()

print('The slope of the basic fit is', slope_basic, 'with a standard error of', std_err_basic)
print('The slope of the acidic fit is', slope_acidic , 'with a standard error of', std_err_acidic)

epsilon_basic = slope_basic
epsilon_acidic = slope_acidic

lst_max_wavelength_basic = []
lst_max_wavelength_acidic = []
lst_max_wavelength_buffer = []

for i in basic_absorbance:
    lst_max_wavelength_basic.append(basic_wavelength[0][np.argmax(i)])

for i in acidic_absorbance:
    lst_max_wavelength_acidic.append(acidic_wavelength[0][np.argmax(i)])

for i in buffer_absorbance:
    lst_max_wavelength_buffer.append(buffer_wavelength[0][np.argmax(i)])

max_wavelength_basic = np.mean(lst_max_wavelength_basic)
max_wavelength_acidic = np.mean(lst_max_wavelength_acidic)
max_wavelength_buffer = np.mean(lst_max_wavelength_buffer)


#max_wavelength_basic = basic_wavelength[0][np.argmax(basic_absorbance[0])]
#max_wavelength_acidic = acidic_wavelength[0][np.argmax(acidic_absorbance[0])]
print('The wavelength of the maximum absorbance of the basic solution is', max_wavelength_basic, 'nm')
print('The wavelength of the maximum absorbance of the acidic solution is', max_wavelength_acidic, 'nm')
print('The wavelength of the maximum absorbance of the buffer solution is', max_wavelength_buffer, 'nm')


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

for i in range(len(basic_absorbance)):
    lst_fwhm_basic.append(calculate_fwhm(basic_wavelength[i], basic_absorbance[i]))

for i in range(len(acidic_absorbance)):
    lst_fwhm_acidic.append(calculate_fwhm(acidic_wavelength[i], acidic_absorbance[i]))



fwhm_basic = np.mean(lst_fwhm_basic)
fwhm_acidic = np.mean(lst_fwhm_acidic)

print('The FWHM of the basic solution spectrum is', abs(fwhm_basic), 'nm')
print('The FWHM of the acidic solution spectrum is', abs(fwhm_acidic), 'nm')

# transition dipole moment

def calculate_transition_dipole(epsilon, fwhm, lambda_max):

    transition_dipole = np.sqrt(0.0092*epsilon*((fwhm)/(lambda_max)))

    return transition_dipole

# error propagation for transition dipole moment

# Given values and errors
epsilon_basic = slope_basic  # Epsilon for the basic solution
epsilon_basic_error = std_err_basic  # Error on epsilon

fwhm_basic = fwhm_basic
fwhm_basic_error = np.sqrt(2)

lambda_max_basic = max_wavelength_basic
lambda_max_basic_error = 1.0

# Calculate the transition dipole moment for the basic solution
transition_dipole_basic = calculate_transition_dipole(epsilon_basic, fwhm_basic, lambda_max_basic)

# Calculate the partial derivatives with respect to epsilon, FWHM, and lambda_max
partial_mu_epsilon = np.sqrt(0.0092 * (fwhm_basic / lambda_max_basic))

# Handle cases where the denominator is close to zero to avoid NaN values
partial_mu_fwhm = np.where(np.abs(fwhm_basic) < 1e-10, 0.0, np.sqrt(0.0092 * epsilon_basic * lambda_max_basic / (fwhm_basic ** 3)))
partial_mu_lambda_max = np.where(np.abs(lambda_max_basic) < 1e-10, 0.0, np.sqrt(0.0092 * epsilon_basic * fwhm_basic / (lambda_max_basic ** 3)))

# Use error propagation formulas to find the error on mu
mu_error = np.sqrt((partial_mu_epsilon * epsilon_basic_error)**2 +
                   (partial_mu_fwhm * fwhm_basic_error)**2 + (partial_mu_lambda_max * lambda_max_basic_error)**2)

print('The transition dipole moment of the basic solution is', transition_dipole_basic, 'Debye')
print('The error on the transition dipole moment is', mu_error, 'Debye')


# Given values and errors for the acidic solution

epsilon_acidic = slope_acidic
epsilon_acidic_error = std_err_acidic

fwhm_acidic = fwhm_acidic
fwhm_acidic_error = np.sqrt(2)

lambda_max_acidic = max_wavelength_acidic
lambda_max_acidic_error = 1.0

# Calculate the transition dipole moment for the acidic solution
transition_dipole_acidic = calculate_transition_dipole(epsilon_acidic, fwhm_acidic, lambda_max_acidic)

# Calculate the partial derivatives with respect to epsilon, FWHM, and lambda_max for the acidic solution
partial_mu_epsilon_acidic = np.sqrt(0.0092 * (fwhm_acidic / lambda_max_acidic))

# Handle cases where the denominator is close to zero to avoid NaN values
partial_mu_fwhm_acidic = np.where(np.abs(fwhm_acidic) < 1e-10, 0.0, np.sqrt(0.0092 * epsilon_acidic * lambda_max_acidic / (fwhm_acidic ** 3)))
partial_mu_lambda_max_acidic = np.where(np.abs(lambda_max_acidic) < 1e-10, 0.0, np.sqrt(0.0092 * epsilon_acidic * fwhm_acidic / (lambda_max_acidic ** 3)))

# Use error propagation formulas to find the error on mu for the acidic solution
mu_error_acidic = np.sqrt((partial_mu_epsilon_acidic * epsilon_acidic_error)**2 +
                          (partial_mu_fwhm_acidic * fwhm_acidic_error)**2 +
                          (partial_mu_lambda_max_acidic * lambda_max_acidic_error)**2)

print('The transition dipole moment of the acidic solution is', transition_dipole_acidic, 'Debye')
print('The error on the transition dipole moment for the acidic solution is', mu_error_acidic, 'Debye')

# Error Propagation with corrected linear fit for the acidic solution

# Given values and errors for the acidic solution
epsilon_acidic_2 = slope_acidic_2
epsilon_acidic_error_2 = std_err_acidic_2

fwhm_acidic_2 = fwhm_acidic
fwhm_acidic_error_2 = np.sqrt(2)

lambda_max_acidic_2 = max_wavelength_acidic
lambda_max_acidic_error = 1.0

# Calculate the transition dipole moment for the acidic solution
transition_dipole_acidic_2 = calculate_transition_dipole(epsilon_acidic_2, fwhm_acidic, lambda_max_acidic_2)

# Calculate the partial derivatives with respect to epsilon, FWHM, and lambda_max for the acidic solution
partial_mu_epsilon_acidic_2 = np.sqrt(0.0092 * (fwhm_acidic / lambda_max_acidic_2))

# Handle cases where the denominator is close to zero to avoid NaN values
partial_mu_fwhm_acidic_2 = np.where(np.abs(fwhm_acidic) < 1e-10, 0.0, np.sqrt(0.0092 * epsilon_acidic_2 * lambda_max_acidic_2 / (fwhm_acidic ** 3)))
partial_mu_lambda_max_acidic_2 = np.where(np.abs(lambda_max_acidic_2) < 1e-10, 0.0, np.sqrt(0.0092 * epsilon_acidic_2 * fwhm_acidic / (lambda_max_acidic_2 ** 3)))

# Use error propagation formulas to find the error on mu for the acidic solution
mu_error_acidic_2 = np.sqrt((partial_mu_epsilon_acidic_2 * epsilon_acidic_error_2)**2 +
                          (partial_mu_fwhm_acidic_2 * fwhm_acidic_error_2)**2 +
                          (partial_mu_lambda_max_acidic_2 * lambda_max_acidic_error)**2)

print('The transition dipole moment of the acidic solution without the solution 10^-4 M is', transition_dipole_acidic_2, 'Debye')
print('The error on the transition dipole moment for the acidic solution without the solution 10^-4 M is', mu_error_acidic_2, 'Debye')
