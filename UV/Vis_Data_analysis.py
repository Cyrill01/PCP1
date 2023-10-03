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
print('\nThe wavelength of the maximum absorbance of the basic solution is', max_wavelength_basic, 'nm')
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

print('\nThe FWHM of the basic solution spectrum is', abs(fwhm_basic), 'nm')
print('The FWHM of the acidic solution spectrum is', abs(fwhm_acidic), 'nm')

# transition dipole moment

def calculate_transition_dipole(epsilon, fwhm, lambda_max):

    transition_dipole = np.sqrt(0.0092*epsilon*((fwhm)/(lambda_max)))

    return transition_dipole

# error propagation for transition dipole moment

# Define values and errors for the basic solution
epsilon_basic, epsilon_basic_error = slope_basic, std_err_basic
fwhm_basic, fwhm_basic_error = fwhm_basic, np.sqrt(2)
lambda_max_basic, lambda_max_basic_error = max_wavelength_basic, 1.0

# Calculate transition dipole moment for the basic solution
transition_dipole_basic = calculate_transition_dipole(epsilon_basic, fwhm_basic, lambda_max_basic)

# Define values and errors for the acidic solution
epsilon_acidic, epsilon_acidic_error = slope_acidic, std_err_acidic
fwhm_acidic, fwhm_acidic_error = fwhm_acidic, np.sqrt(2)
lambda_max_acidic, lambda_max_acidic_error = max_wavelength_acidic, 1.0

# Calculate transition dipole moment for the acidic solution
transition_dipole_acidic = calculate_transition_dipole(epsilon_acidic, fwhm_acidic, lambda_max_acidic)

# Define values and errors for the acidic solution without 10^-4 M
epsilon_acidic_2, epsilon_acidic_error_2 = slope_acidic_2, std_err_acidic_2
fwhm_acidic_2, fwhm_acidic_error_2 = fwhm_acidic, np.sqrt(2)
lambda_max_acidic_2, lambda_max_acidic_error = max_wavelength_acidic, 1.0

# Calculate transition dipole moment for the acidic solution without 10^-4 M
transition_dipole_acidic_2 = calculate_transition_dipole(epsilon_acidic_2, fwhm_acidic, lambda_max_acidic_2)


# Function to calculate error using error propagation
def calculate_error(epsilon, epsilon_error, fwhm, fwhm_error, lambda_max, lambda_max_error):
    partial_mu_epsilon = np.sqrt(0.0092 * (fwhm / lambda_max))
    partial_mu_fwhm = np.where(np.abs(fwhm) < 1e-10, 0.0, np.sqrt(0.0092 * epsilon * lambda_max / (fwhm ** 3)))
    partial_mu_lambda_max = np.where(np.abs(lambda_max) < 1e-10, 0.0,
                                     np.sqrt(0.0092 * epsilon * fwhm / (lambda_max ** 3)))

    mu_error = np.sqrt((partial_mu_epsilon * epsilon_error) ** 2 + (partial_mu_fwhm * fwhm_error) ** 2 + (
                partial_mu_lambda_max * lambda_max_error) ** 2)

    return mu_error


# Calculate errors for the basic and acidic solutions
mu_error_basic = calculate_error(epsilon_basic, epsilon_basic_error, fwhm_basic, fwhm_basic_error, lambda_max_basic,
                                 lambda_max_basic_error)
mu_error_acidic = calculate_error(epsilon_acidic, epsilon_acidic_error, fwhm_acidic, fwhm_acidic_error,
                                  lambda_max_acidic, lambda_max_acidic_error)
mu_error_acidic_2 = calculate_error(epsilon_acidic_2, epsilon_acidic_error_2, fwhm_acidic_2, fwhm_acidic_error_2,
                                    lambda_max_acidic_2, lambda_max_acidic_error)

# Print results
print('\nBasic Solution:')
print('Transition Dipole Moment:', transition_dipole_basic, 'Debye')
print('Error on Transition Dipole Moment:', mu_error_basic, 'Debye')

print('\nAcidic Solution:')
print('Transition Dipole Moment:', transition_dipole_acidic, 'Debye')
print('Error on Transition Dipole Moment:', mu_error_acidic, 'Debye')

print('\nAcidic Solution without 10^-4 M:')
print('Transition Dipole Moment:', transition_dipole_acidic_2, 'Debye')
print('Error on Transition Dipole Moment:', mu_error_acidic_2, 'Debye')


# Calculate the pKa value

def calculate_pka(pH, epsilon_Ind, epsilon_HInd, lambda_max_Ind, lambda_max_HInd):
    pka = pH - (np.log10((lambda_max_Ind/epsilon_Ind)/(lambda_max_HInd/epsilon_HInd)))
    return pka

pka1 = calculate_pka(5.08, epsilon_basic, epsilon_acidic, max_wavelength_basic, max_wavelength_acidic)

pka2 = calculate_pka(5.29, epsilon_basic, epsilon_acidic, max_wavelength_basic, max_wavelength_acidic)


print('\nThe pKa value for the first transition is', pka1)
print('The pKa value for the second transition is', pka2)

pka1_without_10_4 = calculate_pka(5.08, epsilon_basic, epsilon_acidic_2, max_wavelength_basic, max_wavelength_acidic)

pka2_without_10_4 = calculate_pka(5.29, epsilon_basic, epsilon_acidic_2, max_wavelength_basic, max_wavelength_acidic)

print('\nThe pKa value for the first transition without 10^-4 M is', pka1_without_10_4)
print('The pKa value for the second transition without 10^-4 M is', pka2_without_10_4)

k_a1 = 10**(-pka1)
k_a2 = 10**(-pka2)
k_a1_without_10_4 = 10**(-pka1_without_10_4)
k_a2_without_10_4 = 10**(-pka2_without_10_4)

print('\nThe Ka value for the first transition is', k_a1)
print('The Ka value for the second transition is', k_a2)
print('The Ka value for the first transition without 10^-4 M is', k_a1_without_10_4)
print('The Ka value for the second transition without 10^-4 M is', k_a2_without_10_4)
