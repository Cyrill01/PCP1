import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

# Load data with my Matrikelnummer

data = np.loadtxt("Daten_21-730-742.dat", skiprows=2, dtype=float, usecols=(0, 1, 2))

# Define column names

resistance = data[:, 0]
temperature = data[:, 1]
temperature_error = data[:, 2]

# Plot data

plt.errorbar(resistance, temperature, yerr=temperature_error, color='black', fmt='.', label='Data') # Plot data with error bars
plt.xlabel('Resistance / Ohm')
plt.ylabel('Temperature / °C')
#plt.title('Resistance vs Temperature')
#plt.legend()
plt.savefig('Resistance_vs_Temperature.pdf')
plt.show()

# Perform linear fit

slope, intercept, r_value, p_value, std_err = stats.linregress(resistance, temperature) # Perform linear fit and get slope, intercept and standard error (for slope)
delta_intercept = std_err * np.sqrt(np.mean(resistance**2)) # Calculate error for intercept with square root of mean of resistance squared (Feherfortpflanzungsgesetz)

# Plot linear fit

plt.errorbar(resistance, temperature, yerr=temperature_error, color='black', fmt='.', label='Data')
plt.plot(resistance, slope*resistance + intercept, color='red', label=f'Linear Fit: y = {slope:.5f} ± {std_err:.5f})x + {intercept:.2f} ± {delta_intercept:.2f})')
plt.xlabel('Resistance / Ohm')
plt.ylabel('Temperature / °C')
#plt.title('Resistance vs Temperature with linear fit')
#plt.legend()
plt.savefig('Resistance_vs_Temperature_linear_fit.pdf')
plt.show()

# Print linear fit equation with errors

print(f'Linear Fit Equation: y = ({slope:.5f} ± {std_err:.5f})x + ({intercept:.2f} ± {delta_intercept:.2f})')


ohm = [0, 10858, 13000] # define resistance values for interpolation
t_inter = [] # definr empty list for temperature for interpolation
for i in ohm:   # calculate temperature for interpolation
    t_inter.append(intercept-abs(slope)*i) #  a - |b|* Col(''Ohm'')

print(t_inter)

t_max = [] # define empty list for maximal temperature
for i in ohm: # calculate maximal temperature
    t_max.append((intercept + delta_intercept) - (abs(slope) - std_err)*i) # a + delta_a - (|b| - delta_b)*Col(''Ohm'')

print(t_max)

t_min = [] # define empty list for minimal temperature
for i in ohm: # calculate minimal temperature
    t_min.append((intercept - delta_intercept) - (abs(slope) + std_err)*i) # a - delta_a - (|b| + delta_b)*Col(''Ohm'')

print(t_min)

plt.errorbar(resistance, temperature, yerr=temperature_error,  color='black',fmt='.', label='Data')
plt.plot(ohm, t_inter, color='green', label='Intercept') # add t_inter to plot
plt.plot(ohm, t_max, color='orange', label='Max') # add t_max to plot
plt.plot(ohm, t_min, color='blue', label='Min') # add t_min to plot
plt.plot(resistance, slope*resistance + intercept, color='red', linewidth=2.00, label=f'Linear Fit: y = {slope:.5f} ± {std_err:.5f})x + {intercept:.2f} ± {delta_intercept:.2f})')
plt.xlabel('Resistance / Ohm')
plt.ylabel('Temperature / °C')
#plt.title('Resistance vs Temperature')
#plt.legend()
plt.savefig('Resistance_vs_Temperature_interpolated.pdf')
plt.show()

#calculate temperature at 0 Ohm

t_0 = intercept - abs(slope)*0
print(f'Temperature at 0 Ohm: ({t_0:.2f} ± {delta_intercept:.2f})°C')

# plot for only in range of linear fit

# Define the range of x-values for the linear fit
linear_fit_range = np.linspace(min(resistance), max(resistance), 100)

# Calculate the corresponding temperature values for the linear fit range
linear_fit_temperature = slope * linear_fit_range + intercept

# Plot data within the range of the linear fit
plt.errorbar(resistance, temperature, yerr=temperature_error, color='black', fmt='.', label='Data')
plt.plot(ohm, t_inter, color='green', label='Intercept', linewidth=3.00)  # add t_inter to plot
plt.plot(ohm, t_max, color='orange', label='Max')  # add t_max to plot
plt.plot(ohm, t_min, color='blue', label='Min')  # add t_min to plot
plt.plot(linear_fit_range, linear_fit_temperature, color='red', label=f'Linear Fit: y = {slope:.5f} ± {std_err:.5f})x + {intercept:.2f} ± {delta_intercept:.2f})')
plt.xlabel('Resistance / Ohm')
plt.ylabel('Temperature / °C')
#plt.title('Resistance vs Temperature')

# Set x-axis limits to the range of the linear fit
plt.xlim(min(linear_fit_range), max(linear_fit_range))

#plt.legend()
plt.savefig('Resistance_vs_Temperature_interpolated_limited_range.pdf')
plt.show()


#Aufgabenteil 2

r_inter = np.array([i - 10858 for i in resistance]) #calculate resistance for interpolation


print(r_inter)

#plot of r_inter and temperature with temperature error

plt.errorbar(r_inter, temperature, yerr=temperature_error,  color='black',fmt='.', label='Data')
plt.xlabel('Resistance / Ohm')
plt.ylabel('Temperature / °C')
#plt.title('Resistance vs Temperature')
#plt.legend()
plt.savefig('Resistance_vs_Temperature_interpolated2.pdf')
plt.show()

# Perform linear fit

slope, intercept, r_value, p_value, std_err = stats.linregress(r_inter, temperature)
delta_intercept = std_err * np.sqrt(np.mean(r_inter**2))

# Plot linear fit

plt.errorbar(r_inter, temperature, yerr=temperature_error,  color='black',fmt='.', label='Data')
plt.plot(r_inter, slope*r_inter + intercept, color='red', label=f'Linear Fit: y = {slope:.5f} ± {std_err:.5f})x + {intercept:.2f} ± {delta_intercept:.2f})')
plt.xlabel('Resistance / Ohm')
plt.ylabel('Temperature / °C')
#plt.title('Resistance vs Temperature')
#plt.legend()
plt.savefig('Resistance_vs_Temperature_linear_fit2.pdf')
plt.show()

# Print linear fit equation with standard error

print(f'Linear Fit Equation: y = ({slope:.5f} ± {std_err:.5f})x + ({intercept:.2f} ± {delta_intercept:.2f})')


# Add point to plot at 0 Ohm with errorbars


plt.errorbar(r_inter, temperature, yerr=temperature_error,  color='black', fmt='.', label='Data')
plt.errorbar(0, intercept, yerr=0.1, color='green', marker='.', label='Intercept')
plt.plot(r_inter, slope*r_inter + intercept, color='red', linewidth=2.00, label=f'Linear Fit: y = {slope:.5f} ± {std_err:.5f})x + {intercept:.2f} ± {delta_intercept:.2f})')
plt.xlabel('Resistance / Ohm')
plt.ylabel('Temperature / °C')
#plt.title('Resistance vs Temperature')
#plt.legend()
plt.savefig('Resistance_vs_Temperature_added_point.pdf')
plt.show()


# calculate temperature at 0 Ohm with error

t_0 = intercept - abs(slope)*0
print(f'Temperature at 0 Ohm: ({t_0:.2f} ± {delta_intercept:.2f}) °C')

