import numpy as np
import matplotlib.pyplot as plt

# Read data from "Data1.dat" file
data = np.genfromtxt("Daten1.dat", skip_header=2, dtype=float, usecols=(0, 1), names=['x', 'y'])

# Extract x and y values from the loaded data
x_values = data['x']
y_values = data['y']

# Perform linear fit
coefficients = np.polyfit(x_values, y_values, 1)
slope, intercept = coefficients

# Create a linear model function
linear_model = np.poly1d(coefficients)

# Generate y values for the linear fit
y_fit = linear_model(x_values)

# Plot the data and linear fit
plt.scatter(x_values, y_values, label='Data')
plt.plot(x_values, y_fit, color='red', label=f'Linear Fit: y = {slope:.2f}x + {intercept:.2f}')
plt.xlabel('x-Wert')
plt.ylabel('y-Wert')
plt.title('Linear Fit')
plt.legend()
plt.grid(True)

# Show the plot
plt.show()

# Print the equation of the linear fit
print(f'Linear Fit Equation: y = {slope:.2f}x + {intercept:.2f}')


x_values = data['y']
y_values = data['x']

# Perform linear fit
coefficients = np.polyfit(x_values, y_values, 1)
slope, intercept = coefficients

# Create a linear model function
linear_model = np.poly1d(coefficients)

# Generate y values for the linear fit
y_fit = linear_model(x_values)

# Plot the data and linear fit
plt.scatter(x_values, y_values, label='Data')
plt.plot(x_values, y_fit, color='red', label=f'Linear Fit: y = {slope:.2f}x + {intercept:.2f}')
plt.xlabel('x-Wert')
plt.ylabel('y-Wert')
plt.title('Linear Fit with changed x and y')
plt.legend()
plt.grid(True)

# Show the plot
plt.show()

# Print the equation of the linear fit
print(f'Linear Fit Equation: y = {slope:.2f}x + {intercept:.2f}')
