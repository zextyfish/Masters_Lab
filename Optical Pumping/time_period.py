import csv
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.signal import find_peaks

time = []
x = []

with open("time_period_data.txt", "r") as file:
    reader = csv.reader(file)
    next(reader)  # Skip 2 lines
    next(reader)

    for row in reader:
        if row[1]:  # Add rows with x values available
            time.append(float(row[0]))
            x.append(float(row[1]))

time = np.array(time)
x = np.array(x)
print(time[16])

peaks, _ = find_peaks(x)  # Find local maximas

peak_times = time[peaks]
periods = np.diff(peak_times)
np.set_printoptions(precision=2)
print(periods)
avg_period = np.mean(periods)
print(f"Average Period: {avg_period} seconds")


def exp_sine(t, A, lambd, B, C, D):
    return A * np.exp(-lambd * t) * np.sin(B * t + C) + D


B_guess = 2 * np.pi / avg_period
initial_guess = [0, 0, B_guess, 0, 0]


for _ in range(5):
    noisy_guess = initial_guess + np.random.normal(0, 0.1, 5)
    params, _ = curve_fit(exp_sine, time, x, p0=noisy_guess)
    bounds = (
        [0, 0, 0, -np.pi],  # lower bounds
        [np.inf, 1, 10, np.pi],  # upper bounds
    )

    print(params)

# # Perform the curve fitting
params, covariance = curve_fit(exp_sine, time, x, p0=initial_guess)

# # Extract the fitted parameters
A_fit, lambd_fit, B_fit, C_fit, D_f = params
print(
    f"Fitted parameters: A = {A_fit}, λ = {lambd_fit}, B = {B_fit},C = {C_fit},D={D_f}"
)

# Generate the fitted curve using the optimal parameters
fitted_x = exp_sine(time, *params)

# Calculate residuals (difference between data and fitted curve)
residuals = x - fitted_x

# Calculate R-squared
ss_res = np.sum(residuals**2)
ss_tot = np.sum((x - np.mean(x)) ** 2)
r_squared = 1 - (ss_res / ss_tot)

# Calculate Chi-squared (sum of squared residuals)
chi_squared = np.sum(
    (residuals**2) / fitted_x
)  # Assumes no weighting, you can add weights if available

# Output the fit quality metrics
print(f"R-squared: {r_squared}")
print(f"Chi-squared: {chi_squared}")
# Plot the original data and the fitted curve
plt.scatter(time, x, label="Data", color="blue", s=1)
plt.plot(time, fitted_x, label="Fitted Exponential Decay + Sine Curve", color="red")
plt.xlabel("Time")
plt.ylabel("X")
plt.legend()
plt.show()

# plt.figure()
# plt.scatter(time, residuals, s=1)
# plt.axhline(0, color="red")
# plt.title("Residuals")
# plt.show()
