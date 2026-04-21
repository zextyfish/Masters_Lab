import csv
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.signal import find_peaks


def exp_sine(t, A, lamb, B, C, D, E, F):
    return A * np.exp(-lamb * t) * np.sin(B * t + C) + D * t * t + E * t + F


def eq_drift(t, A, lamb, B, C, D, E, F):
    return D * t * t + E * t + F

time = []
x = []

with open("G_01_25.csv", "r") as file:
    reader = csv.reader(file)
    next(reader)
    next(reader)

    for row in reader:
        if row[1]:
            time.append(float(row[0]))
            x.append(float(row[1]))

time = np.array(time)
x = np.array(x)

plt.scatter(time, x, label="Raw_Data", s=1)

t0 = 100
t1 = 3100
t2 = 3800
t3 = 10000
t4 = 10500
t5 = 17900


plt.axvline(t0, color="black")
plt.axvline(t1, color="black")
plt.axvline(t2, color="black")
plt.axvline(t3, color="black")
plt.axvline(t4, color="black")
plt.axvline(t5, color="black")

seg1 = (time >= t0) & (time < t1)
seg2 = (time >= t2) & (time < t3)
seg3 = (time >= t4) & (time < t5)

segments = [(time[seg1], x[seg1]), (time[seg2], x[seg2]), (time[seg3], x[seg3])]

for i, (t_seg, x_seg) in enumerate(segments, start=1):
    peaks, _ = find_peaks(x_seg, prominence=0.005, distance=16)
    troughs, _ = find_peaks(-x_seg, prominence=0.005, distance=16)
    n = min(len(peaks), len(troughs))

    filtered_peaks = peaks[:n]
    filtered_troughs = troughs[:n]

    x_eq = x_seg[filtered_troughs] + x_seg[filtered_peaks]

    peak_times = t_seg[filtered_peaks]
    periods = np.diff(peak_times)
    avg_period = np.mean(periods)
    A_g = np.max(x_seg) - np.min(x_seg)
    B_g = 2 * np.pi / avg_period
    F_g = np.mean(x_seg)
    initial_guess = [A_g, 0.0001, B_g, 0.0, 0.0, 0.0, F_g]

    params, covariance = curve_fit(
        exp_sine,
        t_seg,
        x_seg,
        p0=initial_guess,
        maxfev=100000,
    )

    print(avg_period)
    print(A_g)

    print(f"Segment {i} params:", params)

    fitted_x_seg = exp_sine(t_seg, *params)
    drift = eq_drift(t_seg, *params)
    plt.scatter(t_seg, x_seg, label="Data", color="blue", s=1)

    plt.plot(
        t_seg,
        fitted_x_seg,
        color="red",
    )
    plt.plot(
        t_seg,
        drift,
        color="red",
    )
plt.show()
