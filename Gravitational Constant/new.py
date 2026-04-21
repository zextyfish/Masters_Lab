import csv
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.signal import find_peaks

# --- 1. Data Loading ---
time = []
x = []

# with open("G_01_25.csv", "r") as file:
with open("Big_Mass.csv", "r") as file:
    reader = csv.reader(file)
    next(reader)  # Skip headers
    next(reader)

    for row in reader:
        if row[1]:
            time.append(float(row[0]))
            x.append(float(row[1]))

time = np.array(time)
x = np.array(x)

# --- 2. Segment Definitions ---
t0, t1 = 100, 3400
t2, t3 = 3800, 9000
t4, t5 = 9700, 15000

seg1 = (time >= t0) & (time < t1)
seg2 = (time >= t2) & (time < t3)
seg3 = (time >= t4) & (time < t5)

# Combine for the global fit
t_total = np.concatenate([time[seg1], time[seg2], time[seg3]])
x_total = np.concatenate([x[seg1], x[seg2], x[seg3]])


# --- 3. Define the Global Model ---
# Shared: D (t^2), E (t)
# Unique: A, lamb, B, C, F for each segment
def global_model(t, D, E, A1, l1, B1, C1, F1, A2, l2, B2, C2, F2, A3, l3, B3, C3, F3):

    y = np.zeros_like(t)

    # Segment 1
    m1 = (t >= t0) & (t < t1)
    y[m1] = (
        A1 * np.exp(-l1 * t[m1]) * np.sin(B1 * t[m1] + C1)
        + D * t[m1] ** 2
        + E * t[m1]
        + F1
    )

    # Segment 2
    m2 = (t >= t2) & (t < t3)
    y[m2] = (
        A2 * np.exp(-l2 * t[m2]) * np.sin(B2 * t[m2] + C2)
        + D * t[m2] ** 2
        + E * t[m2]
        + F2
    )

    # Segment 3
    m3 = (t >= t4) & (t < t5)
    y[m3] = (
        A3 * np.exp(-l3 * t[m3]) * np.sin(B3 * t[m3] + C3)
        + D * t[m3] ** 2
        + E * t[m3]
        + F3
    )

    return y


# --- 4. Generate Initial Guesses via Peak Finding ---
# We start with D=0 and E=0 as the initial guess for the shared drift
initial_guess = [0.0, 0.0]

segments_masks = [seg1, seg2, seg3]

plt.figure(figsize=(12, 6))

for i, mask in enumerate(segments_masks, start=1):
    t_seg = time[mask]
    x_seg = x[mask]

    peaks, _ = find_peaks(x_seg, prominence=0.005, distance=16)
    troughs, _ = find_peaks(-x_seg, prominence=0.005, distance=16)
    peak_time = t_seg[peaks]
    peak_x = x_seg[peaks]

    # plt.scatter(peak_time, peak_x, color="black", label="Peaks", alpha=0.5)

    avg_period = np.mean(np.diff(t_seg[peaks]))
    A_g = (np.max(x_seg) - np.min(x_seg)) / 2
    B_g = 2 * np.pi / avg_period
    F_g = np.mean(x_seg)

    # Append guesses for [A, lamb, B, C, F]
    initial_guess.extend([A_g, 0.0001, B_g, 0.0, F_g])

# --- 5. Perform Global Fit ---
print("Fitting global model...")
popt, pcov = curve_fit(global_model, t_total, x_total, p0=initial_guess, maxfev=200000)

# --- 6. Extract Results ---
D_fit, E_fit = popt[0], popt[1]
# F values are at indices 6, 11, and 16
F1_fit, F2_fit, F3_fit = popt[6], popt[11], popt[16]
avg_offset = np.mean([F1_fit, F2_fit, F3_fit])

print(f"\n--- Results ---")
print(f"Shared Quadratic Drift (D): {D_fit:.4e}")
print(f"Shared Linear Drift (E):    {E_fit:.4e}")
print(f"Average Segment Offset (F): {avg_offset:.4f}")
print(F1_fit, F2_fit, F3_fit)

# --- 7. Visualization ---
plt.scatter(time, x, s=1, color="gray", label="All Raw Data", alpha=0.5)

# Plot the total fitted model
# We use t_total to show the fit only where segments exist
y_fit = global_model(t_total, *popt)
plt.scatter(t_total, y_fit, s=1, color="red", label="Global Fit", zorder=3)

# # Plot the underlying shared drift with the calculated offset
# t_drift = np.linspace(t0, t5, 1000)
# # Adding the average offset ensures the curve sits within the data range
# drift_curve = D_fit * t_drift**2 + E_fit * t_drift + avg_offset

# plt.plot(
#     t_drift,
#     drift_curve,
#     color="black",
#     linestyle="--",
#     linewidth=2,
#     label=f"Common Drift (Offset: {avg_offset:.2f})",
# )

# Optional: Plot the specific drift line for each segment
for i, (start, end, F_val) in enumerate(
    [(t0, t1, F1_fit), (t2, t3, F2_fit), (t4, t5, F3_fit)]
):
    t_s = np.linspace(start, end, 100)
    d_s = D_fit * t_s**2 + E_fit * t_s + F_val
    plt.plot(t_s, d_s, color="blue", alpha=0.6, label=f"Seg {i+1} Trend")

# Formatting
for vline in [t0, t1, t2, t3, t4, t5]:
    plt.axvline(vline, color="green", linestyle=":", alpha=0.4)

plt.title("Multi-Segment Global Fit with Shared Drift & Offset")
plt.xlabel("Time")
plt.ylabel("X")
plt.legend(loc="best", markerscale=5)
plt.grid(True, alpha=0.2)
plt.tight_layout()
plt.show()

# Linear Drift
# import csv
# import numpy as np
# import matplotlib.pyplot as plt
# from scipy.optimize import curve_fit
# from scipy.signal import find_peaks

# # --- 1. Data Loading ---
# # (Assumed same as your snippet)
# time = []
# x = []

# with open("G_01_25.csv", "r") as file:
#     reader = csv.reader(file)
#     next(reader)  # Skip headers
#     next(reader)
#     for row in reader:
#         if row[1]:
#             time.append(float(row[0]))
#             x.append(float(row[1]))

# time = np.array(time)
# x = np.array(x)

# # --- 2. Segment Definitions ---
# t0, t1 = 100, 3100
# t2, t3 = 3800, 10000
# t4, t5 = 10500, 17900

# seg1 = (time >= t0) & (time < t1)
# seg2 = (time >= t2) & (time < t3)
# seg3 = (time >= t4) & (time < t5)

# t_total = np.concatenate([time[seg1], time[seg2], time[seg3]])
# x_total = np.concatenate([x[seg1], x[seg2], x[seg3]])


# # --- 3. Define the Global Model (Linear Drift Only) ---
# # Shared: E (t)
# # Unique: A, lamb, B, C, F for each segment
# def global_model_linear(
#     t, E, A1, l1, B1, C1, F1, A2, l2, B2, C2, F2, A3, l3, B3, C3, F3
# ):

#     y = np.zeros_like(t)

#     # Shared Linear Drift Component: E * t
#     # Note: We apply E*t to all segments

#     # Segment 1
#     m1 = (t >= t0) & (t < t1)
#     y[m1] = A1 * np.exp(-l1 * t[m1]) * np.sin(B1 * t[m1] + C1) + E * t[m1] + F1

#     # Segment 2
#     m2 = (t >= t2) & (t < t3)
#     y[m2] = A2 * np.exp(-l2 * t[m2]) * np.sin(B2 * t[m2] + C2) + E * t[m2] + F2

#     # Segment 3
#     m3 = (t >= t4) & (t < t5)
#     y[m3] = A3 * np.exp(-l3 * t[m3]) * np.sin(B3 * t[m3] + C3) + E * t[m3] + F3

#     return y


# # --- 4. Generate Initial Guesses ---
# initial_guess = [0.0]  # Only one shared parameter now (E)

# segments_masks = [seg1, seg2, seg3]

# for i, mask in enumerate(segments_masks, start=1):
#     t_seg = time[mask]
#     x_seg = x[mask]

#     peaks, _ = find_peaks(x_seg, prominence=0.005, distance=16)

#     avg_period = np.mean(np.diff(t_seg[peaks])) if len(peaks) > 1 else 100.0
#     A_g = (np.max(x_seg) - np.min(x_seg)) / 2
#     B_g = 2 * np.pi / avg_period
#     F_g = np.mean(x_seg)

#     initial_guess.extend([A_g, 0.0001, B_g, 0.0, F_g])

# # --- 5. Perform Global Fit ---
# print("Fitting global model with linear drift...")
# popt, pcov = curve_fit(
#     global_model_linear, t_total, x_total, p0=initial_guess, maxfev=200000
# )

# # --- 6. Extract Results ---
# E_fit = popt[0]
# F1_fit, F2_fit, F3_fit = popt[5], popt[10], popt[15]
# print(f"\n--- Results ---")
# print(f"Shared Linear Drift (E): {E_fit:.4e}")
# print(f"Drift (F1): {F1_fit:.4e}")
# print(f"Drift (F2): {F2_fit:.4e}")
# print(f"Drift (F3): {F3_fit:.4e}")

# # --- 7. Visualization ---
# plt.figure(figsize=(12, 6))
# plt.scatter(time, x, s=1, color="lightgray", label="All Raw Data")

# # Fit Plot
# y_fit = global_model_linear(t_total, *popt)
# plt.scatter(
#     t_total, y_fit, s=1, color="red", label="Global Fit (Linear Drift)", zorder=3
# )

# # # Shared Drift Trend Plot
# # t_drift = np.linspace(t0, t5, 1000)
# # drift_curve = E_fit * t_drift # Linear only: y = E*t
# # plt.plot(t_drift, drift_curve, color='black', linestyle='--', linewidth=2, label='Common Linear Trend')

# plt.title("Multi-Segment Global Fit with Shared Linear Drift")
# plt.xlabel("Time")
# plt.ylabel("X")
# plt.legend()
# plt.grid(True, alpha=0.2)
# plt.show()
