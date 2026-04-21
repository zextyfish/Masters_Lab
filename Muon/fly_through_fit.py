import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import landau, norm
from scipy.signal import fftconvolve
from scipy.optimize import curve_fit

# count_25 = 14568
count_100 = 57569


def langauss(x, loc, scale, sigma, A):
    dx = x[1] - x[0]
    x_ext = np.linspace(x.min() - 5 * sigma, x.max() + 5 * sigma, len(x) * 2)

    L = landau.pdf(x_ext, loc=loc, scale=scale)
    G = norm.pdf(x_ext, loc=0, scale=sigma)

    conv = fftconvolve(L, G, mode="same") * dx
    return A * np.interp(x, x_ext, conv)


data_100 = np.genfromtxt("fly_through_100p.TKA", skip_header=2)
data_50 = np.genfromtxt("fly_through_50p.TKA", skip_header=2)
data_25_raw = np.genfromtxt("fly_through_100.TKA", skip_header=8)
print(len(data_100))
print(len(data_50))
# data_25 = np.genfromtxt("fly_through_25.TKA", skip_header=2)
data_25 = data_25_raw.reshape(-1, 4).sum(axis=1)
# data_100 = data_100 / count_100
# data_100_rebin = data_100_rebin / count_100
# data_100 = data_100[:3001]
# data_25 = data_25[:3001]
# data_100_rebin = data_100_rebin[:200]

channel = np.arange(len(data_100))
# channel_rebin = np.arange(len(data_100_rebin))
# print("fit started")

# --- plot ---
# G = norm.pdf(channel_rebin, loc=50, scale=20)
# L = norm.pdf(channel_rebin, loc=70, scale=20)
# plt.plot(channel_rebin, data_100_rebin, label="Data", alpha=0.6)
# plt.plot(channel_rebin, G, label="Data", alpha=0.6)
# plt.plot(channel_rebin, L, label="Data", alpha=0.6)
# plt.plot(channel, data_25 , label="Data", alpha=0.6)
# plt.plot(channel, langauss(channel, *params), label="Langauss (fit)", linewidth=2)

plt.plot(channel, data_100, label="100", alpha=0.6)
plt.plot(channel, data_50, label="50", alpha=0.6)
plt.plot(channel, data_25, label="25", alpha=0.6)
# plt.plot(channel, L_scaled, "--", label="Landau (scaled)")
# plt.plot(channel, G_scaled, "--", label="Gaussian (scaled)")

plt.legend()
plt.xlabel("Channels")
plt.ylabel("Counts pre second")
# plt.title("Langauss Fit with Components")
plt.grid(True)
plt.show()

# peak_idx = np.argmax(data_100)

# # p0 = [
# #     channel[np.argmax(data_100)],  # loc
# #     20,  # scale
# #     5,  # sigma
# #     np.trapezoid(data_100, channel) / len(channel),
# # ]
# p0 = [0.0, 0.0, 0.0, 0.0]
# # peak = channel[np.argmax(data_100)]

# # bounds = ([peak - 200, 1, 0.5, 0], [peak + 200, 100, 30, np.inf])

# # params, _ = curve_fit(langauss, channel, data_100, p0=p0, bounds=bounds, maxfev=100000)
# params, _ = curve_fit(langauss, channel, data_100, p0=p0, maxfev=100000)

# loc, scale, sigma, A = params

# # --- components (for plotting) ---
# L = landau.pdf(channel, loc=loc, scale=scale)
# G = norm.pdf(channel, loc=loc, scale=sigma)

# # scale them for visualization (important!)
# L_scaled = A * L / np.max(L)
# G_scaled = A * G / np.max(G)
