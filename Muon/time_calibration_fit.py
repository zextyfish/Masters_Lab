import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import landau, norm
from scipy.signal import fftconvolve
from scipy.optimize import curve_fit


def langauss(x, loc, scale, sigma, A):
    dx = x[1] - x[0]
    x_ext = np.linspace(x.min() - 5 * sigma, x.max() + 5 * sigma, len(x) * 2)

    L = landau.pdf(x_ext, loc=loc, scale=scale)
    G = norm.pdf(x_ext, loc=0, scale=sigma)

    conv = fftconvolve(L, G, mode="same") * dx
    return A * np.interp(x, x_ext, conv)


data_4 = np.genfromtxt("r2a4.4.TKA", skip_header=2)

channel = np.arange(len(data_4))
print("Fit started")
# peak_idx = np.argmax(data_100)
# p0 = [
#     channel[np.argmax(data_100)],  # loc
#     20,                             # scale
#     5,                              # sigma
#     np.trapezoid(data_100, channel) / len(channel)
# ]
# peak = channel[np.argmax(data_100)]

# bounds = (
#     [peak - 200,   1,   0.5,   0],
#     [peak + 200, 100,  30,    np.inf]
# )

# params, _ = curve_fit(
#     langauss,
#     channel,
#     data_100,
#     p0=p0,
#     bounds=bounds,
#     maxfev=100000
# )

# loc, scale, sigma, A = params

# --- components (for plotting) ---
L = landau.pdf(channel, loc=loc, scale=scale)
G = norm.pdf(channel, loc=loc, scale=sigma)

# scale them for visualization (important!)
L_scaled = A * L / np.max(L)
G_scaled = A * G / np.max(G)

# --- plot ---
plt.plot(channel, data_100, label="Data", alpha=0.6)
# plt.plot(channel, data_25 , label="Data", alpha=0.6)
plt.plot(channel, langauss(channel, *params), label="Langauss (fit)", linewidth=2)

plt.plot(channel, L_scaled, "--", label="Landau (scaled)")
plt.plot(channel, G_scaled, "--", label="Gaussian (scaled)")

plt.legend()
plt.xlabel("Channel")
plt.ylabel("Counts")
plt.title("Langauss Fit with Components")
plt.grid(True)
plt.show()
