import numpy as np
import scipy
import matplotlib.pyplot as plt
import math
import random
from scipy.signal import welch

# Task 1: Parameters
N_ss = 64  # Number of samples per symbol
R_s = 10e9  # Signal baud rate (Gbaud)
N_sym = 1000  # Number of symbols

np.random.seed(34310)

# Generate a random bits sequence
random_bits = np.random.randint(2, size=N_sym)


# Define Mapper function
def Mapper(bits):
    A = 1  # Amplitude for '1' bit
    return np.where(bits == 1, A, 0)


# Map bits into symbols
OOKsymbols = Mapper(random_bits)

plt.figure()
plt.plot(OOKsymbols[0:10], ".")
plt.grid()
plt.show()

bit_period = 1 / R_s  # Symbol period (Ts)

# Create the OOK signal waveform
time = np.linspace(0, N_sym * bit_period, N_sym)

# Plot the OOK signal
plt.figure("Symbols")
plt.plot(time, OOKsymbols, "ob", linewidth=2)
plt.xlabel("Time (s)")
plt.ylabel("Amplitude")
plt.title("10 Gb/s OOK Signal")
plt.grid(True)
plt.axis([0, time[-1], -0.2, 1.2])

# Zoom into the first 10 bits
zoomed_bits = 10
plt.xlim(0, zoomed_bits * bit_period)
plt.ylim(-0.2, 1.2)  # Assuming you want to keep the y-axis limits the same

plt.show()

# Plot the OOK signal
plt.figure("Symbols")
plt.plot(time, OOKsymbols, "ob", linewidth=2)
plt.xlabel("Time (s)")
plt.ylabel("Amplitude")
plt.title("10 Gb/s OOK Signal")
plt.grid(True)
plt.axis([0, time[-1], -0.2, 1.2])

# Zoom into the first 10 bits
zoomed_bits = 10
plt.xlim(0, zoomed_bits * bit_period)
plt.ylim(-0.2, 1.2)  # Assuming you want to keep the y-axis limits the same

plt.show()
