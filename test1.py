import numpy as np
import matplotlib.pyplot as plt

# Step 1: Set up simulation parameters
Nss = 64  # Number of samples per symbol
Rb = 10e9  # Baud rate (10 Gbaud)
Nsym = 1000  # Number of symbols
Fs = Nss * Rb  # Sampling frequency
Ts = 1 / Rb  # Symbol period
T_total = Nsym * Ts  # Total simulation time
N_total = int(Fs * T_total)  # Total number of samples
t = np.linspace(0, T_total, N_total, endpoint=False)  # Time grid

# Step 2: Initialize random number generator and generate OOK signal
np.random.seed(34310)  # Seed for reproducibility
symbols = np.random.choice([0, 1], Nsym)  # Generate random binary symbols
ook_signal = np.repeat(symbols, Nss)  # NRZ pulse shaping (repeat each symbol Nss times)


# Plot Power Spectral Density (PSD) of OOK signal
def plot_psd(signal, fs, title):
    f, Pxx = plt.psd(signal, NFFT=1024, Fs=fs, scale_by_freq=False)
    plt.title(title)
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Power/Frequency (dB/Hz)")
    plt.show()


plot_psd(ook_signal, Fs, "PSD of OOK Signal")

# Step 3: Add optical noise to the OOK signal
OSNR_dB = 20  # Optical signal-to-noise ratio in dB
OSNR_linear = 10 ** (OSNR_dB / 10)  # Convert OSNR to linear scale
P_signal = np.mean(ook_signal**2)  # Calculate signal power
B_ref = 12.5e9  # Reference bandwidth (~12.5 GHz)
N_ASE = P_signal / (2 * B_ref * OSNR_linear)  # Noise power spectral density
P_noise = N_ASE * Fs  # Total noise power

# Generate complex AWGN
sigma = np.sqrt(P_noise / 2)  # Standard deviation of noise
noise = sigma * (
    np.random.randn(len(ook_signal)) + 1j * np.random.randn(len(ook_signal))
)
noisy_signal = ook_signal + np.real(noise)  # Add noise to the real part of the signal

# Plot PSD of noisy signal
plot_psd(noisy_signal, Fs, "PSD of Noisy OOK Signal")

# Step 4: Mark signal and noise levels
signal_level = 10 * np.log10(P_signal)
noise_level = 10 * np.log10(P_noise)
print(f"Signal Level: {signal_level:.2f} dB")
print(f"Noise Level: {noise_level:.2f} dB")

# Step 5: Change number of samples per symbol and observe effects
Nss_new = 32  # Updated number of samples per symbol
Fs_new = Nss_new * Rb
T_total_new = Nsym * Ts
N_total_new = int(Fs_new * T_total_new)
t_new = np.linspace(0, T_total_new, N_total_new, endpoint=False)
ook_signal_new = np.repeat(symbols, Nss_new)

plot_psd(ook_signal_new, Fs_new, "PSD of OOK Signal (Nss=32)")

# Generate new noisy signal
P_signal_new = np.mean(ook_signal_new**2)
N_ASE_new = P_signal_new / (2 * B_ref * OSNR_linear)
P_noise_new = N_ASE_new * Fs_new
sigma_new = np.sqrt(P_noise_new / 2)
noise_new = sigma_new * (
    np.random.randn(len(ook_signal_new)) + 1j * np.random.randn(len(ook_signal_new))
)
noisy_signal_new = ook_signal_new + np.real(noise_new)

plot_psd(noisy_signal_new, Fs_new, "PSD of Noisy OOK Signal (Nss=32)")

# Compare observations and answer questions
print("Out-of-band noise level is dependent on the sampling rate (Nss).")
print("OSNR primarily refers to in-band signal-to-noise ratio.")

# Q1-2: Apply optical filtering to the noisy signal
# Define Gaussian-shaped optical filter parameters
B_3dB = 4 * Rb  # 3-dB bandwidth of the filter
ord = 1  # Order of the Gaussian filter
F = np.fft.fftfreq(len(noisy_signal), d=1 / Fs)  # Frequency grid
Hf_power = np.exp(-0.5 * ((F / B_3dB) ** (2 * ord)))  # Power transfer function
Hf_field = np.sqrt(Hf_power)  # Field transfer function

# Apply out-of-band rejection ratio
RejectionRatio_dB = 20  # Out-of-band rejection in dB
Hf_min = 1 / (10 ** (RejectionRatio_dB / 20))
Hf_field[Hf_field < Hf_min] = Hf_min

# Plot the power transfer function of the optical filter
plt.figure()
plt.plot(F / 1e9, -20 * np.log10(Hf_field), label="Gaussian Filter")
plt.xlabel("Frequency (GHz)")
plt.ylabel("Rejection (dB)")
plt.title("Gaussian Filter Power Transfer Function")
plt.legend()
plt.grid()
plt.show()

# Apply the optical filter to the noisy signal
noisy_signal_fft = np.fft.fftshift(np.fft.fft(noisy_signal))  # FFT of noisy signal
filtered_signal_fft = noisy_signal_fft * Hf_field  # Apply filter in frequency domain
filtered_signal = np.fft.ifft(
    np.fft.ifftshift(filtered_signal_fft)
)  # IFFT to time domain

# Step 4: Plot filtered signal's PSD
plot_psd(np.real(filtered_signal), Fs, "PSD of Filtered OOK Signal")
