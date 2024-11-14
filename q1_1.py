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

# Mark signal and noise levels
signal_level = 10 * np.log10(P_signal)
noise_level = 10 * np.log10(P_noise)

# Change number of samples per symbol and observe effects
Nss_new = 32  # Updated number of samples per symbol
Fs_new = Nss_new * Rb
T_total_new = Nsym * Ts
N_total_new = int(Fs_new * T_total_new)
t_new = np.linspace(0, T_total_new, N_total_new, endpoint=False)
ook_signal_new = np.repeat(symbols, Nss_new)

# Generate new noisy signal
P_signal_new = np.mean(ook_signal_new**2)
N_ASE_new = P_signal_new / (2 * B_ref * OSNR_linear)
P_noise_new = N_ASE_new * Fs_new
sigma_new = np.sqrt(P_noise_new / 2)
noise_new = sigma_new * (
    np.random.randn(len(ook_signal_new)) + 1j * np.random.randn(len(ook_signal_new))
)
noisy_signal_new = ook_signal_new + np.real(noise_new)


# Plot Power Spectral Density (PSD) of OOK signal
def plot_psd(signal, fs, title):
    f, Pxx = plt.psd(signal, NFFT=1024, Fs=fs, scale_by_freq=False)
    plt.title(title)
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Power/Frequency (dB/Hz)")
    plt.show()


if __name__ == "__main__":
    plot_psd(ook_signal, Fs, "PSD of OOK Signal")
    plot_psd(noisy_signal, Fs, "PSD of Noisy OOK Signal")
    print(f"Signal Level: {signal_level:.2f} dB")
    print(f"Noise Level: {noise_level:.2f} dB")
    plot_psd(ook_signal_new, Fs_new, "PSD of OOK Signal (Nss=32)")
    plot_psd(noisy_signal_new, Fs_new, "PSD of Noisy OOK Signal (Nss=32)")

    # Compare observations and answer questions
    # print("Out-of-band noise level is dependent on the sampling rate (Nss).")
    # print("OSNR primarily refers to in-band signal-to-noise ratio.")
