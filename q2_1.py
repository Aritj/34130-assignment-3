import numpy as np
import matplotlib.pyplot as plt
from q1_1 import Rb, plot_psd

# Step 1: Set up parameters for WDM signal generation
Nss = 64  # Number of samples per symbol
Nsym = 100000  # Number of symbols per channel
F_spacing = 50e9  # Channel spacing in Hz
N_channels = 5  # Number of WDM channels
Fs_wdm = Nss * Rb  # Sampling frequency

# Time grid
T_total = Nsym / Rb  # Total simulation time
N_total = int(Fs_wdm * T_total)  # Total number of samples
t = np.linspace(0, T_total, N_total, endpoint=False)

# Step 2: Generate OOK signal for the base channel
np.random.seed(34310)  # Seed for reproducibility
symbols = np.random.choice([0, 1], Nsym)  # Generate random binary symbols
ook_signal = np.repeat(symbols, Nss)  # NRZ pulse shaping (repeat each symbol Nss times)

# Frequency grid
F = np.fft.fftfreq(N_total, d=1 / Fs_wdm)

# Step 3: Generate WDM signal by frequency-shifting the base channel
wdm_signal = np.zeros(N_total, dtype=complex)
for i in range(N_channels):
    f_shift = (i - (N_channels - 1) / 2) * F_spacing  # Frequency shift for each channel
    carrier = np.exp(
        1j * 2 * np.pi * f_shift * t
    )  # Complex exponential for frequency shifting
    wdm_signal += (
        ook_signal * carrier
    )  # Add the frequency-shifted channel to WDM signal


# Step 4: Add AWGN to the WDM signal
OSNR_per_channel_dB = 20  # OSNR per channel in dB
OSNR_total_dB = OSNR_per_channel_dB + 10 * np.log10(N_channels)  # Total OSNR in dB
OSNR_linear = 10 ** (OSNR_total_dB / 10)  # Convert total OSNR to linear scale
P_signal = np.mean(np.abs(wdm_signal) ** 2)  # Signal power
B_ref = 12.5e9  # Reference bandwidth (~12.5 GHz)
N_ASE = P_signal / (2 * B_ref * OSNR_linear)  # Noise power spectral density
P_noise = N_ASE * Fs_wdm  # Total noise power

# Generate AWGN noise
sigma = np.sqrt(P_noise / 2)
noise = sigma * (
    np.random.randn(len(wdm_signal)) + 1j * np.random.randn(len(wdm_signal))
)
noisy_wdm_signal = wdm_signal + noise


# Step 5: Mark signal and noise levels
signal_level = 10 * np.log10(P_signal)
noise_level = 10 * np.log10(P_noise)

if __name__ == "__main__":
    # Plot PSD of the WDM signal
    plot_psd(np.real(wdm_signal), Fs_wdm, "PSD of WDM Signal")

    # Plot PSD of the noisy WDM signal
    plot_psd(np.real(noisy_wdm_signal), Fs_wdm, "PSD of Noisy WDM Signal")

    print(f"Signal Level: {signal_level:.2f} dB")
    print(f"Noise Level: {noise_level:.2f} dB")

    # Questions:
    # Why the samples per symbol, Nss, is increased compared to single-channel simulation?
    # Answer: Increasing Nss ensures better resolution in the frequency domain, allowing accurate representation of closely spaced WDM channels.

    # Calculate the overall bandwidth of the WDM signal and the sampling frequency
    bandwidth_wdm = N_channels * F_spacing  # Total bandwidth of WDM signal
    print(f"Total Bandwidth of WDM Signal: {bandwidth_wdm / 1e9:.2f} GHz")
    print(f"Sampling Frequency: {Fs_wdm / 1e9:.2f} GHz")
