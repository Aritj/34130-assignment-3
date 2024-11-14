import numpy as np
import matplotlib.pyplot as plt
from q2_1 import noisy_wdm_signal, Fs_wdm, F_spacing, N_channels, plot_psd

# Step 1: Define Gaussian-shaped optical filter
Rb = 10e9  # Baud rate (10 Gbaud)
B_3dB = 2 * Rb  # 3-dB bandwidth of the filter
ord = 1  # Order of the Gaussian filter
F = np.fft.fftfreq(len(noisy_wdm_signal), d=1 / Fs_wdm)  # Frequency grid

# Construct the field transfer function
Hf_power = np.exp(-0.5 * ((F / B_3dB) ** (2 * ord)))  # Power transfer function
Hf_field = np.sqrt(Hf_power)  # Field transfer function

# Apply out-of-band rejection
RejectionRatio_dB = 20  # Out-of-band rejection ratio in dB
Hf_min = 1 / (10 ** (RejectionRatio_dB / 20))
Hf_field[Hf_field < Hf_min] = Hf_min

# Step 2: Apply optical filter to the first channel of the WDM signal
channel_center_freq = (
    -(N_channels - 1) / 2 * F_spacing
)  # Center frequency of the first channel
carrier = np.exp(
    -1j * 2 * np.pi * channel_center_freq * np.arange(len(noisy_wdm_signal)) / Fs_wdm
)
demultiplexed_signal = noisy_wdm_signal * carrier  # Shift first channel to baseband

demultiplexed_signal_fft = np.fft.fftshift(
    np.fft.fft(demultiplexed_signal)
)  # FFT of signal
filtered_signal_fft = demultiplexed_signal_fft * Hf_field  # Apply filter
filtered_signal = np.fft.ifft(
    np.fft.ifftshift(filtered_signal_fft)
)  # IFFT back to time domain

# Step 3: Detect the filtered channel
# Function to plot eyediagram


def plot_eye_diagram(signal, samples_per_symbol, title):
    plt.figure()
    for i in range(0, len(signal) - 2 * samples_per_symbol, 2 * samples_per_symbol):
        trace = signal[i : i + 2 * samples_per_symbol]
        plt.plot(trace, color="blue", alpha=0.5)
    plt.xlabel("Samples")
    plt.ylabel("Amplitude")
    plt.title(title)
    plt.grid()
    plt.show()


if __name__ == "__main__":
    plot_psd(np.real(filtered_signal), Fs_wdm, "PSD of Extracted Channel")

    # Detect using a single photodiode (magnitude squared)
    detected_signal = np.abs(filtered_signal) ** 2

    # Plot PSD of detected signal
    plot_psd(detected_signal, Fs_wdm, "PSD of Detected Signal")

    # Plot eyediagram
    samples_per_symbol = int(Fs_wdm / Rb)  # Samples per symbol
    plot_eye_diagram(
        np.real(filtered_signal),
        samples_per_symbol,
        title="Eyediagram of Filtered Signal",
    )

    # Questions:
    # Do you see clear eyediagrams? What components are visible in the power spectral density?
    # What is the limiting factor for performance?
