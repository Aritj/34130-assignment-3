import numpy as np
import matplotlib.pyplot as plt

from q1_1 import Rb, noisy_signal, Fs, plot_psd

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

# Apply the optical filter to the noisy signal
noisy_signal_fft = np.fft.fftshift(np.fft.fft(noisy_signal))  # FFT of noisy signal
filtered_signal_fft = noisy_signal_fft * Hf_field  # Apply filter in frequency domain
filtered_signal = np.fft.ifft(
    np.fft.ifftshift(filtered_signal_fft)
)  # IFFT to time domain


def plot_power_transfer_function(field_transfer_function):
    # Plot the power transfer function of the optical filter
    plt.figure()
    plt.plot(
        F / 1e9,
        -20 * np.log10(field_transfer_function),
        label="Gaussian Filter",
    )
    plt.xlabel("Frequency (GHz)")
    plt.ylabel("Rejection (dB)")
    plt.title("Gaussian Filter Power Transfer Function")
    plt.legend()
    plt.grid()
    plt.show()


if __name__ == "__main__":
    plot_power_transfer_function(Hf_field)

    # Step 4: Plot filtered signal's PSD
    plot_psd(np.real(filtered_signal), Fs, "PSD of Filtered OOK Signal")
