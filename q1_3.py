import numpy as np
import matplotlib.pyplot as plt
from q1_1 import Rb, Fs, symbols, ook_signal, noisy_signal
from q1_2 import filtered_signal, plot_psd

# Step 1: Setup parameters for BER analysis
OSNR_values_dB = np.arange(
    16, 5, -1
)  # OSNR values from 16 dB to 6 dB, decrement by 1 dB
BER_results = []  # To store BER results


# Function to calculate BER
def calculate_ber(original_symbols, detected_symbols):
    errors = np.sum(original_symbols != detected_symbols)
    return errors / len(original_symbols)


# Step 2: Iterate through OSNR values
for OSNR_dB in OSNR_values_dB:
    OSNR_linear = 10 ** (OSNR_dB / 10)

    # Recalculate noise power
    P_signal = np.mean(ook_signal**2)  # Signal power
    B_ref = 12.5e9  # Reference bandwidth
    N_ASE = P_signal / (2 * B_ref * OSNR_linear)
    P_noise = N_ASE * Fs

    # Generate new noise and add to signal
    sigma = np.sqrt(P_noise / 2)
    noise = sigma * (
        np.random.randn(len(ook_signal)) + 1j * np.random.randn(len(ook_signal))
    )
    noisy_signal_iter = ook_signal + np.real(noise)

    # Apply filtering
    noisy_signal_fft = np.fft.fftshift(np.fft.fft(noisy_signal_iter))
    filtered_signal_fft = noisy_signal_fft * np.sqrt(
        np.exp(
            -0.5 * ((np.fft.fftfreq(len(noisy_signal_iter), d=1 / Fs) / (4 * Rb)) ** 2)
        )
    )
    filtered_signal_iter = np.fft.ifft(np.fft.ifftshift(filtered_signal_fft))

    # Step 3: Perform detection (downsampling and thresholding)
    downsampled_signal = filtered_signal_iter[::64]  # Downsample by Nss
    detected_symbols = (np.real(downsampled_signal) > 0.5).astype(
        int
    )  # Simple thresholding

    # Calculate BER
    BER = calculate_ber(symbols[: len(detected_symbols)], detected_symbols)
    BER_results.append(BER)

# Step 4: Plot OSNR-BER curve
plt.figure()
plt.plot(OSNR_values_dB, -np.log10(BER_results), "-*")
plt.title("OSNR vs -log10(BER)")
plt.xlabel("OSNR (dB)")
plt.ylabel("-log10(BER)")
plt.grid()
plt.show()

# Observations and comparisons
print("BER results:")
for osnr, ber in zip(OSNR_values_dB, BER_results):
    print(f"OSNR: {osnr} dB, BER: {ber:.2e}")
