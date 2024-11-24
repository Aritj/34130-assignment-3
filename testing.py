import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import lfilter

np.random.seed(34310)

# Step 1: Generate random bits and map to OOK symbols
L = 40  # Number of random bits
bits = np.random.randint(0, 2, L)
symbols = bits * 2 - 1  # Map to OOK symbols (+1, -1)

# Step 2: NRZ shaping
N = 64  # Number of samples per symbol (upsampling factor)
nrz_waveform = np.repeat(symbols, N)

# Upsample symbols
upsampled_symbols = np.zeros(len(symbols) * N)
upsampled_symbols[::N] = symbols

spectrum_xlim = (-0.5, 0.5)  # Normalized frequency range

def generate_rrc_filter(roll_off, filter_length, upsampling_factor):
    """
    Generate RRC filter coefficients in the time domain.
    """
    t = np.arange(-filter_length / 2, filter_length / 2 + 1 / upsampling_factor, 1 / upsampling_factor)
    rrc_filter = np.zeros_like(t)

    for i, tau in enumerate(t):
        if tau == 0:
            rrc_filter[i] = 1 - roll_off + (4 * roll_off / np.pi)
        elif abs(tau) == 1 / (4 * roll_off):
            rrc_filter[i] = (
                roll_off / np.sqrt(2)
                * ((1 + 2 / np.pi) * np.sin(np.pi / (4 * roll_off))
                   + (1 - 2 / np.pi) * np.cos(np.pi / (4 * roll_off)))
            )
        else:
            numerator = np.sin(np.pi * tau * (1 - roll_off)) + 4 * roll_off * tau * np.cos(np.pi * tau * (1 + roll_off))
            denominator = np.pi * tau * (1 - (4 * roll_off * tau) ** 2)
            rrc_filter[i] = numerator / denominator

    return rrc_filter


# Generate RRC filter
roll_off = 0.5
filter_length = 10  # in symbols
rrc_filter = generate_rrc_filter(roll_off, filter_length, N)

# Normalize the filter
#rrc_filter /= np.sqrt(np.sum(rrc_filter**2))

# Apply RRC filter to upsampled symbols
rrc_waveform = np.convolve(upsampled_symbols, rrc_filter, mode='full')

# Center the RRC filter by compensating the delay
group_delay = (len(rrc_filter) - N) // 2  # Group delay in samples
rrc_waveform_aligned = rrc_waveform[group_delay : group_delay + len(nrz_waveform)]

# Plot NRZ and RRC waveforms (overlapping for comparison)
plt.figure()
plt.plot(nrz_waveform, label="NRZ Waveform", linewidth=2)
plt.plot(rrc_waveform_aligned, label="RRC Waveform (Aligned)", linewidth=2, alpha=0.8)
plt.title("Aligned RRC Shaped OOK Waveform")
plt.xlabel("Samples")
plt.ylabel("Amplitude")
plt.legend()
plt.grid(True)
plt.ylim(top=2)
plt.xlim(0, len(nrz_waveform))
plt.show()


# Step 4: Change roll-off factor and filter length, observe waveform and spectrum
roll_offs = [0.2, 0.5, 0.8]
filter_lengths = [5, 10, 15]

for roll_off in roll_offs:
    plt.figure(figsize=(12, 6))

    for i, filter_length in enumerate(filter_lengths):
        rrc_filter = generate_rrc_filter(roll_off, filter_length, N)
        rrc_waveform = lfilter(rrc_filter, 1.0, upsampled_symbols)
        
        # Center the RRC filter by compensating the delay
        group_delay = (len(rrc_filter) - N) // 2  # Group delay in samples
        rrc_waveform_aligned = rrc_waveform[group_delay : group_delay + len(nrz_waveform)]

        # Calculate spectrum
        spectrum = np.fft.fftshift(np.abs(np.fft.fft(rrc_waveform)))
        freqs = np.fft.fftshift(np.fft.fftfreq(len(rrc_waveform), 1 / N))
        
        # Plot waveform
        plt.subplot(2, 3, i + 1)
        plt.plot(rrc_waveform_aligned)
        plt.plot(nrz_waveform)
        plt.title(f"Waveform (roll_off={roll_off}, len={filter_length})")
        plt.xlabel("Samples")
        plt.ylabel("Amplitude")
        plt.grid()
        plt.xlim(0, len(rrc_waveform_aligned))
        
        # Plot spectrum
        plt.subplot(2, 3, i + 4)
        plt.plot(freqs, 20 * np.log10(spectrum))
        plt.title(f"Spectrum (roll_off={roll_off}, len={filter_length})")
        plt.xlabel("Frequency")
        plt.ylabel("Magnitude (dB)")
        plt.grid()

    plt.tight_layout()
    plt.show()


# Step 5: Adapt to 4PAM
pam4_symbols = np.random.randint(0, 4, L) * 2 - 3  # Map to 4PAM levels (-3, -1, 1, 3)
pam4_nrz_waveform = np.repeat(pam4_symbols, N)  # NRZ waveform for 4PAM

# Upsample 4PAM symbols
pam4_upsampled = np.zeros(len(pam4_symbols) * N)
pam4_upsampled[::N] = pam4_symbols

# Apply RRC shaping for 4PAM
rrc_waveform_4pam = np.convolve(pam4_upsampled, rrc_filter, mode="full")

# Compensate for group delay
group_delay = (len(rrc_filter) - N) // 2  # Group delay in samples
rrc_waveform_4pam_aligned = rrc_waveform_4pam[group_delay : group_delay + len(pam4_nrz_waveform)]

# Plot NRZ and RRC waveforms for 4PAM
plt.figure()
plt.plot(pam4_nrz_waveform, label="4PAM NRZ Waveform", linewidth=2)
plt.plot(rrc_waveform_4pam_aligned, label="4PAM RRC Waveform (Aligned)", linewidth=2, alpha=0.8)
plt.title("Aligned RRC Shaped 4PAM Waveform")
plt.xlabel("Samples")
plt.ylabel("Amplitude")
plt.legend()
plt.ylim(top=6)
plt.xlim(0, len(rrc_waveform_4pam_aligned))
plt.grid(True)
plt.show()
