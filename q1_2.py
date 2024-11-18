import numpy as np
import matplotlib.pyplot as plt
from q1_1 import (
    seed,
    SimulationConfig,
    SimulationResults,
    compute_psd,
    calculate_results,
)

np.random.seed(seed)  # Set seed for reproducability

# Set figure DPI to 300 (increasing plot resolution)
plt.rcParams["savefig.dpi"] = 300


# Function to create a Gaussian optical filter
def create_gaussian_filter(freqs, center_freq, bw_3dB, rejection_ratio_dB):
    f_1e_width = bw_3dB / (2 * np.sqrt(2 * np.log(2)))
    power_transfer = np.exp(-0.5 * ((freqs - center_freq) / f_1e_width) ** 2)
    field_transfer = np.sqrt(power_transfer)

    # Out-of-band rejection
    rejection_ratio_linear = 1 / (10 ** (rejection_ratio_dB / 20))
    field_transfer[field_transfer < rejection_ratio_linear] = rejection_ratio_linear

    return field_transfer


# Function to apply the Gaussian filter
def apply_optical_filter(signal, filter_transfer):
    # FFT of the signal
    signal_fft = np.fft.fftshift(np.fft.fft(signal))

    # Apply the filter in frequency domain
    filtered_fft = signal_fft * filter_transfer

    # IFFT back to time domain
    filtered_signal = np.fft.ifft(np.fft.ifftshift(filtered_fft))
    return filtered_signal


def plot_filter_transfer_function(freqs, filter_transfer):
    plt.figure()
    plt.plot(
        freqs / 1e9,
        10 * np.log10(np.abs(filter_transfer) ** 2),
        label="Filter Transfer Function (Power)",
        color="blue",
    )
    plt.title("Gaussian Filter Transfer Function")
    plt.xlabel("Frequency (GHz)")
    plt.ylabel("Rejection (dB)")
    plt.xlim(freqs.min() / 1e9, freqs.max() / 1e9)
    plt.grid()
    plt.show()


def plot_signals(results: SimulationResults, config: SimulationConfig):
    plt.figure()
    plt.plot(
        results.time * 1e9,
        results.noisy_signal,
        label="Noisy OOK signal",
        color="blue",
    )
    plt.plot(
        results.time * 1e9,
        results.filtered_signal,
        label="Filtered OOK signal",
        linestyle="-.",
        color="magenta",
    )
    plt.plot(
        results.time * 1e9,
        results.signal,
        linestyle="--",
        label="OOK signal",
        color="darkorange",
    )
    plt.xlabel("Time (ns)")
    plt.ylabel("Amplitude")
    plt.title(f"OOK and noisy OOK signals (N_ss={config.N_samples_per_symbol})")
    plt.grid()
    plt.legend()
    plt.ylim(-0.2, 1.4)
    plt.xlim(0, 1e9 * 10 / (config.baud_rate))
    plt.show()


# Function to plot eye diagram
def plot_eye_diagram(signal, samples_per_symbol, title):
    plt.figure()
    num_symbols = 200  # Number of symbols to display in the eye diagram
    signal = signal[: num_symbols * samples_per_symbol]
    time_axis = np.linspace(
        0, 1, samples_per_symbol, endpoint=False
    )  # One symbol period
    for i in range(num_symbols):
        plt.plot(
            time_axis,
            signal[i * samples_per_symbol : (i + 1) * samples_per_symbol].real,
            color="blue",
            alpha=0.5,
        )
    plt.title(title)
    plt.xlabel("Time (Symbol Periods)")
    plt.ylabel("Amplitude")
    plt.grid()
    plt.show()


"""
# Function to plot waveform
def plot_waveform(signal, time, title):
    plt.figure()
    total_time = time[-1]  # Total duration of the signal
    start = 0
    start_time = total_time * start / 100
    end_time = total_time * (start + 1) / 100

    mask = (time >= start_time) & (time < end_time)
    plt.plot(time[mask] * 1e9, signal[mask].real)  # Convert time to ns for readability
    plt.title(title)
    plt.xlabel("Time (ns)")
    plt.ylabel("Amplitude")
    plt.grid()
    plt.show()
"""


def plot_filtered_psd_with_levels(results: SimulationResults, config: SimulationConfig):
    freqs_original, psd_original = results.psd_original
    freqs_noisy, psd_noisy = results.psd_noisy
    freqs_filtered, psd_filtered = results.psd_filtered

    plt.figure(figsize=(18, 6))

    # Subplot for OOK PSD
    plt.subplot(1, 3, 1)
    plt.plot(
        freqs_original / 1e9,
        psd_original,
        label="OOK PSD",
        color="darkorange",
    )
    plt.axhline(
        10 * np.log10(results.signal_power),
        color="g",
        linestyle="--",
        label="Signal Level",
    )
    plt.axhline(
        10 * np.log10(results.noise_power),
        color="r",
        linestyle="--",
        label="Noise Level",
    )
    plt.ylim(-60, 50)
    plt.xlim(freqs_original.min() / 1e9, freqs_original.max() / 1e9)
    plt.title(f"PSD of OOK Signal (N_ss={config.N_samples_per_symbol})")
    plt.xlabel("Frequency (GHz)")
    plt.ylabel("PSD (dB/Hz)")
    plt.legend()
    plt.grid()

    # Subplot for noisy OOK PSD
    plt.subplot(1, 3, 2)
    plt.plot(freqs_noisy * 1e-9, psd_noisy, label="Noisy OOK PSD", color="blue")
    plt.axhline(
        10 * np.log10(results.signal_power),
        color="g",
        linestyle="--",
        label="Signal Level",
    )
    plt.axhline(
        10 * np.log10(results.noise_power),
        color="r",
        linestyle="--",
        label="Noise Level",
    )
    plt.ylim(-60, 50)
    plt.xlim(freqs_noisy.min() * 1e-9, freqs_noisy.max() * 1e-9)
    plt.title(f"PSD of Noisy OOK Signal (N_ss={config.N_samples_per_symbol})")
    plt.xlabel("Frequency (GHz)")
    plt.ylabel("PSD (dB/Hz)")
    plt.legend()
    plt.grid()

    # Subplot for filtered OOK PSD
    plt.subplot(1, 3, 3)
    plt.plot(
        freqs_filtered * 1e-9, psd_filtered, label="Filtered OOK PSD", color="magenta"
    )
    plt.axhline(
        10 * np.log10(results.signal_power),
        color="g",
        linestyle="--",
        label="Signal Level",
    )
    plt.axhline(
        10 * np.log10(results.noise_power),
        color="r",
        linestyle="--",
        label="Noise Level",
    )
    plt.ylim(-60, 50)
    plt.xlim(freqs_filtered.min() * 1e-9, freqs_filtered.max() * 1e-9)
    plt.title(f"PSD of Filtered OOK Signal (N_ss={config.N_samples_per_symbol})")
    plt.xlabel("Frequency (GHz)")
    plt.ylabel("PSD (dB/Hz)")
    plt.legend()
    plt.grid()

    plt.tight_layout()
    plt.show()


# Main function
def main():
    config = SimulationConfig()
    results = SimulationResults()

    results = calculate_results(config, results)

    # Compute PSD of noisy signal
    freqs, _ = results.psd_noisy

    # Create Gaussian filter
    filter_transfer = create_gaussian_filter(
        freqs, center_freq=0, bw_3dB=4 * config.baud_rate, rejection_ratio_dB=20
    )

    # Apply the filter to the noisy signal
    results.filtered_signal = apply_optical_filter(
        results.noisy_signal, filter_transfer
    )

    # Compute PSD of filtered signal
    results.psd_filtered = compute_psd(results.filtered_signal, results.sampling_rate)

    # Plot signals
    plot_signals(results, config)

    # Plot transfer function
    plot_filter_transfer_function(freqs, filter_transfer)

    # Plot PSD results
    plot_filtered_psd_with_levels(results, config)

    # Plot waveforms
    # plot_waveform(results.noisy_signal, results.time, title="Waveform of Noisy Signal")
    # plot_waveform(
    #    results.filtered_signal, results.time, title="Waveform of Filtered Signal"
    # )

    # Plot eye diagrams
    plot_eye_diagram(
        results.noisy_signal,
        config.N_samples_per_symbol,
        title="Eye Diagram of Noisy Signal",
    )
    plot_eye_diagram(
        results.filtered_signal,
        config.N_samples_per_symbol,
        title="Eye Diagram of Filtered Signal",
    )


if __name__ == "__main__":
    main()
