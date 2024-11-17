import numpy as np
import matplotlib.pyplot as plt
from q1_1 import (
    seed,
    SimulationConfig,
    SimulationResults,
    generate_OOK_signal,
    calculate_signal_power,
    add_awgn,
    compute_psd,
    plot_psd_with_levels,
)

np.random.seed(seed)  # Set seed for reproducability


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
    )
    plt.title("Gaussian Filter Transfer Function")
    plt.xlabel("Frequency (GHz)")
    plt.ylabel("Rejection (dB)")
    plt.grid()
    plt.legend()
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


# Main function
def main():
    config = SimulationConfig()
    results = SimulationResults()

    # Generate OOK signal and add noise
    results.time, results.signal, bits = generate_OOK_signal(config)
    results.signal_power = calculate_signal_power(results.signal)
    results.noisy_signal = add_awgn(results.signal, results.signal_power, config)

    # Compute PSD of noisy signal
    sampling_rate = config.baud_rate * config.N_samples_per_symbol
    freqs, psd_noisy = compute_psd(results.noisy_signal, sampling_rate)

    # Create Gaussian filter
    bw_3dB = 4 * config.baud_rate  # 3-dB bandwidth
    rejection_ratio_dB = 20  # Rejection ratio in dB
    filter_transfer = create_gaussian_filter(
        freqs, center_freq=0, bw_3dB=bw_3dB, rejection_ratio_dB=rejection_ratio_dB
    )

    # Apply the filter to the noisy signal
    filtered_signal = apply_optical_filter(results.noisy_signal, filter_transfer)

    # Compute PSD of filtered signal
    freqs, psd_filtered = compute_psd(filtered_signal, sampling_rate)

    plot_filter_transfer_function(freqs, filter_transfer)

    # Plot results
    plot_psd_with_levels(
        freqs,
        psd_noisy,
        results.signal_power,
        results.signal_power / (10 ** (config.OSNR_dB / 10)),
        title="PSD of Noisy Signal",
    )
    plot_psd_with_levels(
        freqs,
        psd_filtered,
        results.signal_power,
        results.signal_power / (10 ** (config.OSNR_dB / 10)),
        title="PSD of Filtered Signal",
    )

    # Plot waveforms
    plot_waveform(results.noisy_signal, results.time, title="Waveform of Noisy Signal")
    plot_waveform(filtered_signal, results.time, title="Waveform of Filtered Signal")

    # Plot eye diagrams
    plot_eye_diagram(
        results.noisy_signal,
        config.N_samples_per_symbol,
        title="Eye Diagram of Noisy Signal",
    )
    plot_eye_diagram(
        filtered_signal,
        config.N_samples_per_symbol,
        title="Eye Diagram of Filtered Signal",
    )


if __name__ == "__main__":
    main()
