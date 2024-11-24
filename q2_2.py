from q2_1 import WDMConfig, generate_wdm_signal
from q1_1 import add_noise
from q1_2 import plot_filter_response, create_gaussian_filter, filter_signal
import numpy as np
import matplotlib.pyplot as plt


def apply_optical_filter(signal, Hf_field):
    """Apply optical filter to signal."""
    nfft = len(signal)
    signal_fft = np.fft.fftshift(np.fft.fft(signal))
    filtered_fft = signal_fft * Hf_field
    return np.fft.ifft(np.fft.ifftshift(filtered_fft))


def calculate_psd(signal, Fs):
    """Calculate power spectral density."""
    nfft = len(signal)
    f = np.fft.fftshift(np.fft.fftfreq(nfft, 1 / Fs))
    signal_fft = np.fft.fftshift(np.fft.fft(signal)) / nfft
    psd = (
        10 * np.log10(np.abs(signal_fft) ** 2 + np.finfo(float).eps) - 30
    )  # Convert to dBm/Hz
    return f, psd


def plot_detected_signal(signal, config: WDMConfig):
    """Plot detected signal waveform for first 10 symbols."""
    plt.figure(figsize=(12, 4))

    # Calculate time vector in picoseconds
    plot_symbols = 10
    samples_to_plot = plot_symbols * config.N_samples_per_symbol
    t = np.arange(samples_to_plot) / (config.baud_rate * config.N_samples_per_symbol)
    t_ps = t * 1e12  # Convert to picoseconds

    # Plot only first 10 symbols without normalization
    plt.plot(t_ps, signal[:samples_to_plot])
    plt.xlabel("Time (ps)")
    plt.ylabel("Electrical Signal Power")
    plt.title("Directly Detected WDM Signal")
    plt.xlim(0, 1000)
    plt.grid()
    plt.show()


def plot_psd(signal, Fs, title, max_freq=None):
    """Plot power spectral density."""
    f, psd = calculate_psd(signal, Fs)

    plt.figure(figsize=(10, 6))
    plt.plot(f / 1e9, psd, linewidth=1)
    if max_freq:
        plt.xlim(-max_freq / 1e9, max_freq / 1e9)
    plt.xlabel("Frequency (GHz)")
    plt.ylabel("Power Spectral Density (dBm/Hz)")
    plt.title(title)
    plt.grid(True)
    plt.show()


def plot_eye_diagram(signal, samples_per_symbol, title, num_symbols=200):
    """Plot eye diagram using optical power."""
    plt.figure(figsize=(8, 6))
    signal_section = signal[: samples_per_symbol * num_symbols]
    signal_power = np.abs(signal_section) ** 2

    for i in range(num_symbols):
        t = np.arange(0, samples_per_symbol) / samples_per_symbol
        plt.plot(t, signal_power[i * samples_per_symbol : (i + 1) * samples_per_symbol])

    plt.xlabel("Symbol Time")
    plt.ylabel("Optical Power")
    plt.title(title)
    plt.grid(True)
    plt.xlim(0, t.max())
    plt.show()


def calculate_signal_power(signal):
    """Calculate average signal power."""
    return np.mean(np.abs(signal) ** 2)


def main():
    # Initialize configuration
    config = WDMConfig()

    print("\nGenerating WDM signal...")
    wdm_signal, channel_bits = generate_wdm_signal(config)

    # Calculate sampling rate
    Fs = config.baud_rate * config.N_samples_per_symbol

    # Plot original WDM signal spectrum
    print("\nPlotting original WDM spectrum...")
    plot_psd(wdm_signal, Fs, "Original WDM Signal Spectrum", max_freq=200e9)

    # Add noise to signal
    noisy_signal = add_noise(wdm_signal, config.OSNR_dB)

    # Plot noisy WDM spectrum
    print("\nPlotting noisy WDM spectrum...")
    plot_psd(noisy_signal, Fs, "Noisy WDM Signal Spectrum", max_freq=200e9)

    # Plot results
    print("\nPlotting detected signal...")
    plot_detected_signal(np.abs(noisy_signal) ** 2, config)

    # Calculate frequency grid
    f = np.fft.fftshift(np.fft.fftfreq(len(wdm_signal), 1 / Fs))

    # Create optical filter for first channel
    b3db = 2 * config.baud_rate
    channel_index = 0
    channel_offset = (
        channel_index - (config.N_channels - 1) / 2
    ) * config.channel_spacing
    filter_bandwidth = 2 * config.baud_rate

    # Create and apply optical filter
    print("\nApplying optical filter...")
    Hf_field, Hf_power = create_gaussian_filter(f, channel_offset, filter_bandwidth)
    filtered_signal = filter_signal(noisy_signal, Hf_field)

    # Plot filter response
    plot_filter_response(f, Hf_power, channel_offset, b3db)

    # Plot filtered signal spectrum
    print("\nPlotting filtered signal spectrum...")
    plot_psd(filtered_signal, Fs, "Filtered Channel Spectrum", max_freq=200e9)

    # Direct detection
    detected_signal = np.abs(filtered_signal) ** 2

    # Apply baseband electrical filter
    bb_filter, bb_power = create_gaussian_filter(f, 0, b3db, order=2)
    electrical_signal = np.real(filter_signal(detected_signal, bb_filter))

    # Plot eye diagrams
    print("\nPlotting eye diagrams...")
    plot_eye_diagram(
        filtered_signal,
        config.N_samples_per_symbol,
        "Eye Diagram Before Electrical Filtering",
    )
    plot_eye_diagram(
        electrical_signal,
        config.N_samples_per_symbol,
        "Eye Diagram After Electrical Filtering",
    )


if __name__ == "__main__":
    main()
