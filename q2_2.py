import numpy as np
import matplotlib.pyplot as plt
from q1_1 import (
    seed,
    calculate_signal_power,
    add_awgn,
)
from q1_2 import (
    create_gaussian_filter,
    apply_optical_filter,
    plot_eye_diagram,
    plot_filter_transfer_function,
)
from q1_3 import calculate_ber
from q2_1 import WDMConfig, generate_wdm_signal

np.random.seed(seed)


def make_decisions(detected_signal, config: WDMConfig):
    """Make bit decisions from detected signal."""
    # Reshape signal into symbol periods
    signal_reshape = detected_signal.reshape(-1, config.N_samples_per_symbol)

    # Take mean of each symbol period
    symbol_values = np.mean(signal_reshape, axis=1)

    # Normalize
    symbol_values = symbol_values / np.max(symbol_values)

    # Decision threshold at 0.5
    decisions = (symbol_values > 0.5).astype(int)

    return decisions


def detect_wdm_channel(wdm_signal: np.ndarray, config: WDMConfig, target_channel: int):
    """Extract and detect a single channel from WDM signal."""
    # Calculate frequency grid
    sampling_rate = config.baud_rate * config.N_samples_per_symbol
    freqs = np.fft.fftshift(np.fft.fftfreq(len(wdm_signal), 1 / sampling_rate))

    # Calculate center frequency for target channel
    channel_offset = (
        target_channel - (config.N_channels - 1) / 2
    ) * config.channel_spacing

    # Create optical filter centered on target channel
    filter_transfer = create_gaussian_filter(
        freqs,
        center_freq=channel_offset,
        bw_3dB=2 * config.baud_rate,  # 2*Rs bandwidth
        rejection_ratio_dB=20,
    )

    # Plot filter transfer function
    plot_filter_transfer_function(freqs, filter_transfer)

    # Apply optical filter to extract channel
    filtered_signal = apply_optical_filter(wdm_signal, filter_transfer)

    return filtered_signal


def plot_signal_spectrum(signal, config: WDMConfig, title: str):
    """Plot power spectral density of a signal."""
    # Calculate frequency grid
    sampling_rate = config.baud_rate * config.N_samples_per_symbol
    freqs = np.fft.fftshift(np.fft.fftfreq(len(signal), 1 / sampling_rate))

    # Calculate PSD
    signal_fft = np.fft.fftshift(np.fft.fft(signal)) / len(signal)
    psd = 10 * np.log10(np.abs(signal_fft) ** 2) - 30  # Convert to dBm/Hz

    plt.figure(figsize=(10, 6))
    plt.plot(freqs / 1e9, psd, linewidth=1)

    # Set axis limits
    """
    max_psd = np.max(psd)
    plt.axis(
        [
            -sampling_rate / 4e9,  # Show smaller frequency range for better detail
            sampling_rate / 4e9,
            max_psd - 70,
            max_psd + 10,
        ]
    )
    """

    plt.xlabel("Frequency (GHz)")
    plt.ylabel("Power spectral density (dBm/Hz)")
    plt.title(title)
    plt.grid(True)
    plt.show()


def detect_signal(signal, config: WDMConfig):
    """Perform direct detection on the optical signal."""
    # Square law detection
    detected_signal = np.abs(signal) ** 2

    # Apply baseband filter to limit electrical bandwidth
    sampling_rate = config.baud_rate * config.N_samples_per_symbol
    freqs = np.fft.fftshift(np.fft.fftfreq(len(signal), 1 / sampling_rate))

    # Create and apply electrical filter
    bb_filter = create_gaussian_filter(
        freqs, center_freq=0, bw_3dB=2 * config.baud_rate, rejection_ratio_dB=20
    )

    # Apply filter in frequency domain
    signal_fft = np.fft.fftshift(np.fft.fft(detected_signal))
    filtered_fft = signal_fft * bb_filter
    filtered_signal = np.real(np.fft.ifft(np.fft.ifftshift(filtered_fft)))

    return filtered_signal


def analyze_channel(
    wdm_signal: np.ndarray, original_bits: list, config: WDMConfig, target_channel: int
):
    """Analyze a single WDM channel through filtering and detection."""
    print(f"\nAnalyzing Channel {target_channel + 1}:")

    # Extract channel using optical filter
    filtered_signal = detect_wdm_channel(wdm_signal, config, target_channel)

    # Plot optical spectrum before and after filtering
    plot_signal_spectrum(wdm_signal, config, "WDM Signal Spectrum Before Filtering")
    plot_signal_spectrum(
        filtered_signal, config, f"Extracted Channel {target_channel+1} Spectrum"
    )

    # Detect the filtered signal
    detected_signal = detect_signal(filtered_signal, config)

    # Make decisions and calculate BER
    decisions = make_decisions(detected_signal, config)
    ber = calculate_ber(decisions, original_bits[target_channel])

    return detected_signal, decisions, ber


def main():
    # Setup configuration
    config = WDMConfig()

    # Generate WDM signal
    print("Generating WDM signal...")
    wdm_signal, channel_bits = generate_wdm_signal(config)

    # Add some noise
    signal_power = calculate_signal_power(wdm_signal)
    config.OSNR_dB = 20  # Set OSNR for demonstration
    noisy_signal = add_awgn(wdm_signal, signal_power, config)

    # Analyze first channel (as an example)
    target_channel = 0  # First channel
    detected_signal, decisions, ber = analyze_channel(
        noisy_signal, channel_bits, config, target_channel
    )
    print(f"Channel {target_channel+1}: {ber}")

    # Plot eye diagram
    plot_eye_diagram(
        detected_signal[:20000],
        config.N_samples_per_symbol,
        f"Eye Diagram - Channel {target_channel+1}",
    )


if __name__ == "__main__":
    main()
