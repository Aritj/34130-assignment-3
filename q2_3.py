from q1_1 import seed
from q2_1 import WDMConfig, generate_wdm_signal
from q2_2 import create_gaussian_filter, apply_optical_filter
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator, FuncFormatter

np.random.seed(seed)


def add_noise_calibrated(signal, OSNR_dB_per_channel, config: WDMConfig):
    """Add noise following OSNR per channel definition."""
    OSNR_linear = 10 ** (OSNR_dB_per_channel / 10)
    ref_bandwidth = 12.5e9  # Reference bandwidth
    bandwidth_ratio = ref_bandwidth / (2 * config.baud_rate)
    adjusted_OSNR = OSNR_linear / bandwidth_ratio

    signal_power = np.mean(np.abs(signal) ** 2)
    noise_power = signal_power / adjusted_OSNR
    noise_scaling = np.sqrt(noise_power)

    noise = noise_scaling * (
        np.random.randn(len(signal)) + 1j * np.random.randn(len(signal))
    )
    return signal + noise


def detect_and_decide(signal, N_ss):
    """Detection and decision process for single channel."""
    # Convert to power
    detected_power = np.abs(signal) ** 2

    # Sample at symbol centers and normalize
    symbol_centers = np.arange(N_ss // 2, len(detected_power), N_ss)
    sampled_signal = detected_power[symbol_centers]
    normalized_signal = sampled_signal / np.max(sampled_signal)

    # Make decisions with 0.2 threshold (as in Q1-3)
    decisions = (normalized_signal > 0.2).astype(int)

    return decisions


def process_wdm_channel(wdm_signal, config: WDMConfig, channel_idx: int):
    """Process a single WDM channel including filtering and detection."""
    # Calculate frequency grid
    Fs = config.baud_rate * config.N_samples_per_symbol
    f = np.fft.fftshift(np.fft.fftfreq(len(wdm_signal), 1 / Fs))

    # Calculate channel frequency offset
    channel_offset = (
        channel_idx - (config.N_channels - 1) / 2
    ) * config.channel_spacing

    # Create and apply optical filter
    optical_filter = create_gaussian_filter(
        f,
        channel_offset,
        bw_3dB=2 * config.baud_rate,  # 2*Rs bandwidth
        rejection_db=20,
    )
    filtered_signal = apply_optical_filter(wdm_signal, optical_filter)

    # Create and apply electrical filter
    electrical_filter = create_gaussian_filter(
        f,
        0,  # Centered at DC
        bw_3dB=2 * config.baud_rate,
        order=2,  # 2nd order filter
        rejection_db=20,
    )

    # Direct detection and electrical filtering
    detected_signal = np.abs(filtered_signal) ** 2
    electrical_signal = np.real(
        apply_optical_filter(detected_signal, electrical_filter)
    )

    # Make decisions
    decisions = detect_and_decide(electrical_signal, config.N_samples_per_symbol)

    return decisions


def calculate_ber(decisions, original_bits):
    """Calculate BER between decisions and original bits."""
    errors = np.sum(decisions != original_bits)
    return max(errors / len(original_bits), 1e-6)  # Minimum BER floor


def simulate_wdm_system(osnr_range_db):
    """Simulate WDM system for a range of OSNR values."""
    config = WDMConfig()

    # Store BER results for each channel
    ber_results = np.zeros((config.N_channels, len(osnr_range_db)))

    for osnr_idx, osnr_db in enumerate(osnr_range_db):
        print(f"\nSimulating OSNR = {osnr_db} dB")

        # Generate WDM signal
        wdm_signal, channel_bits = generate_wdm_signal(config)

        # Add noise
        noisy_signal = add_noise_calibrated(wdm_signal, osnr_db, config)

        # Process each channel
        for channel_idx in range(config.N_channels):
            # Detect channel and make decisions
            decisions = process_wdm_channel(noisy_signal, config, channel_idx)

            # Calculate BER
            ber = calculate_ber(decisions, channel_bits[channel_idx])
            ber_results[channel_idx, osnr_idx] = ber

            print(f"Channel {channel_idx + 1}: BER = {ber:.2e}")

    return ber_results


def plot_ber_curves(osnr_range_db, ber_results):
    """Plot BER vs OSNR curves for all channels."""
    fig, ax = plt.subplots(figsize=(10, 6))
    markers = ["o", "s", "D", "^", "v"]

    for ch in range(ber_results.shape[0]):
        log_ber = -np.log10(ber_results[ch])
        ax.semilogy(
            osnr_range_db,
            log_ber,
            f"-{markers[ch]}",
            label=f"Channel {ch+1}",
            markerfacecolor="white",
            linewidth=1,
            markersize=6,
        )

    # Configure plot
    ax.set_xlim(4, 16)
    ax.set_ylim(5, 1)
    ax.xaxis.set_major_locator(MultipleLocator(2))
    ax.xaxis.set_minor_locator(MultipleLocator(1))
    ax.yaxis.set_major_locator(MultipleLocator(1))
    ax.yaxis.set_major_formatter(FuncFormatter(lambda x, _: f"{int(x)}"))

    plt.xlabel("OSNR per channel (dB)")
    plt.ylabel("-log$_{10}$(BER)")
    plt.xlim(4, 12)
    plt.ylim(5, 1)
    plt.legend()
    plt.title("BER Performance of WDM Channels")
    plt.tight_layout()
    plt.show()


def main():
    # OSNR range from 4 to 12 dB with 1 dB steps
    osnr_range_db = np.arange(12, 4, -1)

    # Simulate system and get BER results
    ber_results = simulate_wdm_system(osnr_range_db)

    # Plot results
    plot_ber_curves(osnr_range_db, ber_results)


if __name__ == "__main__":
    main()
