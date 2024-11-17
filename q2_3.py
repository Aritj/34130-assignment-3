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
)
from q1_3 import calculate_ber
from q2_1 import WDMConfig, generate_wdm_signal
from q2_2 import (
    detect_wdm_channel,
    detect_signal,
    make_decisions,
)

np.random.seed(seed)


def simulate_wdm_performance(config: WDMConfig, target_channel: int):
    """Simulate WDM system performance for a range of OSNR values."""
    osnr_range = np.arange(4, 13, 1)  # 4 to 12 dB
    ber_results = []

    # Generate WDM signal once
    wdm_signal, channel_bits = generate_wdm_signal(config)
    signal_power = calculate_signal_power(wdm_signal)

    # Create optical filter once
    sampling_rate = config.baud_rate * config.N_samples_per_symbol
    freqs = np.fft.fftshift(np.fft.fftfreq(len(wdm_signal), 1 / sampling_rate))
    channel_offset = (
        target_channel - (config.N_channels - 1) / 2
    ) * config.channel_spacing

    optical_filter = create_gaussian_filter(
        freqs,
        center_freq=channel_offset,
        bw_3dB=2 * config.baud_rate,  # 2*Rs bandwidth as specified
        rejection_ratio_dB=20,
    )

    # Simulate for each OSNR value
    for osnr_db in osnr_range:
        print(f"Simulating OSNR = {osnr_db} dB...")

        # Set current OSNR
        config.OSNR_dB = osnr_db

        # Add noise
        noisy_signal = add_awgn(wdm_signal, signal_power, config)

        # Filter target channel
        filtered_signal = apply_optical_filter(noisy_signal, optical_filter)

        # Detect signal
        detected_signal = detect_signal(filtered_signal, config)

        # Make decisions
        decisions = make_decisions(detected_signal, config)

        # Calculate BER
        ber = calculate_ber(decisions, channel_bits[target_channel])
        ber_results.append((osnr_db, ber))

        print(f"BER: {ber:.2e}")

    return np.array(ber_results)


def plot_ber_vs_osnr(results, labels):
    """Plot BER vs OSNR curves."""
    plt.figure(figsize=(10, 6))
    for result, label in zip(results, labels):
        osnr_db, ber = result.T
        plt.semilogy(osnr_db, ber, "-*", label=label)

    plt.grid(True, which="both")
    plt.grid(True, which="minor", alpha=0.2)
    plt.xlabel("OSNR per channel (dB)")
    plt.ylabel("BER")
    plt.legend()
    plt.show()


def main():
    # Setup configuration
    config = WDMConfig()
    config.N_symbols = int(1e5)  # Use large number of symbols for accurate BER

    # Simulate all channels
    all_results = []
    channel_labels = []

    for channel in range(config.N_channels):
        print(f"\nSimulating Channel {channel + 1}")
        results = simulate_wdm_performance(config, channel)
        all_results.append(results)
        channel_labels.append(f"Channel {channel + 1}")

    # Plot results
    plot_ber_vs_osnr(all_results, channel_labels)

    # Print analysis
    print("\nAnalysis of WDM System Performance:")
    print("1. Channel Performance:")
    for i, results in enumerate(all_results):
        min_ber = np.min(results[:, 1])
        osnr_at_min = results[np.argmin(results[:, 1]), 0]
        print(f"   Channel {i+1}:")
        print(f"   - Best BER: {min_ber:.2e} at OSNR = {osnr_at_min} dB")

    print("\n2. Performance Factors:")
    print("   - Channel spacing: 50 GHz")
    print("   - Filter bandwidth: 2*Rs = 20 GHz")
    print("   - Samples per symbol: 64")

    print("\n3. Improvement Suggestions:")
    print("   - Optimize filter bandwidth")
    print("   - Adjust channel spacing")
    print("   - Increase sampling rate")


if __name__ == "__main__":
    main()
