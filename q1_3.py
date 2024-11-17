import numpy as np
import matplotlib.pyplot as plt
from q1_1 import (
    seed,
    SimulationConfig,
    generate_OOK_signal,
    calculate_signal_power,
    add_awgn,
)
from q1_2 import create_gaussian_filter, apply_optical_filter

np.random.seed(seed)  # Set seed for reproducability


# Function to detect signal and make symbol decisions
def detect_and_decide(signal, samples_per_symbol):
    # Power detection (square law detection)
    detected_signal = np.abs(signal) ** 2

    # Downsample to symbol rate using mean of each symbol period
    detected_reshape = detected_signal.reshape(-1, samples_per_symbol)
    symbol_decisions = np.mean(detected_reshape, axis=1)

    # Normalize to maximum value
    symbol_decisions = symbol_decisions / np.max(symbol_decisions)

    # Decision threshold at 0.5 for OOK
    decisions = (symbol_decisions > 0.5).astype(int)
    return decisions


# Function to calculate BER
def calculate_ber(decisions, original_bits):
    errors = np.sum(decisions != original_bits)
    # return max(errors / len(original_bits), 1e-7)  # Limit minimum BER for plotting with max() function
    return errors / len(original_bits)


# Function to simulate system performance for given OSNR
def simulate_performance(
    base_config: SimulationConfig, filter_bw_factor, use_filter=True
):
    # Generate original signal and bits
    time, signal, original_bits = generate_OOK_signal(base_config)

    # Calculate signal power
    signal_power = calculate_signal_power(signal)

    # Create optical filter if needed
    if use_filter:
        sampling_rate = base_config.baud_rate * base_config.N_samples_per_symbol
        freqs = np.fft.fftshift(np.fft.fftfreq(len(signal), 1 / sampling_rate))
        filter_transfer = create_gaussian_filter(
            freqs,
            center_freq=0,
            bw_3dB=filter_bw_factor * base_config.baud_rate,
            rejection_ratio_dB=20,
        )

    ber_results = []
    for osnr_db in np.arange(6, 17, 1):  # 6 to 16 dB
        # Create a new config with current OSNR
        current_config = SimulationConfig()
        current_config.N_samples_per_symbol = base_config.N_samples_per_symbol
        current_config.baud_rate = base_config.baud_rate
        current_config.N_symbols = base_config.N_symbols
        current_config.OSNR_dB = osnr_db

        # Add noise
        noisy_signal = add_awgn(signal, signal_power, current_config)

        # Apply optical filtering if requested
        if use_filter:
            processed_signal = apply_optical_filter(noisy_signal, filter_transfer)
        else:
            processed_signal = noisy_signal

        # Detect and make decisions
        decisions = detect_and_decide(
            processed_signal, current_config.N_samples_per_symbol
        )

        # Calculate BER
        ber = calculate_ber(decisions, original_bits)
        ber_results.append((osnr_db, ber))

        print(f"OSNR: {osnr_db} dB, BER: {ber:.2e}")

    return np.array(ber_results)


def plot_ber_curves(osnr_results, labels):
    plt.figure(figsize=(10, 6))
    h = plt.gca()

    for result, label in zip(osnr_results, labels):
        osnr_db, ber = result.T
        plt.semilogy(osnr_db, ber, "-*", label=label)

    # Configure plot style
    h.grid(True, which="both")  # Add both major and minor gridlines
    h.grid(True, which="minor", alpha=0.2)  # Make minor gridlines lighter

    # Set font properties
    h.set_xlabel("OSNR (dB)", fontsize=14)
    h.set_ylabel("BER", fontsize=14)

    plt.legend()
    plt.show()


def main():
    # Set up configuration
    config = SimulationConfig()
    config.N_samples_per_symbol = 16
    config.N_symbols = int(1e5)  # Large number of symbols for accurate BER

    results = []
    print("Simulating without filtering...")
    results.append(simulate_performance(config, 2, use_filter=False))

    print("Simulating with 2*Rs filter bandwidth...")
    results.append(simulate_performance(config, 2, use_filter=True))

    print("Simulating with Rs filter bandwidth...")
    results.append(simulate_performance(config, 1, use_filter=True))

    print("Simulating with 0.75*Rs filter bandwidth...")
    results.append(simulate_performance(config, 0.75, use_filter=True))

    print("Simulating with 0.5*Rs filter bandwidth...")
    results.append(simulate_performance(config, 0.5, use_filter=True))

    print("Simulating with 0.25*Rs filter bandwidth...")
    results.append(simulate_performance(config, 0.25, use_filter=True))

    labels = [
        "No filtering",
        "Filter BW = 2Rs",
        "Filter BW = Rs",
        "Filter BW = 0.75Rs",
        "Filter BW = 0.5Rs",
        "Filter BW = 0.25Rs",
    ]
    plot_ber_curves(results, labels)


if __name__ == "__main__":
    main()
