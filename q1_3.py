from q1_1 import seed, SignalParameters, generate_ook_signal, apply_MZM
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator, FuncFormatter

np.random.seed(seed)

# Set figure DPI to 300 (increasing plot resolution)
plt.rcParams["savefig.dpi"] = 300


def create_filter_for_chunk(f_chunk, bandwidth, order=1, rejection_db=20):
    """Create Gaussian filter for a specific chunk size."""
    F0 = bandwidth / (2 * np.sqrt(2 * np.log(2)) ** (1 / order))
    Hf_min = 1 / (10 ** (rejection_db / 20))

    Hf_field = np.exp(-(1 / 4) * ((f_chunk) / F0) ** (2 * order))
    Hf_field[Hf_field <= Hf_min] = Hf_min

    return Hf_field


def filter_chunk(chunk, Fs, bandwidth):
    """Filter a single chunk of signal."""
    nfft = len(chunk)
    f_chunk = np.fft.fftshift(np.fft.fftfreq(nfft, 1 / Fs))

    # Create filter for this chunk size
    Hf_field = create_filter_for_chunk(f_chunk, bandwidth)

    # Apply filter in frequency domain
    chunk_fft = np.fft.fftshift(np.fft.fft(chunk))
    filtered_fft = chunk_fft * Hf_field

    return np.fft.ifft(np.fft.ifftshift(filtered_fft))


def add_noise_calibrated(signal, OSNR_dB, Rs):
    """Add noise following the MATLAB implementation."""
    OSNR_linear = 10 ** (OSNR_dB / 10)
    ref_bandwidth = 12.5e9  # Reference bandwidth
    bandwidth_ratio = ref_bandwidth / (2 * Rs)
    adjusted_OSNR = OSNR_linear / bandwidth_ratio

    signal_power = np.mean(np.abs(signal) ** 2)
    noise_power = signal_power / adjusted_OSNR
    noise_scaling = np.sqrt(noise_power)

    noise = noise_scaling * (
        np.random.randn(len(signal)) + 1j * np.random.randn(len(signal))
    )
    return signal + noise


def detect_and_decide(signal, current_chunk, N_ss):
    """Detection process following MATLAB implementation."""
    detected_power = np.abs(signal) ** 2

    # Dynamic level normalization
    max_level = np.mean(detected_power[current_chunk > 0.5])
    min_level = np.mean(detected_power[current_chunk <= 0.5])
    normalized_signal = (detected_power - min_level) / (max_level - min_level)

    # Sample at symbol centers with 0.2 threshold
    symbol_centers = np.arange(N_ss // 2, len(normalized_signal), N_ss)
    sampled_signal = normalized_signal[symbol_centers]
    decisions = (sampled_signal > 0.2).astype(int)

    return decisions


def simulate_ber_curve(signal, params, filter_bandwidth=None):
    """Simulate BER curve for given filter bandwidth."""
    OSNR_range = np.arange(17, 3, -1)
    BER_results = []

    # Process in chunks
    chunk_size = 16000
    n_chunks = len(signal) // chunk_size

    for OSNR_dB in OSNR_range:
        all_decisions = []
        original_bits = []

        for i in range(n_chunks):
            start_idx = i * chunk_size
            end_idx = start_idx + chunk_size
            chunk = signal[start_idx:end_idx]

            # Add noise
            noisy_chunk = add_noise_calibrated(chunk, OSNR_dB, params.Rs)

            # Apply filter if specified
            if filter_bandwidth is not None:
                processed_chunk = filter_chunk(noisy_chunk, params.Fs, filter_bandwidth)
            else:
                processed_chunk = noisy_chunk

            # Get decisions for this chunk
            decisions = detect_and_decide(processed_chunk, chunk, params.N_ss)
            all_decisions.extend(decisions)

            # Get original bits for this chunk
            chunk_bits = chunk[:: params.N_ss] > 0.5
            original_bits.extend(chunk_bits)

        # Calculate BER
        errors = np.sum(np.array(all_decisions) != np.array(original_bits))
        ber = max(errors / len(original_bits), 1e-6)
        BER_results.append(ber)

        print(f"OSNR: {OSNR_dB} dB, BER: {ber:.2e}")

    return OSNR_range, BER_results


def main():
    # Initialize parameters
    params = SignalParameters(N_ss=16, Rs=10e9, Nsym=100000)

    # Generate signal with sqrt(2) normalization
    electrical_signal = generate_ook_signal(params) * np.sqrt(2)
    optical_signal = apply_MZM(electrical_signal)

    # Filter configurations
    filter_configs = {
        "No Filter": None,
        "Filter BW = 2Rs": 2.0 * params.Rs,
        "Filter BW = 0.75Rs": 0.75 * params.Rs,
        "Filter BW = 0.5Rs": 0.5 * params.Rs,
    }

    # Create plot
    fig, ax = plt.subplots(figsize=(10, 6))
    markers = ["o", "s", "D", "^"]

    # Run simulation for each configuration
    for (label, bandwidth), marker in zip(filter_configs.items(), markers):
        print(f"\nSimulating {label}...")
        osnr_range, ber_results = simulate_ber_curve(optical_signal, params, bandwidth)

        # Plot results
        log_ber = -np.log10(ber_results)
        ax.semilogy(
            osnr_range,
            log_ber,
            "-" + marker,
            label=label,
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
    plt.xlabel("OSNR (dB)")
    plt.ylabel("-log$_{10}$(BER)")
    plt.legend()
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
