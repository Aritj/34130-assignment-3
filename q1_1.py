import numpy as np
import matplotlib.pyplot as plt

seed = 34310
np.random.seed(seed)  # Set seed for reproducability

# Set figure DPI to 300 (increasing plot resolution)
plt.rcParams["savefig.dpi"] = 300


# Module for configuration and results storage
class SimulationConfig:
    def __init__(self):
        self.N_samples_per_symbol = 64
        self.baud_rate = 10e9  # 10 Gbaud
        self.N_symbols = 1000
        self.OSNR_dB = 20  # OSNR in dB


class SimulationResults:
    def __init__(self):
        self.time = None
        self.signal = None
        self.noisy_signal = None
        self.filtered_signal = None
        self.signal_power = None
        self.noise_power = None
        self.psd_original = None
        self.psd_noisy = None
        self.psd_filtered = None
        self.sampling_rate = None


# Function to generate an OOK signal with NRZ pulse shaping
def generate_OOK_signal(config: SimulationConfig):
    bits = np.random.randint(0, 2, config.N_symbols)
    N_samples = config.N_samples_per_symbol * config.N_symbols
    signal = np.repeat(bits, config.N_samples_per_symbol)
    time = np.linspace(
        0, config.N_symbols / config.baud_rate, N_samples, endpoint=False
    )
    return time, signal, bits


# Function to calculate signal power
def calculate_signal_power(signal):
    return np.mean(np.abs(signal) ** 2)


# Function to add AWGN to the signal
def add_awgn(signal, signal_power, config):
    OSNR_linear = 10 ** (config.OSNR_dB / 10)
    noise_power = signal_power / OSNR_linear
    noise_variance = noise_power / 2
    noise = np.sqrt(noise_variance) * (
        np.random.randn(len(signal)) + 1j * np.random.randn(len(signal))
    )
    return signal + noise


# Function to compute the Power Spectral Density (PSD)
def compute_psd(signal, sampling_rate):
    fft_signal = np.fft.fftshift(np.fft.fft(signal))
    psd = np.abs(fft_signal) ** 2 / len(signal)
    freqs = np.fft.fftshift(np.fft.fftfreq(len(signal), 1 / sampling_rate))
    return freqs, 10 * np.log10(psd)


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


# Function to plot results
def plot_psd_with_levels(results: SimulationResults, config: SimulationConfig):
    freqs_original, psd_original = results.psd_original
    freqs_noisy, psd_noisy = results.psd_noisy

    plt.figure(figsize=(12, 6))

    # Subplot for OOK PSD
    plt.subplot(1, 2, 1)
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
    plt.subplot(1, 2, 2)
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

    plt.tight_layout()
    plt.show()


def calculate_results(config: SimulationConfig, results: SimulationResults):
    # Compute OOK signal
    results.time, results.signal, bits = generate_OOK_signal(config)

    # Compute signal and noise power
    results.signal_power = calculate_signal_power(results.signal)
    results.noise_power = results.signal_power / (10 ** (config.OSNR_dB / 10))

    # Compute noisy OOK signal
    results.noisy_signal = add_awgn(results.signal, results.signal_power, config)

    # Compute PSDs
    results.sampling_rate = config.baud_rate * config.N_samples_per_symbol
    results.psd_original = compute_psd(results.signal, results.sampling_rate)
    results.psd_noisy = compute_psd(results.noisy_signal, results.sampling_rate)

    return results


def calculate_and_plot_PSD(config: SimulationConfig, results: SimulationResults):
    # Calculate results
    results = calculate_results(config, results)

    # Plot OOK and noisy OOK signals
    plot_signals(results, config)

    # Plot PSD
    plot_psd_with_levels(results, config)

    print(f"N_ss = {config.N_samples_per_symbol}")
    print(f"Signal Power (dB): {10 * np.log10(results.signal_power):.2f}")
    print(f"Noise Power (dB): {10 * np.log10(results.noise_power):.2f}")


# Main function to execute the simulation
def main():
    config = SimulationConfig()
    results = SimulationResults()

    # Compute the OOK signal, noise and PSD and plot them
    calculate_and_plot_PSD(config, results)

    # Optional part
    config.N_samples_per_symbol = 32
    calculate_and_plot_PSD(config, results)


if __name__ == "__main__":
    main()
