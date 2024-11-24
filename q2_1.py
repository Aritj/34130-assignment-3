import numpy as np
import matplotlib.pyplot as plt
from q1_1 import seed

np.random.seed(seed)  # Set seed for reproducability

# Set figure DPI to 300 (increasing plot resolution)
plt.rcParams["savefig.dpi"] = 300


class WDMConfig:
    def __init__(self):
        self.N_samples_per_symbol = 64
        self.baud_rate = 10e9  # 10 Gbaud
        self.N_symbols = int(1e5)  # Number of symbols
        self.channel_spacing = 50e9  # 50 GHz
        self.N_channels = 5
        self.OSNR_dB = 20  # OSNR per channel in dB


def generate_wdm_channel(config: WDMConfig, channel_index: int):
    """Generate a single WDM channel with frequency shift. Returns signal and bits."""
    # Generate random bits for this channel
    bits = np.random.randint(0, 2, config.N_symbols)

    # Generate NRZ signal
    N_samples = config.N_samples_per_symbol * config.N_symbols
    signal = np.repeat(bits, config.N_samples_per_symbol)

    # Calculate time grid
    time = np.linspace(
        0, config.N_symbols / config.baud_rate, N_samples, endpoint=False
    )

    # Calculate frequency offset for this channel
    channel_offset = (
        channel_index - (config.N_channels - 1) / 2
    ) * config.channel_spacing

    # Apply frequency shift
    shifted_signal = signal * np.exp(1j * 2 * np.pi * channel_offset * time)

    return shifted_signal, bits


def generate_wdm_signal(config: WDMConfig):
    """Generate complete WDM signal by combining all channels. Returns signal and all bits."""
    N_samples = config.N_samples_per_symbol * config.N_symbols
    wdm_signal = np.zeros(N_samples, dtype=np.complex128)
    channel_bits = []

    # Generate and combine all channels
    for i in range(config.N_channels):
        channel_signal, bits = generate_wdm_channel(config, i)
        wdm_signal += channel_signal
        channel_bits.append(bits)

    return wdm_signal, channel_bits


def plot_wdm_spectrum(signal, config: WDMConfig):
    """Plot the power spectral density of the WDM signal."""
    # Calculate sampling rate and frequency grid
    sampling_rate = config.baud_rate * config.N_samples_per_symbol
    freqs = np.fft.fftshift(np.fft.fftfreq(len(signal), 1 / sampling_rate))

    # Calculate PSD with proper normalization
    signal_fft = np.fft.fftshift(np.fft.fft(signal)) / len(signal)
    psd = 10 * np.log10(np.abs(signal_fft) ** 2) - 30  # Convert to dBm/Hz

    # Create plot
    plt.figure(figsize=(10, 6))
    plt.plot(freqs / 1e9, psd, linewidth=1, color=[0, 0, 1, 0.2])

    # Set axis limits
    max_psd = np.max(psd)
    plt.axis(
        [
            -sampling_rate / 2e9,
            sampling_rate / 2e9,
            max_psd - 70,  # Show 70 dB range
            max_psd + 10,
        ]
    )

    # Add channel markers
    plt.text(0, max_psd + 7, "WDM channel", horizontalalignment="center")
    for i in range(config.N_channels):
        channel_freq = (i - (config.N_channels - 1) / 2) * config.channel_spacing
        plt.text(
            channel_freq / 1e9,
            max_psd + 2,
            f"{i+1}\n↓",
            horizontalalignment="center",
            verticalalignment="baseline",
        )

    plt.xlabel("Frequency (GHz)")
    plt.ylabel("Power spectral density (dBm/Hz)")
    plt.grid(True)
    plt.show()


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


def main():
    config = WDMConfig()
    fs = config.baud_rate * config.N_samples_per_symbol
    total_bw = config.channel_spacing * (config.N_channels - 1)
    min_nss = np.ceil(2 * total_bw / config.baud_rate)

    # Generate and plot WDM signal
    print("\nGenerating WDM signal...")
    wdm_signal, bits = generate_wdm_signal(config)
    plot_wdm_spectrum(wdm_signal, config)

    # Plot results
    print("\nPlotting detected signal...")
    plot_detected_signal(np.abs(wdm_signal) ** 2, config)

    total_osnr_db = 10 * np.log10(config.N_channels) + config.OSNR_dB

    print(fs)
    print(total_bw)
    print(min_nss)
    print(f"OSNR for entire WDM signal: {total_osnr_db:.1f} dB")


if __name__ == "__main__":
    main()
