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


def main():
    # Answer Question 1
    print("\nQuestion 1 Analysis:")
    print("a) Why is Nss increased to 64?")
    print(
        "   - For WDM signals, we need higher sampling rate to accommodate multiple channels"
    )
    print(
        "   - Total bandwidth = Channel spacing * (N_channels - 1) = 50 GHz * 4 = 200 GHz"
    )
    print("   - Minimum sampling rate = 2 * Total bandwidth = 400 GHz")
    print(
        "   - With Rs = 10 GHz, Nss = 64 gives sampling rate of 640 GHz, which is sufficient"
    )

    print("\nb) Calculate overall bandwidth and sampling frequency:")
    config = WDMConfig()
    fs = config.baud_rate * config.N_samples_per_symbol
    total_bw = config.channel_spacing * (config.N_channels - 1)
    print(f"   - Total bandwidth = {total_bw/1e9:.1f} GHz")
    print(f"   - Sampling frequency = {fs/1e9:.1f} GHz")

    print("\nc) Why Nss = 64 instead of 16:")
    print("   - Nyquist sampling theorem requires fs > 2 * signal_bandwidth")
    print("   - With 16 samples/symbol: fs = 16 * 10 GHz = 160 GHz")
    print("   - This is less than required 2 * 200 GHz = 400 GHz")
    print("   - With 64 samples/symbol: fs = 640 GHz, which satisfies Nyquist")

    print("\nd) Minimum Nss required:")
    min_nss = np.ceil(2 * total_bw / config.baud_rate)
    print(f"   - Minimum Nss = ceil(2 * total_bw / Rs) = {min_nss}")

    # Generate and plot WDM signal
    print("\nGenerating WDM signal...")
    wdm_signal, bits = generate_wdm_signal(config)
    plot_wdm_spectrum(wdm_signal, config)


if __name__ == "__main__":
    main()
