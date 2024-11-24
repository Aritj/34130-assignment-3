import numpy as np
import matplotlib.pyplot as plt

# Constants
SPEED_OF_LIGHT = 3e8  # m/s
WAVELENGTH = 1550e-9  # m
REFERENCE_BW = 12.5e9  # Hz (0.1nm reference bandwidth around 1550nm)
VPI = np.pi  # Half-wave voltage of MZM
V_BIAS = VPI / 2  # Bias voltage (quadrature point)
seed = 34310
np.random.seed(34310)

# Set figure DPI to 300 (increasing plot resolution)
plt.rcParams["savefig.dpi"] = 300


class SignalParameters:
    """Class to hold signal parameters"""

    def __init__(self, N_ss=32, Rs=10e9, Nsym=1000):
        self.N_ss = N_ss  # Number of samples per symbol
        self.Rs = Rs  # Signal baud rate
        self.Nsym = Nsym  # Number of symbols
        self.Ts = 1 / Rs  # Symbol period
        self.Fs = N_ss / self.Ts  # Sampling frequency
        self.t = np.arange(0, N_ss * Nsym) / self.Fs  # Time vector


def generate_ook_signal(params):
    """
    Generate OOK signal with NRZ pulse shaping.

    Args:
        params (SignalParameters): Signal parameters

    Returns:
        ndarray: Electrical signal
    """
    # Generate random binary data
    data = np.random.randint(0, 2, params.Nsym)

    # NRZ pulse shaping
    pulse = np.ones(params.N_ss)
    electrical_signal = np.zeros(params.N_ss * params.Nsym)

    # Apply pulse shaping
    for i in range(params.Nsym):
        if data[i] == 1:
            electrical_signal[i * params.N_ss : (i + 1) * params.N_ss] = pulse

    return electrical_signal


def apply_MZM(electrical_signal):
    """
    Apply Mach-Zehnder Modulator transfer function with proper normalization.
    """
    # MZM transfer function
    optical_field = np.sin(np.pi / 2 * electrical_signal)

    # Convert to complex and normalize to unit power
    return optical_field / np.sqrt(np.mean(optical_field[electrical_signal > 0.5] ** 2))


def calculate_psd(signal, Fs):
    """
    Calculate Power Spectral Density using FFT.

    Args:
        signal (ndarray): Input signal
        Fs (float): Sampling frequency

    Returns:
        tuple: Frequency array and PSD
    """
    nfft = len(signal)
    f = np.fft.fftshift(np.fft.fftfreq(nfft, 1 / Fs))
    signal_fft = np.fft.fftshift(np.fft.fft(signal)) / nfft
    psd = np.abs(signal_fft) ** 2
    return f, psd


def add_noise(signal, OSNR_db=20):
    """Add noise with correct reference bandwidth scaling."""
    B_ref = 12.5e9  # Reference bandwidth (0.1nm @ 1550nm)
    OSNR_linear = 10 ** (OSNR_db / 10)

    # Calculate signal power
    signal_power = np.mean(np.abs(signal) ** 2)

    # Calculate ASE noise spectral density
    N_ASE = signal_power / (2 * B_ref * OSNR_linear)

    # Generate noise
    L = len(signal)
    noise = np.sqrt(N_ASE * B_ref) * (np.random.randn(L) + 1j * np.random.randn(L))

    return signal + noise


def plot_signals(f, psd_original, psd_noisy, params):
    """Plot original and noisy signals with power levels."""
    plt.figure(figsize=(10, 6))

    # Convert to dB
    psd_original_db = 10 * np.log10(psd_original + np.finfo(float).eps)
    psd_noisy_db = 10 * np.log10(psd_noisy + np.finfo(float).eps)

    # Plot signals
    plt.plot(f / 1e9, psd_noisy_db, "r", label="Noisy Signal", linewidth=1.5)
    plt.plot(f / 1e9, psd_original_db, "b", label="Original Signal", linewidth=1.5)

    # Calculate and mark signal and noise levels
    signal_level = 10 * np.log10(np.max(psd_original))
    noise_level = 10 * np.log10(np.mean(psd_noisy[np.abs(f) > 5e9]))

    # Draw horizontal lines for levels
    plt.axhline(y=signal_level, color="r", linestyle="--", label="Signal Level")
    plt.axhline(y=noise_level, color="g", linestyle="--", label="Noise Level")

    # Formatting
    plt.title(f"Power Spectral Density of OOK Signal for N_ss={params.N_ss}")
    plt.xlabel("Frequency (GHz)")
    plt.ylabel("PSD (dB/Hz)")
    plt.ylim([-120, 0])
    plt.xlim([-params.Fs / 2e9, params.Fs / 2e9])
    plt.grid(True)
    plt.legend()
    plt.show()

    # Print diagnostic information
    print(f"Signal Level: {signal_level:.2f} dB")
    print(f"Noise Level: {noise_level:.2f} dB")
    print(f"Calculated OSNR: {signal_level - noise_level:.2f} dB")

    return signal_level, noise_level


def compute_for_N_ss(N_ss):
    # Initialize parameters
    params = SignalParameters(N_ss=N_ss, Rs=10e9, Nsym=1000)

    # Generate OOK signal
    electrical_signal = generate_ook_signal(params)

    # Apply MZM
    complex_ook_signal = apply_MZM(electrical_signal)

    # Calculate original PSD
    f, psd_original = calculate_psd(complex_ook_signal, params.Fs)

    # Add noise (20 dB OSNR)
    noisy_signal = add_noise(complex_ook_signal)

    # Calculate noisy PSD
    _, psd_noisy = calculate_psd(noisy_signal, params.Fs)

    # Verification prints
    print("\nSignal Level Verification:")
    print(f"Electrical signal levels: {np.unique(electrical_signal)}")
    optical_levels = np.unique(np.real(complex_ook_signal))
    print(f"Optical field levels: {[f'{level:.4f}' for level in optical_levels]}")
    power_levels = np.unique(np.abs(complex_ook_signal) ** 2)
    print(f"Optical power levels: {[f'{level:.4f}' for level in power_levels]}")

    # Create plots
    plt.figure()
    t_sample = params.t[: 10 * params.N_ss]
    e_sample = electrical_signal[: 10 * params.N_ss]
    plt.plot(
        t_sample * 1e9,
        np.abs(noisy_signal[: 10 * params.N_ss]) ** 2,
        label="Noisy signal",
    )
    plt.plot(t_sample * 1e9, e_sample, label="Input signal")
    plt.title(f"Input vs Noisy signal (First 10 Symbols) for N_ss={params.N_ss}")
    plt.xlabel("Time (ns)")
    plt.ylabel("Amplitude")
    plt.legend()
    plt.xlim(0, 1)
    plt.grid()
    plt.show()

    plot_signals(f, psd_original, psd_noisy, params)


def main():
    compute_for_N_ss(64)
    compute_for_N_ss(32)


if __name__ == "__main__":
    main()
