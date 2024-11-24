import numpy as np
import matplotlib.pyplot as plt

# Constants
SPEED_OF_LIGHT = 3e8  # m/s
WAVELENGTH = 1550e-9  # m
REFERENCE_BW = 12.5e9  # Hz (0.1nm reference bandwidth around 1550nm)
VPI = np.pi  # Half-wave voltage of MZM
V_BIAS = VPI / 2  # Bias voltage (quadrature point)

class SignalParameters:
    """Class to hold signal parameters"""
    def __init__(self, N_ss=32, Rs=10e9, Nsym=1000):
        self.N_ss = N_ss  # Number of samples per symbol
        self.Rs = Rs      # Signal baud rate
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
    np.random.seed(34310)
    data = np.random.randint(0, 2, params.Nsym)
    
    # NRZ pulse shaping
    pulse = np.ones(params.N_ss)
    electrical_signal = np.zeros(params.N_ss * params.Nsym)
    
    # Apply pulse shaping
    for i in range(params.Nsym):
        if data[i] == 1:
            electrical_signal[i * params.N_ss:(i + 1) * params.N_ss] = pulse
            
    return electrical_signal

def apply_MZM(electrical_signal):
    """
    Apply Mach-Zehnder Modulator transfer function with proper normalization.
    """
    # MZM transfer function
    optical_field = np.sin(np.pi/2 * electrical_signal)
    
    # Convert to complex and normalize to unit power
    complex_signal = optical_field
    complex_signal = complex_signal / np.sqrt(np.max(np.abs(complex_signal)**2))
    
    return complex_signal

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

def add_noise(signal, OSNR_db = 20):
    signal_power = np.mean(np.abs(signal) ** 2)
    noise_power = signal_power / (10 ** (OSNR_db / 10))
    noise = np.sqrt(noise_power / 2) * (np.random.randn(len(signal)) + 1j * np.random.randn(len(signal)))
    return signal + noise

def plot_signals(f, psd_original, psd_noisy, params):
    """Plot original and noisy signals with power levels."""
    plt.figure(figsize=(10, 6))
    
    # Convert to dB
    psd_original_db = 10 * np.log10(psd_original + np.finfo(float).eps)
    psd_noisy_db = 10 * np.log10(psd_noisy + np.finfo(float).eps)
    
    # Plot signals
    plt.plot(f / 1e9, psd_noisy_db, 'r', label='Noisy Signal', linewidth=1.5)
    plt.plot(f / 1e9, psd_original_db, 'b', label='Original Signal', linewidth=1.5)
    
    # Calculate and mark signal and noise levels
    signal_level = 10 * np.log10(np.max(psd_original))
    noise_level = 10 * np.log10(np.mean(psd_noisy[np.abs(f) > 5e9]))
    
    # Draw horizontal lines for levels
    plt.axhline(y=signal_level, color='r', linestyle='--', label='Signal Level')
    plt.axhline(y=noise_level, color='g', linestyle='--', label='Noise Level')
    
    # Formatting
    plt.title('Power Spectral Density of OOK Signal')
    plt.xlabel('Frequency (GHz)')
    plt.ylabel('PSD (dB/Hz)')
    plt.ylim([-120, 0])
    plt.xlim([-params.Fs / 2e9, params.Fs / 2e9])
    plt.grid(True)
    plt.legend()
    
    # Print diagnostic information
    print(f"Signal Level: {signal_level:.2f} dB")
    print(f"Noise Level: {noise_level:.2f} dB")
    print(f"Calculated OSNR: {signal_level - noise_level:.2f} dB")
    
    return signal_level, noise_level

def main():
    # Initialize parameters
    params = SignalParameters(N_ss=32, Rs=10e9, Nsym=1000)
    
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
    power_levels = np.unique(np.abs(complex_ook_signal)**2)
    print(f"Optical power levels: {[f'{level:.4f}' for level in power_levels]}")
    
    # Create verification plots
    plt.figure(figsize=(15, 10))
    plt.subplot(2, 2, 1)
    # Plot first 10 symbols of electrical signal
    t_sample = params.t[:10*params.N_ss]
    e_sample = electrical_signal[:10*params.N_ss]
    plt.plot(t_sample*1e9, e_sample)
    plt.title('Electrical Signal (First 10 Symbols)')
    plt.xlabel('Time (ns)')
    plt.ylabel('Amplitude')
    plt.grid(True)
    
    plt.subplot(2, 2, 2)
    # Plot first 10 symbols of optical field
    o_sample = np.real(complex_ook_signal[:10*params.N_ss])
    plt.plot(t_sample*1e9, o_sample)
    plt.title('Optical Field (First 10 Symbols)')
    plt.xlabel('Time (ns)')
    plt.ylabel('Field Amplitude')
    plt.grid(True)
    
    plt.subplot(2, 2, 3)
    # Plot PSDs
    plt.plot(f/1e9, 10*np.log10(psd_original + np.finfo(float).eps), 'b', label='Original')
    plt.plot(f/1e9, 10*np.log10(psd_noisy + np.finfo(float).eps), 'r', label='Noisy')
    plt.title('Power Spectral Density')
    plt.xlabel('Frequency (GHz)')
    plt.ylabel('PSD (dB/Hz)')
    plt.grid(True)
    plt.legend()
    
    plt.subplot(2, 2, 4)
    # Plot optical power of first 10 symbols
    p_sample = np.abs(complex_ook_signal[:10*params.N_ss])**2
    plt.plot(t_sample*1e9, p_sample)
    plt.title('Optical Power (First 10 Symbols)')
    plt.xlabel('Time (ns)')
    plt.ylabel('Power')
    plt.grid(True)
    
    plt.tight_layout()
    
    # Additional verification plot for noise
    plt.figure(figsize=(10, 5))
    # Plot small section of noisy signal
    plt.plot(t_sample*1e9, np.abs(noisy_signal[:10*params.N_ss])**2)
    plt.title('Noisy Optical Power (First 10 Symbols)')
    plt.xlabel('Time (ns)')
    plt.ylabel('Power')
    plt.grid(True)
    
    # Calculate and print OSNR verification
    signal_power = np.mean(np.abs(complex_ook_signal)**2)
    noise_power = np.var(noisy_signal - complex_ook_signal)
    measured_osnr = 10*np.log10(signal_power/noise_power)
    print(f"\nOSNR Verification:")
    print(f"Target OSNR: 20 dB")
    print(f"Measured OSNR: {measured_osnr:.2f} dB")
    
    # Calculate and print noise statistics
    noise = noisy_signal - complex_ook_signal
    print(f"\nNoise Statistics:")
    print(f"Noise mean: {np.mean(noise):.2e}")
    print(f"Noise std: {np.std(noise):.2e}")
    print(f"Noise power: {np.mean(np.abs(noise)**2):.2e}")
    
    plt.show()

if __name__ == "__main__":
    main()