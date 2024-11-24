from test import (
    SignalParameters, generate_ook_signal, 
    apply_MZM, add_noise
)
import numpy as np
import matplotlib.pyplot as plt

def create_gaussian_filter(f, f0, B3dB, order=1, rejection_db=20):
    """Create Gaussian optical filter transfer function."""
    F0 = B3dB / (2 * np.sqrt(2 * np.log(2)) ** (1 / order))
    Hf_min = 1 / (10 ** (rejection_db / 20))
    
    Hf_field = np.exp(-(1 / 4) * ((f - f0) / F0) ** (2 * order))
    Hf_field[Hf_field <= Hf_min] = Hf_min
    Hf_power = -20 * np.log10(Hf_field)
    
    return Hf_field, Hf_power

def plot_filter_response(f, Hf_power, channel_offset, B3dB):
    """Plot filter power transfer function."""
    plt.figure(figsize=(10, 6))
    f_ghz = f / 1e9
    
    fwhm_left_ghz = -B3dB / 2e9 + channel_offset / 1e9
    fwhm_right_ghz = B3dB / 2e9 + channel_offset / 1e9
    
    plt.plot(f_ghz, Hf_power, label='Power Transfer Function')
    plt.scatter([fwhm_left_ghz, fwhm_right_ghz], [3, 3], color='red', label='FWHM Points')
    
    plt.text(fwhm_left_ghz, 3, 'FWHM point, left →', 
            horizontalalignment='right', verticalalignment='center')
    plt.text(fwhm_right_ghz, 3, '← FWHM point, right', 
            horizontalalignment='left', verticalalignment='center')
    
    plt.axvline(fwhm_left_ghz, color='red', linestyle='--', linewidth=0.8)
    plt.axvline(fwhm_right_ghz, color='red', linestyle='--', linewidth=0.8)
    
    plt.gca().invert_yaxis()
    plt.xlabel('Frequency (GHz)')
    plt.ylabel('Rejection (dB)')
    plt.title('Power Transfer Function of Gaussian Optical Filter')
    plt.xlim(f.min() / 1e9, f.max() / 1e9)
    plt.grid(True)
    plt.legend()

def filter_signal(signal, Hf_field):
    """Apply optical filter to signal."""
    nfft = len(signal)
    # Apply filter in frequency domain
    signal_fft = np.fft.fftshift(np.fft.fft(signal))
    filtered_fft = signal_fft * Hf_field
    return np.fft.ifft(np.fft.ifftshift(filtered_fft))

def plot_eyediagram(signal, samples_per_symbol, num_symbols, title):
    """Plot eye diagram using optical power."""
    plt.figure(figsize=(8, 6))
    signal_section = signal[:samples_per_symbol * num_symbols]
    
    # Convert to power for optical signals
    signal_power = np.abs(signal_section)**2
    
    for i in range(num_symbols):
        t = np.arange(0, samples_per_symbol) / samples_per_symbol
        plt.plot(t, signal_power[i * samples_per_symbol:(i + 1) * samples_per_symbol])
    
    plt.xlabel('Symbol Time')
    plt.ylabel('Optical Power')
    plt.title(title)
    plt.grid(True)

def plot_psd(signal, f, title, use_db=True):
    """
    Plot power spectral density.
    
    Args:
        signal: Input signal
        f: Frequency array
        title: Plot title
        use_db: If True, plot in dB scale
    """
    plt.figure(figsize=(10, 6))
    nfft = len(signal)
    signal_fft = np.fft.fftshift(np.fft.fft(signal)) / nfft
    psd = np.abs(signal_fft) ** 2
    
    if use_db:
        psd_plot = 10 * np.log10(psd + np.finfo(float).eps)
        ylabel = 'PSD (dB/Hz)'
    else:
        psd_plot = psd
        ylabel = 'PSD (W/Hz)'
    
    plt.plot(f / 1e9, psd_plot)
    plt.xlabel('Frequency (GHz)')
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(True)

def main():
    # Initialize parameters
    params = SignalParameters(N_ss=64, Rs=10e9, Nsym=1000)
    
    # Generate and modulate signal
    electrical_signal = generate_ook_signal(params)
    optical_signal = apply_MZM(electrical_signal)
    
    # Add noise
    noisy_signal = add_noise(optical_signal)
    
    # Calculate frequency grid
    nfft = len(noisy_signal)
    f = np.fft.fftshift(np.fft.fftfreq(nfft, 1/params.Fs))
    
    # Create and apply optical filter
    B3dB = 4 * params.Rs
    Hf_field, Hf_power = create_gaussian_filter(f, 0, B3dB)
    filtered_signal = filter_signal(noisy_signal, Hf_field)
    
    # Print signal statistics for verification
    print("\nSignal Statistics:")
    print("Original signal power levels:", 
          np.unique(np.abs(optical_signal)**2))
    print("Noisy signal mean power:", 
          np.mean(np.abs(noisy_signal)**2))
    print("Filtered signal mean power:", 
          np.mean(np.abs(filtered_signal)**2))
    
    # Plot filter response
    plot_filter_response(f, Hf_power, 0, B3dB)
    
    # Plot eye diagrams
    plot_eyediagram(noisy_signal, params.N_ss, 200, 
                   'Eyediagram of Unfiltered Noisy Signal (Power)')
    plot_eyediagram(filtered_signal, params.N_ss, 200, 
                   'Eyediagram of Filtered Noisy Signal (Power)')
    
    # Plot PSDs
    plot_psd(noisy_signal, f, 'PSD of Unfiltered Noisy Signal')
    plot_psd(filtered_signal, f, 'PSD of Filtered Noisy Signal')
    
    # Plot time domain signals (first 1/100 of signal)
    display_length = len(params.t) // 100
    
    plt.figure(figsize=(10, 4))
    plt.plot(params.t[:display_length], 
            np.abs(noisy_signal[:display_length])**2)
    plt.xlabel('Time (s)')
    plt.ylabel('Optical Power')
    plt.title('Waveform of Unfiltered Noisy Signal (Power)')
    plt.grid(True)
    
    plt.figure(figsize=(10, 4))
    plt.plot(params.t[:display_length], 
            np.abs(filtered_signal[:display_length])**2)
    plt.xlabel('Time (s)')
    plt.ylabel('Optical Power')
    plt.title('Waveform of Filtered Noisy Signal (Power)')
    plt.grid(True)
    
    plt.show()

if __name__ == "__main__":
    main()