from test import SignalParameters, generate_ook_signal, apply_MZM
from test2 import create_gaussian_filter, filter_signal
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator

def add_noise_calibrated(signal, OSNR_dB):
    """
    Add calibrated AWGN noise for given OSNR.
    Using reference bandwidth of 12.5 GHz (0.1nm @ 1550nm)
    """
    OSNR_linear = 10 ** (OSNR_dB / 10)
    signal_power = np.mean(np.abs(signal) ** 2)
    
    # Using reference bandwidth
    Bref = 12.5e9  # 0.1nm @ 1550nm
    Nase = signal_power / (2 * Bref * OSNR_linear)
    
    # Generate complex noise
    noise = np.sqrt(Nase * Bref) * (np.random.randn(len(signal)) + 
                                   1j * np.random.randn(len(signal)))
    
    return signal + noise

def detect_and_decide(signal, samples_per_symbol):
    """Detect and make symbol decisions for OOK signal"""
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

def calculate_ber(decisions, original_bits):
    """Calculate BER"""
    errors = np.sum(decisions != original_bits)
    return max(errors / len(original_bits), 1e-6)

def simulate_ber_curve(optical_signal, original_bits, params, f, filter_bandwidth=None):
    """Simulate BER curve for given filter bandwidth"""
    OSNR_range = np.arange(4, 17, 1)
    BER_results = []
    
    # Create filter if bandwidth specified
    if filter_bandwidth is not None:
        Hf_field, _ = create_gaussian_filter(f, 0, filter_bandwidth)
    
    for OSNR_dB in OSNR_range:
        # Add noise
        noisy_signal = add_noise_calibrated(optical_signal, OSNR_dB)
        
        # Apply filter if specified
        if filter_bandwidth is not None:
            signal_to_detect = filter_signal(noisy_signal, Hf_field)
        else:
            signal_to_detect = noisy_signal
            
        # Detect and calculate BER
        decisions = detect_and_decide(signal_to_detect, params.N_ss)
        ber = calculate_ber(decisions, original_bits)
        
        BER_results.append(ber)
        print(f"OSNR: {OSNR_dB} dB, BER: {ber:.2e}")
    
    return OSNR_range, BER_results

def main():
    # Initialize parameters
    params = SignalParameters(N_ss=16, Rs=10e9, Nsym=100000)
    
    # Generate data and signal
    electrical_signal = generate_ook_signal(params)
    optical_signal = apply_MZM(electrical_signal)
    original_bits = electrical_signal[::params.N_ss]
    
    # Setup frequency grid
    nfft = len(optical_signal)
    f = np.fft.fftshift(np.fft.fftfreq(nfft, 1/params.Fs))
    
    # Define filter bandwidths to test (same as reference plot)
    filter_bandwidths = {
        'Without Filter': None,
        'Filter 2·Rs': 2 * params.Rs,
        'Filter 0.75·Rs': 0.75 * params.Rs,
        'Filter 0.5·Rs': 0.5 * params.Rs
    }
    
    # Create plot
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Markers for different curves
    markers = ['o', 's', 'D', '^']
    
    # Simulate and plot for each bandwidth
    for (label, bandwidth), marker in zip(filter_bandwidths.items(), markers):
        print(f"\nSimulating {label}...")
        OSNR_range, BER_results = simulate_ber_curve(optical_signal, original_bits, 
                                                    params, f, bandwidth)
        
        # Convert BER to -log10 and plot
        log_ber = -np.log10(BER_results)
        ax.plot(OSNR_range, log_ber, '-' + marker, label=label, markerfacecolor='white',
                linewidth=1, markersize=6)
    
    # Set axis limits and ticks
    ax.set_xlim(4, 16)
    ax.set_ylim(5, 0)
    
    # Major and minor ticks
    ax.xaxis.set_major_locator(MultipleLocator(2))
    ax.xaxis.set_minor_locator(MultipleLocator(1))
    ax.yaxis.set_major_locator(MultipleLocator(1))
    ax.yaxis.set_minor_locator(MultipleLocator(0.1))
    
    # Grid
    ax.grid(True, which='major', linestyle='-', alpha=0.7)
    ax.grid(True, which='minor', linestyle=':', alpha=0.4)
    
    # Labels and legend
    ax.set_xlabel('OSNR (dB)', fontsize=14)
    ax.set_ylabel('-log$_{10}$(BER)', fontsize=14)
    ax.legend(fontsize=12)
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()