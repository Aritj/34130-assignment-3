# Fiber Optic Communication Systems Assignment - WDM System Analysis

This repository contains solutions for the Fundamentals of Fiber-Optic Communication Systems course assignment, focusing on Wavelength Division Multiplexing (WDM) systems simulation and analysis.

## Assignment Overview

The assignment is divided into two main exercises:

### Exercise 1: Basics of Optical Filter
1. **Q1-1**: Adding noise to optical signal
   - Implementation of OSNR calculation
   - Generation of OOK signal with NRZ pulse shaping
   - Addition of AWGN according to specified OSNR
   - Visualization of signal spectra

2. **Q1-2**: Optical Filtering
   - Implementation of Gaussian-shaped optical filter
   - Application of filter to noisy signal
   - Analysis of filtering effects through eye diagrams and spectra

3. **Q1-3**: Performance Evaluation
   - BER analysis for different filter configurations
   - Performance comparison of various filter bandwidths
   - Study of signal quality vs OSNR

### Exercise 2: WDM System
1. **Q2-1**: WDM Signal Generation
   - Generation of 5-channel WDM signal
   - 50 GHz channel spacing
   - 10 Gbaud per channel
   - Spectral analysis of combined signal

2. **Q2-2**: WDM Signal Detection
   - Channel extraction using optical filtering
   - Direct detection implementation
   - Eye diagram analysis
   - BER calculation

3. **Q2-3**: Complete WDM System Analysis
   - Full system performance evaluation
   - BER vs OSNR analysis for all channels
   - Comparison of channel performance
   - System optimization suggestions

## Code Structure
- `q1_1.py`: Basic signal generation and noise addition
- `q1_2.py`: Optical filtering implementation
- `q1_3.py`: BER performance analysis
- `q2_1.py`: WDM signal generation
- `q2_2.py`: Single channel detection
- `q2_3.py`: Complete WDM system simulation

## Dependencies
- NumPy
- Matplotlib

## Results
The code generates various visualizations:
- Power Spectral Density plots
- Eye diagrams
- BER vs OSNR curves
- Filter transfer functions

## Usage
1. Install dependencies:
```bash
pip install numpy matplotlib
```

2. Run individual questions:
```bash
python q1_1.py
python q1_2.py
python q1_3.py
python q2_1.py
python q2_2.py
python q2_3.py
```

## Key Findings
1. **Filter Bandwidth Effects**:
   - 2Rs bandwidth provides optimal performance
   - Narrower filters cause signal distortion
   - Wider filters allow more noise

2. **WDM Performance**:
   - Channel spacing adequately prevents interference
   - Center and edge channels show different performance
   - OSNR requirements vary by channel position

3. **System Optimization**:
   - Filter bandwidth critically affects performance
   - Trade-off between noise reduction and signal distortion
   - Channel spacing impacts inter-channel interference

## Future Improvements
- Implementation of advanced modulation formats
- Addition of fiber propagation effects
- Enhanced filtering techniques
- Chromatic dispersion compensation
