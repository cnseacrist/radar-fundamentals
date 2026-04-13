"""
Radar Range Equation Tool
=========================
Computes single-pulse SNR vs. range for a monostatic radar.

Author: Charles Seacrist
"""

import numpy as np

#----------------------------------------
# Physical Constants
#----------------------------------------
c = 299_792_458  # Speed of light in m/s    
K_BOLTZMANN = 1.380649e-23  # Boltzmann's constant in J/K
T_0 = 290  # Standard temperature in K 

#----------------------------------------
# Helper: convert dB to linear and vice versa
#----------------------------------------
def db_to_linear(value_dB):
    """Convert a value in dB to a linear ratio."""
    return 10 ** (value_dB / 10)

def linear_to_db(value_linear):
    """Convert a linear ratio to dB."""
    return 10 * np.log10(value_linear)

#----------------------------------------
# SNR calculation
#----------------------------------------
def compute_snr(Pt_W, G_dBi, freq_Hz, sigma_m2, range_m, NF_dB, Bn_Hz, L_sys_dB):
    """
    Compute single-pule SNR in dB.

    Parameters
    ----------
    Pt_W      : Peak transmit power [W]
    G_dBi      : Antenna gain [dBi]
    freq_Hz   : Operating frequency [Hz]
    sigma_m2  : Target radar cross section [m^2]
    range_m   : Range to target [m]
    NF_dB     : Receiver noise figure [dB]
    Bn_Hz     : Noise bandwidth [Hz]
    L_sys_dB  : Total system losses [dB]

    Returns
    -------
    snr_dB : SNR in dB
    """
    # Step 1: Derive wavelength from frequency
    wavelength_m = c / freq_Hz

    # Step 2: Convert dB quantities to linear ratios
    G_linear = db_to_linear(G_dBi)
    NF_linear = db_to_linear(NF_dB)
    L_sys_linear = db_to_linear(L_sys_dB)

    # Step 3: Compute system noise temperature  
    # T_sys = T_0 * NF_linear is the simplified form.
    # This assumes is looking at a scene near 290K,
    # which is reasonable for a ground-based detector looking at the horizon.
    T_sys = T_0 * NF_linear

    # Step 4: Build the radar range equation
    numerator = Pt_W * G_linear**2 * wavelength_m**2 * sigma_m2

    denominator = (4 * np.pi)**3 * range_m**4 * K_BOLTZMANN * T_sys * Bn_Hz * L_sys_linear

    snr_linear = numerator / denominator
    snr_db = linear_to_db(snr_linear)

    return snr_db

#----------------------------------------
# Max Detection Range Finder
#----------------------------------------  
def find_max_range(Pt_W, G_dBi, freq_Hz, sigma_m2, NF_dB, Bn_Hz, L_sys_dB, SNR_min_dB=13.0, max_search_m=50000, step_m=1.0):
    
    """
    Sweep through ranges and find where SNR drops below the detection threshold.

    Returns max range in meters. If the target is never detectable, returns 0.
    """
    # Start at 100m (can't detect at 0 range - division by zero)
    ranges = np.arange(100, max_search_m, step_m)

    for R in ranges:
        snr = compute_snr(Pt_W, G_dBi, freq_Hz, sigma_m2, R, NF_dB, Bn_Hz, L_sys_dB)
        if snr < SNR_min_dB:
            return R - step_m # Return the last range where it was detectable

    return max_search_m # If we exhaust the search range, return the max search range as the limit


#----------------------------------------
# Main 
#----------------------------------------
def main():
    # === define radar parameters ===
    # These represent a modest X-band surveillance radar.
    Pt_W = 1000       # 1kW peak power
    G_dBi = 30.0      # 30 dBi antenna gain (~1m dish at X-band)
    freq_Hz = 10e9    # 10 GHz (X-band)
    NF_dB = 4.0       # 4 dB noise figure
    Bn_Hz = 1e6       # 1 MHz noise bandwidth
    L_sys_dB = 6.0    # 6 dB system losses
    SNR_min_dB = 13.0 # Minimum SNR for detection

    # === print system summary ===
    wavelength_m = c / freq_Hz

    print("=" * 55)
    print("RADAR RANGE EQUATION TOOL")
    print("=" * 55)
    print(f"  Peak Power:       {Pt_W} W  ({linear_to_db(Pt_W):.1f} dBW)")
    print(f"  Antenna Gain:     {G_dBi} dBi")
    print(f"  Frequency:        {freq_Hz / 1e9} GHz")
    print(f"  Wavelength:       {wavelength_m * 100:.2f} cm")
    print(f"  Noise Figure:     {NF_dB} dB")
    print(f"  Noise Bandwidth:  {Bn_Hz / 1e6} MHz")
    print(f"  System Losses:    {L_sys_dB} dB")
    print(f"  Detection Thresh: {SNR_min_dB} dB")
    print("=" * 55)

    # === Compute SNR at a specific range (sanity check) ===
    test_range = 2000  # 2 km
    test_rcs = 0.1     # 0.1 m^2
    snr = compute_snr(Pt_W, G_dBi, freq_Hz, test_rcs, test_range, NF_dB, Bn_Hz, L_sys_dB)
    print(f"\nSanity check: SNR at {test_range}m for σ={test_rcs} m² = {snr:.1f} dB")

    # === Max detection range for several RCS values ===
    rcs_values = [0.001, 0.01, 0.1, 1.0]

    print(f"\n{'RCS [m²]':>12}  {'Max Detection Range':>22}")
    print("-" * 38)

    for sigma in rcs_values:
        r_max = find_max_range(Pt_W, G_dBi, freq_Hz, sigma, NF_dB, Bn_Hz, L_sys_dB, SNR_min_dB=SNR_min_dB)
        print(f"  {sigma:>10.3f}    {r_max / 1000:>18.2f} km")

    print()


if __name__ == "__main__":
    main()