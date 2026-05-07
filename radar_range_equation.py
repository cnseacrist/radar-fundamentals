"""
Radar Range Equation Tool
=========================
Computes single-pulse SNR vs. range for a monostatic radar.

Author: Charles Seacrist
"""

import numpy as np
import matplotlib.pyplot as plt

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
# Atmospheric Attentuation (simplified ITU-R P.676)
#----------------------------------------
def atmospheric_loss_dB(range_m, alpha_dB_per_km=0.012):
    """
    Compute two-way atmostpheric attenuation in dB.

    Parameters
    ----------
    range_m          : Range to target in meters
    alpha_dB_per_km  : One way specific attenuation in [dB/km]
                       (default 0.012 dB/km for X-band)

    Returns
    -------
    loss_dB          : Two-way atmospheric loss in dB
    """
    range_km = range_m / 1000.0
    loss_dB = 2.0 * alpha_dB_per_km * range_km
    return loss_dB

#---------------------------------------
# SNR calculation
#----------------------------------------
def compute_snr(Pt_W, G_dBi, freq_Hz, sigma_m2, range_m, NF_dB, Bn_Hz,
                L_sys_dB, alpha_dB_per_km=None):
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
    alpha_dB_per_km : One way specific attenuation in [dB/km]

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

    # Step 4: Atmospheric loss (if enabled)
    if alpha_dB_per_km is not None:
        L_atm_dB = atmospheric_loss_dB(range_m, alpha_dB_per_km)
        L_atm_linear = db_to_linear(L_atm_dB)
    else:
        L_atm_linear = 1.0  # No atmospheric loss

    # Step 4: Build the radar range equation
    numerator = Pt_W * G_linear**2 * wavelength_m**2 * sigma_m2

    denominator = (4 * np.pi)**3 * range_m**4 * K_BOLTZMANN * T_sys * Bn_Hz * L_sys_linear * L_atm_linear

    snr_linear = numerator / denominator
    snr_db = linear_to_db(snr_linear)

    return snr_db

#----------------------------------------
# Max Detection Range Finder
#----------------------------------------  
def find_max_range(Pt_W, G_dBi, freq_Hz, sigma_m2, NF_dB, Bn_Hz, L_sys_dB, SNR_min_dB=13.0,
                   max_search_m=50000, step_m=1.0, alpha_dB_per_km=None):
    
    """
    Sweep through ranges and find where SNR drops below the detection threshold.

    Returns max range in meters. If the target is never detectable, returns 0.
    """
    # Start at 100m (can't detect at 0 range - division by zero)
    ranges = np.arange(100, max_search_m, step_m)

    for R in ranges:
        snr = compute_snr(Pt_W, G_dBi, freq_Hz, sigma_m2, R, NF_dB, Bn_Hz, L_sys_dB,
                          alpha_dB_per_km=alpha_dB_per_km)
        if snr < SNR_min_dB:
            return R - step_m # Return the last range where it was detectable

    return max_search_m # If we exhaust the search range, return the max search range as the limit

#----------------------------------------
# Plot
#----------------------------------------
def plot_snr_vs_range(Pt_W, G_dBi, freq_Hz, rcs_values, NF_dB, Bn_Hz, L_sys_dB,
                      SNR_min_dB=13.0, alpha_dB_per_km=None, max_range_km=20.0):
    """
    Plot SNR vs. range for multiple RCS values.

    Parameters
    __________
    rcs_values  : list of RCS values [m^2] to plot
    max_range_km: maximum range on the x_axis [km]
    (other params same as compute_snr)

    Saves the plot as 'snr_vs_range.png' in the current directory
    """

    # create an array of 500 range values from 100m to max_range_km
    ranges_m = np.linspace(100, max_range_km * 1000, 500)

    # create the figure and axes
    fig, ax = plt.subplots(figsize=(10,6))

    # compute snr for each rcs in rcs_values
    for sigma in rcs_values:
        snr = compute_snr(Pt_W, G_dBi, freq_Hz, sigma, ranges_m, NF_dB, Bn_Hz,
                          L_sys_dB, alpha_dB_per_km=None)
        # label = f"= {sigma} m^2"
        label = f"\u03c3 = {sigma} m\u00b2"
        ax.plot(ranges_m / 1000, snr, linewidth=2, label=label)

    # drawing the detection threshold
    ax.axhline(y=SNR_min_dB, color='red', linestyle='--', linewidth=1.5,
               label=f"Detection Threshold ({SNR_min_dB} dB)")

    # labeling axes
    ax.set_xlabel("Range [km]", fontsize=12)
    ax.set_ylabel("SNR [dB]", fontsize=12)
    ax.set_title("SNR vs. Range - X-band Radar", fontsize=14)

    # Adding the legend and grid
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    # Setting lower limit to -10dB. SNRs at ranges beyond this are unhelpful for this graph
    ax.set_ylim(bottom=-10.0)

    # Save figure to file
    plt.tight_layout()
    fig.savefig("snr_vs_range.png", dpi=500, bbox_inches='tight')
    print("  Saved: snr_vs_range.png")
    plt.close(fig)

def plot_uas_scenario(Pt_W, G_dBi, freq_Hz, NF_dB, Bn_Hz, L_sys_dB, SNR_min_dB=13.0,
                      alpha_dB_per_km=0.012, max_range_km=15.0):
    """
    C-UAS scenario: compare detection of two common commercial drones.

    Saves the plot as 'cuas_scenario.png' in the current directory.
    """
    # Define the two targets
    # These are approximate X-band RCS values from open literature.
    targets = { 
        "DJI Mavic 3  (\u03c3 \u2248 0.01 m\u00b2)": 0.01,
        "DJI Matrice 300 (\u03c3 \u2248 0.1 m\u00b2)": 0.1,
    }

    ranges_m = np.linspace(100, max_range_km * 1000, 500)

    # Create a figure with two side-by-side panels
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

    # Left panel: SNR vs Range for each drone ---
    colors = ['#e74c3c', '#2ecc71']

    for (name, sigma), color in zip(targets.items(), colors):
        snr = compute_snr(Pt_W, G_dBi, freq_Hz, sigma, ranges_m, NF_dB, Bn_Hz, L_sys_dB,
                          alpha_dB_per_km=alpha_dB_per_km)
        ax1.plot(ranges_m / 1000, snr, linewidth=2.5, label=name, color=color)

        # Find and mark the max detection range
        r_max = find_max_range(Pt_W, G_dBi, freq_Hz, sigma, NF_dB, Bn_Hz, L_sys_dB,
                               SNR_min_dB=SNR_min_dB, alpha_dB_per_km=alpha_dB_per_km)

        if r_max > 0:
            #draw a vertical dotted line at max range
            ax1.axvline(x=r_max / 1000, color=color, linestyle=':', alpha=0.7, linewidth=1.5)

            # Add a text annotation with an arrow
            if sigma < 0.05:
                # Mavic 3 (small RCS) — place label lower-right
                text_x = r_max / 1000 - 2.0
                text_y = SNR_min_dB - 8
            else:
                # Matrice 300 (larger RCS) — place label upper-right
                text_x = r_max / 1000
                text_y = SNR_min_dB + 8
            ax1.annotate(f"R_max = {r_max/1000:.1f} km",
                         xy=(r_max / 1000, SNR_min_dB),
                         xytext=(text_x, text_y),
                         fontsize=9, color=color,
                         arrowprops=dict(arrowstyle='->', color=color))

    ax1.axhline(y=SNR_min_dB, color='gray', linestyle='--', linewidth=1.5,
                label=f"Threshhold ({SNR_min_dB} dB)")
    ax1.set_xlabel("Range [km]", fontsize=12)
    ax1.set_ylabel("SNR [dB]", fontsize=12)
    ax1.set_title("C-UAS Detection: SNR vs. Range", fontsize=13)
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(bottom=-10)
    
    # --- Right panel: atmospheric attenuation vs range ---
    atm_loss_dB = np.array([atmospheric_loss_dB(r, alpha_dB_per_km) for r in ranges_m])
    ax2.plot(ranges_m / 1000, atm_loss_dB, linewidth=2, color='#3498bd')
    ax2.set_xlabel("Range [km]", fontsize=12)
    ax2.set_ylabel("Two-way atmostpheric loss [dB]", fontsize=12)
    ax2.set_title(f"Atmospheric attenuation (\u03b1 = {alpha_dB_per_km} dB/km)", fontsize=13)
    ax2.grid(True)

    # Add an overall title and save
    fig.suptitle("C-UAS Scenario: Generic X-band Radar vs. Small UAS targets", 
                 fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    fig.savefig("cuas_scenario.png", dpi=150, bbox_inches='tight')
    print(" Saved: cuas_scenario.png")
    plt.close(fig)

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

    # Atmospheric parameter
    alpha = 0.012 # dB/km, clear X-band

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
    print(f"  Atm Attenuation:  {alpha} dB/km (one-way)")
    print("=" * 55)


    # === Max detection range for several RCS values ===
    rcs_values = [0.001, 0.01, 0.1, 1.0]

    print(f"\n{'RCS [m²]':>12}  {'Free-space':>18}  {'With atm.':>16}  {'Range lost':>14}")
    print("-" * 66)

    for sigma in rcs_values:
        r_free = find_max_range(Pt_W, G_dBi, freq_Hz, sigma, NF_dB, Bn_Hz, L_sys_dB,
                                SNR_min_dB=SNR_min_dB, alpha_dB_per_km=None)
        r_atm = find_max_range(Pt_W, G_dBi, freq_Hz, sigma, NF_dB, Bn_Hz, L_sys_dB,
                               SNR_min_dB=SNR_min_dB, alpha_dB_per_km=alpha)
        lost = r_free - r_atm
        print(f"  {sigma:>10.3f}    {r_free / 1000:>12.2f} km    {r_atm / 1000:>12.2f} km    {lost:>10.0f} m")

    # === Show atmospheric loss at key ranges ===
    print(f"\nAtmospheric Loss at Key Ranges (alpha={alpha} dB/km):")
    print(f"  {'Range':>10}  {'Two-way loss':>14}")
    print(f"  {'-'*10}  {'-'*14}")
    for r_km in [1, 2, 5, 10, 20]:
        loss = atmospheric_loss_dB(r_km * 1000, alpha)
        print(f"  {r_km:>8} km {loss:>12.3f} dB")

    print()

    print("Generating Plot...")
    plot_snr_vs_range(Pt_W, G_dBi, freq_Hz, rcs_values, NF_dB, Bn_Hz, L_sys_dB, 
                      SNR_min_dB=SNR_min_dB, alpha_dB_per_km=alpha, max_range_km=20.0)

    # === C-UAS Scenario ===
    print("\n" + "=" * 55)
    print("C-UAS SCENARIO ANALYSIS")
    print("=" * 55)

    cuas_targets = {"DJI Mavic 3": 0.01, "DJI Matrice 300": 0.1}

    for name, sigma in cuas_targets.items():
        r_max = find_max_range(Pt_W, G_dBi, freq_Hz, sigma, NF_dB, Bn_Hz, L_sys_dB, 
                               SNR_min_dB=SNR_min_dB, alpha_dB_per_km=alpha)
        print(f"    {name} (\u03c3 \u2248 {sigma} m\u00b2): R_max = {r_max/1000:.2f} km")

    plot_uas_scenario(Pt_W, G_dBi, freq_Hz, NF_dB, Bn_Hz, L_sys_dB,
                      SNR_min_dB=SNR_min_dB, alpha_dB_per_km=alpha)


if __name__ == "__main__":
    main()
