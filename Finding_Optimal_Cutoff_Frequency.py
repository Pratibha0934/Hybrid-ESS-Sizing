import numpy as np
import pandas as pd
from scipy import signal
import matplotlib.pyplot as plt
from scipy.optimize import minimize_scalar
import os

def load_data(filename):
    
    # Check file existence
    if not os.path.exists(filename):
        raise FileNotFoundError(f"File not found: {filename}")

    # Read data
    if filename.endswith('.csv'):
        df = pd.read_csv(filename)
    elif filename.endswith('.xlsx') or filename.endswith('.xls'):
        df = pd.read_excel(filename)
    else:
        raise ValueError("Unsupported file format. Must be CSV or Excel.")

    # Assumes columns "Time_Hours", "Original_Power"
    time_hours = df['Time_Hours'].values
    power = df['Original_Power'].values

    # Convert hours to seconds
    time_seconds = time_hours * 3600
    return time_seconds, power

def calculate_storage_characteristics(power_profile, fs, soc_min, soc_max, efficiency=0.9):
   
    # R(t) = η^(-sgn(P(t))) * P(t)
    sgn_p = np.sign(power_profile)
    R_t = np.power(efficiency, -sgn_p) * power_profile

    # Pn = max|R(t)|
    nominal_power = np.max(np.abs(R_t))

    # E(t) = ∫ R(t) dt 
    energy_profile = np.cumsum(R_t) / fs

    # En = [max(E(t)) - min(E(t))] / (soc_max - soc_min)
    E_swing = np.max(energy_profile) - np.min(energy_profile)
    nominal_capacity = E_swing / (soc_max - soc_min)

    # Rn = max |R(t) - R(t-1/fs)|
    if len(R_t) > 1:
        ramp_rate = np.max(np.abs(np.diff(R_t)))
    else:
        ramp_rate = 0

    # Mode changes
    if len(power_profile) > 1:
        mode_changes = np.sum(np.diff(np.signbit(power_profile)) != 0)
    else:
        mode_changes = 0

    return {
        'nominal_power': nominal_power,
        'nominal_capacity': nominal_capacity,
        'ramp_rate': ramp_rate,
        'mode_changes': mode_changes
    }

def low_pass_filter(data, cutoff_freq, fs):
    
    nyquist = 0.5 * fs
    if cutoff_freq >= nyquist:
        return data.copy()

    normal_cutoff = cutoff_freq / nyquist
    b, a = signal.butter(4, normal_cutoff, btype='low', analog=False)
    return signal.filtfilt(b, a, data)

def find_optimal_cutoff_frequency(power_profile, fs, c1, c2, c3):
   
    # 1) Gather ratio maxima over candidate frequencies
    fc_candidates = np.linspace(0.002, 0.5 * fs, 50)  # 50 steps
    ratio_En = []
    ratio_Pn = []
    ratio_Rn = []

    En1_list, En2_list = [], []
    Pn1_list, Pn2_list = [], []
    Rn1_list, Rn2_list = [], []

    for fc in fc_candidates:
        low_freq_profile = low_pass_filter(power_profile, fc, fs)
        high_freq_profile = power_profile - low_freq_profile

        # Battery
        batt = calculate_storage_characteristics(
            low_freq_profile, fs, soc_min=0.10, soc_max=0.90 ,efficiency=0.9
        )
        # Flywheel
        fly = calculate_storage_characteristics(
            high_freq_profile, fs, soc_min=0.25, soc_max=1.00 ,efficiency=0.9
        )

        En1_list.append(batt['nominal_capacity'])
        En2_list.append(fly['nominal_capacity'])
        Pn1_list.append(batt['nominal_power'])
        Pn2_list.append(fly['nominal_power'])
        Rn1_list.append(batt['ramp_rate'])
        Rn2_list.append(fly['ramp_rate'])

    # Filter out zeros to avoid inf
    valid_inds = [
        i for i in range(len(fc_candidates))
        if En1_list[i] != 0 and En2_list[i] != 0
        and Pn1_list[i] != 0 and Pn2_list[i] != 0
        and Rn1_list[i] != 0 and Rn2_list[i] != 0
    ]
    if not valid_inds:
        # If all are invalid, fallback
        return 0.001

    # Compute ratio max
    ratio_En = [En2_list[i]/En1_list[i] for i in valid_inds]
    ratio_Pn = [Pn1_list[i]/Pn2_list[i] for i in valid_inds]
    ratio_Rn = [Rn1_list[i]/Rn2_list[i] for i in valid_inds]

    max_En_ratio = max(ratio_En)
    max_Pn_ratio = max(ratio_Pn)
    max_Rn_ratio = max(ratio_Rn)

    # 2) Define normalized objective function
    def normalized_obj(fc):
        low_freq_profile = low_pass_filter(power_profile, fc, fs)
        high_freq_profile = power_profile - low_freq_profile

        batt = calculate_storage_characteristics(low_freq_profile, fs,  0.10, 0.90, 0.9)
        fly = calculate_storage_characteristics(high_freq_profile, fs,  0.25, 1.00, 0.9)

        En1 = batt['nominal_capacity']
        En2 = fly['nominal_capacity']
        Pn1 = batt['nominal_power']
        Pn2 = fly['nominal_power']
        Rn1 = batt['ramp_rate']
        Rn2 = fly['ramp_rate']

        if any(x == 0 for x in [En1, En2, Pn1, Pn2, Rn1, Rn2]):
            return np.inf

        term1 = (En2/En1) / max_En_ratio
        term2 = (Pn1/Pn2) / max_Pn_ratio
        term3 = (Rn1/Rn2) / max_Rn_ratio
        return c1*term1 + c2*term2 + c3*term3

    # 3) Minimize
    res = minimize_scalar(normalized_obj, bounds=(0.001, 0.5*fs), method='bounded')
    return res.x

def analyze_hybrid_ess(power_profile, fs,  c1, c2, c3, fc=None):
  
    if fc is None:
        fc = find_optimal_cutoff_frequency(power_profile, fs, c1, c2, c3)

    # Split into low and high freq
    low_freq_profile = low_pass_filter(power_profile, fc, fs)
    high_freq_profile = power_profile - low_freq_profile

    battery_chars = calculate_storage_characteristics(
        low_freq_profile, fs,  soc_min=0.10, soc_max=0.90 ,efficiency=0.9
    )
    flywheel_chars = calculate_storage_characteristics(
        high_freq_profile, fs,  soc_min=0.25, soc_max=1.00, efficiency=0.9
    )

    return {
        'cutoff_frequency': fc,
        'battery': battery_chars,
        'flywheel': flywheel_chars,
        'low_freq_profile': low_freq_profile,
        'high_freq_profile': high_freq_profile
    }

def plot_results(time, power_profile, results):
    plt.figure(figsize=(15, 9))
    # 1) Power curves
    plt.subplot(2, 1, 1)
    plt.plot(time, power_profile, 'k', label='Original Power Profile')
    plt.plot(time, results['low_freq_profile'], 'b', label='Battery (Low Freq.)')
    plt.plot(time, results['high_freq_profile'], 'r', label='Flywheel (High Freq.)')
    plt.title(f"Power Profiles with Cutoff={results['cutoff_frequency']:.6f} Hz")
    plt.xlabel("Time [s]")
    plt.ylabel("Power [kW]")
    plt.legend()
    plt.grid(True)

    # 2) Characteristics bar chart
    plt.subplot(2, 1, 2)
    metrics = ['nominal_power', 'nominal_capacity', 'ramp_rate', 'mode_changes']
    battery_vals = [results['battery'][m] for m in metrics]
    flywheel_vals = [results['flywheel'][m] for m in metrics]

    x = np.arange(len(metrics))
    width = 0.35
    plt.bar(x - width/2, battery_vals, width, label='Battery')
    plt.bar(x + width/2, flywheel_vals, width, label='Flywheel')
    plt.title("Storage Characteristics")
    plt.xticks(x, metrics)
    plt.ylabel("Value")
    plt.legend()
    plt.grid(True)
    plt.yscale('symlog', linthresh=1e-2) 
    plt.tight_layout()
    plt.show()
    

def compute_soc_profile(profile, fs, soc_min, soc_max, efficiency=0.9):
    """
    Given a power profile (kW) and storage limits [soc_min, soc_max],
    returns the SOC time‐series between soc_min and soc_max.
    """
    # R(t) = η^(−sgn(P)) * P
    sgn = np.sign(profile)
    R = profile * np.power(efficiency, -sgn)
    E = np.cumsum(R) / fs               # energy trajectory
    E_min, E_max = E.min(), E.max()
    E_swing = E_max - E_min
    # normalize to [soc_min, soc_max]
    soc = soc_min + (E - E_min) / E_swing * (soc_max - soc_min)
    return soc


if __name__ == "__main__":
    # 1) File path
    filename = r"C:\Users\prach\standard_motif_data.csv"

    # 2) Load data
    time, load_power = load_data(filename)
    
    # 3) Compute average and define RESIDUAL power
    avg_power = np.mean(load_power)
    power_profile = load_power-avg_power

    # 3) Compute sampling frequency
    if len(time) > 1:
        fs = 1 / ((time[-1] - time[0]) / (len(time) - 1))
    else:
        fs = 1.0

    print(f"Calculated sampling frequency: {fs} Hz")

    # 4) Decide weighting (higher c2 or c3 => push more HF to flywheel)
    c1, c2, c3 = 1.0, 20.0, 20.0

    # 5) Analyze
    results = analyze_hybrid_ess(power_profile, fs,  c1=c1, c2=c2, c3=c3,fc=None)

    # 6) Print & plot
    print(f"Optimal cutoff frequency: {results['cutoff_frequency']:.6f} Hz\n")
    print("Battery characteristics:")
    for k, v in results['battery'].items():
        print(f"  {k}: {v:.2f}")

    print("\nFlywheel characteristics:")
    for k, v in results['flywheel'].items():
        print(f"  {k}: {v:.2f}")

    plot_results(time, power_profile, results)
    
    # ------------------------------------------------------------------
    # 7) Scenario‐comparison plot for the *battery* power profile:
    #    - Hybrid:    low_freq_profile
    #    - Batt-only: full power_profile
    #    - No ESS:    zero
    # ------------------------------------------------------------------
    batt_hybrid = results['low_freq_profile']
    fly_hybrid  = results['high_freq_profile']
    batt_only   = power_profile.copy()
    no_ess      = np.zeros_like(power_profile)

    plt.figure(figsize=(12, 4))
    plt.plot(time, fly_hybrid, label='Hybrid ESS → Flywheel')
    plt.plot(time, batt_only,   label='Battery-Only (handles all)')
    plt.plot(time, batt_hybrid, label='Hybrid ESS → Battery')
    plt.title("Battery Power Profile Under Different ESS Scenarios")
    plt.xlabel("Time [s]")
    plt.ylabel("Power [kW]")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    # ------------------------------------------------------------------
    # 8) SOC‐vs‐time for battery & flywheel in the HYBRID case
    # ------------------------------------------------------------------
    soc_batt = compute_soc_profile(
        results['low_freq_profile'], fs,
        soc_min=0.10, soc_max=0.90, efficiency=0.9
    )
    soc_fly = compute_soc_profile(
        results['high_freq_profile'], fs,
        soc_min=0.25, soc_max=1.00, efficiency=0.9
    )

plt.figure(figsize=(12, 4))
plt.plot(time, soc_batt, label='Battery SOC')
plt.plot(time, soc_fly,  label='Flywheel SOC')

# horizontal dotted lines for battery SOC limits (0.10 & 0.90) in the same color as the batt curve
plt.axhline(0.10, color='C0', linestyle='--')
plt.axhline(0.90, color='C0', linestyle='--')

# horizontal dotted lines for flywheel SOC limits (0.25 & 1.00) in the same color as the fly curve
plt.axhline(0.25, color='C1', linestyle='--')
plt.axhline(1.00, color='C1', linestyle='--')

plt.title("State of Charge Over Time (Hybrid ESS)")
plt.xlabel("Time [s]")
plt.ylabel("SOC (fraction)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# --- after your existing code that computes `results` and prints opt_fc ---

# 1) Re-build the cutoff list and recompute all six curves
fc_candidates = np.linspace(0.001, 0.5 * fs, 50)      # same as in find_optimal_cutoff_frequency
En_batt, P_batt, R_batt = [], [], []
En_fly,  P_fly,  R_fly  = [], [], []

for fc in fc_candidates:
    low = low_pass_filter(power_profile, fc, fs)
    high = power_profile - low

    b = calculate_storage_characteristics(low,  fs, soc_min=0.10, soc_max=0.90, efficiency=0.9)
    f = calculate_storage_characteristics(high, fs, soc_min=0.25, soc_max=1.00, efficiency=0.9)

    En_batt.append(b['nominal_capacity'])
    P_batt.append(b['nominal_power'])
    R_batt.append(b['ramp_rate'])

    En_fly.append(f['nominal_capacity'])
    P_fly.append(f['nominal_power'])
    R_fly.append(f['ramp_rate'])

# 2) Find the index of the optimum fc in our candidate list
opt_idx = np.argmin(np.abs(fc_candidates - results['cutoff_frequency']))

# 3) Plot 3×2 subplots
fig, axes = plt.subplots(3, 2, figsize=(12, 10), sharex=True)
axes = axes.flatten()

labels = [
    ("Battery Capacity", En_batt, "Capacity"),
    ("Battery Power",    P_batt,  "Power"),
    ("Battery Ramp Rate",R_batt,  "Ramp Rate"),
    ("Flywheel Capacity",En_fly,  "Capacity"),
    ("Flywheel Power",   P_fly,   "Power"),
    ("Flywheel Ramp Rate",R_fly,  "Ramp Rate"),
]

for i, (title, data, ylabel) in enumerate(labels):
    ax = axes[i]
    is_batt = i < 3
    color = 'tab:blue' if is_batt else 'tab:orange'

    # plot the curve
    ax.plot(fc_candidates * 1000, data, color=color)
    # mark the optimum
    ax.scatter(fc_candidates[opt_idx] * 1000,
               data[opt_idx],
               color='red', zorder=5)

    # annotate
    ax.set_title(f"({chr(97+i)}) {title}")
    ax.set_ylabel(f"{ylabel}")
    ax.grid(True)

# 4) common X label (bottom row)
for ax in axes[4:6]:
    ax.set_xlabel("Cut‐off Frequency (mHz)")

plt.tight_layout()
plt.show()
