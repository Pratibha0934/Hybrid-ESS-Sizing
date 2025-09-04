import numpy as np
import pandas as pd
from scipy import signal
import matplotlib.pyplot as plt
from scipy.optimize import minimize_scalar
import os
from PyEMD import EMD

# --- original helpers unchanged ---
def load_data(filename):
    if not os.path.exists(filename):
        raise FileNotFoundError(f"File not found: {filename}")
    if filename.endswith('.csv'):
        df = pd.read_csv(filename)
    elif filename.lower().endswith(('.xls', '.xlsx')):
        df = pd.read_excel(filename)
    else:
        raise ValueError("Unsupported file format. Must be CSV or Excel.")
    time_hours = df['Time_Hours'].values
    power = df['Original_Power'].values
    return time_hours * 3600, power


def calculate_storage_characteristics(power_profile, fs, soc_min, soc_max, efficiency=0.9):
    sgn_p = np.sign(power_profile)
    R_t = np.power(efficiency, -sgn_p) * power_profile
    nominal_power = np.max(np.abs(R_t))
    energy_profile = np.cumsum(R_t) / fs
    E_swing = np.max(energy_profile) - np.min(energy_profile)
    nominal_capacity = E_swing / (soc_max - soc_min)
    ramp_rate = np.max(np.abs(np.diff(R_t))) if len(R_t) > 1 else 0
    mode_changes = int(np.sum(np.diff(np.signbit(power_profile)) != 0)) if len(power_profile) > 1 else 0
    return {'nominal_power': nominal_power,
            'nominal_capacity': nominal_capacity,
            'ramp_rate': ramp_rate,
            'mode_changes': mode_changes}


def low_pass_filter(data, cutoff_freq, fs):
    nyquist = 0.5 * fs
    if cutoff_freq >= nyquist:
        return data.copy()
    normal_cutoff = cutoff_freq / nyquist
    b, a = signal.butter(4, normal_cutoff, btype='low')
    return signal.filtfilt(b, a, data)


def find_optimal_cutoff_frequency(power_profile, fs, c1, c2, c3):
    fc_candidates = np.linspace(0.002, 0.5 * fs, 50)
    En1, En2, Pn1, Pn2, Rn1, Rn2 = [], [], [], [], [], []
    for fc in fc_candidates:
        low = low_pass_filter(power_profile, fc, fs)
        high = power_profile - low
        b = calculate_storage_characteristics(low, fs, 0.10, 0.90)
        f = calculate_storage_characteristics(high, fs, 0.25, 1.00)
        En1.append(b['nominal_capacity']); En2.append(f['nominal_capacity'])
        Pn1.append(b['nominal_power']);    Pn2.append(f['nominal_power'])
        Rn1.append(b['ramp_rate']);       Rn2.append(f['ramp_rate'])
    valid = [i for i in range(len(fc_candidates)) if all(X[i] > 0 for X in (En1,En2,Pn1,Pn2,Rn1,Rn2))]
    maxE = max(En2[i]/En1[i] for i in valid)
    maxP = max(Pn1[i]/Pn2[i] for i in valid)
    maxR = max(Rn1[i]/Rn2[i] for i in valid)
    def obj(fc):
        low = low_pass_filter(power_profile, fc, fs)
        high = power_profile - low
        b = calculate_storage_characteristics(low,  fs, 0.10, 0.90)
        f = calculate_storage_characteristics(high, fs, 0.25, 1.00)
        t1 = (f['nominal_capacity']/b['nominal_capacity'])/maxE
        t2 = (b['nominal_power']/f['nominal_power'])/maxP
        t3 = (b['ramp_rate']/f['ramp_rate'])/maxR
        return np.inf if any(np.isinf(x) for x in (t1,t2,t3)) else c1*t1 + c2*t2 + c3*t3
    res = minimize_scalar(obj, bounds=(0.001,0.5*fs), method='bounded')
    return res.x


def analyze_hybrid_ess(power_profile, fs, c1, c2, c3, fc=None):
    if fc is None:
        fc = find_optimal_cutoff_frequency(power_profile, fs, c1, c2, c3)
    low = low_pass_filter(power_profile, fc, fs)
    high = power_profile - low
    return {'cutoff_frequency': fc,
            'battery': calculate_storage_characteristics(low, fs, 0.10, 0.90),
            'flywheel': calculate_storage_characteristics(high, fs, 0.25, 1.00),
            'low_freq_profile': low,
            'high_freq_profile': high}


# --- NEW: EMD-based split ---
def analyze_emd_ess(power_profile, fs, k_c=1):
    # time vector in seconds
    t = np.arange(len(power_profile)) / fs
    imfs = EMD().emd(power_profile, t)
    fast = np.sum(imfs[:k_c], axis=0)
    slow = power_profile - fast
    return {'battery': calculate_storage_characteristics(slow, fs, 0.10, 0.90),
            'flywheel': calculate_storage_characteristics(fast, fs, 0.25, 1.00),
            'k_c': k_c}


if __name__ == "__main__":
    # load & prepare
    filename = r"C:\Users\prach\standard_motif_data.csv"
    time, load_power = load_data(filename)
    residual = load_power - np.mean(load_power)
    fs = 1.0 if len(time)<2 else 1.0/((time[-1]-time[0])/(len(time)-1))
    print(f"Sampling freq: {fs:.3f} Hz")

    # LPF-method
    c1, c2, c3 = 1.0, 10.0, 20.0
    res_lpf = analyze_hybrid_ess(residual, fs, c1, c2, c3)

    # EMD-method (first 2 IMFs as fast)
    res_emd = analyze_emd_ess(residual, fs, k_c=1)

    # tabulated comparison
    metrics = ['nominal_power','nominal_capacity','ramp_rate','mode_changes']
    print("\nMethod | Component | " + " | ".join(metrics))
    print("-------|-----------|" + "|".join(["-------"]*len(metrics)))
    for method, res in [('LPF', res_lpf), ('EMD', res_emd)]:
        for comp in ['battery','flywheel']:
            vals = [res[comp][m] for m in metrics]
            print(f"{method:>3}   | {comp:>9} | " + " | ".join(f"{v:7.2f}" for v in vals))

    # bar‐chart comparison
    x = np.arange(len(metrics))
    width = 0.35
    b_lpf = [res_lpf['battery'][m] for m in metrics]
    b_emd = [res_emd['battery'][m] for m in metrics]
    f_lpf = [res_lpf['flywheel'][m] for m in metrics]
    f_emd = [res_emd['flywheel'][m] for m in metrics]

    fig, (ax1, ax2) = plt.subplots(2,1, figsize=(10,8))
    # battery
    ax1.bar(x-width/2, b_lpf, width, label='LPF')
    ax1.bar(x+width/2, b_emd, width, label='EMD')
    ax1.set_xticks(x); ax1.set_xticklabels(metrics, rotation=20)
    ax1.set_title('Battery Comparison'); ax1.legend(); ax1.set_yscale('symlog', linthresh=1e-2)
    ax1.grid(True)
    # flywheel
    ax2.bar(x-width/2, f_lpf, width, label='LPF')
    ax2.bar(x+width/2, f_emd, width, label='EMD')
    ax2.set_xticks(x); ax2.set_xticklabels(metrics, rotation=20)
    ax2.set_title('Flywheel Comparison'); ax2.legend(); ax2.set_yscale('symlog', linthresh=1e-2)
    ax2.grid(True)
    plt.tight_layout()
    plt.show()
