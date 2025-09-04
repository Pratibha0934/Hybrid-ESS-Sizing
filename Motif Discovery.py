import os
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import norm
from fastdtw import fastdtw
from scipy.spatial.distance import euclidean
import multiprocessing as mp
from functools import partial

# --- Data Loading and Preprocessing ---

def load_day_data(file_path):
    """Load and preprocess data efficiently"""
    df = pd.read_csv(file_path, low_memory=False)
    df['UpdateTime'] = pd.to_datetime(df['UpdateTime'])
    # Convert to float32 and ensure 1-D array
    ts = np.asarray(df['kPt'].values, dtype=np.float32).reshape(-1)
    return ts

def normalize_series(ts):
    ts = np.asarray(ts, dtype=np.float32).reshape(-1)
    ts_min = np.min(ts)
    ts_max = np.max(ts)
    if ts_max - ts_min > 0:
        return (ts - ts_min) / (ts_max - ts_min), ts_min, ts_max
    return ts - ts_min, ts_min, ts_max

def denormalize_series(normalized_ts, ts_min, ts_max):
    """Convert normalized values back to original scale"""
    return normalized_ts * (ts_max - ts_min) + ts_min

def ensure_equal_length(series_list):
    """Ensure all time series have the same length by padding with zeros"""
    max_length = max(len(s) for s in series_list)
    padded_series = []
    for s in series_list:
        # Ensure input is 1-D array
        s = np.asarray(s, dtype=np.float32).reshape(-1)
        if len(s) < max_length:
            # Pad with zeros at the end
            padded = np.pad(s, (0, max_length - len(s)), mode='constant', constant_values=0)
        else:
            padded = s
        padded_series.append(padded)
    return padded_series

# --- Dimensionality Reduction with PAA and SAX Transformation ---

def process_series(args):
    """Process a single time series through PAA and SAX"""
    ts, num_frames, alphabet_size = args
    # Ensure input is 1-D array
    ts = np.asarray(ts, dtype=np.float32).reshape(-1)
    paa_vals = paa(ts, num_frames)
    sax_word = sax_transform(paa_vals, alphabet_size)
    return sax_word

def paa(series, num_frames):
    """Vectorized PAA implementation"""
    # Ensure input is 1-D array
    series = np.asarray(series, dtype=np.float32).reshape(-1)
    n = len(series)
    frame_size = n // num_frames
    # Reshape and compute mean in one operation
    return np.mean(series[:num_frames*frame_size].reshape(-1, frame_size), axis=1)

def sax_transform(paa_vals, alphabet_size):
    """Vectorized SAX transformation"""
    # Ensure input is 1-D array
    paa_vals = np.asarray(paa_vals, dtype=np.float32).reshape(-1)
    breakpoints = norm.ppf(np.linspace(0, 1, alphabet_size + 1)[1:-1])
    # Vectorized comparison
    return np.searchsorted(breakpoints, paa_vals)

# --- Candidate Motif Discovery using Random Projection ---

def random_projection(sax_words, mask_length):
    word_length = len(sax_words[0])
    mask_indices = np.sort(np.random.choice(word_length, mask_length, replace=False))
    # Convert to numpy array for vectorized operations
    sax_array = np.array(sax_words)
    masked = sax_array[:, mask_indices]
    # Use numpy's unique to find groups
    unique, counts = np.unique(masked, axis=0, return_counts=True)
    return [np.where((masked == u).all(axis=1))[0] for u in unique[counts > 1]]

# --- Fast DTW for Candidate Refinement ---

def dtw_distance_fast(s1, s2, radius=10):
    # Ensure inputs are 1-D float32 arrays
    s1 = np.asarray(s1, dtype=np.float32).reshape(-1)
    s2 = np.asarray(s2, dtype=np.float32).reshape(-1)
    
    # Verify the inputs are 1-D
    if s1.ndim != 1 or s2.ndim != 1:
        raise ValueError(f"Input vectors must be 1-D. Got s1.ndim={s1.ndim}, s2.ndim={s2.ndim}")
    
    n = len(s1)
    m = len(s2)
    
    # Initialize DTW matrix
    dtw_matrix = np.full((n+1, m+1), np.inf, dtype=np.float32)
    dtw_matrix[0, 0] = 0
    
    # Compute DTW matrix with smaller cost sensitivity
    for i in range(1, n+1):
        for j in range(max(1, i-radius), min(m+1, i+radius+1)):
            cost = abs(s1[i-1] - s2[j-1]) * 0.1  
            dtw_matrix[i, j] = cost + min(
                dtw_matrix[i-1, j],    # insertion
                dtw_matrix[i, j-1],    # deletion
                dtw_matrix[i-1, j-1]   # match
            )
    
    # Return the normalized DTW distance
    return dtw_matrix[n, m] / (n + m)  # Normalize by length

def compute_group_dtw(args):
    group, day_series = args
    n = len(group)
    if n < 2:
        return np.inf
        
    # Pre-allocate distance matrix
    distances = np.zeros((n, n), dtype=np.float32)
    
    # Compute only upper triangle
    for i in range(n):
        for j in range(i+1, n):
            try:
                # Ensure inputs are 1-D arrays
                s1 = np.asarray(day_series[group[i]], dtype=np.float32).reshape(-1)
                s2 = np.asarray(day_series[group[j]], dtype=np.float32).reshape(-1)
                # Use a smaller radius for faster computation
                distances[i,j] = dtw_distance_fast(s1, s2, radius=5)
            except (ValueError, MemoryError) as e:
                print(f"Error comparing series {group[i]} and {group[j]}: {str(e)}")
                distances[i,j] = np.inf
    
    # Return mean of non-zero distances
    valid_distances = distances[distances > 0]
    return np.mean(valid_distances) if len(valid_distances) > 0 else np.inf

def refine_candidates(day_series, candidate_groups, dtw_threshold):
    """Optimized candidate refinement with parallel processing"""
    candidate_motifs = []
    
    # Filter out small groups
    candidate_groups = [g for g in candidate_groups if len(g) >= 2]
    
    if not candidate_groups:
        return candidate_motifs
    
    # Ensure all time series have the same length and are 1-D
    day_series = ensure_equal_length(day_series)
    
    # Process in parallel with progress tracking
    with mp.Pool(processes=mp.cpu_count()) as pool:
        args = [(group, day_series) for group in candidate_groups]
        avg_dtw = pool.map(compute_group_dtw, args)
    
    # Filter and create motifs
    for group, avg in zip(candidate_groups, avg_dtw):
        if avg < dtw_threshold:
            motif_group = np.array([day_series[i] for i in group])
            candidate_motifs.append(motif_group)
    
    return candidate_motifs

# --- Compute Standard Motif (80th Percentile) ---

def compute_standard_motif(candidate_motifs):
    if not candidate_motifs:
        return None
    
    # Find the group with most occurrences
    best_group = max(candidate_motifs, key=lambda x: x.shape[0])
    
    # Compute 80th percentile efficiently
    standard_motif = np.percentile(best_group, 80, axis=0)
    return standard_motif

# --- Main Execution for 30-Day Motif Discovery ---

if __name__ == "__main__":
    # Parameters adjusted for voltage/kPt data
    num_frames = 24       
    alphabet_size = 5     
    mask_length = 15      
    dtw_threshold = 0.1  
    dtw_radius = 3       
    
    # Data directory
    data_dir = r"C:\Users\prach\Downloads\data-BESS\2022-Sep_HT9_meter_data"
    file_pattern = os.path.join(data_dir, "*.csv")
    files = sorted(glob.glob(file_pattern))
    
    if len(files) < 30:
        print("Warning: Found less than 30 files. Please verify the data directory.")
    
    print(f"Processing {len(files)} files...")
    """print(f"Parameters: num_frames={num_frames}, alphabet_size={alphabet_size}, "
          f"mask_length={mask_length}, dtw_threshold={dtw_threshold}, dtw_radius={dtw_radius}")
    """
    # Process files in parallel
    with mp.Pool(processes=mp.cpu_count()) as pool:
        # Load and normalize data
        print("Loading and normalizing data...")
        day_series = pool.map(load_day_data, files)
        normalized_series = []
        original_mins = []
        original_maxs = []
        for ts in day_series:
            norm_ts, ts_min, ts_max = normalize_series(ts)
            normalized_series.append(norm_ts)
            original_mins.append(ts_min)
            original_maxs.append(ts_max)
        day_series = normalized_series
        
        # Ensure all series have the same length and are 1-D
        day_series = ensure_equal_length(day_series)
        
        # Process series through PAA and SAX
        print("Computing PAA and SAX transformations...")
        process_args = [(ts, num_frames, alphabet_size) for ts in day_series]
        sax_words = pool.map(process_series, process_args)
    
  
    candidate_groups = random_projection(sax_words, mask_length)
   
    print("Refining candidates...")
    candidate_motifs = refine_candidates(day_series, candidate_groups, dtw_threshold)
   
    
    print("\nComputing standard motif...")
    standard_motif = compute_standard_motif(candidate_motifs)
    
    if standard_motif is not None:
        # Create a figure with two subplots
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 16), dpi=100)
        
        # Plot 1: All Motifs
        best_group = max(candidate_motifs, key=lambda x: x.shape[0])
        time_hours = np.linspace(0, 24, best_group.shape[1])
        
        # Plot individual motifs
        for i, motif in enumerate(best_group):
            ax1.plot(time_hours, motif, alpha=0.2, linewidth=1)
        
       #Plot 1-> plot all recurring motifs
        ax1.set_title(f"All Recurring Motifs\n", 
                 fontsize=16, pad=20)
        ax1.set_xlabel("Time of Day (Hours)", fontsize=14)
        ax1.set_ylabel("Normalized Value", fontsize=14)
        ax1.grid(True, linestyle='--', alpha=0.7)
        ax1.legend(fontsize=12, loc='upper right')
        ax1.set_xticks(np.arange(0, 25, 3))
        ax1.set_xlim(0, 24)
        
        # Plot 2: Most Recurring Motif (Standard Motif)
        time_hours = np.linspace(0, 24, len(standard_motif))
        ax2.plot(time_hours, standard_motif, 'b-', linewidth=2, marker='.', markersize=4)
        ax2.set_title("Most Recurring Daily Pattern", fontsize=16, pad=20)
        ax2.set_xlabel("Time of Day (Hours)", fontsize=14)
        ax2.set_ylabel("Normalized Value", fontsize=14)
        ax2.grid(True, linestyle='--', alpha=0.7)
        
        # Add min/max annotations
        min_idx = np.argmin(standard_motif)
        max_idx = np.argmax(standard_motif)
        ax2.annotate(f'Min: {standard_motif[min_idx]:.2f}', 
                    (time_hours[min_idx], standard_motif[min_idx]),
                    xytext=(0, -20), textcoords='offset points',
                    ha='center', bbox=dict(facecolor='white', edgecolor='red', alpha=0.7))
        ax2.annotate(f'Max: {standard_motif[max_idx]:.2f}', 
                    (time_hours[max_idx], standard_motif[max_idx]),
                    xytext=(0, 20), textcoords='offset points',
                    ha='center', bbox=dict(facecolor='white', edgecolor='green', alpha=0.7))
        
        ax2.set_xticks(np.arange(0, 25, 3))
        ax2.set_xlim(0, 24)
        
        plt.tight_layout()
        plt.show()
        
        # Calculate average min and max from original data
        avg_min = np.mean(original_mins)
        avg_max = np.mean(original_maxs)
        
        # Convert standard motif back to original scale
        original_standard_motif = denormalize_series(standard_motif, avg_min, avg_max)
        
        # Create DataFrame with both normalized and original values
        data_df = pd.DataFrame({
            'Time_Hours': time_hours,
            'Normalized_Power': standard_motif,
            'Original_Power': original_standard_motif
        })
        
        # Export to CSV
        output_file = 'standard_motif_data.csv'
        output_path = os.path.abspath(output_file)
        data_df.to_csv(output_file, index=False)
        print(f"\nData exported to: {output_path}")
        
        # Print arrays
        print("\nTime array (hours):")
        print(time_hours)
        print("\nNormalized power values array:")
        print(standard_motif)
        print("\nOriginal power values array:")
        print(original_standard_motif)
    else:
        print("No candidate motifs found with the given parameters.")
        print("\nSuggestions for voltage/kPt data:")
        print("1. Try increasing dtw_threshold (currently {})".format(dtw_threshold))
        print("2. Try adjusting num_frames to match data frequency (currently {})".format(num_frames))
        print("3. Check data quality and missing values")
        print("4. Consider analyzing voltage and kPt separately")
