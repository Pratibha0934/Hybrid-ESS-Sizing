PROCESS FLOW
1. Motif_Discovery.py (MAIN SCRIPT - Run First)
Purpose:
● Discover the most recurring motif (pattern) in real power data.
● Generate a graph of the standard motif.
● Save a CSV file named standard_motif.csv containing timestamp vs
real power values.
Steps:
● In main, update the data_dir variable in the script to point to
the path of the folder containing the dataset.
● Run the script.
Output:
● standard_motif.csv
● Graph showing the most recurring pattern (motif).
How to Run:
python Motif_Discovery.py
Make sure the input data is in the expected format and located in the
correct path if required.
2. Finding_Optimal_Cutoff_Frequency.py
   Purpose:
● Load the standard motif.
● Apply a low-pass filter (LPF) with various cutoff frequencies.
● Evaluate optimal cutoff based on filtering results.
Steps:
● In main, update the filename variable in the script to point to
the path of standard_motif.csv.
● Run the script.
Output:
● Multiple graphs showing filtering results
● Text-based results printed to the terminal.
How to Run:
python Finding_Optimal_Cutoff_Frequency.py
3. Alternate_Approach.py
Purpose:
● Compare two signal processing approaches:
○ Low-Pass Filtering (LPF)
○ Empirical Mode Decomposition (EMD)
● Display a tabular result in the terminal.
● Plot a comparison graph.
Steps:
● In the main function, update the filename variable to the path of
standard_motif.csv.
● Run the script.
Output:
● Terminal display of tabular results comparing LPF and EMD
methods.
● A comparison graph between the two techniques.
How to Run:
python Alternate_Approach.py
Prerequisites
Ensure you have the following Python libraries installed:
● numpy
● pandas
● matplotlib
● scipy
● PyEMD (for Alternate_Approach.py)
You can install them using pip:
pip install numpy pandas matplotlib scipy EMD-signal,
