# A script to calculate the thermodynamic properties of randomly chosen locations from a csv file (single column with sequences)
# Calculates: GC content, dG, dH, and dS

# Input: input sequences, thermodynamic raw data (csv file), output folder
# Output: a csv file, with GCcontent, dH, dS, dG, fragment size for each input sequence

import pandas as pd
from tqdm import tqdm

# Load nearest-neighbor thermodynamic values
def load_nn_thermo_properties(csv_file):
    """Load nearest-neighbor thermodynamic properties from a CSV file."""
    df = pd.read_csv(csv_file)
    nn_thermo = df.set_index('Pair')
    return nn_thermo

# Calculate thermodynamic properties for a given sequence
def calculate_thermo_properties(sequence, nn_thermo, property):
    """Calculate thermodynamic properties (dH, dS, dG) for a given sequence."""
    total_value = 0
    sequence_upper = sequence.upper()
    
    for i in range(len(sequence_upper) - 1):
        pair = sequence_upper[i:i+2]
        if pair in nn_thermo.index:
            total_value += nn_thermo.loc[pair][property]
        else:
            total_value += 0  # Handle invalid pairs
    
    return total_value

# Calculate the GC content
def calculate_gc_content(sequence):
    """Calculate the GC content of a given sequence."""
    sequence_upper = sequence.upper()
    gc_count = sequence_upper.count('G') + sequence_upper.count('C')
    return (gc_count / len(sequence_upper)) * 100 if sequence_upper else 0

# Main function to process sequences and calculate all properties
def process_sequences(input_csv, nn_thermo_csv, output_csv):
    nn_thermo = load_nn_thermo_properties(nn_thermo_csv)
    sequences_df = pd.read_csv(input_csv)
    
    results = []
    for _, row in tqdm(sequences_df.iterrows(), total=len(sequences_df), desc="Processing sequences"):
        sequence = row['Sequence']
        gc_content = calculate_gc_content(sequence)
        dH = calculate_thermo_properties(sequence, nn_thermo, 'dH')
        dS = calculate_thermo_properties(sequence, nn_thermo, 'dS')
        dG = calculate_thermo_properties(sequence, nn_thermo, 'dG')
        
        results.append({
            'Sequence': sequence,
            'GC_Content': gc_content,
            'dH': dH,
            'dS': dS,
            'dG': dG
        })
    
    # Save results to a new CSV
    results_df = pd.DataFrame(results)
    results_df.to_csv(output_csv, index=False)
    print(f"Saved results to {output_csv}")

# Example usage
input_csv = "/XXX/XXX/10bp_DNA_combinations.csv"  # Input file with sequences
nn_thermo_csv = "/XXX/XXX/therdym_NNbp_santalucia1998.csv"  # CSV with the experimental thermodynamic values
output_csv = "/XXX/XXX/10bp_DNA_thermo_results.csv"  # Output file

process_sequences(input_csv, nn_thermo_csv, output_csv)
