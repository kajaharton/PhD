# Calculating cumulative entropy (shannon) for a csv file with sequences
# It was devised for 10 bp sequence - alter in process_csv function as needed

# The calculations are based on Gabrielian and Bolshoy, 1999 doi.org/10.1016/S0097-8485(99)00007-8
# Briefly, shannon entropy is cumulated as a sum of probabilities that a k-mer occurs in a sequence S times the actual number of occurances. 
# The results are normalised by a maximum entropy to account for the sequence length 

import csv
import math
from collections import Counter
from tqdm import tqdm
import pandas as pd

def shannon_entropy(sequence, k):
    """
    Calculate the Shannon entropy for k-mers in a given sequence.
    """
    kmers = [sequence[i:i+k] for i in range(len(sequence) - k + 1)]
    freq = Counter(kmers)
    total_kmers = len(kmers)
    
    entropy = 0
    for count in freq.values():
        p = count / total_kmers
        entropy -= p * math.log2(p)
    
    return entropy

def calculate_cumulative_entropy(sequence):
    """
    Calculate Shannon entropy for k-mers (1 to 9) and the cumulative entropy.
    """
    max_k = len(sequence)
    entropies = []
    
    for k in range(1, max_k):  # Calculate for k = 1 to 9
        entropy = shannon_entropy(sequence, k)
        max_entropy = math.log2(4**k)  # Maximum entropy for k-mers
        normalized_entropy = entropy / max_entropy if max_entropy > 0 else 0
        entropies.append(normalized_entropy)
    
    # Cumulative entropy is the sum of normalized entropies
    cumulative_entropy = sum(entropies)
    entropies.append(cumulative_entropy)
    
    return entropies

def process_csv(input_csv, output_csv):
    """
    Read the input CSV file, compute entropies, and write to the output CSV file.
    """
    # Read the input CSV
    df = pd.read_csv(input_csv)
    sequences = df.iloc[:, 0].values  # Assume sequences are in the first column
    
    results = []
    for seq in tqdm(sequences, desc="Processing Sequences"):
        if len(seq) == 10:  # Ensure sequence length is 10
            entropies = calculate_cumulative_entropy(seq)
            results.append([seq] + entropies)
        else:
            print(f"Skipping sequence (not length 10): {seq}")
    
    # Prepare column names
    columns = ['Sequence'] + [f'Entropy_k={k}' for k in range(1, 10)] + ['Cumulative_Entropy']
    
    # Write the results to a new CSV file
    output_df = pd.DataFrame(results, columns=columns)
    output_df.to_csv(output_csv, index=False)
    print(f"Results written to {output_csv}")

# Example Usage
input_csv = "/XXX/XXX/10bp_DNA_combinations.csv"  # Replace with your input file path for any set of sequences
output_csv = "/XXX/XXX/10bp_DNA_combinations_shannonentropy.csv"  # Replace with your desired output file path
process_csv(input_csv, output_csv)
