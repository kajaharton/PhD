# A code to generate a permutational ensemble
# Creates a csv file with all possible DNA combinatinos (A, T, G, C) for an n-mer of defined length

import itertools
import pandas as pd

# parameters
bases = ['A', 'T', 'G', 'C']
k = 4 # Change to the desired size
chunk_size = 500_000  # write in chunks to avoid memory issues
path_nmer = '/Users/kajaharton/Desktop/paper4_genomiccontext/thermodynamics/4bp_combinations/4bp_DNA_combinations.csv'

# open CSV file for writing
with open(path_nmer, 'w') as f:
    f.write('Sequence\n')  # header
    chunk = []
    count = 0
    for p in itertools.product(bases, repeat=k):
        chunk.append(''.join(p))
        count += 1
        if count % chunk_size == 0:
            f.write('\n'.join(chunk) + '\n')
            chunk = []
    # write any remaining sequences
    if chunk:
        f.write('\n'.join(chunk) + '\n')

path_nmer
