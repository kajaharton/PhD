# A script to calculate the thermodynamic properties of randomly chosen locations from a reference genome (.fna or .fa format)
# Step 1: randomly select coordinates from the genome (all chromosomes included)
# Step 2: calculate the GC content, dG, dH, and dS for fragments around the selected coordinates

# Input: reference genome (.fna or .fa); thermodynamic raw data (csv file); output folder
# Output: random_locations file with the coordinates used; csv files for each combination of GC content, dH, dS, dG x fragment size

# Customisable elements: number of locations, fragment sizes 
# Runtime: on macbook pro, 10k locations for 10bp, 50, 100, 500...100kb fragments runs about 20h 
# The script can pick up from half-ran data (uses the same random_locations, as long as the output folder is set the same)


import random
from Bio import SeqIO
import pandas as pd
from tqdm import tqdm
import os

# Load the nearest-neighbor thermodynamic values
def load_nn_thermo_properties(csv_file):
    """Load nearest-neighbor thermodynamic properties from a CSV file."""
    df = pd.read_csv(csv_file)
    nn_thermo = df.set_index('Pair')
    return nn_thermo

# Calculate thermodynamic properties for a given sequence and property
def calculate_thermo_properties(sequence, nn_thermo, property):
    """Calculate thermodynamic properties (dH, dS, dG) for a given sequence."""
    total_value = 0
    sequence_upper = sequence.upper()
    
    for i in range(len(sequence_upper) - 1):
        pair = sequence_upper[i:i+2]
        
        if pair in nn_thermo.index:
            total_value += nn_thermo.loc[pair][property]
        else:
            total_value += 0  # Handle invalid pairs if needed

    return total_value

# Calculate the GC content of a given sequence
def calculate_gc_content(sequence):
    """Calculate the GC content of a given sequence."""
    sequence_upper = sequence.upper()
    gc_count = sequence_upper.count('G') + sequence_upper.count('C')
    valid_bases = sequence_upper.count('A') + sequence_upper.count('T') + gc_count
    return (gc_count / valid_bases) * 100 if valid_bases > 0 else 0

# Extract random locations from the reference genome and save them
def extract_and_save_random_locations(reference_genome, num_locations, output_file):
    """Extract random locations ensuring each chromosome is represented and save to CSV."""
    locations = []
    chromosomes = list(SeqIO.parse(reference_genome, "fasta"))
    
    for record in tqdm(chromosomes, desc="Extracting random locations"):
        chromosome = record.id
        chr_length = len(record.seq)
        location = random.randint(0, chr_length - 1)
        locations.append((chromosome, location, chr_length))

    remaining_locations = num_locations - len(locations)
    
    if remaining_locations > 0:
        for _ in tqdm(range(remaining_locations), desc="Adding more random locations"):
            record = random.choice(chromosomes)
            chromosome = record.id
            chr_length = len(record.seq)
            location = random.randint(0, chr_length - 1)
            locations.append((chromosome, location, chr_length))

    # Save the locations to a CSV file
    df_locations = pd.DataFrame(locations, columns=['chromosome', 'location', 'chr_length'])
    df_locations.to_csv(output_file, index=False)
    print(f"Saved locations to {output_file}")

# Process each location and calculate GC content
def process_gc_content(locations_csv, reference_genome, sizes, output_folder):
    """Calculate GC content for each location and size."""
    locations_df = pd.read_csv(locations_csv)
    chromosomes = list(SeqIO.parse(reference_genome, "fasta"))

    for size in tqdm(sizes, desc="Processing GC content"):
        results = []

        for _, row in locations_df.iterrows():
            chromosome, loc_position, _ = row['chromosome'], row['location'], row['chr_length']

            for record in chromosomes:
                if record.id == chromosome:
                    chr_seq = str(record.seq)
                    chr_length = len(chr_seq)

                    start = max(loc_position - size // 2, 0)
                    end = min(loc_position + size // 2, chr_length)
                    sequence = chr_seq[start:end]

                    gc_content = calculate_gc_content(sequence)

                    results.append({
                        'chromosome': chromosome,
                        'start': start,
                        'end': end,
                        'size': size,
                        'gc_content': gc_content
                    })
                    break
        
        # Save GC content results for each size
        df_results = pd.DataFrame(results)
        output_file = f"{output_folder}/gc_content_{size}bp.csv"
        df_results.to_csv(output_file, index=False)
        print(f"Saved {output_file}")

# Process each location and calculate thermodynamic properties
def process_thermo_property(locations_csv, reference_genome, sizes, nn_thermo_csv, output_folder, property):
    """Calculate thermodynamic property (dH, dS, or dG) for each location and size."""
    nn_thermo = load_nn_thermo_properties(nn_thermo_csv)
    locations_df = pd.read_csv(locations_csv)
    chromosomes = list(SeqIO.parse(reference_genome, "fasta"))

    for size in tqdm(sizes, desc=f"Processing {property}"):
        results = []

        for _, row in locations_df.iterrows():
            chromosome, loc_position, _ = row['chromosome'], row['location'], row['chr_length']

            for record in chromosomes:
                if record.id == chromosome:
                    chr_seq = str(record.seq)
                    chr_length = len(chr_seq)

                    start = max(loc_position - size // 2, 0)
                    end = min(loc_position + size // 2, chr_length)
                    sequence = chr_seq[start:end]

                    thermo_value = calculate_thermo_properties(sequence, nn_thermo, property)

                    results.append({
                        'chromosome': chromosome,
                        'start': start,
                        'end': end,
                        'size': size,
                        property: thermo_value
                    })
                    break
        
        # Save thermodynamic property results for each size
        df_results = pd.DataFrame(results)
        output_file = f"{output_folder}/thermo_properties_{size}bp_{property}.csv"
        df_results.to_csv(output_file, index=False)
        print(f"Saved {output_file}")

# Main function to process locations sequentially
def main_process(reference_genome, num_locations, sizes, nn_thermo_csv, output_folder):
    locations_csv = f"{output_folder}/random_locations.csv"
    
    # Step 1: Extract random locations and save them (skip if already exists)
    if not os.path.exists(locations_csv):
        extract_and_save_random_locations(reference_genome, num_locations, locations_csv)
    else:
        print(f"Locations already exist at {locations_csv}, skipping extraction.")

    # Step 2: Process GC content (skip sizes already processed)
    for size in sizes:
        gc_output_file = f"{output_folder}/gc_content_{size}bp.csv"
        if not os.path.exists(gc_output_file):
            process_gc_content(locations_csv, reference_genome, [size], output_folder)
        else:
            print(f"GC content file for size {size}bp already exists, skipping.")

    # Step 3: Process each thermodynamic property sequentially (skip already processed files)
    properties = ['dH', 'dS', 'dG']
    for prop in properties:
        for size in sizes:
            thermo_output_file = f"{output_folder}/thermo_properties_{size}bp_{prop}.csv"
            if not os.path.exists(thermo_output_file):
                process_thermo_property(locations_csv, reference_genome, [size], nn_thermo_csv, output_folder, prop)
            else:
                print(f"Thermo property file for {prop} at size {size}bp already exists, skipping.")
                
if __name__ == "__main__":
    reference_genome = "/XXX/XXX/refgenome.fna" # Change as needed
    num_locations = 10000 # Change as needed
    nn_thermo_csv = "/XXX/XXX/therdym_NNbp_santalucia1998.csv" # Change as needed
    output_folder = "/XXX/XXX/random_location_thermodynamics" # Change as needed
    
    sizes = [10, 50, 100, 500, 1000, 5000, 10000, 50000, 100000] # Adjust as needed
    
    main_process(reference_genome, num_locations, sizes, nn_thermo_csv, output_folder)
