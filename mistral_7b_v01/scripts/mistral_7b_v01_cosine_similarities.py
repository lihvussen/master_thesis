import json
import numpy as np
import torch
from itertools import combinations
import os
import time

# Parameters
number_of_iterations = 5
number_of_examples = 100

# Define model name for reuse in file paths
model = "mistral_7b_v01"

# Select device
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Pooling options used for representation extraction
options = {"all": "averages", "last": "last", "weighted": "weighted"}

# Iterate through pooling options
for option_1, option_2 in options.items():

    # Perform multiple runs (e.g., for statistical robustness)
    for iteration in range(number_of_iterations):

        start_iter_time = time.time()

        # File name for hidden states input
        file_name = f"flores_101_hidden_layers_{model}_{iteration+1}_iteration_{number_of_examples}_examples_{option_1}"

        # Skip if result file already exists
        output_path = f"{model}/cosine_similarities/{option_1}/cosine_similarities_{model}_{iteration+1}_iteration_{number_of_examples}_examples_{option_1}.json"
        if os.path.exists(output_path):
            print(f"Calculations for iteration {iteration+1} already {option_1} done")
            continue

        print(f"Iteration {iteration+1} for option {option_1} starts.")

        # Load hidden state tensors from file
        with open(f"{model}/hidden_states_{option_2}/{file_name}.json", "r") as f:
            averaged_tensors = json.load(f)

        print("Starting cosine similarity calculations")

        all_similarities = {}

        # Iterate over text examples
        for text_nr, languages_results in averaged_tensors.items():

            languages = list(languages_results.keys())
            layers = list(languages_results[languages[0]].keys())

            layer_similarities = {}

            for layer in layers:

                language_pair_similarities = {}

                # Compute pairwise cosine similarities
                for language_pair in combinations(languages, 2):
                    try:
                        # Handle case where data is stored as nested list (e.g., [ [vec] ])
                        vector_1 = torch.tensor(languages_results[language_pair[0]][layer][0])
                        vector_2 = torch.tensor(languages_results[language_pair[1]][layer][0])
                    except:
                        # Fallback if stored as single list (e.g., [vec])
                        vector_1 = torch.tensor(languages_results[language_pair[0]][layer])
                        vector_2 = torch.tensor(languages_results[language_pair[1]][layer])

                    # Compute cosine similarity
                    sim = torch.nn.functional.cosine_similarity(vector_1.unsqueeze(0), vector_2.unsqueeze(0)).item()
                    language_pair_similarities[f"{language_pair[0]}-{language_pair[1]}"] = sim

                layer_similarities[layer] = language_pair_similarities

            all_similarities[text_nr] = layer_similarities

        # Save similarity results to JSON
        os.makedirs(f"{model}/cosine_similarities/{option_1}", exist_ok=True)
        with open(output_path, "w") as f_2:
            json.dump(all_similarities, f_2, indent=4)

        end_iter_time = time.time()
        print(f"Iteration {iteration+1} for {model} {option_1} done in {round((end_iter_time - start_iter_time)/60, 2)} minutes")
