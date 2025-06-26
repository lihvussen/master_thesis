import json
from collections import defaultdict
import time
import os
import torch
import numpy as np
from itertools import combinations
import torch.nn.functional as F

# Configuration
model = "mistral_7b_v01"  # Model name used in paths
number_of_examples = 100
number_of_iterations = 5

# Use GPU if available
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

def compute_mexa_score(embeddings_L1, embeddings_L2):
    """Compute MEXA alignment score for a single layer using cosine similarity."""
    assert embeddings_L1.shape == embeddings_L2.shape, "Embeddings must have the same shape"
    
    # Compute cosine similarity matrix between each pair of examples
    similarity_matrix = F.cosine_similarity(embeddings_L1.unsqueeze(1), embeddings_L2.unsqueeze(0), dim=-1)

    n = similarity_matrix.shape[0]  # Number of examples
    correct_count = 0

    for i in range(n):
        c_ii = similarity_matrix[i, i]  # Diagonal similarity (aligned pair)
        max_row = torch.max(similarity_matrix[i, torch.arange(n) != i])  # Max of other pairs in row
        max_col = torch.max(similarity_matrix[torch.arange(n) != i, i])  # Max of other pairs in column

        # Count if the correct alignment is stronger than any mismatched alignment
        if c_ii > max(max_row, max_col):
            correct_count += 1

    return correct_count / n

def transform_nested_dict(data):
    """
    Transpose the structure from:
    layer -> lang -> example -> embedding
    to:
    layer -> example -> lang -> embedding
    """
    new_dict = defaultdict(lambda: defaultdict(dict))
    for main_key, first_level in data.items():
        for first_nested_key, second_level in first_level.items():
            for second_nested_key, value_list in second_level.items():
                new_dict[second_nested_key][first_nested_key][main_key] = value_list
    return {k: dict(v) for k, v in new_dict.items()}

# Run MEXA scoring for several iterations
for iteration in range(number_of_iterations):
    start_time_iteration = time.time()

    for embedding_type in ["last", "weighted", "all"]:
        # Setup paths dynamically
        hidden_state_dir = {
            "last": f"{model}/hidden_states_last",
            "weighted": f"{model}/hidden_states_weighted",
            "all": f"{model}/hidden_states_averages",
        }[embedding_type]

        result_dir = {
            "last": f"{model}/mexa/last_general_results",
            "weighted": f"{model}/mexa/weighted_general_results",
            "all": f"{model}/mexa/all_general_results",
        }[embedding_type]

        input_path = f"{hidden_state_dir}/flores_101_hidden_layers_{model}_{iteration+1}_iteration_{number_of_examples}_examples_{embedding_type}.json"
        output_path = f"{result_dir}/{model}_mexa_results_general_{embedding_type}_iteration_{iteration+1}_examples_{number_of_examples}.json"

        if os.path.exists(output_path):
            continue  # Skip if already computed

        # Load hidden state embeddings
        with open(input_path, "r") as f:
            current_similarities = json.load(f)

        new_similarities = transform_nested_dict(current_similarities)

        print(f"Calculations for iteration {iteration+1} for {embedding_type} start.")

        layers_dict = {}

        for layer, languages in new_similarities.items():
            languages_dict = {lang: [] for lang in languages}
            layer_scores = {}

            # Aggregate representations for each language
            for language, examples in languages.items():
                for _, embedding in examples.items():
                    languages_dict[language].append(embedding)

            # Compute MEXA scores for each language pair
            for lang1, lang2 in combinations(languages, 2):
                rep1 = torch.from_numpy(np.squeeze(np.array(languages_dict[lang1])))
                rep2 = torch.from_numpy(np.squeeze(np.array(languages_dict[lang2])))

                pair_key = f"{lang1}_{lang2}"
                layer_scores[pair_key] = compute_mexa_score(rep1, rep2)

            layers_dict[layer] = layer_scores
            print(f"Layer {layer} for iteration {iteration+1} for {embedding_type} done.")

        # Save results
        os.makedirs(result_dir, exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(layers_dict, f, indent=4)

        print(f"Calculations for iteration {iteration+1} for {embedding_type} done.")

        del new_similarities  # Free memory

    print(f"Iteration {iteration+1} completed.")
    print(f"Time needed for iteration {iteration+1}: {round((time.time() - start_time_iteration)/60, 2)} minutes")
