import time
import os
import json
import numpy as np
from collections import defaultdict
import torch

start_time = time.time()

def transform_nested_dict(data):
    """
    Transforms a nested dictionary from
    {main_key: {first_nested_key: {second_nested_key: value_list}}}
    to
    {second_nested_key: {first_nested_key_main_key: value_list}}
    """
    new_dict = defaultdict(lambda: defaultdict(dict))
    
    for main_key, first_level in data.items():
        for first_nested_key, second_level in first_level.items():
            for second_nested_key, value_list in second_level.items():
                new_dict[second_nested_key][f"{first_nested_key}_{main_key}"] = value_list
                
    # Convert defaultdict to regular dict before returning
    return {k: dict(v) for k, v in new_dict.items()}

# Determine if CUDA is available and set device accordingly
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Define model name for flexible reuse
model = "bloom_3b"

number_of_iterations = 5
number_of_examples = 100
number_of_languages = 12

# Mapping of options from retrieval folder to hidden_states folder naming
options = {"all": "averages", "last": "last", "weighted": "weighted"}

for option_1, option_2 in options.items():

    for iteration in range(number_of_iterations):

        # Check if the output file already exists to avoid redundant computation
        output_path = f"{model}/retrieval/{option_1}/cosine_similarities_retrieval_{model}_{iteration+1}_iteration_{number_of_examples}_examples_{option_1}.json"
        if os.path.exists(output_path):
            print(f"Iteration {iteration+1} option {option_1} already done")
            continue

        start_time_inter = time.time()
        
        # Load hidden states for the current iteration and option
        input_path = f"{model}/hidden_states_{option_2}/flores_101_hidden_layers_{model}_{iteration+1}_iteration_{number_of_examples}_examples_{option_1}.json"
        with open(input_path, "r") as f:
            iteration_results = json.load(f)
        
        print(f"Iteration {iteration+1} for option {option_1} starts")

        # Transform nested dictionary for easier processing
        new_data = transform_nested_dict(iteration_results)
        del iteration_results

        print(f"Start of calculations for iteration {iteration+1} option {option_1}")

        layer_dict = {}
        for layer, languages_examples in new_data.items():

            keys = list(languages_examples.keys())
            try:
                # Extract vectors assuming values are lists inside a list: [vector]
                vectors = np.array([languages_examples[k][0] for k in keys], dtype=np.float32)
                tensor_vectors = torch.tensor(vectors, device=device)

                dot_products = torch.matmul(tensor_vectors, tensor_vectors.t())

                norms = torch.norm(tensor_vectors, p=2, dim=1, keepdim=True)  # shape: (num_items, 1)
            except:
                # If previous assumption fails, try directly using the value
                vectors = np.array([languages_examples[k] for k in keys], dtype=np.float32)
                tensor_vectors = torch.tensor(vectors, device=device)

                dot_products = torch.matmul(tensor_vectors, tensor_vectors.t())

                norms = torch.norm(tensor_vectors, p=2, dim=1, keepdim=True)

            # Compute cosine similarity matrix
            norm_matrix = torch.matmul(norms, norms.t())
            epsilon = 1e-8  # Small constant to avoid division by zero
            cosine_similarity_matrix = dot_products / (norm_matrix + epsilon)

            # Exclude self-similarity by setting diagonal to -1
            cosine_similarity_matrix.fill_diagonal_(-1)

            # Number of top similar vectors to keep (excluding self)
            top_k = number_of_examples * number_of_languages - 1
            top_values, top_indices = torch.topk(cosine_similarity_matrix, k=top_k, dim=1)

            # Move tensors to CPU for JSON serialization
            top_values = top_values.cpu().numpy()
            top_indices = top_indices.cpu().numpy()

            # Build dictionary of top similarities for each key
            result = {}
            for i, key in enumerate(keys):
                similar_dict = {keys[top_indices[i][j]]: float(top_values[i][j]) for j in range(top_k)}
                result[key] = similar_dict

            layer_dict[layer] = result
            print(f"Layer {layer} for iteration {iteration+1} done.")

        # Save the cosine similarity retrieval results per iteration and option
        with open(output_path, "w") as f:
            json.dump(layer_dict, f, indent=4)

        intermediate_time_end = time.time()
        print(f"Iteration {iteration+1} option {option_1} finished in {round((intermediate_time_end - start_time_inter)/60, 2)} minutes")

    end_time = time.time()
    print(f"Time needed after option {option_1} iteration {iteration+1}: {round((end_time - start_time)/60, 2)} minutes")
