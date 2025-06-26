import json
import statistics
import time

# Parameters
iterations = 5
nr_examples = 100
options = ["all", "last", "weighted"]

# Define model name for reuse
model = "llama2_7b"

start = time.time()

# Loop over different embedding strategies
for option in options:
    for iteration in range(iterations):

        # Load cosine similarity results for current iteration and option
        input_path = f"{model}/cosine_similarities/{option}/cosine_similarities_{model}_{iteration+1}_iteration_{nr_examples}_examples_{option}.json"
        with open(input_path, "r") as f:
            cosine_similaritites = json.load(f)

        # Extract layer and language pair info from the first entry
        all_layers = list(cosine_similaritites[list(cosine_similaritites.keys())[0]].keys())
        all_lang_pairs = list(cosine_similaritites[list(cosine_similaritites.keys())[0]][all_layers[0]].keys())

        # Initialize structure to collect similarities per language pair per layer
        layers_dict = {layer_nr: {lang_pair: [] for lang_pair in all_lang_pairs} for layer_nr in all_layers}

        # Aggregate cosine similarities for each language pair in each layer
        for example, layers in cosine_similaritites.items():
            for layer, language_pairs in layers.items():
                for language_pair, similarity in language_pairs.items():
                    layers_dict[layer][language_pair].append(similarity)

        # Compute average similarity per language pair per layer
        new_layers_dict = {}
        for layer, lang_pairs in layers_dict.items():
            new_layers_dict[layer] = {
                lang_pair: round(statistics.mean(similarities), 3)
                for lang_pair, similarities in lang_pairs.items()
            }

        # Save aggregated results to output file
        output_path = f"{model}/cosine_alignment/{option}_general_results/{model}_cosine_results_general_{option}_iteration_{iteration+1}_examples_{nr_examples}.json"
        with open(output_path, "w") as f:
            json.dump(new_layers_dict, f, ensure_ascii=False, indent=4)
