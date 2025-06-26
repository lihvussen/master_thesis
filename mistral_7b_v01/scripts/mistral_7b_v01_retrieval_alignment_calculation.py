import json
from itertools import combinations
import statistics
import time

# Number of examples per iteration
nr_examples = 100
# Number of iterations to run
iterations = 5

start = time.time()

# Model name used in file paths
model = "mistral_7b_v01"

# List of languages to analyze, grouped loosely by language families
languages_to_analyze = [
    "Swahili", "Yoruba",  # African Languages
    "Hindi", "Urdu",
    "German", "English",  # Germanic Languages
    "Italian", "Spanish",  # Romance Languages
    "Polish", "Croatian", "Russian",  # Slavic Languages
    "Hungarian",  # Other European Languages
]

# Generate all unique language pairs joined by '_'
language_pairs = ["_".join(pair) for pair in combinations(languages_to_analyze, 2)]

# Embedding pooling options to process
options = ["all", "last", "weighted"]

# Loop over each option (embedding pooling method)
for option in options:
    # Loop over each iteration
    for iteration in range(iterations):

        # Load the retrieval cosine similarity data from JSON for current option and iteration
        with open(
            f"{model}/retrieval/{option}/"
            f"cosine_similarities_retrieval_{model}_{iteration+1}_iteration_{nr_examples}_examples_{option}.json",
            "r"
        ) as f:
            layers = json.load(f)

        # Initialize dictionary to store aggregated results per layer
        layers_dict = {}

        # Iterate through each layer in the loaded data
        for layer, example_languages in layers.items():

            # Initialize a dictionary for this layer with all language pairs as keys and empty lists to collect rank positions
            layer_dict = {language_pair: [] for language_pair in language_pairs}

            # Iterate through the example languages in this layer
            for example_language, example_language_results in example_languages.items():

                # Iterate through each retrieved example and its similarity score
                for i, (example_language_result, similarity) in enumerate(example_language_results.items()):

                    # Check if the language code suffix (after '_') matches between example and retrieved example
                    if example_language.split('_')[1] == example_language_result.split('_')[1]:

                        # Extract language prefixes (before '_') for example and retrieved example
                        ex_lang = example_language.split('_')
                        ex_lang_res = example_language_result.split('_')

                        # Attempt to append the rank (i+1) to the correct language pair key in consistent order
                        try:
                            layer_dict[f"{ex_lang[0]}_{ex_lang_res[0]}"].append(i + 1)
                        except KeyError:
                            # If that pair key not found, try reversed order
                            layer_dict[f"{ex_lang_res[0]}_{ex_lang[0]}"].append(i + 1)

            # For each language pair, compute the average rank and convert it to a score between 0 and 1
            new_layer_dict = {}
            for language_pair, places in layer_dict.items():
                # Average rank normalized by max rank (1200), then inverted and rounded to 3 decimals
                new_layer_dict[language_pair] = round(1 - (statistics.mean(places) / 1200), 3)

            # Store the computed scores per language pair for this layer
            layers_dict[layer] = new_layer_dict

        # Save the aggregated retrieval alignment results for this iteration and option to a JSON file
        with open(
            f"{model}/retrieval_alignment/{option}_general_results/"
            f"{model}_retrieval_results_general_{option}_iteration_{iteration+1}_examples_{nr_examples}.json", "w"
        ) as f:
            json.dump(layers_dict, f, indent=4, ensure_ascii=False)

        print(f"Iteration {iteration+1} for option {option} finished.")

    end = time.time()
    print(f"Option {option} finished in {round((end - start) / 60, 2)} minutes.")
