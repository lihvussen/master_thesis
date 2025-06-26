import json

# Define model name as a variable for easy reuse and modification
model = "mistral_7b_v01"

number_of_iterations = 5
number_of_examples = 100

options = ["all", "last", "weighted"]

def reverse_language_pair(pair: str) -> str:
    """Helper function to reverse a language pair string 'lang1-lang2' to 'lang2-lang1'."""
    parts = pair.split("-")
    if len(parts) == 2:
        return f"{parts[1]}-{parts[0]}"
    return pair

for option in options:
    for iteration in range(number_of_iterations):
        # Load cosine similarity data for the current option and iteration
        input_path = f"{model}/cosine_similarities/{option}/cosine_similarities_{model}_{iteration+1}_iteration_{number_of_examples}_examples_{option}.json"
        with open(input_path, "r") as f:
            cosine_similarities = json.load(f)

        # Get a sample key to initialize dictionaries for layers and language pairs
        first_key = list(cosine_similarities.keys())[0]

        # Initialize dictionaries to store aggregated similarities per layer and language pair
        all_layers = {layer: [] for layer in cosine_similarities[first_key].keys()}
        all_language_pairs = {lp: [] for lp in cosine_similarities[first_key]["0"].keys()}
        all_language_pairs_layers = {
            lp: {layer: [] for layer in cosine_similarities[first_key].keys()}
            for lp in cosine_similarities[first_key]["0"].keys()
        }

        # Iterate through all texts, layers, and language pairs to aggregate similarities
        for text_nr, layers in cosine_similarities.items():
            for layer, language_pairs in layers.items():
                for language_pair, similarity in language_pairs.items():
                    # Try appending similarity to the normal language pair key
                    if language_pair in all_language_pairs:
                        all_language_pairs[language_pair].append(similarity)
                        all_language_pairs_layers[language_pair][layer].append(similarity)
                    else:
                        # If key not found, try reversed language pair key
                        reversed_pair = reverse_language_pair(language_pair)
                        if reversed_pair in all_language_pairs:
                            all_language_pairs[reversed_pair].append(similarity)
                            all_language_pairs_layers[reversed_pair][layer].append(similarity)
                        else:
                            # Language pair not found either way; can log or ignore
                            pass

        # Save the aggregated similarities organized by language pair and layer
        output_path = f"{model}/language_pairs_layers_similarities/{option}/{model}_language_pairs_layers_similarities_{iteration+1}_iteration_{number_of_examples}_examples_{option}.json"
        with open(output_path, "w") as f:
            json.dump(all_language_pairs_layers, f, indent=4)
