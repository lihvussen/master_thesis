import json
import matplotlib.pyplot as plt

# Set the model name as a variable for reusability
model = "llama2_7b"

# Number of iterations and number of examples used per run
number_of_iterations = 5
number_of_examples = 100

# Different embedding pooling strategies
options = ["all", "last", "weighted"]

# Loop through each embedding option
for option in options:

    # Loop through each iteration
    for iteration in range(number_of_iterations):

        # Load cosine similarity data for all language pairs and layers
        with open(
            f"{model}/language_pairs_layers_similarities/{option}/"
            f"{model}_language_pairs_layers_similarities_{iteration+1}_iteration_{number_of_examples}_examples_{option}.json", "r"
        ) as f:
            language_pairs_layers = json.load(f)

        averages_per_layer = {}

        # Compute average cosine similarity per layer for each language pair
        for language_pair, layers in language_pairs_layers.items():
            layer_averages = {}
            for layer, similarities in layers.items():
                layer_averages[layer] = round(sum(similarities) / len(similarities), 4)
            averages_per_layer[language_pair] = layer_averages

        # Print averages for inspection
        print(averages_per_layer)

        # Save the per-language-pair averages per layer
        with open(
            f"{model}/plots/{option}/data/"
            f"{model}_averages_per_layer_for_each_language_pair_{iteration+1}_iteration_{number_of_examples}_examples_{option}.json", "w"
        ) as f:
            json.dump(averages_per_layer, f, indent=4)

        # Plot cosine similarities per layer for each language pair
        plt.figure(figsize=(12, 15))  # Larger height for clarity

        for language_pair, values in averages_per_layer.items():
            x = list(map(int, values.keys()))
            y = list(values.values())
            plt.plot(x, y, marker='o', label=language_pair)

        # Plot styling
        plt.ylim(-1, 1.2)
        plt.xlabel('Layer')
        plt.ylabel('Cosine Similarity')
        plt.title(f'Cosine Similarity Across Layers for {len(language_pairs_layers.keys())} Language Pairs ({model} - {option} embeddings)')
        plt.grid(alpha=0.3)

        # Place legend outside plot
        plt.legend(loc='center left', bbox_to_anchor=(1, 0.5), fontsize='small')
        plt.tight_layout()

        # Save the plot
        plt.savefig(
            f"{model}/plots/{option}/"
            f"{model}_average_cosine_similarity_{len(language_pairs_layers.keys())}_language_pairs_"
            f"{iteration+1}_iteration_{number_of_examples}_examples_{option}.png", 
            bbox_inches='tight'
        )
        plt.show()

        # Reorganize data: sort language pairs by similarity for each layer
        all_layers = list(averages_per_layer["Croatian-English"].keys())  # assumes this pair is always present

        layers_dict = {}
        for layer in all_layers:
            language_pairs_dict = {
                language_pair: averages_per_layer[language_pair][layer]
                for language_pair in averages_per_layer
            }
            # Sort language pairs by similarity descending
            layers_dict[layer] = dict(sorted(language_pairs_dict.items(), key=lambda item: item[1], reverse=True))

        # Print sorted layer-wise dictionary
        print(layers_dict)

        # Save the sorted per-layer results
        with open(
            f"{model}/plots/{option}/data/"
            f"{model}_averages_per_language_pair_for_each_layer_{iteration+1}_iteration_{number_of_examples}_examples_{option}.json", "w"
        ) as f:
            json.dump(layers_dict, f, indent=4)
