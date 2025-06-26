import json
import matplotlib.pyplot as plt

# Number of runs and evaluation samples
number_of_iterations = 5
number_of_examples = 100

# Different embedding strategies
options = ["all", "last", "weighted"]

# Short model identifier for path consistency
model_name_short = "aya_8b"

# Loop over each embedding type
for option in options:
    for iteration in range(number_of_iterations):

        # Load cosine similarity scores per sentence per layer
        input_path = (
            f"{model_name_short}/language_pairs_layers_retrieval/{option}/"
            f"{model_name_short}_language_pairs_layers_retrieval_{iteration+1}_iteration_"
            f"{number_of_examples}_examples_{option}.json"
        )
        with open(input_path, "r") as f:
            language_pairs_layers = json.load(f)

        # Compute average similarity score per layer for each language pair
        averages_per_layer = {}
        for language_pair, layers in language_pairs_layers.items():
            layer_averages = {}
            for layer, similarities in layers.items():
                layer_averages[layer] = round(sum(similarities) / len(similarities), 4)
            averages_per_layer[language_pair] = layer_averages

        # Save per-language-pair per-layer averages
        output_data_path = (
            f"{model_name_short}/plots_retrieval_language_pair_alignment/{option}/data/"
            f"{model_name_short}_averages_per_layer_for_each_language_pair_{iteration+1}_iteration_"
            f"{number_of_examples}_examples_{option}.json"
        )
        with open(output_data_path, "w") as f:
            json.dump(averages_per_layer, f, indent=4)

        # Plot sentence retrieval alignment across layers
        plt.figure(figsize=(12, 15))  # Larger height for many language pairs
        for language_pair, values in averages_per_layer.items():
            x = list(map(int, values.keys()))  # Convert layer indices to int
            y = list(values.values())
            plt.plot(x, y, marker='o', label=language_pair)

        plt.ylim(0, 1.2)
        plt.xlabel('Layer')
        plt.ylabel('Sentence Retrieval Score')
        plt.title(
            f'Sentence Retrieval Score Across Layers for {len(language_pairs_layers)} Language Pairs\n'
            f'{model_name_short.upper()} using {option} embeddings'
        )
        plt.grid(alpha=0.3)
        plt.legend(loc='center left', bbox_to_anchor=(1, 0.5), fontsize='small')
        plt.tight_layout()

        output_plot_path = (
            f"{model_name_short}/plots_retrieval_language_pair_alignment/{option}/"
            f"{model_name_short}_average_cosine_similarity_{len(language_pairs_layers)}_language_pairs_"
            f"{iteration+1}_iteration_{number_of_examples}_examples_{option}.png"
        )
        plt.savefig(output_plot_path, bbox_inches='tight')
        plt.show()

        # Sort and save per-layer view: which language pairs are most aligned at each layer
        all_layers = list(averages_per_layer["English-Croatian"].keys())
        sorted_per_layer = {}

        for layer in all_layers:
            language_scores = {
                lang_pair: averages_per_layer[lang_pair][layer]
                for lang_pair in averages_per_layer
            }
            sorted_per_layer[layer] = dict(
                sorted(language_scores.items(), key=lambda item: item[1], reverse=True)
            )

        output_sorted_path = (
            f"{model_name_short}/plots_retrieval_language_pair_alignment/{option}/data/"
            f"{model_name_short}_averages_per_language_pair_for_each_layer_{iteration+1}_iteration_"
            f"{number_of_examples}_examples_{option}.json"
        )
        with open(output_sorted_path, "w") as f:
            json.dump(sorted_per_layer, f, indent=4)
