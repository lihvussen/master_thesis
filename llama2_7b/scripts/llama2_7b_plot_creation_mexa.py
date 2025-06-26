import json
import matplotlib.pyplot as plt

# Define number of iterations and examples used per evaluation
number_of_iterations = 5
number_of_examples = 100

# Embedding strategies to evaluate: non-weighted average ("all"), last token ("last"), and weighted average ("weighted")
options = ["all", "last", "weighted"]

# Short model name used for consistent file path generation
model_name_short = "llama2_7b"

for option in options:
    for iteration in range(number_of_iterations):
        # Load layer-wise results for one iteration of one embedding type
        input_path = f"{model_name_short}/mexa/{option}_general_results/{model_name_short}_mexa_results_general_{option}_iteration_{iteration+1}_examples_{number_of_examples}.json"

        with open(input_path, "r") as f:
            layers_dict = json.load(f)

        # Reformat from layer-wise to language-pair-wise for plotting convenience
        averages_per_layer = {}
        for layer, language_pairs in layers_dict.items():
            for language_pair, avg_value in language_pairs.items():
                if language_pair not in averages_per_layer:
                    averages_per_layer[language_pair] = {}
                averages_per_layer[language_pair][layer] = round(avg_value, 4)

        # Save reformatted data for future analysis or inspection
        output_data_path = (
            f"{model_name_short}/plots_mexa/{option}/data/"
            f"{model_name_short}_averages_per_layer_for_each_language_pair_{iteration+1}_iteration_{number_of_examples}_examples_{option}.json"
        )
        with open(output_data_path, "w") as f:
            json.dump(averages_per_layer, f, indent=4)

        # Begin plotting MEXA scores across layers for each language pair
        plt.figure(figsize=(12, 15))  # Tall figure for many language pair lines

        for language_pair, values in averages_per_layer.items():
            x = list(map(int, values.keys()))  # Layer numbers as x-axis
            y = list(values.values())          # Cosine similarity scores as y-axis
            plt.plot(x, y, marker='o', label=language_pair)

        plt.ylim(0, 1.2)
        plt.xlabel('Layer')
        plt.ylabel('MEXA score')
        plt.title(
            f'MEXA Score Across Layers for {len(averages_per_layer)} Language Pairs '
            f'({model_name_short.upper()} using {option} embeddings)'
        )
        plt.grid(alpha=0.3)
        plt.legend(loc='center left', bbox_to_anchor=(1, 0.5), fontsize='small')
        plt.tight_layout()

        # Save plot to disk
        output_plot_path = (
            f"{model_name_short}/plots_mexa/{option}/"
            f"{model_name_short}_average_cosine_similarity_{len(averages_per_layer)}_language_pairs_"
            f"{iteration+1}_iteration_{number_of_examples}_examples_{option}.png"
        )
        plt.savefig(output_plot_path, bbox_inches='tight')
        plt.show()

        # Collect list of layers
        all_layers = list(layers_dict.keys())

        # Sort language pairs for each layer by descending score (for heatmap prep, ranking, etc.)
        averages_sorted_per_layer = {}
        for layer in all_layers:
            language_pairs_dict = {
                language_pair: averages_per_layer[language_pair][layer]
                for language_pair in averages_per_layer
            }
            averages_sorted_per_layer[layer] = dict(
                sorted(language_pairs_dict.items(), key=lambda item: item[1], reverse=True)
            )

        # Save sorted-per-layer version to disk
        output_sorted_path = (
            f"{model_name_short}/plots_mexa/{option}/data/"
            f"{model_name_short}_averages_per_language_pair_for_each_layer_{iteration+1}_iteration_"
            f"{number_of_examples}_examples_{option}.json"
        )
        with open(output_sorted_path, "w") as f:
            json.dump(averages_sorted_per_layer, f, indent=4)
