import os
import json
import matplotlib.pyplot as plt
import numpy as np

# List of models to analyze
models = [
    "deepseek_7b_base", "gemma_7b", "llama2_7b", 
    "llama3_1_8b", "aya_8b", "bloom_3b", "mistral_7b_v01"
]

# Different types of embedding pooling strategies or score extraction methods
score_types = ["all", "last", "weighted"]

# Different alignment measurement methods to process
measurements = ["mexa", "cosine", "retrieval_language_pair_alignment"]

# Base directory where data is stored
base_path = "/work/mlichwa"

# Directory to save aggregated plots and data
save_dir = "general_results/alignment_aggregated"
os.makedirs(save_dir, exist_ok=True)  # Ensure save directory exists

for measurement in measurements:
    # Create a figure with one subplot per score_type arranged vertically,
    # sharing the same x-axis (layers)
    fig, axs = plt.subplots(len(score_types), 1, figsize=(20, 6 * len(score_types)), sharex=True)

    # If only one score_type, axs is not a list, so convert to list for uniform handling
    if len(score_types) == 1:
        axs = [axs]

    for idx, score_type in enumerate(score_types):
        all_model_data = {}  # To store average scores per layer for all models

        for model in models:
            # Construct file path based on measurement type and score_type
            if measurement == "retrieval_language_pair_alignment":
                file_path = f"{base_path}/{model}/plots_{measurement}/{score_type}/data/{model}_averages_per_layer_for_each_language_pair_1_iteration_100_examples_{score_type}.json"
            elif measurement == "cosine":
                file_path = f"{base_path}/{model}/plots/{score_type}/data/{model}_averages_per_layer_for_each_language_pair_1_iteration_100_examples_{score_type}.json"
            else:
                file_path = f"{base_path}/{model}/plots_{measurement}/{score_type}/data/{model}_averages_per_layer_for_each_language_pair_1_iteration_100_examples_{score_type}.json"

            # If data file is missing, warn and skip this model
            if not os.path.exists(file_path):
                print(f"[!] Missing: {file_path}")
                continue

            # Load JSON data containing alignment scores per language pair and layer
            with open(file_path, "r") as f:
                data = json.load(f)

            layer_scores = {}
            # Aggregate scores by layer across all language pairs for this model
            for lang_pair in data:
                for layer_str, score in data[lang_pair].items():
                    layer = int(layer_str)
                    layer_scores.setdefault(layer, []).append(score)

            # Compute average score per layer, rounded to 3 decimals
            avg_per_layer = {layer: round(np.mean(scores), 3) for layer, scores in sorted(layer_scores.items())}
            all_model_data[model] = avg_per_layer

        # Collect all layers present in any model to unify x-axis range
        all_layers = set()
        for scores in all_model_data.values():
            all_layers.update(scores.keys())
        all_layers = sorted(all_layers)

        ax = axs[idx]
        # Plot average alignment score per layer for each model
        for model, scores in all_model_data.items():
            layers_to_plot = []
            values_to_plot = []
            for layer in all_layers:
                if layer in scores:
                    layers_to_plot.append(layer)
                    values_to_plot.append(scores[layer])

            ax.plot(layers_to_plot, values_to_plot, marker='o', label=model)
            
        # Set y-axis limits and labels
        ax.set_ylim(0, 1)
        ax.set_ylabel("Avg Alignment Score")

        # Set title per score type with descriptive text
        if score_type == "all":
            ax.set_title(f"All models average results for non-weighted average embedding representation")
        elif score_type == "last":
            ax.set_title(f"All models average results for last token embedding representation")
        else:
            ax.set_title(f"All models average results for weighted average embedding representation")

        ax.grid(True)
        ax.legend(loc='upper left')

    # Label x-axis only on bottom plot (last subplot)
    axs[-1].set_xlabel("Layer")

    # Add a main title to the figure summarizing the measurement type
    fig.suptitle(f"Alignment per Layer ({measurement.upper()})", fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.97])

    # Save the combined plot to file
    plot_filename = f"alignment_per_layer_{measurement}_all_score_types.png"
    plot_path = os.path.join(save_dir, plot_filename)
    plt.savefig(plot_path)
    plt.close()
    print(f"[✓] Saved combined plot: {plot_path}")

    # Also save the aggregated data for each score_type separately in JSON files
    for score_type in score_types:
        all_model_data = {}
        # Reload data to ensure fresh reads (could reuse from above if optimized)
        for model in models:
            if measurement == "retrieval_language_pair_alignment":
                file_path = f"{base_path}/{model}/plots_{measurement}/{score_type}/data/{model}_averages_per_layer_for_each_language_pair_1_iteration_100_examples_{score_type}.json"
            elif measurement == "cosine":
                file_path = f"{base_path}/{model}/plots/{score_type}/data/{model}_averages_per_layer_for_each_language_pair_1_iteration_100_examples_{score_type}.json"
            else:
                file_path = f"{base_path}/{model}/plots_{measurement}/{score_type}/data/{model}_averages_per_layer_for_each_language_pair_1_iteration_100_examples_{score_type}.json"

            if not os.path.exists(file_path):
                continue

            with open(file_path, "r") as f:
                data = json.load(f)

            layer_scores = {}
            for lang_pair in data:
                for layer_str, score in data[lang_pair].items():
                    layer = int(layer_str)
                    layer_scores.setdefault(layer, []).append(score)

            avg_per_layer = {layer: round(np.mean(scores), 3) for layer, scores in sorted(layer_scores.items())}
            all_model_data[model] = avg_per_layer

        # Save the aggregated average scores per layer to JSON file for this measurement and score_type
        json_filename = f"alignment_per_layer_{measurement}_{score_type}.json"
        json_path = os.path.join(save_dir, json_filename)
        with open(json_path, "w") as f:
            json.dump(all_model_data, f, indent=2)
        print(f"[✓] Saved data: {json_path}")
