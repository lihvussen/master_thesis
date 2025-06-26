import matplotlib.pyplot as plt
import json
import statistics
import numpy as np
import os
from PIL import Image  # For merging images

# List of language models to analyze
models = ["aya_8b", "bloom_3b", "deepseek_7b_base", "gemma_7b", "llama2_7b", "llama3_1_8b", "mistral_7b_v01"]

# Evaluation tasks for which results exist
tasks = ["belebele", "mmlu", "translation"]

# Pooling averaging options for embeddings
averaging_options = ["all", "last", "weighted"]

# Different alignment extraction methods to compare
alignment_extractions = ["mexa", "cosine", "retrieval"]

# Colors for plotting max and mean pooling bars
colors = {
    "max_pooling": "#4C72B0",
    "mean_pooling": "#DD8452",
}

# Loop over each alignment extraction method
for alignment_extraction in alignment_extractions:
    # Loop over each evaluation task
    for task in tasks:
        # Construct the input path to load Pearson correlation results JSON file
        input_path = f"/work/mlichwa/general_results/{alignment_extraction}/{task}/pearson_correlation_all_models_{alignment_extraction}.json"
        
        # Load the results JSON (structured by model -> averaging option -> iteration -> pooling -> correlation score)
        with open(input_path, "r") as f:
            results = json.load(f)

        # Dictionary to hold averaged correlation scores by averaging option for each model
        averaging_options_dict = {}

        # Loop over each averaging option ('all', 'last', 'weighted')
        for avg_option in averaging_options:
            averaged = {}

            # Loop over each model's results for this averaging option
            for model, options in results.items():
                if avg_option in options:
                    max_scores = []  # Collect all max_pooling scores
                    mean_scores = []  # Collect all mean_pooling scores

                    # Each iteration contains pooling scores, accumulate them across iterations
                    for pooling_scores in options[avg_option].values():
                        for pooling, score in pooling_scores.items():
                            if pooling == "max_pooling":
                                max_scores.append(score)
                            elif pooling == "mean_pooling":
                                mean_scores.append(score)

                    # Compute mean scores across iterations, round to 2 decimals
                    averaged[model] = {
                        "max_pooling": round(statistics.mean(max_scores), 2) if max_scores else 0,
                        "mean_pooling": round(statistics.mean(mean_scores), 2) if mean_scores else 0,
                    }

            averaging_options_dict[avg_option] = averaged

        # Set up subplots for each averaging option
        fig, axes = plt.subplots(len(averaging_options), 1, figsize=(10, 12), sharex=True)
        bar_width = 0.35  # Width of bars in bar chart
        x = np.arange(len(models))  # X-axis positions for models

        # Plot bar charts for each averaging option on a separate subplot
        for i, avg_option in enumerate(averaging_options):
            ax = axes[i]
            avg_data = averaging_options_dict[avg_option]

            # Extract max and mean pooling values in model order
            max_vals = [avg_data.get(model, {}).get("max_pooling", 0) for model in models]
            mean_vals = [avg_data.get(model, {}).get("mean_pooling", 0) for model in models]

            # Plot bars side by side for max and mean pooling
            ax.bar(x - bar_width / 2, max_vals, bar_width, label="Max Pooling", color=colors["max_pooling"])
            ax.bar(x + bar_width / 2, mean_vals, bar_width, label="Mean Pooling", color=colors["mean_pooling"])

            # Set y-label and limits
            ax.set_ylabel(avg_option.capitalize())
            ax.set_ylim(0, 1)

            # Label x-ticks with model names, rotate for readability
            ax.set_xticks(x)
            ax.set_xticklabels(models, rotation=45, ha="right")

            # Add grid lines and legend
            ax.grid(axis="y", linestyle="--", alpha=0.7)
            ax.legend()

        # Set overall title and x-label
        fig.suptitle(f'Pooling Comparison Across Models - {task} ({alignment_extraction})')
        plt.xlabel("Language Models")

        # Adjust layout to prevent overlap
        plt.tight_layout()

        # Create output directory if it doesn't exist
        output_dir = f"/work/mlichwa/general_results/{alignment_extraction}/{task}"
        os.makedirs(output_dir, exist_ok=True)

        # Save the plot to a PNG file with high resolution
        output_file = f"{output_dir}/pooling_comparison_plot_{alignment_extraction}.png"
        plt.savefig(output_file, bbox_inches="tight", dpi=300)

        # Close the figure to free memory
        plt.close()

        print(f"Saved plot to {output_file}")

# After plotting all, merge the plots horizontally for each task
for task in tasks:
    images = []

    # Collect all alignment extraction plots for the current task
    for alignment_extraction in alignment_extractions:
        plot_path = f"/work/mlichwa/general_results/{alignment_extraction}/{task}/pooling_comparison_plot_{alignment_extraction}.png"
        if os.path.exists(plot_path):
            images.append(Image.open(plot_path))
        else:
            print(f"Warning: {plot_path} does not exist and will be skipped.")

    if images:
        # Find minimum height among images to scale all to same height
        min_height = min(img.height for img in images)

        # Resize images proportionally to have the same height
        resized_images = [
            img.resize((int(img.width * min_height / img.height), min_height)) for img in images
        ]

        # Calculate total width of merged image by summing widths
        total_width = sum(img.width for img in resized_images)

        # Create new blank image with total width and min height
        merged_image = Image.new('RGB', (total_width, min_height))

        # Paste images side by side
        x_offset = 0
        for img in resized_images:
            merged_image.paste(img, (x_offset, 0))
            x_offset += img.width

        # Save the merged image to disk
        merged_output_path = f"/work/mlichwa/general_results/{task}/merged_pooling_comparison_{task}.png"
        merged_image.save(merged_output_path)
        print(f"Merged image saved to {merged_output_path}")
