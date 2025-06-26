import os
from PIL import Image

# === USER INPUTS ===
# List of model names whose plots we want to merge
model_names = [
    "aya_8b", "bloom_3b", "deepseek_7b_base", "gemma_7b",
    "llama2_7b", "llama3_1_8b", "mistral_7b_v01"
]

# Different calculation types (averaging options) used in plot filenames
calculation_types = ["all", "last", "weighted"]


def merge_and_save(model_names, calculation_types, base_dir, output_dir, variant="1_iteration_100", suffix=""):
    """
    For each model, loads images corresponding to different calculation types,
    merges them horizontally into a single row image, and saves the combined image.

    Parameters:
        model_names (list): List of model names
        calculation_types (list): List of calculation types (e.g., "all", "last", "weighted")
        base_dir (str): Directory where source images are located (relative path)
        output_dir (str): Directory where merged images will be saved
        variant (str): Variant string to select specific images (part of filename)
        suffix (str): Optional suffix to add to the output filename (e.g., "_appendix")
    """
    images = []

    # Load images per model and calculation type, store in nested list (rows)
    for model_name in model_names:
        row = []
        for calculation in calculation_types:
            # Construct image filepath based on naming convention
            img_path = f"{model_name}/{base_dir}/{calculation}/{model_name}_average_cosine_similarity_66_language_pairs_{variant}_examples_{calculation}.png"
            # Open and append image to current row
            row.append(Image.open(img_path))
        images.append(row)

    # Ensure the output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # For each model, merge images horizontally and save the combined image
    for model_idx, model_name in enumerate(model_names):
        row_imgs = images[model_idx]

        # Get width and height of one image (assuming all images have the same size)
        img_width, img_height = row_imgs[0].size

        spacing = 10  # spacing between images in pixels

        # Calculate total width for merged row: 3 images + 2 gaps of spacing
        row_img_width = len(calculation_types) * img_width + (len(calculation_types) - 1) * spacing
        row_img_height = img_height

        # Create a new blank white image for the merged row
        combined_row = Image.new("RGB", (row_img_width, row_img_height), color="white")

        # Paste each image side by side with spacing
        for col_idx, img in enumerate(row_imgs):
            x = col_idx * (img_width + spacing)
            combined_row.paste(img, (x, 0))

        # Construct output filepath and save merged image
        output_path = os.path.join(output_dir, f"{model_name}{suffix}.png")
        combined_row.save(output_path)


# === COSINE PLOTS ===
cosine_dir = "plots"  # source folder for cosine similarity plots
cosine_out = "general_results/cosine/plots_merged_rows"  # output folder for merged cosine plots
# Merge plots for first and second variant, saving also an appendix version
merge_and_save(model_names, calculation_types, cosine_dir, cosine_out, "1_iteration_100")

# === RETRIEVAL PLOTS ===
retrieval_dir = "plots_retrieval_language_pair_alignment"  # source folder for retrieval plots
retrieval_out = "general_results/retrieval/plots_merged_rows"  # output folder for merged retrieval plots
merge_and_save(model_names, calculation_types, retrieval_dir, retrieval_out, "1_iteration_100")

# === MEXA PLOTS ===
mexa_dir = "plots_mexa"  # source folder for MEXA plots
mexa_out = "general_results/mexa/plots_merged_rows"  # output folder for merged MEXA plots
merge_and_save(model_names, calculation_types, mexa_dir, mexa_out, "1_iteration_100")

print("All plots saved (including appendix versions).")
