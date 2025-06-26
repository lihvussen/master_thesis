from datasets import load_dataset
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import json
import regex as re
from comet import download_model, load_from_checkpoint
import random
import time
import os

# Disable parallelism warning from tokenizers
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Set device (GPU if available)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def cut_at_first_occurrence(text):
    """
    Cuts the input text at the earliest occurrence of '.' or '\n'.
    Returns the substring from the start up to that position.
    If none found, returns the entire text.
    """
    dot_index = text.find('.')
    newline_index = text.find('\n')

    # Filter out -1 (not found) and find minimum index
    min_index = min(filter(lambda x: x != -1, [dot_index, newline_index]), default=None)

    return text[:min_index] if min_index is not None else text

# Set seed for reproducibility
random.seed(43)

# Model name variable for flexible path usage
model_name_all = "gemma_7b"

# Download and load the COMET evaluation model
eval_model_path = download_model("Unbabel/wmt22-comet-da")  # Use appropriate model checkpoint
eval_model = load_from_checkpoint(eval_model_path)

def evaluate_translation(original, translated, reference):
    """
    Uses the COMET model to evaluate a translation against
    the original source and reference translation.
    Returns a score (higher is better).
    """
    data = [{
        "src": original, 
        "mt": translated, 
        "ref": reference
    }]
    scores = eval_model.predict(data)
    return scores[0]

# Load translation results JSON
with open(f"{model_name_all}/mexa/task_results/{model_name_all}_translation_results.json", "r", encoding="utf-8") as f:
    translations = json.load(f)

# Load Flores language data JSON for source and reference sentences
with open('flores_101.json', "r", encoding="utf-8") as f:
    languages_flores = json.load(f)

eval_dict = {}
start = time.time()

# Iterate over examples and their language pair translations
for index, (example, language_pairs) in enumerate(translations.items()):
    example_dict = {}
    for language_pair, translation in language_pairs.items():
        try:
            # Remove leading newline if present
            if translation and translation[0] == "\n":
                new_translation = translation[1:]
            else:
                new_translation = translation
            # Clean translation by cutting at first '.' or newline
            clean_translation = cut_at_first_occurrence(new_translation)
        except:
            # Fallback if translation processing fails
            new_translation = "No text"
            clean_translation = cut_at_first_occurrence(new_translation)

        # Extract source and reference sentences from Flores data
        source_lang, target_lang = language_pair.split("-")
        source_sentence = languages_flores[example][source_lang].split('.')[0]
        reference_sentence = languages_flores[example][target_lang].split('.')[0]

        # Evaluate translation and store score
        example_dict[language_pair] = evaluate_translation(
            source_sentence,
            clean_translation,
            reference_sentence
        )
    eval_dict[example] = example_dict

    # Save partial results every 10 examples and print progress
    if (index + 1) % 10 == 0:
        with open(f"{model_name_all}/mexa/task_results/{model_name_all}_translation_evaluation.json", "w", encoding="utf-8") as f:
            json.dump(eval_dict, f, ensure_ascii=False, indent=4)
        current_time = time.time()
        print(f"{index+1} examples done in {round((current_time - start) / 60, 2)} minutes")
