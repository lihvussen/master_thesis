import time
import pickle
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import json
import random
import os
import numpy as np

random.seed(42)

# Set the model name as a variable for reuse in file paths and naming
model = "deepseek_7b_base"

number_of_outputs = 100
number_of_iterations = 5

# Check for GPU availability and set device accordingly
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Load language data from FLORES-101 dataset JSON file
with open('flores_101.json', "r", encoding="utf-8") as f:
    languages_flores = json.load(f)

# Define the set of languages to analyze (subset of FLORES languages)
languages_to_analyze = [
    "Swahili", "Yoruba",  # African Languages
    "Hindi", "Urdu",
    "English", "German",  # Germanic Languages
    "Italian", "Spanish",  # Romance Languages
    "Polish", "Croatian", "Russian",  # Slavic Languages
    "Hungarian",  # Other European Languages
]

def get_all_hidden_states(sentence, model, tokenizer):
    # Tokenize input sentence and send tensors to device (GPU/CPU)
    inputs = tokenizer(sentence, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True)
    # Return hidden states moved back to CPU as numpy arrays for saving
    return [state.cpu().numpy() for state in outputs.hidden_states]

def position_weighted_average(hidden_states):
    # Compute position-weighted average of token embeddings
    T = hidden_states.shape[0]  # Number of tokens in the sequence
    weights = np.arange(1, T + 1) / np.sum(np.arange(1, T + 1))  # Positional weights normalized
    weighted_embedding = np.sum(weights[:, np.newaxis] * hidden_states, axis=0)
    return weighted_embedding

# Initialize tokenizer and model with the access token
access_token = "#####"
model_name = "deepseek-aideepseek-ai/deepseek-llm-7b-base"
tokenizer = AutoTokenizer.from_pretrained(model_name, token=access_token)
model = AutoModelForCausalLM.from_pretrained(model_name, token=access_token, output_hidden_states=True)
model.to(device)  # Move model to the selected device (GPU/CPU)

iteration_results = {}

for iteration in range(number_of_iterations):

    # Randomly sample example indices for this iteration
    random_numbers = random.sample(range(2009), number_of_outputs)

    iteration_name = f"flores_101_hidden_layers_{model}_{iteration+1}_iteration_{number_of_outputs}_examples"

    # Check if this iteration's results already exist; skip if yes
    output_path = f"{model}/hidden_states_averages/{iteration_name}_all.json"
    if os.path.exists(output_path):
        print(f"Iteration {iteration+1} already done.")
        continue

    # Load previous intermediate results if available, else start fresh
    pkl_path = f"{model}/hidden_states/{iteration_name}.pkl"
    try:
        with open(pkl_path, "rb") as f:
            current_output = pickle.load(f)
    except FileNotFoundError:
        current_output = {}

    start_time_iteration = time.time()

    # Process each example in the random sample
    for i, example_nr in enumerate(random_numbers):
        example = str(example_nr)

        # If example data already present, check languages and update if needed
        if example in current_output:
            # Remove any languages not in the target set
            for existing_language in list(current_output[example].keys()):
                if existing_language not in languages_to_analyze:
                    del current_output[example][existing_language]
                    print(f"{existing_language} example deleted from {example_nr}")

            # Add missing languages' hidden states
            for necessary_language in languages_to_analyze:
                if necessary_language not in current_output[example]:
                    hidden_states = get_all_hidden_states(languages_flores[example][necessary_language], model, tokenizer)
                    current_output[example][necessary_language] = hidden_states
                    print(f"{necessary_language} example added for {example_nr}")

            # Periodic checkpoint saving every 10 examples
            if (i + 1) % 10 == 0:
                print(f"{i + 1} examples updated")
                with open(pkl_path, "wb") as f:
                    pickle.dump(current_output, f)
                print(list(current_output.keys()))
            continue  # Skip to next example after updating existing data

        # Process new example from scratch
        start_time_example = time.time()
        dict_example = {}
        translations = languages_flores[example]

        # Compute hidden states for each relevant language
        for language, translation in translations.items():
            if language not in {"URL", "topic"} and language in languages_to_analyze:
                try:
                    hidden_states = get_all_hidden_states(translation, model, tokenizer)
                    dict_example[language] = hidden_states
                except Exception as e:
                    print(f"Error for {example} in {language}: {e}")

        current_output[example] = dict_example

        # Save periodically every 101 examples (to avoid long delay)
        if (i + 1) % 101 == 0:
            print(f"{i + 1} examples done")
            with open(pkl_path, "wb") as f:
                pickle.dump(current_output, f)
            print(list(current_output.keys()))

        end_time_example = time.time()
        print(f"Generation for {example} done in {round(end_time_example - start_time_example, 2)} seconds")

    iteration_results[iteration_name] = current_output

    end_time_iteration = time.time()
    print(f"Generation for iteration {iteration + 1} done in {round(end_time_iteration - start_time_iteration, 2)} seconds")

    # After gathering all hidden states, compute various embeddings per example and language

    averaged_tensors = {}
    averaged_tensors_last = {}
    averaged_tensors_weighted = {}

    for example, languages in current_output.items():

        for_example_tensors = {}
        for_example_tensors_last = {}
        for_example_tensors_weighted = {}

        for language, tensors in languages.items():

            for_language_tensors = {}
            for_language_tensors_last = {}
            for_language_tensors_weighted = {}

            for i, tensor in enumerate(tensors):
                # Average over all tokens for each hidden state layer
                for_language_tensors[i] = np.mean(tensor, axis=1).tolist()

                # Extract last token embedding for each hidden state layer
                for_language_tensors_last[i] = tensor[:, -1, :].tolist()

                # Position-weighted average embedding (usually weighted by token positions)
                for_language_tensors_weighted[i] = position_weighted_average(tensor[0]).tolist()

            for_example_tensors[language] = for_language_tensors
            for_example_tensors_last[language] = for_language_tensors_last
            for_example_tensors_weighted[language] = for_language_tensors_weighted

        averaged_tensors[example] = for_example_tensors
        averaged_tensors_last[example] = for_example_tensors_last
        averaged_tensors_weighted[example] = for_example_tensors_weighted

    # Save the different embedding types as JSON files
    with open(f"{model}/hidden_states_last/{iteration_name}_last.json", "w") as f_2:
        json.dump(averaged_tensors_last, f_2, indent=4)
    print(f"Last token embedding saved")

    with open(f"{model}/hidden_states_averages/{iteration_name}_all.json", "w") as f_2:
        json.dump(averaged_tensors, f_2, indent=4)
    print(f"Average embedding saved")

    with open(f"{model}/hidden_states_weighted/{iteration_name}_weighted.json", "w") as f_2:
        json.dump(averaged_tensors_weighted, f_2, indent=4)
    print(f"Average weighted embedding saved")

    print(f"Saved: {iteration_name}")
