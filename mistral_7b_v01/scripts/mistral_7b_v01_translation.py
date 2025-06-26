from datasets import load_dataset
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import json
import regex as re
import random
import time

# Set a seed for reproducibility
random.seed(43)

# Select 100 random example numbers from 1 to 2000 (inclusive)
selected_numbers = random.sample(range(1, 2001), 100)

model_name_all = "mistral_7b_v01"

# Setup device (GPU if available)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Load tokenizer and model with authentication token (replace with your token)
access_token = "#####"
model_name = "mistralai/Mistral-7B-v0.1"

tokenizer = AutoTokenizer.from_pretrained(model_name, token=access_token)
model = AutoModelForCausalLM.from_pretrained(model_name, token=access_token, output_hidden_states=True)
model.to(device)  # Move model to GPU if available

# Load Flores 101 language sentences
with open('flores_101.json', "r", encoding="utf-8") as f:
    languages_flores = json.load(f)

# List of languages to generate translations for
languages_to_analyze = [
    "Swahili", "Yoruba",    # African Languages
    "Hindi", "Urdu",        # South Asian Languages
    "German",               # Germanic Languages
    "Italian", "Spanish",   # Romance Languages
    "Polish", "Croatian", "Russian",  # Slavic Languages
    "Hungarian",            # Other European Languages
]

def format_prompt(language_1, language_2, language_1_sentence):
    """
    Create prompt for translating a sentence from language_1 to language_2.
    Cuts at first period to avoid overly long input.
    """
    prompt = f"Translate the following text in {language_1} into {language_2}. Do not return anything besides the translation.\n"
    prompt += f"Text in {language_1}: {language_1_sentence.split('.')[0]}\n"
    prompt += "Translation: "
    return prompt

def extract_answer(response):
    """
    Extract translation after 'Translation: ' keyword in the model output.
    Returns None if keyword not found.
    """
    keyword = "Translation: "
    index = response.find(keyword)
    return response[index + len(keyword):].strip() if index != -1 else None

start_time = time.time()
translations_dict = {}

for i, number in enumerate(selected_numbers):
    number_dict = {}
    for language in languages_to_analyze:
        # English -> Target language translation
        prompt_1 = format_prompt("English", language, languages_flores[str(number)]["English"])
        inputs_1 = tokenizer(prompt_1, return_tensors="pt").to(device)
        with torch.no_grad():
            output = model.generate(**inputs_1, max_new_tokens=50)
        response_1 = tokenizer.decode(output[0], skip_special_tokens=True)
        answer_1 = extract_answer(response_1)
        number_dict[f"English-{language}"] = answer_1

        # Target language -> English translation
        prompt_2 = format_prompt(language, "English", languages_flores[str(number)][language])
        inputs_2 = tokenizer(prompt_2, return_tensors="pt").to(device)
        with torch.no_grad():
            output = model.generate(**inputs_2, max_new_tokens=50)
        response_2 = tokenizer.decode(output[0], skip_special_tokens=True)
        answer_2 = extract_answer(response_2)
        number_dict[f"{language}-English"] = answer_2


    translations_dict[number] = number_dict
    print(f"Example {i+1} done")

    # Print progress every 10 examples
    if (i + 1) % 10 == 0:
        intermediate = time.time()
        elapsed_minutes = round((intermediate - start_time) / 60, 2)
        print(f"{i + 1} examples done in {elapsed_minutes} minutes.")

# Save all generated translations to JSON file
with open(f"{model_name_all}/mexa/task_results/{model_name_all}_translation_results.json", "w", encoding="utf-8") as f:
    json.dump(translations_dict, f, indent=4, ensure_ascii=False)

end_time = time.time()
total_minutes = round((end_time - start_time) / 60, 2)
print(f"Translations done in {total_minutes} minutes.")
