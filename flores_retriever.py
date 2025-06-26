from datasets import load_dataset
import pandas as pd
import json

with open("flores_101_languages.json", "r") as f:
    lang_dict = json.load(f)

# Load the dataset
ds_101 = load_dataset("gsarti/flores_101", "all")

# Merge all splits into a single DataFrame
df_list = [ds_101[split].to_pandas() for split in ds_101.keys()]
merged_df = pd.concat(df_list, ignore_index=True)

columns_to_extract = ["URL"] + ["topic"] + list(merged_df.columns)[6:]

new_dict = {}

for index, row in merged_df.iterrows():

    texts = {}

    for column in columns_to_extract:
        if "sentence" in column:
            try:
                texts[lang_dict[column.split("_", 1)[1]]] = row[column]
            except:
                texts[column.split("_", 1)[1]] = row[column]
        else:
            texts[column] = row[column]

    new_dict[index] = texts

with open("flores_101.json", "w", encoding="utf-8") as fp:
    json.dump(new_dict, fp, ensure_ascii=False, indent = 4)