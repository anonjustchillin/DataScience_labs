import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForMaskedLM
import os
from get_data import get_data
from verbalizers import *
from analyze_results import get_result_analysis
import warnings
warnings.filterwarnings(action='ignore')

PROJECT_PATH = 'D:\\uni\\3курс\\Data_Science\\Data_science_labs\\lab4\\data'
URL_1 = 'https://rozetka.com.ua/ua/sony-playstation-5-slim-digital-edition/p410219112/comments/'
NAME_1 = 'playstation'

# на сковорідку більше негативних відгуків
URL_2 = 'https://rozetka.com.ua/ua/holmer_fp_22330_swmbl_star_chef/p381166296/comments/'
NAME_2 = 'pan'

tokenizer = AutoTokenizer.from_pretrained(
    "HPLT/hplt_gpt_bert_base_3_0_ukr_Cyrl",
)
model = AutoModelForMaskedLM.from_pretrained(
    "HPLT/hplt_gpt_bert_base_3_0_ukr_Cyrl",
    trust_remote_code=True,
    use_safetensors=False,
)
categories = ['позитивний', 'нейтральний', 'негативний']
verbalizers = {
    categories[0]: POSITIVE,
    categories[1]: NEUTRAL,
    categories[2]: NEGATIVE
}

label_token_ids = {}
for category, words in verbalizers.items():
    token_ids = [tokenizer.encode(word, add_special_tokens=False)[0] for word in words]
    label_token_ids[category] = token_ids

def classify_text(input_text):
    prompt = f'Отримано відгук: "{input_text}". За тональністю цей відгук є {tokenizer.mask_token}'
    input_text = tokenizer(prompt, return_tensors="pt")

    mask_token_idx = torch.where(input_text.input_ids == tokenizer.mask_token_id)[1]

    with torch.no_grad():
        output = model(**input_text)
        logits = output.logits

    mask_token_logits = logits[0, mask_token_idx[0], :]
    category_scores = {}
    for category, token_ids in label_token_ids.items():
        scores = mask_token_logits[token_ids]
        category_scores[category] = scores.mean().item()

    best = max(category_scores, key=category_scores.get)
    return best

def review_reviews(url, name):
    get_data(url, name)

    print()
    filename = name + '_df_cleaned.csv'
    filepath = os.path.join(PROJECT_PATH, filename)
    df = pd.read_csv(filepath, index_col=0)
    df.rename(columns={'0': 'Comments'}, inplace=True)
    print(df.head())
    print()

    model_output = []
    for i in range(len(df)):
        text = df.iloc[i, 0]
        output = classify_text(text)
        model_output.append(output)

    df['Category'] = model_output

    cols = list(df.columns)
    a, b = cols.index('Comments'), cols.index('Category')
    cols[b], cols[a] = cols[a], cols[b]
    df = df[cols]

    print(df.head())
    print(df.tail())
    print()

    filename = os.path.join(PROJECT_PATH, name + '_result.csv')
    df.to_csv(filename)

    get_result_analysis(filename, name)


if __name__ == '__main__':
    review_reviews(URL_2, NAME_2)

