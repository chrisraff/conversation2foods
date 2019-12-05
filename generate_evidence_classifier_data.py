"""
Uses 'chunks_to_foods.json' (from 'parse_json_labels.py') to create
a dataset of bertified vectors of the chunks and a corrospending
boolean label
"""
import json
import pandas as pd
import numpy as np
from tqdm import tqdm
from bertinator import get_bert_vector


chunks_path = 'chunks_to_foods.json'


if __name__ == "__main__":
    with open(chunks_path, 'r') as f:
        data = json.load(f)

    bert_vectors = []
    labels = []

    for chunk_item in tqdm(data):
        chunk = chunk_item["chunk"]
        foods = chunk_item["foods"]

        label = len(foods) > 0

        bert_vector = get_bert_vector(chunk)

        bert_vectors += [bert_vector.cpu().detach().numpy()]
        labels += [label]

    bert_vectors_vector = np.array(bert_vectors)

    df_bert = pd.DataFrame(bert_vectors_vector, columns=[f'b{x}' for x in range(bert_vectors_vector.shape[1])])
    df_label = pd.DataFrame({'labels': labels})

    df = df_label.join(df_bert)

    print('saving file')
    df.to_csv('bert_data.csv')

    # print('appending to dataframe')
    # df_old = pd.read_csv('bert_data.csv', index_col=0)
    # df = df_old.append(df, ignore_index=True)
    # df.to_csv('bert_data.csv')
