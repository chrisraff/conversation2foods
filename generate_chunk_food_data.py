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
from sklearn.preprocessing import OneHotEncoder


chunks_path = 'chunks_to_foods.json'


if __name__ == "__main__":
    with open(chunks_path, 'r') as f:
        data = json.load(f)

    # load the foods
    with open('all_foods.txt', 'r') as f:
        all_foods = f.readlines()

    food_to_idx = {food.strip(): i for i, food in enumerate(all_foods)}

    bert_vectors = []
    food_vectors = np.zeros((len(data), len(all_foods)))

    for i, chunk_item in tqdm(enumerate(data), total=len(data)):
        chunk = chunk_item["chunk"]
        foods = chunk_item["foods"]

        bert_vector = get_bert_vector(chunk)

        bert_vectors += [bert_vector.cpu().detach().numpy()]

        for food in foods:
            food_vectors[i, food_to_idx[food.lower()]] = 1

    bert_vectors_vector = np.array(bert_vectors)

    df_bert = pd.DataFrame(bert_vectors_vector, columns=[f'b{x}' for x in range(bert_vectors_vector.shape[1])])
    df_label = pd.DataFrame(food_vectors, columns=all_foods)

    df = df_label.join(df_bert)

    print('saving file')
    df.to_csv('bert_food_vectors_data.csv')

    # print('appending to dataframe')
    # df_old = pd.read_csv('bert_food_vectors_data.csv', index_col=0)
    # df = df_old.append(df, ignore_index=True)
    # df.to_csv('bert_food_vectors_data.csv')
