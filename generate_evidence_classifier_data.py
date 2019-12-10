"""
Uses the chunks from each transcript (from 'parse_json_labels.py')
to create a dataset of bertified vectors of the chunks and a
corrospending boolean label
"""
import json
import pandas as pd
import numpy as np
from tqdm import tqdm
from bertinator import get_bert_vector
from pathlib import Path


chunks_path = Path('chunks/')
output_path = Path('bert_evidence/')

# chunks_path = Path('chunks_augmented/')
# output_path = Path('bert_evidence_augmented/')

bert_vector_size = get_bert_vector('').cpu().detach().numpy().shape[0]


if __name__ == "__main__":
    for chunk_path in tqdm(list(chunks_path.glob('*.json'))):
        with open(chunk_path, 'r') as f:
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

        if len(bert_vectors) > 0:

            bert_vectors_vector = np.array(bert_vectors)

            df_bert = pd.DataFrame(bert_vectors_vector, columns=[f'b{x}' for x in range(bert_vector_size)])
            df_label = pd.DataFrame({'labels': labels})
        else:
            # create empty df with column names
            df_bert = pd.DataFrame(columns=[f'b{x}' for x in range(bert_vector_size)])
            df_label = pd.DataFrame(columns=['labels'])

        df = df_label.join(df_bert)
            
        fname = chunk_path.parts[-1][:-4] + 'csv'
        df.to_csv(output_path / fname)

        # print('appending to dataframe')
        # df_old = pd.read_csv('bert_data.csv', index_col=0)
        # df = df_old.append(df, ignore_index=True)
        # df.to_csv('bert_data.csv')
