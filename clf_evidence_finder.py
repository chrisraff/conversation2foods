'''
use the neural network from nn.py to find evidence in a file
uses cleaned files from clean_hslld.py
'''
import numpy as np
from transformers import *
import torch
import pickle


tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
bert_model = DistilBertModel.from_pretrained('distilbert-base-uncased')


def transcript_to_chunks(transcript, radius=1):
    transcript_lines = transcript.splitlines()

    chunks = [None] * len(transcript_lines)

    for i in range(len(transcript_lines)):
        chunks[i] = '\n'.join(transcript_lines[max(0, i - radius) : min(len(transcript_lines), i + radius + 1)])
    
    return chunks
    


def find_evidence(clf, transcript, radius=1):
    chunks = transcript_to_chunks(transcript, radius=radius)

    bert_vectors = []

    for chunk in chunks:
        # get the bert vector
        input_ids = torch.tensor(tokenizer.encode(f"[CLS] {chunk} [SEP]")).unsqueeze(0)
        outputs = bert_model(input_ids)
        bert_vector = outputs[0][0, -1, :]

        bert_vectors += [bert_vector.detach().numpy()]
    
    X = np.array(bert_vectors)

    y = clf.predict(X)

    for i, is_evidence in enumerate(y):
        if is_evidence == 1:
            print(chunks[i])
            print('-'*48)


if __name__ == "__main__":
    fname = 'transcripts/remmt1.cha'

    with open(fname, 'r') as f:
        transcript = f.read()

    # load neural network
    with open('clf.pickle', 'rb') as f:
        clf = pickle.load(f)

    find_evidence(clf, transcript)
    
