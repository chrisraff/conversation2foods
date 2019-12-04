'''
use the neural network from nn.py to find evidence in a file
uses cleaned files from clean_hslld.py
'''
import numpy as np
from transformers import *
import torch
from nn import Net
from torch.autograd import Variable
from bertinator import get_bert_vector


def transcript_to_chunks(transcript, radius=1):
    transcript_lines = transcript.splitlines()

    chunks = [None] * len(transcript_lines)

    for i in range(len(transcript_lines)):
        chunks[i] = '\n'.join(transcript_lines[max(0, i - radius) : min(len(transcript_lines), i + radius + 1)])
    
    return chunks
    


def find_evidence(model, transcript, radius=1):
    chunks = transcript_to_chunks(transcript, radius=radius)

    bert_vectors = []

    for chunk in chunks:
        bert_vector = get_bert_vector(chunk)

        bert_vectors += [bert_vector.detach().numpy()]
    
    X = np.array(bert_vectors)
    X = Variable(torch.Tensor(X).float())

    predict_out = model(X)
    _, y_pred = torch.max(predict_out, 1)

    for i, is_evidence in enumerate(y_pred):
        if is_evidence.data == 1:
            print(chunks[i])
            print('-'*48)


if __name__ == "__main__":
    fname = 'transcripts/remmt1.cha'

    with open(fname, 'r') as f:
        transcript = f.read()

    # load neural network
    model = Net()
    model.load_state_dict(torch.load('evidencenet.nn'))
    model.eval()

    find_evidence(model, transcript)
