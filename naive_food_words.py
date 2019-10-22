'''
Check the entire corpus to find out which food words show up
Note that we don't care about what contexts the food words
show up in here, this is just to rule out foods that aren't
ever used
'''
from utils import get_all_transcript_strings
from extract_speech import extract_speech_string, pickle_path
from tqdm import tqdm
import pickle
import spacy


if __name__ == "__main__":
    print('starting spacy')
    nlp = spacy.load("en")

    print('loading transcripts')
    transcripts = get_all_transcript_strings()

    print('cleaning transcripts')
    transcripts = [extract_speech_string(s) for s in transcripts]

    print('REDUCING SIZE FOR TESTING')
    transcripts = transcripts[:10]

    print('parsing transcripts')
    # note that this does not benefit from multithreading
    nlp_trancripts = [None] * len(transcripts)
    for i, t in tqdm(enumerate(transcripts), total=len(transcripts)):
        new_doc = nlp(t)

        nlp_trancripts[i] = new_doc

    print('counting occurences of lemmas')
    lemma_counts = {}
    for transcript in tqdm(nlp_trancripts):
        for token in transcript:
            if token.lemma_ not in lemma_counts:
                lemma_counts[token.lemma_] = 0
            lemma_counts[token.lemma_] += 1

    with open(pickle_path + 'lemma_counts.pickle', 'wb') as f:
        pickle.dump(lemma_counts, f)
