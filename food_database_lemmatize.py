'''
Lemmatize individual words in the food phrase database so they
can be matched with lemmatized sentences later
'''
from utils import get_all_transcript_strings, pickle_path
from extract_speech import extract_speech_string
from tqdm import tqdm
import pickle
import spacy


if __name__ == "__main__":
    print('starting spacy')
    nlp = spacy.load("en")

    print('loading foods')
    with open('../Python/data/food_desc_files/food_names.pickle', 'rb') as f:
        food_names = pickle.load(f)

    # print('REDUCING SIZE FOR TESTING')
    # food_names = { f: food_names[f] for f in list(food_names.keys())[:10] }

    print('parsing food names')
    # note that this does not benefit from multithreading
    lemmatized_food_names = {}
    for name in tqdm(food_names.keys(), total=len(food_names)):
        nlp_food_name = nlp(name)

        lemmatized_name = ' '.join( [token.lemma_ for token in nlp_food_name] )

        lemmatized_food_names[ lemmatized_name ] = food_names[name]

    print('saving lemmatized food names')
    with open(pickle_path + 'lemmatized_food_names.pickle', 'wb') as f:
        pickle.dump(lemmatized_food_names, f)
