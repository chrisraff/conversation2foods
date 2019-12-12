'''
For a given transcript, lemmatize it and then match food
words from the lemmatized food names database
'''
import pickle
from extract_speech import extract_speech_string, data_path
import spacy


# start spacy
nlp = spacy.load('en')

# load food names
with open('pickles/lemmatized_food_names.pickle', 'rb') as f:
    food_names = pickle.load(f)


# takes a speech extracted transcript, returns a dictionary of sentences to food
# def extract_


# Returns a list of potential phrases in the sentence
# problems with this very naive approach:
#  supersets:
#   I want chocolate milk after I'm all done with mine.
#   ['milk', 'chocolate milk', 'chocolate']
def extract_potential_foods(sentence):
    nlp_sentence = nlp(sentence)

    lemmatized_sentence = ' '.join( [ t.lemma_ for t in nlp_sentence ] )

    output = []

    for food_name in food_names.keys():
        # outer spaces are used to avoid substring problems (pieces -> pie)
        if f' {food_name} ' in f' {lemmatized_sentence} ':
            output.append(food_name)
    
    return output


if __name__ == '__main__':
    transcript_path = data_path.joinpath('HV1/MT/admmt1.cha')

    with open(transcript_path, 'r') as f:
        content = f.read()

    sentences = extract_speech_string(content)

    # extract_candidate_sentences(content)
    for sentence in sentences.split('\n'):
        foods = extract_potential_foods(sentence)
        if len(foods):
            print(sentence)
            print(foods)
            print('=' * 30)
