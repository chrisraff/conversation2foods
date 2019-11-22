# conversation2foods
 Extract foods from spoken conversation

## Requirements
Python 3

The spacy python package, which can be installed with: `pip install spacy`

Spacy english data, which can be downloaded with: `python -m spacy download en`

## Pipeline

Run `clean_hslld.py` to clean all the transcripts (create the folder 'transcripts' for them to be saved to)

Run the food database lemmatizer
`python food_database_lemmatizer.py`

Then, you can run `extract_food_candidates.py`
