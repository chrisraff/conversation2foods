# conversation2foods
 Extract foods from spoken conversation

## Requirements
Python 3

The spacy python package, which can be installed with: `pip install spacy`

Spacy english data, which can be downloaded with: `python -m spacy download en`

## Pipeline

Run `clean_hslld.py` to clean all the transcripts (create the folder 'transcripts' for them to be saved to)

Then run `parse_json_labels.py` to generate evidence data

Then run `generate_evidence_classifier_data.py` to convert the evidence data into labeled bert data. A true label indicates that the chunk contains evidence of a food

At this point you can run `train_classifier.py` or `nn.py` to train a classifier or neural network

Run the food database lemmatizer
`python food_database_lemmatizer.py`

Then, you can run `extract_food_candidates.py`
