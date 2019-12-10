# conversation2foods
 Extract foods from spoken conversation

## Requirements
Python 3

The spacy python package, which can be installed with: `pip install spacy`

Spacy english data, which can be downloaded with: `python -m spacy download en`

## Pipeline

Run `clean_hslld.py` to clean all the transcripts (create the folder 'transcripts' for them to be saved to)

Then run `parse_json_labels.py` to generate evidence data (create the destination 'chunks')
This uses the labels and the clean transcripts to create a set of (text_chunk, food) pairs

Run `augment_data.py` to augment the data by swapping in different foods to increase the number of true positives (create the destination folder 'chunks_augmented')

Then run `generate_evidence_classifier_data.py` to convert the evidence data into labeled bert data (create the folder 'bert_evidence'). A true label indicates that the chunk contains evidence of a food
If using augmented data, this should also be run on the augmented chunks (create the folder 'bert_evidence_augmented'). The input and ouput path are declared after the imports so these can be set easily.

Run `train_test_split.py` to create training and desting csv files. Some parameters can be tweaked in this file, like which files end up in the test set, and whether or not the training set is balanced and augmented

At this point you can run `train_classifier.py` or `nn_evidence.py` to train a classifier or neural network

<!-- Run the food database lemmatizer
`python food_database_lemmatizer.py`

Then, you can run `extract_food_candidates.py` -->
