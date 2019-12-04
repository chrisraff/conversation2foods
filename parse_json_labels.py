'''
Using the labels and the clean text, create a dataset of ('lines', [foods]) tuples
'''
from utils import data_path, all_transcripts_glob
from extract_speech import extract_speech_string
from pathlib import Path
import re
import json
import numpy as np
from tqdm import tqdm


evidence_path = Path('labels/')
radius = 1 # number of surrounding lines to take

output_fname = 'chunks_to_foods.json'

file_whitelist = None # if None, all files are accepted
# file_whitelist = ['admmt7.json', 'allmt7.json', 'davmt7.json', 'jebmt7.json']


if __name__ == "__main__":
    dataset = [] # tuples go here

    for fpath in tqdm(evidence_path.glob('*')):
        # check if we should use this file
        if file_whitelist is not None and fpath.parts[-1] not in file_whitelist:
            # print(f'skipping {fpath.parts[-1]}')
            continue

        # loading the labels
        with open(fpath, 'r') as f:
            try:
                food_evidences = json.load(f)
            except json.decoder.JSONDecodeError as e:
                print(f'problem in {fpath}')
                print(e)
                exit()
        
        fe = [food['evidence'] for food in food_evidences]
        evidence_lines = [x for y in fe for x in y]

        # convert the filename from .json to .cha
        cha_fname = fpath.parts[-1][:-4] + 'cha'
        
        # loading the cleaned transcript
        with open(f'transcripts/{cha_fname}') as f:
            transcript_lines = f.read().splitlines()

        # extracting the line chunks and labeling them
        for i in range(len(transcript_lines)):
            foods = []
            potentially_ambiguous = False

            for food in food_evidences:
                if (i + 1) in food['evidence']:
                    foods += [food['food']]

                # if this chunk is near other evidence, it should be ignored
                for n in food['evidence']:
                    # i is 0 indexed, n is 1 indexed
                    if n - radius <= i + 1 <= n + radius:
                        potentially_ambiguous = True
            
            if potentially_ambiguous and len(foods) == 0:
                # ignore it
                continue

            chunk = '\n'.join(transcript_lines[max(0, i - radius) : min(len(transcript_lines), i + radius + 1)])

            dataset += [ {"chunk": chunk, "foods": foods} ]
            
    print('Saving dataset')
    with open(output_fname, 'w') as f:
        json.dump(dataset, f)
