import json
from copy import copy
from tqdm import tqdm
from pathlib import Path


# get all the foods
with open('all_foods.txt', 'r') as f:
    all_foods = f.readlines()

chunks_path = Path('chunks/')
output_path = Path('chunks_augmented/')


# use a seed chunk to generate some synthetic chunks
# excludes the original chunk
def make_more_chunks(chunk_thing):
    chunk_string = chunk_thing['chunk']
    chunk_foods = chunk_thing['foods']

    augmented_chunks = []

    # O(2N) iterations
    for i, chunk_food in enumerate(chunk_foods):
        for fake_food in all_foods:
            if fake_food == chunk_food:
                continue

            # update the string
            augmented_chunk = chunk_string.replace(chunk_food, fake_food)

            # update the label
            augmented_foods = copy(chunk_foods)
            augmented_foods[i] = fake_food

            augmented_chunks += [ {"chunk": augmented_chunk, "foods": augmented_foods} ]

    return augmented_chunks


for chunk_path in tqdm(list(chunks_path.glob('*.json'))):
    with open(chunk_path, 'r') as f:
        dataset = json.load(f)

    augmented_dataset = []
    for chunk_thing in dataset:
        augmented_chunks = make_more_chunks(chunk_thing)
        augmented_dataset += augmented_chunks
        
    with open(output_path / chunk_path.parts[-1], 'w') as f:
        json.dump(augmented_dataset, f)

    # print(f"length has increase from {len(dataset)} to {len(augmented_dataset)}")
