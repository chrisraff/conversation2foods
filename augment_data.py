import json
from copy import copy
from tqdm import tqdm


# read in the data
with open('chunks_to_foods.json') as f:
    dataset = json.load(f)

# extract a list of all the foods
all_foods = set()
for chunk_thing in dataset:
    for food_noun_phrase in chunk_thing['foods']:
        food_noun_phrase = food_noun_phrase.lower()
        # TODO lemmatize it

        all_foods.add(food_noun_phrase)
all_foods = list(all_foods)
# print(all_foods)


# use a seed chunk to generate some synthetic chunks
def make_more_chunks(chunk_thing):
    chunk_string = chunk_thing['chunk']
    chunk_foods = chunk_thing['foods']

    # don't forget to include the original data
    augmented_chunks = [chunk_thing]

    # O(2N) iterations
    for i, chunk_food in enumerate(chunk_foods):
        for fake_food in all_foods:
            # update the string
            augmented_chunk = chunk_string.replace(chunk_food, fake_food)

            # update the label
            augmented_foods = copy(chunk_foods)
            augmented_foods[i] = fake_food

            augmented_chunks += [ {"chunk": augmented_chunk, "foods": augmented_foods} ]

    return augmented_chunks

augmented_dataset = []
for chunk_thing in tqdm(dataset):
    augmented_chunks = make_more_chunks(chunk_thing)
    augmented_dataset += augmented_chunks
# print(augmented_dataset)

print('Saving dataset')
with open('chunks_to_foods_augmented.json', 'w') as f:
    json.dump(augmented_dataset, f)

print(f"length has increase from {len(dataset)} to {len(augmented_dataset)}")
