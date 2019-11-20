import pickle
with open('../Python/data/food_desc_files/food_names.pickle', 'rb') as f:
    food_names = pickle.load(f)

food_tuples = sorted([(food_id, food_name) for (food_name, food_id) in food_names.items()])

giant_string = "\n".join(["{:>5} : {}".format(*x) for x in food_tuples])
print(giant_string)
