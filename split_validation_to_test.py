import json
import os
import math
import random
import numpy

train_json = './toy_dataset/compositional/train.json'
new_train_json = './toy_dataset/compositional/new_train.json'
create_validation_json = './toy_dataset/compositional/validation.json'



with open(train_json) as f:

    validation_ratio = 0.1

    train = json.load(f)
    
    v = len(train)  # number of validation data
    t = math.floor(v * validation_ratio) # number of train data to split

    indices = random.sample(range(v), t)

    train = numpy.array(train)
    train_data = train[indices]
    
    validation_data = numpy.delete(train, indices)


    with open(new_train_json, 'w') as new_train_file:
        json.dump(train_data.tolist(), new_train_file)
    with open(create_validation_json, 'w') as validation_file:
        json.dump(validation_data.tolist(), validation_file)