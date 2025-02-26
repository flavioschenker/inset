import os
import numpy
import pandas
import pickle
from data.featuremap import FeatureMap
from benchmark.parameters import parameters

def feature_maker(data:pandas.Series, feature_map:FeatureMap) -> numpy.ndarray:
    result = []
    l = len(data)
    for row, sequence in enumerate(data):
        abs_max = max([abs(n) for n in sequence])
        feature_sequence = []
        for position, number in enumerate(sequence):
            features = numpy.zeros(len(feature_map))
            features[feature_map.position] = position
            features[feature_map.sign] = int(number < 0)
            features[feature_map.normalized] = number / abs_max if abs_max!=0 else 0
            number = abs(number)
            features[feature_map.log] = numpy.log10(number) if number!=0 else 0
            for i in range(feature_map.digits_first, feature_map.digits_last+1):
                features[i] = number%10
                number = number//10
            feature_sequence.append(features)
        result.append(feature_sequence)
        if row % 10000 == 0:
            print(f"Progress: {(row/l)*100: 2.2f} %", end="\r")
    return numpy.array(result, dtype=numpy.float64)

param = parameters()
directory = param["directory"]
dim_sequence = param["dim_sequence"]
validation_split = param["validation_split"]
debug_fraction = param["debug_dataset_fraction"]

path = os.path.join(directory,"raw","FACT_Dataset_50_eval.pickle")
with open(path, "rb") as file:
    data = pickle.load(file)
    print(len(data), "datapoints loaded.")
    data = data.sample(frac=1).reset_index(drop=True)

path = os.path.join(directory,"data")

dataset_length = len(data)
split = int(param["validation_split"]*dataset_length)
feature_map = FeatureMap(
    position=1,
    sign=1,
    normalized=1,
    log=1,
    digits=15,
    mask=1,
    )
feature_map.save(path)

x = feature_maker(data["sequence"], feature_map)
y = data[["eval_polynomial","eval_periodic","eval_exponential","eval_trigonometric","eval_modulo","eval_prime","eval_finite","complexity_d","complexity_l","complexity_t"]].to_numpy(dtype=numpy.float64)
print("done.                       ")
print("x", x.shape)
print("y", y.shape)

mask = numpy.full(dataset_length, False)
mask[:int(validation_split*dataset_length)] = True
numpy.random.shuffle(mask)
valid_mask = mask
train_mask = ~valid_mask

x_train = x[train_mask]
y_train = y[train_mask]
x_valid = x[valid_mask]
y_valid = y[valid_mask]
x_train_debug = x_train[:int(debug_fraction*len(x_train))]
y_train_debug = y_train[:int(debug_fraction*len(y_train))]
x_valid_debug = x_valid[:int(debug_fraction*len(x_valid)/validation_split)]
y_valid_debug = y_valid[:int(debug_fraction*len(y_valid)/validation_split)]
dataset_length = len(y_train)
dataset_length_debug = len(y_train_debug)
testset_length = len(y_valid)
testset_length_debug = len(y_valid_debug)
print(dataset_length, testset_length, dataset_length_debug, testset_length_debug)

dataset = {
    "dataset_length": dataset_length,
    "dim_sequence": dim_sequence,
    "x": x_train,
    "y": y_train
}
dataset_debug = {
    "dataset_length": dataset_length_debug,
    "dim_sequence": dim_sequence,
    "x": x_train_debug,
    "y": y_train_debug
}
testset = {
    "dataset_length": testset_length,
    "dim_sequence": dim_sequence,
    "x": x_valid,
    "y": y_valid
}
testset_debug = {
    "dataset_length": testset_length_debug,
    "dim_sequence": dim_sequence,
    "x": x_valid_debug,
    "y": y_valid_debug
}

with open(path+"/dataset.pickle", "wb") as file:
    pickle.dump(dataset, file)

with open(path+"/testset.pickle", "wb") as file:
    pickle.dump(testset, file)

with open(path+"/dataset_debug.pickle", "wb") as file:
    pickle.dump(dataset_debug, file)

with open(path+"/testset_debug.pickle", "wb") as file:
    pickle.dump(testset_debug, file)
