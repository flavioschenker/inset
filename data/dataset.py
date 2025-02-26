import numpy
import torch
import pickle
import os
from data.featuremap import FeatureMap

class Dataset(torch.utils.data.Dataset):
    def __init__(self, param, validation=False, debug=False, sweep=False):
        assert not (debug and sweep)
        self.param = param
        self.directory = os.path.join(param["directory"], "data")
        if validation:
            if debug:
                self.filename = "testset_debug.pickle"
                m = param["debug_validation_fraction"]
            else:
                self.filename = "testset.pickle"
                m = param["validation_fraction"]
        else:
            if debug:
                self.filename = "dataset_debug.pickle"
                m = param["debug_training_fraction"]
            else:
                self.filename = "dataset.pickle"
                m = param["training_fraction"]

        path = os.path.join(self.directory, self.filename)
        with open(path, "rb") as file:
            data = pickle.load(file)
            dataset_length = data["dataset_length"]
            print(dataset_length, "datapoints loaded.")

        self.dim_sequence = data["dim_sequence"]
        if sweep:
            n = int(param["sweep_fraction"] * dataset_length)
        else:
            n = int(m * dataset_length)

        mask = numpy.random.choice(dataset_length, n, replace=False)
        self.x = torch.tensor(data["x"][mask])
        self.y = torch.tensor(data["y"][mask])
        self.dim_dataset = len(self.y)
        self.class_count_positive = torch.sum(self.y[:,0:7], dim=0)
        self.class_count_positive[self.class_count_positive == 0] = 1
        self.class_count_negative = self.dim_dataset - self.class_count_positive
        self.class_weights = self.class_count_negative / self.class_count_positive
        del data
        if validation and not debug: assert param["valid_batch_size"] <= self.dim_dataset, "batchsize is too big for the amount of data."
        print(self.dim_dataset, "datapoints used.")
        self.featuremap = FeatureMap()
        self.featuremap.load(self.directory)
        self.features_dim = len(self.featuremap)

        if param["leave_one_out_feature"] is not None and not validation:
            feature = param["leave_one_out_feature"]
            if feature == "position":
                self.x[:,:,self.featuremap.position] = 0
            elif feature == "sign":
                self.x[:,:,self.featuremap.sign] = 0
            elif feature == "normalized":
                self.x[:,:,self.featuremap.normalized] = 0
            elif feature == "log":
                self.x[:,:,self.featuremap.log] = 0
            elif feature == "digits":
                self.x[:,:,self.featuremap.digits_first:self.featuremap.digits_last] = 0
            elif feature == "mask":
                self.x[:,:,self.featuremap.mask] = 0
            else:
                raise ValueError(f"feature {feature} not supported in leave-one-out-analysis.")

    def __getitem__(self, index):
        x = self.x[index]
        y = self.y[index]
        return x, y

    def __len__(self):
        return len(self.y)




