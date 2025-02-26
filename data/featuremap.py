import pickle

class FeatureMap:
    def __init__(self, **allocations):
        self.feature_dimension = 0
        self.features = {}
        for key, value in allocations.items():
            self.features[key] = self.feature_dimension
            self.features[key + "_first"] = self.feature_dimension
            self.feature_dimension += value
            self.features[key + "_last"] = self.feature_dimension - 1

    def __len__(self) -> int:
        return self.feature_dimension

    def __str__(self) -> str:
        return str(self.features)

    def __getattr__(self, name:str) -> int:
        if name in self.features:
            return self.features[name]
        else:
            return self.__getattribute__(name)

    def add(self, name: str, feature_width: int) -> int:
        if name in self.features:
            return self.feature_dimension
        else:
            self.features[name] = self.feature_dimension
            self.features[name + "_first"] = self.feature_dimension
            self.feature_dimension += feature_width
            self.features[name + "_last"] = self.feature_dimension - 1
            return self.feature_dimension

    def save(self, path):
        state = {
            "feature_dimension": self.feature_dimension,
            "features": self.features
        }
        with open(path+"/featuremap.pickle", "wb") as file:
            pickle.dump(state, file)

    def load(self, path):
        with open(path+"/featuremap.pickle", "rb") as file:
            data = pickle.load(file)
        self.feature_dimension = data["feature_dimension"]
        self.features = data["features"]