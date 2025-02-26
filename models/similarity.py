import torch

class SimilarityContrastive(torch.nn.Module):
    def __init__(self, param):
        super().__init__()
        self.dim_sequence = param["dim_sequence"]
        self.dim_features = param["dim_features"]
        self.dim_layers = param["similarity_dim_layers"]
        self.dim_embedding_space = param["similarity_dim_embedding_space"]
        in_dim = self.dim_sequence*self.dim_features
        self.layers = []
        for layer in self.dim_layers:
            self.layers.append(torch.nn.Linear(in_dim, layer))
            in_dim = layer
            self.layers.append(torch.nn.ReLU())
        self.layers = torch.nn.ModuleList(self.layers)
        self.output = torch.nn.Linear(self.dim_layers[-1], self.dim_embedding_space)
        self.output_relu = torch.nn.ReLU()

    def forward(self, tensor:torch.Tensor) -> torch.Tensor:
        tensor = tensor.flatten(start_dim=1)
        for layer in self.layers:
            tensor = layer(tensor)
        output = self.output_relu(self.output(tensor))
        return output