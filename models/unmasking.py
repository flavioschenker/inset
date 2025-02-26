import torch

class UnmaskingRegressive(torch.nn.Module):
    def __init__(self, param):
        super().__init__()
        self.dim_sequence = param["dim_sequence"]
        self.dim_features = param["dim_features"]
        self.dim_layers = param["unmasking_regressive_dim_layers"]
        in_dim = self.dim_sequence*self.dim_features
        self.layers = []
        for layer in self.dim_layers:
            self.layers.append(torch.nn.Linear(in_dim, layer))
            in_dim = layer
            self.layers.append(torch.nn.ReLU())
        self.layers = torch.nn.ModuleList(self.layers)
        self.output = torch.nn.Linear(self.dim_layers[-1], self.dim_sequence*self.dim_features)


    def forward(self, tensor:torch.Tensor) -> torch.Tensor:
        input_shape = tensor.shape
        tensor = tensor.flatten(start_dim=1)
        for layer in self.layers:
            tensor = layer(tensor)
        output = self.output(tensor)
        return output.reshape(input_shape)

class UnmaskingContrastive(torch.nn.Module):
    def __init__(self, param):
        super().__init__()
        self.dim_sequence = param["dim_sequence"]
        self.dim_features = param["dim_features"]
        self.dim_layers = param["unmasking_contrastive_dim_layers"]
        self.dim_embedding_space = param["unmasking_contrastive_dim_embedding_space"]
        in_dim = self.dim_sequence*self.dim_features
        self.layers = []
        for layer in self.dim_layers:
            self.layers.append(torch.nn.Linear(in_dim, layer))
            in_dim = layer
            self.layers.append(torch.nn.ReLU())
        self.layers = torch.nn.ModuleList(self.layers)
        self.output = torch.nn.Linear(self.dim_layers[-1], self.dim_embedding_space)

    def forward(self, tensor:torch.Tensor) -> torch.Tensor:
        tensor = tensor.flatten(start_dim=1)
        for layer in self.layers:
            tensor = layer(tensor)
        output = self.output(tensor)
        return output