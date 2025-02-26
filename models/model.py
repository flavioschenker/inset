import os
import sys
import torch
from data.settings import device

class PretrainModel(torch.nn.Module):
    def __init__(self, param, regime, output_heads, output_required):
        super().__init__()
        self.param = param
        self.regime = regime
        self.directory = os.path.join(param["directory"], "trained_models")
        from models.transformer import TransformerEncoder
        self.base_encoder = TransformerEncoder(param)
        self.output_heads = torch.nn.ModuleDict(output_heads)
        self.output_required = output_required

    def forward(self, tensor:torch.Tensor) -> dict:
        tensor = self.base_encoder(tensor)
        output = {}
        for key, head in self.output_heads.items():
            if self.output_required[key]:
                output[key] = head(tensor)
        return output

    def save(self, model_instance_name: str):
        directory = os.path.join(self.directory,"base_encoder")
        if not os.path.exists(directory):
            os.makedirs(directory)
        path = os.path.join(directory, str(model_instance_name) + ".pt")
        torch.save(self.base_encoder, path)

        directory = os.path.join(self.directory,"full_model", self.regime)
        if not os.path.exists(directory):
            os.makedirs(directory)
        path = os.path.join(directory, str(model_instance_name) + ".pt")
        torch.save(self, path)

    def load(self, model_instance_name: str):
        path = os.path.join(self.directory,"full_model", self.regime, str(model_instance_name) + ".pt")
        try:
            return torch.load(path, map_location=device)
        except OSError as e:
            print(e)
            sys.exit(1)

class TuningModel(torch.nn.Module):
    def __init__(self, param, task_head):
        super().__init__()
        self.param = param
        self.task_head = task_head
        self.directory = os.path.join(param["directory"], "trained_models")
        if param["model"] == "new":
            from models.transformer import TransformerEncoder
            self.base_encoder = TransformerEncoder(param)
        else:
            try:
                path = os.path.join(self.directory, "base_encoder", str(param["model"]) + ".pt")
                self.base_encoder = torch.load(path, map_location=device)
            except OSError as e:
                print(e)
                sys.exit(1)

            if param["model_freeze"]:
                for parameter in self.base_encoder.parameters():
                    parameter.requires_grad = False

    def forward(self, tensor:torch.Tensor) -> torch.Tensor:
        tensor = self.base_encoder(tensor)
        tensor = self.task_head(tensor)
        return tensor

    def save(self, name:str) -> None:
        directory = os.path.join(self.directory, "tuning_models", self.param["task"])
        if not os.path.exists(directory):
            os.makedirs(directory)
        path = os.path.join(directory, str(name) + ".pt")
        torch.save(self, path)