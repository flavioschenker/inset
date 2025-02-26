import torch

class Optimizer():
    def __init__(self, param, model):

        self.param = param
        self.model = model
        self.optimizer_string = self.param["optimizer"]

        if self.optimizer_string == "adam":
            self.optimizer = torch.optim.Adam(model.parameters(), lr=param["lr"], weight_decay=param["l2_regularization"])
        elif self.optimizer_string == "adamw":
            self.optimizer = torch.optim.AdamW(model.parameters(), lr=param["lr"], weight_decay=param["l2_regularization"])
        elif self.optimizer_string == "adadelta":
            self.optimizer = torch.optim.Adadelta(model.parameters(), lr=param["lr"], weight_decay=param["l2_regularization"])
        elif self.optimizer_string == "rmsprop":
            self.optimizer = torch.optim.RMSprop(model.parameters(), lr=param["lr"], weight_decay=param["l2_regularization"])
        else:
            raise Exception(f"unknown optimizer {self.optimizer_string}.")

        self.scheduler = torch.optim.lr_scheduler.ExponentialLR(self.optimizer, gamma=self.param["lr_decay"])

    def __call__(self):
        return (self.optimizer, self.scheduler)

    def step(self):
        self.scheduler.step()