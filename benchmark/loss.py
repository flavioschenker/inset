import torch
from data.settings import deftype, device

class Loss(torch.nn.Module):
    def __init__(self, parameters, featuremap):
        super().__init__()
        self.parameters = parameters
        self.featuremap = featuremap
        self.loss = None

    def forward(self, y_pred, y_true):
        if self.loss is None:
            raise NotImplementedError("loss not implemented.")
        return self.loss(y_pred, y_true)

class MeanSquaredError(Loss):
    def __init__(self, parameters, featuremap):
        super().__init__(parameters, featuremap)
        self.loss = torch.nn.MSELoss()

class BinaryCrossentropy(Loss):
    def __init__(self, parameters, featuremap):
        super().__init__(parameters, featuremap)
        self.loss = torch.nn.BCELoss()

class MultiLabelLoss(Loss):
    def __init__(self, parameters, featuremap, class_weights):
        super().__init__(parameters, featuremap)
        self.loss = torch.nn.BCEWithLogitsLoss(pos_weight=class_weights.to(device))

class MSENormLoss(Loss):
    def __init__(self, parameters, featuremap):
        super().__init__(parameters, featuremap)
        self.loss = torch.nn.MSELoss()

    def forward(self, y_pred, y_true):
        norm = self.featuremap.normalized
        return self.loss(y_pred[:,:,norm], y_true[:,:,norm])

class SIGNLoss(Loss):
    def __init__(self, parameters, featuremap):
        super().__init__(parameters, featuremap)
        self.loss = torch.nn.BCELoss()

    def forward(self, y_pred, y_true):
        sign = self.featuremap.sign
        y_pred_sig = torch.sigmoid(y_pred[:,:,sign])
        return self.loss(y_pred_sig, y_true[:,:,sign])

class ModLogLoss(Loss):
    def __init__(self, parameters, featuremap):
        super().__init__(parameters, featuremap)

    def forward(self, y_pred, y_true):
        log = self.featuremap.log
        y_pred_clamped = y_pred.detach().clone()
        y_pred_clamped[:,:,log] = torch.clamp(y_pred_clamped[:,:,log], min=0, max=16)
        se_log = torch.square(y_pred[:,:,log] - y_true[:,:,log])
        return (se_log * (y_pred_clamped[:,:,log] + 1) * (y_true[:,:,log] + 1)).mean()