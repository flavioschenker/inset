import torch
from tasks.task import Task
from data.settings import device, deftype
import benchmark.loss as L

class ComplexityRegressive(Task):
    def __init__(self, param, featuremap, dataset, testset):
        super().__init__("complexity regressive", param, featuremap, dataset, testset)
        self.loss = L.MeanSquaredError(self.param, self.featuremap)
        self.epoch_train_metrics = {
            "complexity/epoch_train/mse-depth": 0,
            "complexity/epoch_train/mse-literals": 0,
            "complexity/epoch_train/mse-terminals": 0,
        }
        self.epoch_valid_metrics = {
            "complexity/epoch_valid/mse-depth": 0,
            "complexity/epoch_valid/mse-literals": 0,
            "complexity/epoch_valid/mse-terminals": 0,
        }

    def get_model_head(self):
        from models.complexity import ComplexityRegressive
        return ComplexityRegressive(self.param)

    def prepare_batch(self, x, y):
        complexity = y[:,7:10]
        return x, complexity

    def predict(self, x, y, model):
        y_pred = model(x)
        y_true = y
        return y_pred, y_true

    def loss_train(self, y_pred, y_true):
        depth_loss = self.loss(y_pred[:,0], y_true[:,0])
        literals_loss = self.loss(y_pred[:,1], y_true[:,1])
        terminals_loss = self.loss(y_pred[:,2], y_true[:,2])
        self.batch_train_metrics["complexity/batch_train/mse-depth"] = depth_loss.item()
        self.batch_train_metrics["complexity/batch_train/mse-literals"] = literals_loss.item()
        self.batch_train_metrics["complexity/batch_train/mse-terminals"] = terminals_loss.item()
        self.epoch_train_metrics["complexity/epoch_train/mse-depth"] += depth_loss.item()
        self.epoch_train_metrics["complexity/epoch_train/mse-literals"] += literals_loss.item()
        self.epoch_train_metrics["complexity/epoch_train/mse-terminals"] += terminals_loss.item()
        return depth_loss + literals_loss + terminals_loss

    def loss_valid(self, y_pred, y_true):
        depth_loss = self.loss(y_pred[:,0], y_true[:,0])
        literals_loss = self.loss(y_pred[:,1], y_true[:,1])
        terminals_loss = self.loss(y_pred[:,2], y_true[:,2])
        self.batch_valid_metrics["complexity/batch_valid/mse-depth"] = depth_loss.item()
        self.batch_valid_metrics["complexity/batch_valid/mse-literals"] = literals_loss.item()
        self.batch_valid_metrics["complexity/batch_valid/mse-terminals"] = terminals_loss.item()
        self.epoch_valid_metrics["complexity/epoch_valid/mse-depth"] += depth_loss.item()
        self.epoch_valid_metrics["complexity/epoch_valid/mse-literals"] += literals_loss.item()
        self.epoch_valid_metrics["complexity/epoch_valid/mse-terminals"] += terminals_loss.item()
        return depth_loss + literals_loss + terminals_loss

    def validate(self, x, y, y_pred, y_true, model):
        pass