import torch
from data.featuremap import FeatureMap

class Task:
    def __init__(self, name, param, featuremap, dataset, testset):
        self.name = name
        self.param = param
        self.featuremap = featuremap
        self.dataset = dataset
        self.testset = testset
        self.batch_train_metrics = {}
        self.epoch_train_metrics = {}
        self.batch_valid_metrics = {}
        self.epoch_valid_metrics = {}

    def get_model_head(self) -> torch.nn.Module:
        raise NotImplementedError(f"model head output not implemented.")

    def prepare_batch(self, x:torch.Tensor, y:torch.Tensor) -> torch.Tensor:
        raise NotImplementedError(f"batch preparation not implemented.")

    def predict(self, x:torch.Tensor, y:torch.Tensor, model:torch.nn.Module) -> torch.Tensor:
        raise NotImplementedError(f"prediction function not implemented.")

    def loss_train(self, y_pred:torch.Tensor, y_true:torch.Tensor) -> torch.Tensor:
        raise NotImplementedError(f"training loss not implemented.")

    def loss_valid(self, y_pred:torch.Tensor, y_true:torch.Tensor) -> torch.Tensor:
        raise NotImplementedError(f"validation loss not implemented.")

    def validate(self, x:torch.Tensor, y:torch.Tensor, y_pred:torch.Tensor, y_true:torch.Tensor, model:torch.nn.Module) -> None:
        raise NotImplementedError(f"validation not implemented.")

    def evaluate_train_batch(self, batch_step:int) -> dict:
        return self.batch_train_metrics
        
    def evaluate_train_epoch(self, epoch:int, batch_count:int) -> dict:
        result = {k: v/batch_count for k,v in self.epoch_train_metrics.items()}
        self.epoch_train_metrics = dict.fromkeys(self.epoch_train_metrics, 0)
        return result

    def evaluate_valid_batch(self, batch_step:int) -> dict:
        return self.batch_valid_metrics

    def evaluate_valid_epoch(self, epoch:int, batch_count:int) -> dict:
        result = {k: v/batch_count for k,v in self.epoch_valid_metrics.items()}
        self.epoch_valid_metrics = dict.fromkeys(self.epoch_valid_metrics, 0)
        return result

def get_task(param:dict, featuremap:FeatureMap, dataset:torch.utils.data.Dataset, testset:torch.utils.data.Dataset):
    task = param["task"]
    contrastive = param["contrastive"]
    if task=="unmasking":
        if contrastive:
            from tasks.unmasking import UnmaskingContrastive
            return UnmaskingContrastive(param, featuremap, dataset, testset)
        else:
            from tasks.unmasking import UnmaskingRegressive
            return UnmaskingRegressive(param, featuremap, dataset, testset)
    elif task=="classification":
        from tasks.classification import ClassificationRegressive
        return ClassificationRegressive(param, featuremap, dataset, testset)
    elif task=="continuation":
        if contrastive:
            from tasks.continuation import ContinuationContrastive
            return ContinuationContrastive(param, featuremap, dataset, testset)
        else:
            from tasks.continuation import ContinuationRegressive
            return ContinuationRegressive(param, featuremap, dataset, testset)
    elif task=="complexity":
        from tasks.complexity import ComplexityRegressive
        return ComplexityRegressive(param, featuremap, dataset, testset)
    elif task=="similarity":
        from tasks.similarity import SimilarityContrastive
        return SimilarityContrastive(param, featuremap, dataset, testset)
    elif task=="next":
        from tasks.next import NextRegressive
        return NextRegressive(param, featuremap, dataset, testset)
    else:
        raise ValueError(f"task {task} not supported.")