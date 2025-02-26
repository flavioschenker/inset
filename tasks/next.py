import torch
from tasks.task import Task
from data.settings import device, deftype
import benchmark.loss as L

class NextRegressive(Task):
    def __init__(self, param, featuremap, dataset, testset):
        super().__init__("next regressive", param, featuremap, dataset, testset)
        self.loss = L.BinaryCrossentropy(self.param, self.featuremap)
        self.epoch_valid_metrics = {
            "next/epoch_valid/hits": 0,
            "next/epoch_valid/size": 0,
        }

    def get_model_head(self):
        from models.next import NextRegressive
        return NextRegressive(self.param)

    def prepare_batch(self, x, y):
        seq_split = int(self.param["dim_sequence"] / 2)
        batch_size = x.shape[0]
        assert batch_size >= 4, f"batchsize {batch_size} to small for next sequence prediction"
        arange = torch.arange(0, batch_size, dtype=torch.long, requires_grad=False, device=device)
        neg_ids = torch.combinations(arange, 2, with_replacement=False)
        neg_ids = neg_ids[torch.ones(neg_ids.shape[0], device=device).multinomial(batch_size, replacement=False)]
        
        pos_pairs = x
        neg_pairs = torch.cat([x[neg_ids[:,0],:seq_split], x[neg_ids[:,1],seq_split:]], dim=1)
        pairs = torch.cat([pos_pairs, neg_pairs], dim=0)
        agreement = torch.cat([torch.ones(batch_size, device=device), torch.zeros(batch_size, device=device)], dim=0)
        return pairs, agreement

    def predict(self, x: torch.Tensor, y: torch.Tensor, model: torch.nn.Module) -> torch.Tensor:
        y_pred = model(x)
        y_true = y
        return y_pred, y_true

    def loss_train(self, y_pred, y_true):
        return self.loss(y_pred, y_true)

    def loss_valid(self, y_pred, y_true):
        return self.loss(y_pred, y_true)

    def validate(self, x, y, y_pred, y_true, model):
        prediction = (y_pred>0.5).to(torch.float)
        hits = (prediction == y_true).to(torch.float)
        hits_sum = hits.sum()
        batch_size = y_true.shape[0]
        accuracy = hits_sum / batch_size
        self.batch_valid_metrics["next/batch_valid/accuracy"] = accuracy
        self.epoch_valid_metrics["next/epoch_valid/hits"] += hits_sum
        self.epoch_valid_metrics["next/epoch_valid/size"] += batch_size

    def evaluate_valid_epoch(self, epoch: int, batch_count: int) -> dict:
        epoch_accuracy = self.epoch_valid_metrics["next/epoch_valid/hits"] / self.epoch_valid_metrics["next/epoch_valid/size"]
        for key in self.epoch_valid_metrics:
            self.epoch_valid_metrics[key] = 0
        return {"next/epoch_valid/accuracy": epoch_accuracy}