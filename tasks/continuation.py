import torch
from tasks.task import Task
from data.settings import device, deftype
import benchmark.loss as L

# sequence continuation multi-shot
class ContinuationContrastive(Task):
    def __init__(self, param, featuremap, dataset, testset):
        super().__init__("continuation contrastive", param, featuremap, dataset, testset)
        self.epoch_train_metrics = {
            "continuation/epoch_train/agreement loss": 0,
            "continuation/epoch_train/margin loss": 0,
        }
        self.epoch_valid_metrics = {
            "continuation/epoch_top_k_rmse/top1": 0,
            "continuation/epoch_top_k_rmse/top3": 0,
            "continuation/epoch_top_k_rmse/top5": 0,
            "continuation/epoch_valid/agreement loss": 0,
            "continuation/epoch_valid/margin loss": 0,
        }

    def get_model_head(self):
        from models.continuation import ContinuationContrastive
        return ContinuationContrastive(self.param)

    def prepare_batch(self, x, y):
        x_masked = x.clone()
        mask_length = self.param["continuation_mask_length"]
        x_masked[:,-mask_length:] = 0
        x_masked[:,-mask_length:,self.featuremap.mask] = 1
        x[:,-mask_length:,self.featuremap.mask] = 1
        return x_masked, x

    def predict(self, x, y, model):
        df = self.featuremap.digits_first
        dl = self.featuremap.digits_last + 1
        x_unmasked = y
        x_embeddings_masked = model(x)
        x_embeddings_unmasked = model(y)
        del x,y
        # indizes
        batch_size = x_unmasked.shape[0]
        arange = torch.arange(0, batch_size, dtype=torch.long, requires_grad=False, device=device)
        pos_pair_indices = torch.cat([arange.unsqueeze(1), arange.unsqueeze(1)], dim=1)
        neg_pair_indices = torch.combinations(arange, 2, with_replacement=False)
        neg_pair_indices_samples = neg_pair_indices[torch.ones(neg_pair_indices.shape[0], device=device).multinomial(self.param["continuation_neg_pairs_fraction"]*batch_size, replacement=False)]
        pair_indices = torch.cat([pos_pair_indices, neg_pair_indices_samples], dim=0)

        x_embeddings_masked = x_embeddings_masked.unsqueeze(1) # (batch, 1, sequence, features)
        x_embeddings_unmasked = x_embeddings_unmasked.unsqueeze(1)
        
        agreement = torch.all(torch.eq(x_unmasked[pair_indices[:,0],:,df:dl], x_unmasked[pair_indices[:,1],:,df:dl]),dim=2).to(dtype=torch.int)
        agreement = torch.cat([agreement, torch.zeros((agreement.shape[0],1),device=device)], dim=1)
        agreement_fraction = torch.argmin(agreement,dim=1) / self.param["dim_sequence"]

        pairs = torch.cat([x_embeddings_masked[pair_indices[:,0]], x_embeddings_unmasked[pair_indices[:,1]]], dim=1)
        del x_embeddings_masked, x_embeddings_unmasked

        return (pairs, agreement_fraction)

    def loss_train(self, y_pred, y_true):
        margin = self.param["continuation_contrastive_margin"]
        embedding_pairs = y_pred
        agreement_fraction = y_true
        del y_pred, y_true
        embedding_distance_squared = torch.maximum(torch.tensor([ 1e-14 ], device=device), torch.square(embedding_pairs[:,0] - embedding_pairs[:,1]))
        embedding_distance_summed = torch.sum(embedding_distance_squared.flatten(start_dim=1), dim=1)
        del embedding_distance_squared
        embedding_distance = torch.sqrt(embedding_distance_summed)
        agreement_loss = agreement_fraction*embedding_distance_summed
        del embedding_distance_summed
        margin_loss = torch.square((1-agreement_fraction)*torch.maximum(margin - embedding_distance, torch.tensor([0], device=device)))
        del embedding_distance
        contrastive_loss = torch.mean(agreement_loss + margin_loss)
        loss = contrastive_loss
        self.batch_train_metrics["continuation/batch_train/agreement loss"] = agreement_loss.mean().item()
        self.batch_train_metrics["continuation/batch_train/margin loss"] = margin_loss.mean().item()
        self.epoch_train_metrics["continuation/epoch_train/agreement loss"] += agreement_loss.mean().item()
        self.epoch_train_metrics["continuation/epoch_train/margin loss"] += margin_loss.mean().item()
        return loss

    def loss_valid(self, y_pred, y_true):
        margin = self.param["continuation_contrastive_margin"]
        embedding_pairs = y_pred
        agreement_fraction = y_true
        del y_pred, y_true
        embedding_distance_squared = torch.maximum(torch.tensor([ 1e-14 ], device=device), torch.square(embedding_pairs[:,0] - embedding_pairs[:,1]))
        embedding_distance_summed = torch.sum(embedding_distance_squared.flatten(start_dim=1), dim=1)
        del embedding_distance_squared
        embedding_distance = torch.sqrt(embedding_distance_summed)
        agreement_loss = agreement_fraction*embedding_distance_summed
        del embedding_distance_summed
        margin_loss = torch.square((1-agreement_fraction)*torch.maximum(margin - embedding_distance, torch.tensor([0], device=device)))
        del embedding_distance
        contrastive_loss = torch.mean(agreement_loss + margin_loss)
        loss = contrastive_loss
        self.batch_valid_metrics["continuation/batch_valid/agreement loss"] = agreement_loss.mean().item()
        self.batch_valid_metrics["continuation/batch_valid/margin loss"] = margin_loss.mean().item()
        self.epoch_valid_metrics["continuation/epoch_valid/agreement loss"] += agreement_loss.mean().item()
        self.epoch_valid_metrics["continuation/epoch_valid/margin loss"] += margin_loss.mean().item()
        return loss

    def validate(self, x, y, y_pred, y_true, model):
        df = self.featuremap.digits_first
        dl = self.featuremap.digits_last+1
        batch_size = y.shape[0]

        batch_masked_left = x # (batch,sequence,features)
        batch_unmasked_left = y # (batch,sequences,features)

        del x,y
        batch_left_embedded = model(batch_masked_left) # (batch, embedding)
        batch_sequences_left = batch_unmasked_left[:,:,df:dl].unsqueeze(1) # (batch,1,sequence,digits)
        del batch_masked_left, batch_unmasked_left

        search_dataloader = torch.utils.data.DataLoader(self.dataset, batch_size=batch_size, drop_last=True, shuffle=True)
        old_embedding_distance = None
        old_sequences = None

        for subbatch in search_dataloader:
            x,y = subbatch
            batch_unmasked_right = x.to(device) # (batch,sequence,features)
            del x,y,subbatch

            batch_right_expanded = batch_unmasked_right.expand((batch_size,batch_size,batch_unmasked_right.shape[1],batch_unmasked_right.shape[2])) # (batch,batch,sequence,features)
            batch_right_embedded = model(batch_unmasked_right.clone()) # (batch, embedding)
            embedding_distance_squared = torch.square(batch_left_embedded.unsqueeze(1) - batch_right_embedded.unsqueeze(0)) # (batch,batch,embeddings)
            del batch_right_embedded
            embedding_distance_summed = torch.sum(embedding_distance_squared, dim=2) # (batch,batch) of sums
            del embedding_distance_squared
            embedding_distance_together = torch.cat([embedding_distance_summed, old_embedding_distance], dim=1) if old_embedding_distance is not None else embedding_distance_summed
            sequences_together = torch.cat([batch_right_expanded, old_sequences], dim=1) if old_sequences is not None else batch_right_expanded # (batch,batch+5,sequence,features)

            topk_embedding = torch.topk(embedding_distance_together, k=5, dim=1, largest=False, sorted=True) # (2,batch,5)
            topk_embedding_ind = topk_embedding[1]
            topk_embedding_val = topk_embedding[0]
            topk_sequences = torch.stack([sequences_together[i,idx] for i,idx in enumerate(topk_embedding_ind)]) # (batch,5,sequence,features)
            old_sequences = topk_sequences
            old_embedding_distance = topk_embedding_val

        batch_sequences_right = topk_sequences[:,:,:,df:dl] # (batch,5,sequence,digits)
        topk_square_elem_error = torch.mean(torch.square(batch_sequences_left - batch_sequences_right), dim=3) # (batch,5,sequence)
        topk_mean_square_error = torch.sqrt(torch.mean(topk_square_elem_error, dim=2)) # (batch, 5)

        top1_rmse = topk_mean_square_error[:,0] # (batch,)
        top3_rmse = torch.min(topk_mean_square_error[:,0:3],dim=1)[0] # (batch,)
        top5_rmse = torch.min(topk_mean_square_error[:,0:5],dim=1)[0] # (batch,)

        top1_rmse_mean = top1_rmse.mean().item()
        top3_rmse_mean = top3_rmse.mean().item()
        top5_rmse_mean = top5_rmse.mean().item()

        self.batch_valid_metrics["continuation/batch_top_k_rmse/top1"] = top1_rmse_mean
        self.batch_valid_metrics["continuation/batch_top_k_rmse/top3"] = top3_rmse_mean
        self.batch_valid_metrics["continuation/batch_top_k_rmse/top5"] = top5_rmse_mean
        self.epoch_valid_metrics["continuation/epoch_top_k_rmse/top1"] += top1_rmse_mean
        self.epoch_valid_metrics["continuation/epoch_top_k_rmse/top3"] += top3_rmse_mean
        self.epoch_valid_metrics["continuation/epoch_top_k_rmse/top5"] += top5_rmse_mean


class ContinuationRegressive(Task):
    def __init__(self, param, featuremap, dataset, testset):
        super().__init__("continuation regressive", param, featuremap, dataset, testset)
        self.epoch_train_metrics = {
            "continuation_reg/epoch/train_msle": 0
        }
        self.epoch_valid_metrics = {
            "continuation_reg/epoch/valid_msle": 0
        }
        self.loss = L.MeanSquaredError(self.param, self.featuremap)

    def get_model_head(self):
        from models.continuation import ContinuationRegressive
        return ContinuationRegressive(self.param)

    def prepare_batch(self, x, y):
        x_masked = x.clone()
        x_masked[:,-1] = 0
        x_masked[:,-1,self.featuremap.mask] = 1
        x[:,-1,self.featuremap.mask] = 1
        return x_masked, x[:,-1]

    def predict(self, x, y, model):
        y_pred = model(x)
        y_true = y
        return y_pred, y_true

    def loss_train(self, y_pred, y_true):
        log = self.featuremap.log
        y_true_log = y_true[:,log]
        log_loss = self.loss(y_pred, y_true_log)
        self.batch_train_metrics["continuation_reg/batch/train_msle"] = log_loss.item()
        self.epoch_train_metrics["continuation_reg/epoch/train_msle"] += log_loss.item()
        return log_loss

    def loss_valid(self, y_pred, y_true):
        log = self.featuremap.log
        y_true_log = y_true[:,log]
        log_loss = self.loss(y_pred, y_true_log)
        self.batch_valid_metrics["continuation_reg/batch/valid_msle"] = log_loss.item()
        self.epoch_valid_metrics["continuation_reg/epoch/valid_msle"] += log_loss.item()
        return log_loss

    def validate(self, x, y, y_pred, y_true, model):
        pass

