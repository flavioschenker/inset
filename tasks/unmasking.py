import torch
from tasks.task import Task
from data.settings import device, deftype
import benchmark.loss as L
import benchmark.metrics as M

class UnmaskingRegressive(Task):
    def __init__(self, param, featuremap, dataset, testset):
        super().__init__("unmasking regressive", param, featuremap, dataset, testset)
        self.norm_loss = L.MSENormLoss(self.param, self.featuremap)
        self.sign_loss = L.SIGNLoss(self.param, self.featuremap)
        self.log_loss = L.ModLogLoss(self.param, self.featuremap)
        self.epoch_train_metrics = {
            "unmasking/epoch_train/mse of norm": 0,
            "unmasking/epoch_train/bce of sign": 0,
            "unmasking/epoch_train/mod log": 0,
        }
        self.epoch_valid_metrics = {
            "unmasking/epoch_valid/mse of norm": 0,
            "unmasking/epoch_valid/bce of sign": 0,
            "unmasking/epoch_valid/mod log": 0,
            "unmasking/epoch_valid/mae of logs": 0,
            "unmasking/epoch_valid/re of logs": 0,
            "unmasking/epoch_valid/mae of norms": 0,
            "unmasking/epoch_valid/re of norms": 0,
            "unmasking/epoch_valid/accuracy of signs": 0,
        }

    def get_model_head(self):
        from models.unmasking import UnmaskingRegressive
        return UnmaskingRegressive(self.param)

    def prepare_batch(self, x, y):
        x_masked = x.clone()
        mask_prob = self.param["unmasking_mask_probability"]
        mask = torch.empty(size=(x.shape[0],x.shape[1],1), dtype=deftype, device=device).uniform_(0,1)
        mask = (mask < mask_prob).to(dtype=deftype)
        x_masked *= (1-mask)
        x[:,:,self.featuremap.mask] = mask[:,:,0]
        x_masked[:,:,self.featuremap.mask] = mask[:,:,0]           
        return x_masked, x

    def predict(self, x, y, model):
        y_pred = model(x)
        y_true = y
        return y_pred, y_true

    def loss_train(self, y_pred, y_true):
        mse_of_normalized = self.norm_loss(y_pred, y_true)
        bce_of_sign = self.sign_loss(y_pred, y_true)
        mod_log = self.log_loss(y_pred, y_true)

        self.batch_train_metrics["unmasking/batch_train/mse of norm"] = mse_of_normalized.item()
        self.batch_train_metrics["unmasking/batch_train/bce of sign"] = bce_of_sign.item()
        self.batch_train_metrics["unmasking/batch_train/mod log"] = mod_log.item()

        self.epoch_train_metrics["unmasking/epoch_train/mse of norm"] += mse_of_normalized.item()
        self.epoch_train_metrics["unmasking/epoch_train/bce of sign"] += bce_of_sign.item()
        self.epoch_train_metrics["unmasking/epoch_train/mod log"] += mod_log.item()

        loss = mse_of_normalized + bce_of_sign + mod_log
        return loss

    def loss_valid(self, y_pred, y_true):
        mse_of_normalized = self.norm_loss(y_pred, y_true)
        bce_of_sign = self.sign_loss(y_pred, y_true)
        mod_log = self.log_loss(y_pred, y_true)

        self.batch_valid_metrics["unmasking/batch_valid/mse of norm"] = mse_of_normalized.item()
        self.batch_valid_metrics["unmasking/batch_valid/bce of sign"] = bce_of_sign.item()
        self.batch_valid_metrics["unmasking/batch_valid/mod log"] = mod_log.item()
        self.epoch_valid_metrics["unmasking/epoch_valid/mse of norm"] += mse_of_normalized.item()
        self.epoch_valid_metrics["unmasking/epoch_valid/bce of sign"] += bce_of_sign.item()
        self.epoch_valid_metrics["unmasking/epoch_valid/mod log"] += mod_log.item()
        loss = mse_of_normalized + bce_of_sign + mod_log
        return loss

    def validate(self, x, y, y_pred, y_true, model):
        mea_of_logs = M.mae_of_logs(y_pred, y_true, self.featuremap)
        re_of_logs = M.re_of_logs(y_pred, y_true, self.featuremap)
        mae_of_normalized = M.mae_of_normalized(y_pred, y_true, self.featuremap)
        re_of_normalized = M.re_of_normalized(y_pred, y_true, self.featuremap)
        accuracy_of_signs = M.accuracy_of_signs(y_pred, y_true, self.featuremap)
        del y_pred,y_true
        self.batch_valid_metrics["unmasking/batch_valid/mae of logs"] = mea_of_logs
        self.batch_valid_metrics["unmasking/batch_valid/re of logs"] = re_of_logs
        self.batch_valid_metrics["unmasking/batch_valid/mae of norms"] = mae_of_normalized
        self.batch_valid_metrics["unmasking/batch_valid/re of norms"] = re_of_normalized
        self.batch_valid_metrics["unmasking/batch_valid/accuracy of signs"] = accuracy_of_signs

        self.epoch_valid_metrics["unmasking/epoch_valid/mae of logs"] += mea_of_logs
        self.epoch_valid_metrics["unmasking/epoch_valid/re of logs"] += re_of_logs
        self.epoch_valid_metrics["unmasking/epoch_valid/mae of norms"] += mae_of_normalized
        self.epoch_valid_metrics["unmasking/epoch_valid/re of norms"] += re_of_normalized
        self.epoch_valid_metrics["unmasking/epoch_valid/accuracy of signs"] += accuracy_of_signs


class UnmaskingContrastive(Task):
    def __init__(self, param, featuremap, dataset, testset):
        super().__init__("unmasking contrastive", param, featuremap, dataset, testset)
        self.epoch_train_metrics = {
            "unmasking/epoch_train/agreement loss": 0,
            "unmasking/epoch_train/margin loss": 0,
        }
        self.epoch_valid_metrics = {
            "unmasking/epoch_top_k_rmse/top1": 0,
            "unmasking/epoch_top_k_rmse/top3": 0,
            "unmasking/epoch_top_k_rmse/top5": 0,
            "unmasking/epoch_valid/agreement loss": 0,
            "unmasking/epoch_valid/margin loss": 0,
        }

    def get_model_head(self) -> torch.nn.Module:
        from models.unmasking import UnmaskingContrastive
        return UnmaskingContrastive(self.param)

    def prepare_batch(self, x, y):
        x_masked = x.clone()
        mask_prob = self.param["unmasking_mask_probability"]
        mask = torch.empty(size=(x.shape[0],x.shape[1],1), dtype=deftype, device=device).uniform_(0,1)
        mask = (mask < mask_prob).to(dtype=deftype)
        x_masked *= (1-mask)
        x[:,:,self.featuremap.mask] = mask[:,:,0]
        x_masked[:,:,self.featuremap.mask] = mask[:,:,0]           
        return x_masked, x

    def predict(self, x, y, model):
        x_embeddings_masked = model(x)
        x_embeddings = model(y) # (batch, sequence, features)
        del x,y
        # generate a balanced set of indices
        batch_size = x_embeddings_masked.shape[0]
        arange = torch.arange(0, batch_size, dtype=torch.long, requires_grad=False, device=device)
        pos_pair_indices = torch.cat([arange.unsqueeze(1), arange.unsqueeze(1)], dim=1)
        neg_pair_indices = torch.combinations(arange, 2, with_replacement=False)
        neg_pair_indices_samples = neg_pair_indices[torch.ones(neg_pair_indices.shape[0], device=device).multinomial(batch_size, replacement=False)]
        pair_indices = torch.cat([pos_pair_indices, neg_pair_indices_samples], dim=0)

        x_embeddings_masked = x_embeddings_masked.unsqueeze(1) # (batch, 1, sequence, features)
        x_embeddings = x_embeddings.unsqueeze(1)
        
        masked_masked_pairs = torch.cat([x_embeddings_masked[pair_indices[:,0]], x_embeddings_masked[pair_indices[:,1]]], dim=1)
        masked_masked_agreement = torch.cat([torch.ones(batch_size, device=device), torch.zeros(batch_size, device=device)], dim=0)
        masked_unmasked_pairs = torch.cat([x_embeddings_masked[pair_indices[:,0]], x_embeddings[pair_indices[:,1]]], dim=1)
        masked_unmasked_agreement = torch.cat([torch.ones(batch_size, device=device), torch.zeros(batch_size, device=device)], dim=0)
        unmasked_masked_pairs = torch.cat([x_embeddings[pair_indices[:,0]], x_embeddings_masked[pair_indices[:,1]]], dim=1)
        unmasked_masked_agreement = torch.cat([torch.ones(batch_size, device=device), torch.zeros(batch_size, device=device)], dim=0)
        unmasked_unmasked_pairs = torch.cat([x_embeddings[pair_indices[:,0]], x_embeddings[pair_indices[:,1]]], dim=1)
        unmasked_unmasked_agreement = torch.cat([torch.ones(batch_size, device=device), torch.zeros(batch_size, device=device)], dim=0)
        
        pairs = torch.cat([masked_masked_pairs, masked_unmasked_pairs, unmasked_masked_pairs,unmasked_unmasked_pairs], dim=0)
        agreement = torch.cat([masked_masked_agreement, masked_unmasked_agreement, unmasked_masked_agreement,unmasked_unmasked_agreement], dim=0)

        del x_embeddings, x_embeddings_masked, arange, pair_indices, pos_pair_indices, neg_pair_indices, neg_pair_indices_samples
        del masked_masked_pairs, masked_unmasked_pairs, unmasked_masked_pairs
        del masked_masked_agreement, masked_unmasked_agreement, unmasked_masked_agreement

        return (pairs, agreement)

    def loss_train(self, y_pred, y_true):
        margin = self.param["unmasking_contrastive_margin"]
        embedding_pairs = y_pred
        pairs_agreement = y_true
        del y_pred, y_true
        embedding_distance_squared = torch.maximum(torch.tensor([ 1e-14 ], device=device), torch.square(embedding_pairs[:,0] - embedding_pairs[:,1]))
        embedding_distance_summed = torch.sum(embedding_distance_squared.flatten(start_dim=1), dim=1)
        del embedding_distance_squared
        embedding_distance = torch.sqrt(embedding_distance_summed)
        agreement_loss = pairs_agreement*embedding_distance_summed
        del embedding_distance_summed
        margin_loss = torch.square((1-pairs_agreement)*torch.maximum(margin - embedding_distance, torch.tensor([0], device=device)))
        del embedding_distance
        contrastive_loss = torch.mean(agreement_loss + margin_loss)
        loss = contrastive_loss
        self.batch_train_metrics["unmasking/batch_train/agreement loss"] = agreement_loss.mean().item()
        self.batch_train_metrics["unmasking/batch_train/margin loss"] = margin_loss.mean().item()
        self.epoch_train_metrics["unmasking/epoch_train/agreement loss"] += agreement_loss.mean().item()
        self.epoch_train_metrics["unmasking/epoch_train/margin loss"] += margin_loss.mean().item()
        return loss

    def loss_valid(self, y_pred, y_true):
        margin = self.param["unmasking_contrastive_margin"]
        embedding_pairs = y_pred
        pairs_agreement = y_true
        del y_pred, y_true
        embedding_distance_squared = torch.maximum(torch.tensor([ 1e-14 ], device=device), torch.square(embedding_pairs[:,0] - embedding_pairs[:,1]))
        embedding_distance_summed = torch.sum(embedding_distance_squared.flatten(start_dim=1), dim=1)
        del embedding_distance_squared
        embedding_distance = torch.sqrt(embedding_distance_summed)
        agreement_loss = pairs_agreement*embedding_distance_summed
        del embedding_distance_summed
        margin_loss = torch.square((1-pairs_agreement)*torch.maximum(margin - embedding_distance, torch.tensor([0], device=device)))
        del embedding_distance
        contrastive_loss = torch.mean(agreement_loss + margin_loss)
        loss = contrastive_loss
        self.batch_valid_metrics["unmasking/batch_valid/agreement loss"] = agreement_loss.mean().item()
        self.batch_valid_metrics["unmasking/batch_valid/margin loss"] = margin_loss.mean().item()
        self.epoch_valid_metrics["unmasking/epoch_valid/agreement loss"] += agreement_loss.mean().item()
        self.epoch_valid_metrics["unmasking/epoch_valid/margin loss"] += margin_loss.mean().item()
        return loss

    def validate(self, x, y, y_pred, y_true, model):
        df = self.featuremap.digits_first
        dl = self.featuremap.digits_last+1
        batch_size = len(y)
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

        self.batch_valid_metrics["unmasking/batch_top_k_rmse/top1"] = top1_rmse_mean
        self.batch_valid_metrics["unmasking/batch_top_k_rmse/top3"] = top3_rmse_mean
        self.batch_valid_metrics["unmasking/batch_top_k_rmse/top5"] = top5_rmse_mean
        self.epoch_valid_metrics["unmasking/epoch_top_k_rmse/top1"] += top1_rmse_mean
        self.epoch_valid_metrics["unmasking/epoch_top_k_rmse/top3"] += top3_rmse_mean
        self.epoch_valid_metrics["unmasking/epoch_top_k_rmse/top5"] += top5_rmse_mean