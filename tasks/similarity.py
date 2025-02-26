import numpy
import torch
from tasks.task import Task
from data.settings import device, deftype

class SimilarityContrastive(Task):
    def __init__(self, param, featuremap, dataset, testset):
        super().__init__("similarity contrastive", param, featuremap, dataset, testset)
        self.epoch_train_metrics = {
            "similarity/epoch_train/agreement loss": 0,
            "similarity/epoch_train/margin loss": 0,
        }
        self.epoch_valid_metrics = {
            "similarity/epoch_top_k_accuracy/top1": 0,
            "similarity/epoch_top_k_accuracy/top3": 0,
            "similarity/epoch_top_k_accuracy/top5": 0,
            "similarity/epoch_valid/agreement loss": 0,
            "similarity/epoch_valid/margin loss": 0,
        }

    def get_model_head(self):
        from models.similarity import SimilarityContrastive
        return SimilarityContrastive(self.param)

    def prepare_batch(self, x, y):
        return x, y

    def predict(self, x, y, model):
        batch_size = y.shape[0]
        x_embedded = model(x)
        groups = y.shape[1]
        groups_indices = torch.argwhere(y.transpose(0,1))
        del x,y
        pos_pairs_stack = []
        neg_pairs_stack = []
        for group in range(groups):
            pos_group_indices = groups_indices[groups_indices[:,0] == group][:,1]
            if pos_group_indices.shape[0] < 1:
                continue
            neg_group_indices = groups_indices[groups_indices[:,0] != group][:,1]
            pos_pairs = torch.combinations(pos_group_indices, 2, with_replacement=False)
            neg_pairs = torch.stack(torch.meshgrid([pos_group_indices, neg_group_indices], indexing="ij")).transpose(0,2).reshape(-1,2)
            pos_pairs_stack.append(pos_pairs)
            neg_pairs_stack.append(neg_pairs)

        pos_pairs_stack = torch.cat(pos_pairs_stack)
        neg_pairs_stack = torch.cat(neg_pairs_stack)
        pos_pairs_length = pos_pairs_stack.shape[0]
        neg_pairs_length = neg_pairs_stack.shape[0]
        pos_n = self.param["similarity_pos_pairs_batch_multiply"]
        neg_n = self.param["similarity_neg_paris_batch_multiply"]
        pos_mask = numpy.random.choice(pos_pairs_length, min(pos_pairs_length,pos_n*batch_size), replace=False)
        neg_mask = numpy.random.choice(neg_pairs_length, min(neg_pairs_length,neg_n*batch_size), replace=False)
        agreement = torch.cat([torch.ones(min(pos_pairs_length,pos_n*batch_size), device=device), torch.zeros(min(neg_pairs_length,neg_n*batch_size), device=device)])
        pairs_indices = torch.cat([pos_pairs_stack[pos_mask], neg_pairs_stack[neg_mask]])
        del pos_pairs_stack, neg_pairs_stack
        assert agreement.shape[0] == pairs_indices.shape[0]
        pairs = torch.stack([x_embedded[pairs_indices[:,0]], x_embedded[pairs_indices[:,1]]],dim=1)
        del x_embedded, pairs_indices

        return (pairs, agreement)
        
    def loss_train(self, y_pred, y_true):
        margin = self.param["similarity_contrastive_margin"]
        pairs = y_pred
        agreement_mask = y_true.to(torch.bool)
        margin_mask = ~agreement_mask
        del y_pred, y_true
        embedding_distance_squared = torch.clamp(torch.square(pairs[:,0] - pairs[:,1]), min=1e-14)
        del pairs
        embedding_distance_summed = torch.sum(embedding_distance_squared,dim=1)
        del embedding_distance_squared
        margin_loss = torch.square(torch.clamp(margin - embedding_distance_summed[margin_mask], min=0.0,max=None))
        agreement_loss = embedding_distance_summed[agreement_mask]
        contrastive_loss = agreement_loss.sum() + margin_loss.sum()
        loss = contrastive_loss
        self.batch_train_metrics["similarity/batch_train/agreement loss"] = agreement_loss.sum().item()
        self.batch_train_metrics["similarity/batch_train/margin loss"] = margin_loss.sum().item()
        self.epoch_train_metrics["similarity/epoch_train/agreement loss"] += agreement_loss.sum().item()
        self.epoch_train_metrics["similarity/epoch_train/margin loss"] += margin_loss.sum().item()
        return loss

    def loss_valid(self, y_pred, y_true):
        margin = self.param["similarity_contrastive_margin"]
        pairs = y_pred
        agreement_mask = y_true.to(torch.bool)
        margin_mask = ~agreement_mask
        del y_pred, y_true
        embedding_distance_squared = torch.clamp(torch.square(pairs[:,0] - pairs[:,1]), min=1e-14)
        del pairs
        embedding_distance_summed = torch.sum(embedding_distance_squared,dim=1)
        del embedding_distance_squared
        margin_loss = torch.square(torch.clamp(margin - embedding_distance_summed[margin_mask], min=0.0,max=None))
        agreement_loss = embedding_distance_summed[agreement_mask]
        contrastive_loss = agreement_loss.sum() + margin_loss.sum()
        loss = contrastive_loss
        self.batch_valid_metrics["similarity/batch_valid/agreement loss"] = agreement_loss.sum().item()
        self.batch_valid_metrics["similarity/batch_valid/margin loss"] = margin_loss.sum().item()
        self.epoch_valid_metrics["similarity/epoch_valid/agreement loss"] += agreement_loss.sum().item()
        self.epoch_valid_metrics["similarity/epoch_valid/margin loss"] += margin_loss.sum().item()
        return loss

    def validate(self, x, y, y_pred, y_true, model):
        left_embedded = model(x)
        batch_size = x.shape[0]
        left_classes = y.unsqueeze(1) # (L-batch, 1, classes)
        del x,y

        search_dataloader = torch.utils.data.DataLoader(self.dataset, batch_size=batch_size, shuffle=True)
        old_distances = None
        old_classes = None
        for subbatch in search_dataloader:
            x,y = subbatch
            right_embedded = model(x.to(device))
            right_classes = y.to(device).expand((batch_size, y.shape[0], y.shape[1])) # (L-batch, R-batch, classes)
            del x,y,subbatch

            embedding_distance_squared = torch.square(left_embedded.unsqueeze(1) - right_embedded.unsqueeze(0)) # (L-batch, R-batch, embedding)
            embedding_distance_summed = torch.sum(embedding_distance_squared, dim=2) # (L-batch, R-batch)
            if old_distances is not None:
                embedding_distance_together = torch.cat([embedding_distance_summed, old_distances], dim=1) # (L-batch, R-batch+5)
                classes_together = torch.cat([right_classes, old_classes], dim=1) # (L-batch, R-batch+5, classes)
            else:
                embedding_distance_together = embedding_distance_summed # (L-batch, R-batch)
                classes_together = right_classes # (L-batch, R-batch, classes)
            
            topk_val, topk_ind = torch.topk(embedding_distance_together, k=5, dim=1, largest=False, sorted=True) # (L-batch, 5)
            old_distances = topk_val
            old_classes = torch.stack([classes_together[Lid,Rids] for Lid,Rids in enumerate(topk_ind)]) # (L-batch, 5, classes)

        topk_classes = old_classes

        hits_all_classes = left_classes == topk_classes # (L-batch, 5, classes)
        all_classes = torch.all(hits_all_classes, dim=2) # (L-batch, 5)

        top1_hits = all_classes[:,0] # (batch,)
        top3_hits = torch.max(all_classes[:,0:3],dim=1)[0] # (batch,)
        top5_hits = torch.max(all_classes[:,0:5],dim=1)[0] # (batch,)

        top1_accuracy = top1_hits.sum() / batch_size
        top3_accuracy = top3_hits.sum() / batch_size
        top5_accuracy = top5_hits.sum() / batch_size


        self.batch_valid_metrics["similarity/batch_top_k_accuracy/top1"] = top1_accuracy
        self.batch_valid_metrics["similarity/batch_top_k_accuracy/top3"] = top3_accuracy
        self.batch_valid_metrics["similarity/batch_top_k_accuracy/top5"] = top5_accuracy
        self.epoch_valid_metrics["similarity/epoch_top_k_accuracy/top1"] += top1_accuracy
        self.epoch_valid_metrics["similarity/epoch_top_k_accuracy/top3"] += top3_accuracy
        self.epoch_valid_metrics["similarity/epoch_top_k_accuracy/top5"] += top5_accuracy
