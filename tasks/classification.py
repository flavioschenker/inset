import torch
from tasks.task import Task
from data.settings import device, deftype
import benchmark.loss as L

class ClassificationRegressive(Task):
    def __init__(self, param, featuremap, dataset, testset):
        super().__init__("classification regressive", param, featuremap, dataset, testset)
        self.class_weights = self.dataset.class_weights
        self.loss = L.MultiLabelLoss(self.param, self.featuremap, self.class_weights)
        self.classes = ["polynomial","periodic","exponential","trigonometric","modulo","prime","finite"]
        self.epoch_valid_metrics = {
            "polynomial": {"TP":0, "TN":0, "FP":0, "FN":0},
            "periodic": {"TP":0, "TN":0, "FP":0, "FN":0},
            "exponential": {"TP":0, "TN":0, "FP":0, "FN":0},
            "trigonometric": {"TP":0, "TN":0, "FP":0, "FN":0},
            "modulo": {"TP":0, "TN":0, "FP":0, "FN":0},
            "prime": {"TP":0, "TN":0, "FP":0, "FN":0},
            "finite": {"TP":0, "TN":0, "FP":0, "FN":0},
        }

    def get_model_head(self):
        from models.classification import ClassificationRegressive
        return ClassificationRegressive(self.param)

    def prepare_batch(self, x, y):
        classes = y[:,0:7]
        return x, classes

    def predict(self, x, y, model):
        y_pred = model(x)
        y_true = y
        return y_pred, y_true

    def loss_train(self, y_pred, y_true):
        return self.loss(y_pred, y_true)

    def loss_valid(self, y_pred, y_true):
        return self.loss(y_pred, y_true)

    def validate(self, x, y, y_pred, y_true, model):
        y_pred_sig = torch.sigmoid(y_pred)
        del x,y,y_pred
        confidence_threshold = self.param["classification_confidence_threshold"]
        y_pred_classes = torch.zeros(y_pred_sig.shape, device=device)
        y_pred_classes[y_pred_sig >= confidence_threshold] = 1
        del y_pred_sig

        for i, c in enumerate(self.classes):
            true_positive = torch.count_nonzero((y_pred_classes[:,i] == 1) & (y_true[:,i] == 1)).item()
            true_negative = torch.count_nonzero((y_pred_classes[:,i] == 0) & (y_true[:,i] == 0)).item()
            false_positive = torch.count_nonzero((y_pred_classes[:,i] == 1) & (y_true[:,i] == 0)).item()
            false_negative = torch.count_nonzero((y_pred_classes[:,i] == 0) & (y_true[:,i] == 1)).item()
            self.epoch_valid_metrics[c]["TP"] += true_positive
            self.epoch_valid_metrics[c]["TN"] += true_negative
            self.epoch_valid_metrics[c]["FP"] += false_positive
            self.epoch_valid_metrics[c]["FN"] += false_negative

            self.batch_valid_metrics["classification/TP/"+str(c)] = true_positive
            self.batch_valid_metrics["classification/TN/"+str(c)] = true_negative
            self.batch_valid_metrics["classification/FP/"+str(c)] = false_positive
            self.batch_valid_metrics["classification/FN/"+str(c)] = false_negative

    def evaluate_valid_epoch(self, epoch, batch_count):
        result = {}
        global_tp = 0
        global_fp = 0
        global_fn = 0
        global_precision = 0
        global_recall = 0
        for c in self.classes:
            tp = self.epoch_valid_metrics[c]["TP"]
            tn = self.epoch_valid_metrics[c]["TN"]
            fp = self.epoch_valid_metrics[c]["FP"]
            fn = self.epoch_valid_metrics[c]["FN"]
            class_accuracy = (tp+tn) / (tp+tn+fp+fn) if (tp+tn+fp+fn)!=0 else 0
            class_precision = tp / (fp+tp) if (fp+tp) != 0 else 1
            class_recall = tp / (fn+tp) if (fn+tp) != 0 else 1
            class_f1 = (2*class_precision*class_recall) / (class_precision+class_recall) if (class_precision+class_recall) != 0 else 0
            global_tp += tp
            global_fp += fp
            global_fn += fn
            global_precision += class_precision
            global_recall += class_recall
            
            result["classification/accuracy/"+str(c)] = class_accuracy
            result["classification/precision/"+str(c)] = class_precision
            result["classification/recall/"+str(c)] = class_recall
            result["classification/f1_score/"+str(c)] = class_f1

        global_precision /= len(self.classes)
        global_recall /= len(self.classes)

        macro_average_f1 = (2*global_precision*global_recall) / (global_precision+global_recall) if (global_precision+global_recall) != 0 else 0
        micro_precision = global_tp / (global_fp + global_tp) if (global_fp+global_tp) != 0 else 0
        micro_recall = global_tp / (global_fn + global_tp) if (global_fn+global_tp) != 0 else 0
        micro_average_f1 = (2*micro_precision*micro_recall) / (micro_precision+micro_recall) if (micro_precision+micro_recall) != 0 else 0
        result["classification/f1_score/macro_average"] = macro_average_f1
        result["classification/f1_score/micro_average"] = micro_average_f1
        for key in self.epoch_valid_metrics:
            self.epoch_valid_metrics[key] = {"TP":0, "TN":0, "FP":0, "FN":0}
        return result