import torch
from data.settings import deftype, device

# mean absolute error of the predicted log features
def mae_of_logs(y_pred, y_true, featuremap):
    index = featuremap.log
    y_pred_log = y_pred[:,:,index]
    y_true_log = y_true[:,:,index]
    return torch.abs(y_pred_log - y_true_log).mean().item()

# relative error of the predicted log features
def re_of_logs(y_pred, y_true, featuremap):
    index = featuremap.log
    y_pred_log = y_pred[:,:,index]
    y_true_log = y_true[:,:,index]
    ae_of_logs = torch.abs(y_pred_log - y_true_log)
    rel_errors = torch.div(ae_of_logs, y_true_log + 1e-6)
    return torch.clamp(rel_errors, min=0, max=1).mean().item()

# accuracy of the predicted sign features
def accuracy_of_signs(y_pred, y_true, featuremap):
    index = featuremap.sign
    y_pred_sign = torch.sigmoid(y_pred[:,:,index]).round()
    y_true_sign = y_true[:,:,index]
    return ((y_pred_sign == y_true_sign).count_nonzero() / (y_true.shape[0] * y_true.shape[1])).item()

# mean absolute error of the predicted normalized features
def mae_of_normalized(y_pred, y_true, featuremap):
    index = featuremap.normalized
    y_pred_norm = y_pred[:,:,index]
    y_true_norm = y_true[:,:,index]
    return torch.abs(y_pred_norm - y_true_norm).mean().item()

# relative error of the predicted log features
def re_of_normalized(y_pred, y_true, featuremap):
    index = featuremap.normalized
    y_pred_norm = y_pred[:,:,index]
    y_true_norm = y_true[:,:,index]
    ae_of_norm = torch.abs(y_pred_norm - y_true_norm)
    rel_errors = torch.div(ae_of_norm, y_true_norm + 1e-6)
    return torch.clamp(rel_errors, min=0, max=1).mean().item()

#root-mean-squared-error RECREATED from the predicted log features
def rmse_from_logs(y_pred, y_true, featuremap):
    index = featuremap.log
    y_pred_log = y_pred[:,:,index]
    y_true_log = y_true[:,:,index]
    return torch.sqrt(torch.square(torch.pow(10,y_pred_log) - torch.pow(10,y_true_log)).mean()).item()

#root-mean-log-squared-error RECREATED from the predicted log features
def rmsle_from_logs(y_pred, y_true, featuremap):
    index = featuremap.log
    y_pred_log = y_pred[:,:,index]
    y_true_log = y_true[:,:,index]
    return torch.sqrt(torch.square(y_pred_log - y_true_log).mean()).item()

#root-mean-squared-error RECREATED from the predicted digits features
def rmse_from_digits(y_pred, y_true, featuremap):
    log = featuremap.log
    numbers_from_digits = torch.zeros((y_true.shape[0], y_true.shape[1]), device=device, dtype=deftype)
    begin = featuremap.digits_first
    end = featuremap.digits_last
    for i in range(begin, end+1):
        numbers_from_digits += y_pred[:,:,i] * (10**(i-begin))
    return torch.sqrt(torch.square(numbers_from_digits - torch.pow(10, y_true[:,:,log])).mean()).item()

#root-mean-log-squared-error RECREATED from the predicted digits features
def rmlse_from_digits(y_pred, y_true, featuremap):
    assert not torch.isnan(y_pred).any()
    assert not torch.isnan(y_true).any()
    log = featuremap.log
    numbers_from_digits = torch.zeros((y_true.shape[0], y_true.shape[1]), device=device, dtype=deftype)
    begin = featuremap.digits_first
    end = featuremap.digits_last
    for i in range(begin, end+1):
        numbers_from_digits += y_pred[:,:,i] * (10**(i-begin))
    idx = numbers_from_digits != 0.0
    numbers_from_digits[idx] = torch.log10(torch.abs(numbers_from_digits[idx]))  
    return torch.sqrt(torch.square(numbers_from_digits - y_true[:,:,log]).mean()).item()

#root-mean-squared-error RECREATED from the predicted normalized features
def rmse_from_normalized(y_pred, y_true, featuremap):
    log = featuremap.log
    norm = featuremap.normalized
    max_in_seq, max_in_seq_ind = torch.max(y_true[:,:,log], dim=1)
    y_pred_unnormalized = max_in_seq.unsqueeze(1) * y_pred[:,:,norm]
    return torch.sqrt(torch.square(y_pred_unnormalized - torch.pow(10, y_true[:,:,log])).mean()).item()

#root-mean-log-squared-error RECREATED from the predicted normalized features
def rmsle_from_normalized(y_pred, y_true, featuremap):
    log = featuremap.log
    norm = featuremap.normalized
    max_in_seq, max_in_seq_ind = torch.max(y_true[:,:,log], dim=1)
    y_pred_unnormalized = max_in_seq.unsqueeze(1) * y_pred[:,:,norm]
    idx = y_pred_unnormalized != 0
    y_pred_unnormalized[idx] = torch.log10(torch.abs(y_pred_unnormalized[idx]))
    return torch.sqrt(torch.square(y_pred_unnormalized - y_true[:,:,log]).mean()).item()