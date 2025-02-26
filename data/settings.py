import torch

deftype = torch.float64
torch.set_default_dtype(deftype)
device = "cuda" if torch.cuda.is_available() else "cpu"

