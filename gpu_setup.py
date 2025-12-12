import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
gpu_dtypes = [torch.float16, torch.bfloat16]
