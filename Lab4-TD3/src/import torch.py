import torch
print(torch.cuda.device_count())  # 查看可用 GPU 數量
print(torch.cuda.get_device_name(0))  # 查看第一張 GPU 名稱12