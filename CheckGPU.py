import torch 

print(torch.cuda.is_available(),
torch.cuda.device_count(),
torch.cuda.current_device(),
# torch.cuda.device(2),
# torch.cuda.get_device_name(2)
)