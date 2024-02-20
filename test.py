import torch

print(torch.__version__)

if torch.cuda.is_available():
    print("CUDA is available. Details:")
    print(f"Number of GPUs: {torch.cuda.device_count()}")
    print(f"Current CUDA Device ID: {torch.cuda.current_device()}")
    print(f"Current CUDA Device Name: {torch.cuda.get_device_name(torch.cuda.current_device())}")
else:
    print("CUDA is not available.")
