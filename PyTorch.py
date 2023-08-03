import torch
import numpy as np

# #directly form data
# data = [[2,3,4,5],[6,7,8,9]]
# x_data = torch.tensor(data)

# print(x_data)

# #from numpy
# np_array = np.array(data)
# x_np = torch.from_numpy(np_array)

# print(x_np)

tensor = torch.rand(3,4)

print(f"Shape of tensor: {tensor.shape}")
print(f"Datatype of tensor: {tensor.dtype}")
print(f"Device tensor is stored on: {tensor.device}")

device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)
print(f"Using {device} device")