# import torch
# from config import GPTConfig as Config
# from Model.gpt import GPTModel

# params = Config()

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# print("Using device:", device)
# if device.type == "cuda":
#     print("GPU:", torch.cuda.get_device_name(0))

# # model = GPTModel(
# #     params.vocab_size,
# #     params.n_embd,
# #     params.block_size,
# #     params.n_head,
# #     params.n_layer
# # ).to(device)

# # dummy_inputs = torch.randint(
# #     low=0,
# #     high=params.vocab_size,
# #     size=(params.batch_size, params.block_size),
# #     dtype=torch.long,
# #     device=device
# # )

# # with torch.no_grad():
# #     output = model(dummy_inputs)

# # print("Output shape:", output.shape)
# # print("Output device:", output.device)

import torch
import sys

print("Python executable:", sys.executable)
print("PyTorch version:", torch.__version__)
print("PyTorch CUDA version:", torch.version.cuda)
print("CUDA available:", torch.cuda.is_available())
print("CUDA device count:", torch.cuda.device_count())

if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))