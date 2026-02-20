import torch

# Load the checkpoint
checkpoint = torch.load("sac_unc_cr_att_noise_posonly_20/qf_target.pt", map_location="cpu")

# Print the type and keys (if it's a dict)
print(type(checkpoint))

if isinstance(checkpoint, dict):
    print("Keys in checkpoint:", checkpoint.keys())