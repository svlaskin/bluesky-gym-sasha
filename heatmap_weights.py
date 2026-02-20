import torch
import matplotlib.pyplot as plt
import seaborn as sns

pt_file = "sac_unc_cr_att_noise_posonly_20/actor.pt"
attn_tensor = torch.load(pt_file).cpu().numpy()
shape = attn_tensor.shape
print(f"Attention tensor shape: {shape}")

# create default labels if you don't have agent names
if len(shape) == 2:
    num_queries, num_keys = shape
    labels = [f"A{i}" for i in range(num_queries)]
    plt.figure(figsize=(6,5))
    sns.heatmap(attn_tensor, annot=True if num_queries<=10 else False,
                xticklabels=labels, yticklabels=labels, cmap="viridis")
    plt.xlabel("Key")
    plt.ylabel("Query")
    plt.title("Attention Weights")
    plt.tight_layout()
    plt.show()

elif len(shape) == 3:
    num_heads, num_queries, num_keys = shape
    labels = [f"A{i}" for i in range(num_queries)]
    fig, axs = plt.subplots(1, num_heads, figsize=(5*num_heads, 4))
    if num_heads == 1:
        axs = [axs]
    for h in range(num_heads):
        sns.heatmap(attn_tensor[h], annot=True if num_queries<=10 else False,
                    xticklabels=labels, yticklabels=labels, cmap="viridis", ax=axs[h], cbar=h==0)
        axs[h].set_title(f"Head {h}")
        axs[h].set_xlabel("Key")
        axs[h].set_ylabel("Query")
    plt.tight_layout()
    plt.show()

else:
    raise ValueError(f"Unsupported attention tensor shape: {shape}")
