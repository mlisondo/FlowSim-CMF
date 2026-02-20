import torch

import numpy as np
import os

import matplotlib.pyplot as plt


def probe(o, name=None):
    obj = type(o)
    header = f"Object '{name}'"
    # print(f"\n{header}: {obj.__module__}.{obj.__name__}")
    print("\n")
    print(f"beginning of {header}".ljust(75, "="))

    print(f"{obj.__module__}.{obj.__name__}")

    # NumPy-style introspection
    if hasattr(o, 'shape'):
        print(f"shape: {o.shape}")
    if hasattr(o, 'ndim'):
        print(f"ndim: {o.ndim}")
    if hasattr(o, 'dtype'):
        print(f"dtype: {o.dtype}")

    # size attribute
    if hasattr(o, 'size') and not callable(o.size):
        print(f"size: {o.size}")

    # Pythonic length
    try:
        print(f"len: {len(o)}")
    except Exception:
        pass

    # Recursive descent into lists
    try:
        if isinstance(o, (list, tuple)):
            for idx, item in enumerate(o):
                print(f"".center(75, "~"))
                probe(item, f"{name}[{idx}]")
    except Exception:
        pass

    # PyTorch tensors
    if isinstance(o, torch.Tensor):
        print(f"shape: {tuple(o.size())}")
        print(f"dtype: {o.dtype}")
        print(f"numel: {o.numel()}")

        print(f"shape: {tuple(o.size())}")
        print(f"dtype: {o.dtype}")
        print(f"numel: {o.numel()}")
        print(f"device: {o.device}")

    print(f"end of {header}: {obj.__module__}.{obj.__name__}".ljust(75, "="))
    print("\n")


file_path = "data/gen_ttbar_400k_final.npy"
    
if os.path.exists(file_path):
    data = np.load(file_path)

    probe(data, name="ttbar events")
else:
    raise Exception("bruh, file_path not properly specified")
print(data[:200, 22])

# raise Exception("break")


features = {
    0: "pt",
    1: "eta",
    2: "phi",
    3: "m",
    4: "flavour",
    5: "btag",
    6: "recoPt",
    7: "recoPhi",
    8: "recoEta",
    9: "muonsInJet",
    10: "recoNConstituents",
    11: "nef",
    12: "nhf",
    13: "cef",
    14: "chf",
    15: "qgl",
    16: "jetId", 
    17: "ncharged",
    18: "nneutral",
    19: "ctag",
    20: "nSV",
    21: "recoMass",
    22: "extra",
}

# for i in range(data.shape[1]):

#     values = data[:, i]

#     values = values[~np.isnan(values)] # remove nans before doing anything. this is something the code doesnt do (removes nans and ininites after dividing)
    
#     mean = np.mean(values)
#     std = np.std(values)
#     min_val = np.min(values)
#     max_val = np.max(values)
#     median = np.median(values)
#     count = len(values)
#     n_unique = len(np.unique(values))

#     fig, (ax_hist, ax_text) = plt.subplots(
#         2, 1, 
#         figsize=(6, 5),
#         gridspec_kw={"height_ratios": [4, 1]}
#     )

#     ax_hist.hist(values, bins=100)
#     ax_hist.set_title(f"{features[i]}, idx={i}")
#     ax_hist.set_ylabel("Count")

#     ax_text.axis("off")
#     stats_text = (
#         f"Count: {count}\n"
#         f"Mean: {mean:.4f}\n"
#         f"Std: {std:.4f}\n"
#         f"Median: {median:.4f}\n"
#         f"Min: {min_val:.4f}\n"
#         f"Max: {max_val:.4f}\n"
#         f"Unique values: {n_unique}"
#     )
#     ax_text.text(0.01, 0.5, stats_text, fontsize=10, va="center")

#     ax_hist.axvline(mean, color="red", linestyle="--", label="Mean")
#     ax_hist.axvline(median, color="green", linestyle="--", label="Median")
#     ax_hist.legend()

#     plt.tight_layout()
#     plt.show()


plt.plot(np.linspace(0, 100, 100), data[:100, 22])
plt.show()