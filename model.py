from huggingface_hub import hf_hub_download
import torch
import torch.nn as nn


class ResBlock(nn.Module):
    def __init__(self, n_feats=64, res_scale=0.1):
        super().__init__()
        self.res_scale = res_scale
        self.body = nn.Sequential(
            nn.Conv2d(n_feats, n_feats, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(n_feats, n_feats, 3, padding=1),
        )

    def forward(self, x):
        return x + self.body(x) * self.res_scale

class EDSR_X1_Refiner(nn.Module):
    # predicts residual; final output will be x + residual
    def __init__(self, in_ch=1, out_ch=1, n_feats=64, n_resblocks=16, res_scale=0.1):
        super().__init__()
        self.head = nn.Conv2d(in_ch, n_feats, 3, padding=1)
        self.body = nn.Sequential(*[ResBlock(n_feats, res_scale) for _ in range(n_resblocks)])
        self.body_tail = nn.Conv2d(n_feats, n_feats, 3, padding=1)
        self.tail = nn.Conv2d(n_feats, out_ch, 3, padding=1)

    def forward(self, x):
        f = self.head(x)
        b = self.body(f)
        b = self.body_tail(b)
        f = f + b
        r = self.tail(f)
        return x + r


def load_pretrained_backbone(model, pretrained_state):
    """Load backbone weights from 2x EDSR into 1x Refiner, handling module. prefix and architecture mismatch."""
    ms = model.state_dict()

    # Unwrap nested dicts if needed
    if isinstance(pretrained_state, dict):
        for key in ["state_dict", "model", "params", "net"]:
            if key in pretrained_state and isinstance(pretrained_state[key], dict):
                pretrained_state = pretrained_state[key]
                break

    # Strip 'module.' prefix (DataParallel artifact)
    cleaned_state = {}
    for k, v in pretrained_state.items():
        if k.startswith("module."):
            k = k[7:]  # remove 'module.' prefix
        cleaned_state[k] = v

    # Load backbone layers (head, body, body_tail, tail)
    # Skip sub_mean/add_mean since our model doesn't have them
    matched = {}
    for k, v in cleaned_state.items():
        if k in ms and ms[k].shape == v.shape:
            matched[k] = v
            print(f"  Loaded: {k}")

    model.load_state_dict(matched, strict=False)
    print(f"\nLoaded {len(matched)}/{len(ms)} tensors (backbone transfer learning).")
    return model


if __name__ == "__main__":
    ckpt_path = hf_hub_download(
        repo_id="eugenesiow/edsr-base",
        filename="pytorch_model_2x.pt",
    )
    state = torch.load(ckpt_path, map_location="cpu")
    print(f"Checkpoint type: {type(state)}")
    print(f"Checkpoint keys (first 10): {list(state.keys())[:10]}\n")

    model = EDSR_X1_Refiner(in_ch=1, out_ch=1, n_feats=64, n_resblocks=16)
    print(f"Initialized model with {sum(p.numel() for p in model.parameters())} parameters.\n")
    
    print("Loading pretrained backbone from 2x EDSR:")
    model = load_pretrained_backbone(model, state)