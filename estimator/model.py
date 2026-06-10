"""Residual CNN channel estimator (ChannelNet-style refinement).

INPUT CONTRACT: the 2-channel (re/im) LS linear-interpolated grid — NOT the
sparse pilot grid. The network predicts a residual correction: out = x + f(x).
Sparse-input training was tried first and lost to LS by ~1 dB at 1600 Hz;
residual refinement over the LS grid is the standard, much easier task.
(Downstream: the ONNX/TensorRT engine in Prompts 3-5 takes the LS grid as
input; LS interp is a cheap fixed pre-processing stage outside the network.)

Parameter count vs latency: 6 conv3x3 layers at width 48 -> ~85k params,
~120 MFLOPs per (14, 64) slot. Receptive field 13x13 — chosen to span the
9-symbol gap between pilot symbols [2, 11] so every data RE sees both pilot
symbols. Small enough that an FP16 TensorRT engine costs single-digit us on
a datacenter GPU (measured in Prompt 4/5, not asserted here) — a fraction
of the 35.7 us symbol budget, vs >500k-param U-Nets that blow it.
"""
import torch
import torch.nn as nn


class ChannelCNN(nn.Module):
    def __init__(self, width=48, depth=6):
        super().__init__()
        layers = [nn.Conv2d(2, width, 3, padding=1), nn.ReLU(inplace=True)]
        for _ in range(depth - 2):
            layers += [nn.Conv2d(width, width, 3, padding=1),
                       nn.BatchNorm2d(width), nn.ReLU(inplace=True)]
        layers += [nn.Conv2d(width, 2, 3, padding=1)]
        self.net = nn.Sequential(*layers)

    def forward(self, x):          # (B, 2, S, F) LS grid -> refined (B, 2, S, F)
        return x + self.net(x)


def load_model(ckpt_path, device="cpu"):
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=True)
    model = ChannelCNN(width=ckpt["width"], depth=ckpt["depth"])
    model.load_state_dict(ckpt["state_dict"])
    return model.to(device).eval()


if __name__ == "__main__":
    m = ChannelCNN()
    n_params = sum(p.numel() for p in m.parameters())
    y = m(torch.randn(4, 2, 14, 64))
    assert y.shape == (4, 2, 14, 64)
    print(f"ChannelCNN: {n_params/1e3:.1f}k params, out shape {tuple(y.shape)}")
