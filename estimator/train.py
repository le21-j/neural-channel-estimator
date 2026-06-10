"""Train ChannelCNN on (LS-interpolated grid -> true_H) minimizing NMSE.

Model input is the LS linear-interp grid (see model.py INPUT CONTRACT);
computed once from rx_pilots + snr_db and cached as ls_interp.npy next to
the dataset. Doppler-stratified validation split: per Doppler bin, a fixed
fraction is held out, so val NMSE is balanced across slow/fast fading.
Logs to W&B (offline by default; set WANDB_MODE=online to sync).

Usage: python estimator/train.py [--epochs 30] [--data data/train]
"""
import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch

from model import ChannelCNN

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "data"))


def ls_input_cached(d):
    """LS linear-interp grids for the dataset at d, cached as ls_interp.npy."""
    cache = d / "ls_interp.npy"
    if cache.exists():
        return torch.from_numpy(np.load(cache))
    from baselines import Baselines, load_cfgs
    yp = np.load(d / "rx_pilots.npy")
    snr = torch.from_numpy(np.load(d / "snr_db.npy"))
    y = torch.complex(torch.from_numpy(yp[:, 0]), torch.from_numpy(yp[:, 1]))
    bl = Baselines(*load_cfgs())
    outs = []
    for i in range(0, len(y), 500):
        no = 10.0 ** (-snr[i: i + 500] / 10.0)
        h_ls = bl.ls(y[i: i + 500], no)
        outs.append(torch.stack([h_ls.real, h_ls.imag], dim=1))
    x = torch.cat(outs).to(torch.float32)
    np.save(cache, x.numpy())
    return x


def nmse_loss(pred, target):
    return (pred - target).pow(2).sum() / target.pow(2).sum()


def stratified_split(doppler, val_frac, seed):
    rng = np.random.default_rng(seed)
    val_mask = np.zeros(len(doppler), dtype=bool)
    for d in np.unique(doppler):
        idx = np.where(doppler == d)[0]
        rng.shuffle(idx)
        val_mask[idx[: int(len(idx) * val_frac)]] = True
    return ~val_mask, val_mask


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=str, default=str(ROOT / "data" / "train"))
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--val-frac", type=float, default=0.15)
    ap.add_argument("--width", type=int, default=48)
    ap.add_argument("--depth", type=int, default=6)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=str,
                    default=str(ROOT / "estimator" / "checkpoints" / "model.pt"))
    args = ap.parse_args()

    d = Path(args.data)
    x = ls_input_cached(d)
    h = torch.from_numpy(np.load(d / "true_H.npy"))
    dop = np.load(d / "doppler_hz.npy")
    tr, va = stratified_split(dop, args.val_frac, args.seed)
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    x_va, h_va = x[va].to(device), h[va].to(device)
    x_tr, h_tr = x[tr], h[tr]
    print(f"train {len(x_tr)} / val {len(x_va)} on {device}")

    import wandb
    run = wandb.init(project="neural-channel-estimator", mode="offline",
                     config=vars(args))

    torch.manual_seed(args.seed)
    model = ChannelCNN(width=args.width, depth=args.depth).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt, T_max=args.epochs, eta_min=args.lr / 20)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    best = float("inf")
    for ep in range(args.epochs):
        t0 = time.time()
        model.train()
        perm = torch.randperm(len(x_tr))
        tot = 0.0
        for i in range(0, len(x_tr), args.batch):
            b = perm[i: i + args.batch]
            xb, hb = x_tr[b].to(device), h_tr[b].to(device)
            loss = nmse_loss(model(xb), hb)
            opt.zero_grad()
            loss.backward()
            opt.step()
            tot += loss.item() * len(b)
        sched.step()
        model.eval()
        with torch.no_grad():
            val = nmse_loss(model(x_va), h_va).item()
        if val < best:
            best = val
            torch.save({"state_dict": model.state_dict(), "val_nmse": val,
                        "width": args.width, "depth": args.depth}, out)
        run.log({"train_nmse": tot / len(x_tr), "val_nmse": val,
                 "lr": sched.get_last_lr()[0]})
        print(f"epoch {ep+1:3d}/{args.epochs}  train {tot/len(x_tr):.4f}  "
              f"val {val:.4f}  best {best:.4f}  ({time.time()-t0:.1f}s)")
    run.finish()
    print(f"best val NMSE {best:.4f} -> {out}")


if __name__ == "__main__":
    main()
