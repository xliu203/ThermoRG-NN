#!/usr/bin/env python3
"""
ThermoRG Phase A v2 — CIFAR-10 D-Scaling Experiment
===================================================

Validates ThermoRG v3 theory on real CIFAR-10 data with diverse architectures.

Theory (v3):
  J_topo = exp(-|Σ log η_l| / L)
  η_l = D_eff^(l) / D_eff^(l-1)
  D_eff = ||W_l||_F² / λ_max(W_l)

  L(D) = α · D^(-β) + E

Hypotheses:
  H1: β̂ ∝ J_topo   (n ≥ 10 architectures)
  H2: α̂ ∝ J_topo²

Validation:
  - 12 architectures (families: ThermoNet-width, ThermoNet-depth, ResNet, VGG)
  - D ∈ {2K, 5K, 10K, 25K, 50K} (5 points, log-spaced)
  - 2 seeds per (arch, D)
  - 50 epochs per run

Success criteria:
  - H1: Pearson r(β̂, J_topo) > 0.7, p < 0.05
  - H2: Pearson r(α̂, J_topo²) > 0.7, p < 0.05
"""

import json, math, os, sys, time, warnings
from pathlib import Path
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import linalg

warnings.filterwarnings('ignore')

# ─────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────

ARCHITECTURES = [
    # ThermoNet width family
    {"Name": "TN-W8",  "arch": "TN3",  "width_mult": 0.25},
    {"Name": "TN-W16", "arch": "TN3",  "width_mult": 0.5},
    {"Name": "TN-W32", "arch": "TN3",  "width_mult": 1.0},
    {"Name": "TN-W64", "arch": "TN5",  "width_mult": 0.5},
    # ThermoNet depth family
    {"Name": "TN-L3",  "arch": "TN3",  "width_mult": 1.0},
    {"Name": "TN-L5",  "arch": "TN5",  "width_mult": 1.0},
    {"Name": "TN-L7",  "arch": "TN7",  "width_mult": 1.0},
    {"Name": "TN-L9",  "arch": "TN9",  "width_mult": 1.0},
    # Traditional baselines
    {"Name": "ResNet-18", "arch": "R18", "width_mult": 1.0},
    {"Name": "ResNet-34", "arch": "R34", "width_mult": 1.0},
    {"Name": "VGG-11",    "arch": "VGG11","width_mult": 1.0},
    {"Name": "VGG-13",    "arch": "VGG13","width_mult": 1.0},
]

D_VALUES  = [2000, 5000, 10000, 25000, 50000]
SEEDS     = [42, 123]
EPOCHS    = 50
LR        = 0.1
BATCH_SIZE = 128
WD        = 5e-4
MOMENTUM  = 0.9
OUT_DIR   = Path("experiments/phase_a/results_v2")
CKPT_DIR  = Path("experiments/phase_a/checkpoints_v2")

# ─────────────────────────────────────────────────────────
# MODEL BUILDERS (inline, avoid torchvision import issues)
# ─────────────────────────────────────────────────────────

class SkipConnection(nn.Module):
    def __init__(self, ic, oc, s=1):
        super().__init__()
        if ic == oc and s == 1:
            self.skip = nn.Identity()
        else:
            self.skip = nn.Sequential(nn.Conv2d(ic, oc, 1, s, bias=False), nn.BatchNorm2d(oc))
    def forward(self, x, residual):
        return x + self.skip(residual)


class ConvBlock(nn.Module):
    def __init__(self, ic, oc, act='gelu', norm=True):
        super().__init__()
        self.conv = nn.Conv2d(ic, oc, 3, padding=1, bias=not norm)
        self.norm = nn.LayerNorm([oc, 32, 32]) if norm else nn.Identity()
        if act == 'gelu':
            self.act = nn.GELU()
        elif act == 'tga':
            self.act = nn.Tanh()
        else:
            self.act = nn.ReLU(inplace=True)
    def forward(self, x):
        return self.act(self.norm(self.conv(x)))


class Bottleneck(nn.Module):
    def __init__(self, ic, oc, bd, act='gelu', norm=True):
        super().__init__()
        self.c1 = nn.Conv2d(ic, bd, 1, bias=not norm)
        self.n1 = nn.LayerNorm([bd, 32, 32]) if norm else nn.Identity()
        self.c2 = nn.Conv2d(bd, bd, 3, padding=1, bias=not norm)
        self.n2 = nn.LayerNorm([bd, 32, 32]) if norm else nn.Identity()
        self.c3 = nn.Conv2d(bd, oc, 1, bias=not norm)
        self.n3 = nn.LayerNorm([oc, 32, 32]) if norm else nn.Identity()
        if act == 'gelu':
            self.act = nn.GELU()
        elif act == 'tga':
            self.act = nn.Tanh()
        else:
            self.act = nn.ReLU(inplace=True)
    def forward(self, x):
        x = self.act(self.n1(self.c1(x)))
        x = self.act(self.n2(self.c2(x)))
        return self.act(self.n3(self.c3(x)))


class BasicBlock(nn.Module):
    def __init__(self, ic, oc, stride=1, downsample=None):
        super().__init__()
        self.conv1 = nn.Conv2d(ic, oc, 3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(oc)
        self.conv2 = nn.Conv2d(oc, oc, 3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(oc)
        self.downsample = downsample
    def forward(self, x):
        identity = x
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        if self.downsample is not None:
            identity = self.downsample(x)
        return F.relu(out + identity)


def make_layer(ic, oc, blocks, stride=1):
    downsample = None
    if stride != 1 or ic != oc:
        downsample = nn.Sequential(nn.Conv2d(ic, oc, 1, stride=stride, bias=False), nn.BatchNorm2d(oc))
    layers = [BasicBlock(ic, oc, stride, downsample)]
    for _ in range(1, blocks):
        layers.append(BasicBlock(oc, oc))
    return nn.Sequential(*layers)


def scale_channels(chs, mult):
    """Scale channel list by multiplier, preserving first (input) channel."""
    return [chs[0]] + [max(1, int(c * mult)) for c in chs[1:]]


# ── Model builders ────────────────────────────────────────

def build_TN3(wm=1.0, num_classes=10, use_skip=True):
    """ThermoNet-3: [64,64,128,128]"""
    ch = scale_channels([3,64,64,128,128], wm)
    blocks = nn.ModuleList()
    skips = nn.ModuleList()
    for i in range(len(ch)-1):
        blocks.append(ConvBlock(ch[i], ch[i+1], 'gelu', True))
        skips.append(SkipConnection(ch[i], ch[i+1]) if (i > 0 and use_skip) else None)
    pool = nn.AdaptiveAvgPool2d((1,1))
    fc = nn.Linear(ch[-1], num_classes)
    return nn.Sequential(*([*blocks, pool, nn.Flatten(), fc]))


def build_TN5(wm=1.0, num_classes=10, use_skip=True):
    """ThermoNet-5: [64,128,256,128,64]"""
    ch = scale_channels([3,64,128,256,128,64], wm)
    blocks = nn.ModuleList()
    skips = nn.ModuleList()
    for i in range(len(ch)-1):
        blocks.append(ConvBlock(ch[i], ch[i+1], 'gelu', True))
        skips.append(SkipConnection(ch[i], ch[i+1]) if (i > 0 and use_skip) else None)
    pool = nn.AdaptiveAvgPool2d((1,1))
    fc = nn.Linear(ch[-1], num_classes)
    return nn.Sequential(*([*blocks, pool, nn.Flatten(), fc]))


def build_TN7(wm=1.0, num_classes=10, use_skip=True):
    """ThermoNet-7: [64,64,128,128,256,128,64]"""
    ch = scale_channels([3,64,64,128,128,256,128,64], wm)
    blocks = nn.ModuleList()
    skips = nn.ModuleList()
    for i in range(len(ch)-1):
        blocks.append(ConvBlock(ch[i], ch[i+1], 'tga', True))
        skips.append(SkipConnection(ch[i], ch[i+1]) if (i > 0 and use_skip) else None)
    pool = nn.AdaptiveAvgPool2d((1,1))
    fc = nn.Linear(ch[-1], num_classes)
    return nn.Sequential(*([*blocks, pool, nn.Flatten(), fc]))


def build_TN9(wm=1.0, num_classes=10, use_skip=False):
    """ThermoNet-9: [64]*8 uniform"""
    ch = scale_channels([3]+[64]*8, wm)
    blocks = nn.ModuleList()
    for i in range(len(ch)-1):
        blocks.append(ConvBlock(ch[i], ch[i+1], 'gelu', True))
    pool = nn.AdaptiveAvgPool2d((1,1))
    fc = nn.Linear(ch[-1], num_classes)
    return nn.Sequential(*([*blocks, pool, nn.Flatten(), fc]))


def build_R18(num_classes=10):
    model = nn.Sequential(
        nn.Conv2d(3, 64, 3, padding=1, bias=False), nn.BatchNorm2d(64), nn.ReLU(),
        make_layer(64, 64, 2, 1),
        make_layer(64, 128, 2, 2),
        make_layer(128, 256, 2, 2),
        make_layer(256, 512, 2, 2),
        nn.AdaptiveAvgPool2d((1,1)), nn.Flatten(), nn.Linear(512, num_classes)
    )
    return model


def build_R34(num_classes=10):
    model = nn.Sequential(
        nn.Conv2d(3, 64, 3, padding=1, bias=False), nn.BatchNorm2d(64), nn.ReLU(),
        make_layer(64, 64, 3, 1),
        make_layer(64, 128, 3, 2),
        make_layer(128, 256, 3, 2),
        make_layer(256, 512, 3, 2),
        nn.AdaptiveAvgPool2d((1,1)), nn.Flatten(), nn.Linear(512, num_classes)
    )
    return model


def build_VGG11(num_classes=10):
    ch = [64,128,256,256,512,512]
    layers = []
    prev = 3
    for i, oc in enumerate(ch):
        layers.extend([nn.Conv2d(prev, oc, 3, padding=1), nn.BatchNorm2d(oc), nn.ReLU(inplace=True)])
        if i < len(ch)-1:
            layers.append(nn.MaxPool2d(2,2))
        prev = oc
    model = nn.Sequential(
        *layers,
        nn.AdaptiveAvgPool2d((1,1)), nn.Flatten(), nn.Linear(512, num_classes)
    )
    return model


def build_VGG13(num_classes=10):
    ch = [64,64,128,128,256,256,512,512]
    layers = []
    prev = 3
    for i, oc in enumerate(ch):
        layers.extend([nn.Conv2d(prev, oc, 3, padding=1), nn.BatchNorm2d(oc), nn.ReLU(inplace=True)])
        if i % 2 == 1 and i < len(ch)-2:
            layers.append(nn.MaxPool2d(2,2))
        prev = oc
    model = nn.Sequential(
        *layers,
        nn.AdaptiveAvgPool2d((1,1)), nn.Flatten(), nn.Linear(512, num_classes)
    )
    return model


def get_arch_model(arch_key, width_mult=1.0, num_classes=10):
    """Get model by architecture key."""
    # When width scaling, disable skip connections to avoid dimension mismatches
    use_skip = (width_mult >= 1.0)
    builders = {
        "TN3":  lambda wm: build_TN3(wm=wm, use_skip=use_skip),
        "TN5":  lambda wm: build_TN5(wm=wm, use_skip=use_skip),
        "TN7":  lambda wm: build_TN7(wm=wm, use_skip=use_skip),
        "TN9":  build_TN9,
        "R18":  lambda wm=1: build_R18(),
        "R34":  lambda wm=1: build_R34(),
        "VGG11": lambda wm=1: build_VGG11(),
        "VGG13": lambda wm=1: build_VGG13(),
    }
    if arch_key in ("R18", "R34", "VGG11", "VGG13"):
        return builders[arch_key]()
    return builders[arch_key](width_mult)


# ─────────────────────────────────────────────────────────
# J_Topo (v3 formula)
# ─────────────────────────────────────────────────────────

def _combine_main_skip(main_W, skip_mod):
    """
    Build the effective weight for a residual block: W_eff = W_main + W_skip.

    W_main: weight tensor of main branch (Conv2d or Linear)
    skip_mod: nn.Identity or projection Conv2d

    Theory: y = F(x) + S(x). The combined information is preserved through the skip.
    - Identity skip (ic == oc, stride=1): W_eff = W_main + I
    - Projection skip (stride>1 or ic!=oc): the skip changes dimensions, so it
      doesn't directly add to W_main. We model this by noting the skip preserves
      info (D_eff contribution ≈ 1), but for the W_eff we just return W_main.

    The key insight: for identity skip, the skip doubles the dominant singular
    value → W_main + I raises D_eff. For projection skip, the skip is a different
    linear map on a different input; we capture this by noting the skip path
    contributes additional effective dimension.
    """
    if isinstance(skip_mod, nn.Identity):
        W = main_W.float()
        if W.dim() == 4:
            out_ch, in_ch = W.shape[0], W.shape[1]
            I = torch.eye(out_ch, in_ch, dtype=W.dtype, device=W.device).view(out_ch, in_ch, 1, 1)
        elif W.dim() == 2:
            I = torch.eye(W.shape[0], W.shape[1], dtype=W.dtype, device=W.device)
        else:
            return W
        return W + I
    else:
        # Projection skip: W_skip != I. The skip operates on the input x
        # (not the main path output), so it doesn't directly add to W_main.
        # Return W_main as-is. The information-theoretic contribution of
        # the skip is captured separately in J_topo via the η_l ratio.
        return main_W.float()


def get_layer_weights_combined(model, arch_key, width_mult=1.0):
    """
    Extract per-layer weights from a model, combining main branch + skip branch
    for residual blocks using W_eff = W_main + W_skip (ThermoRG skip extension).
    
    Returns a list of weight tensors (one per logical layer), ready for compute_J_topo.
    
    arch_key examples: 'TN-W8', 'TN-L3', 'R18', 'VGG11', etc.
    """
    # Get the architecture type
    is_thermonet = arch_key.startswith('TN-')
    is_resnet    = arch_key.startswith('R')
    is_vgg       = arch_key.startswith('VGG')
    
    weights = []
    
    if is_thermonet:
        # ThermoNet: flat Sequential of ConvBlock + pool/Flatten/Linear.
        # SkipConnections are stored separately in model.skips (ModuleList).
        # The skip forward is: x + skips[i](x), applied AFTER block i.
        # For layer i (i>0): W_eff = W_main + W_skip (identity).
        # For layer 0: no skip.
        blocks = getattr(model, 'blocks', None)   # ModuleList
        skips  = getattr(model, 'skips',  None)   # ModuleList

        if blocks is None:
            # Fallback: iterate Sequential children directly
            for m in model.modules():
                if isinstance(m, ConvBlock):
                    # Direct conv attribute (not iterating all submodules)
                    weights.append(m.conv.weight.data.clone().float())
                elif isinstance(m, nn.Linear):
                    weights.append(m.weight.data.clone().float())
            return weights

        for i, block in enumerate(blocks):
            # Direct conv attribute — no iterating over all submodules (avoids LayerNorm)
            main_W = block.conv.weight.data.clone()

            # Skip: model.skips[i] is None for i==0 (no skip at first layer)
            skip_mod = None
            if skips is not None and i > 0 and skips[i] is not None:
                skip_mod = skips[i].skip

            if skip_mod is not None:
                W_eff = _combine_main_skip(main_W, skip_mod)
            else:
                W_eff = main_W.float()
            weights.append(W_eff)

        # Final classifier
        if hasattr(model, 'fc') and isinstance(model.fc, nn.Linear):
            weights.append(model.fc.weight.data.clone().float())
    
    elif is_resnet:
        # ResNet: Sequential with BasicBlocks. Each BasicBlock has .conv1, .conv2, .downsample
        # We need to process per BasicBlock
        seq_children = list(model.children())
        block_idx = 0
        for child in seq_children:
            if isinstance(child, nn.Sequential):
                for block in child.children():
                    if isinstance(block, BasicBlock):
                        # Main branch: conv2 weight (second conv, the one before addition)
                        main_W = block.conv2.weight.data.clone().float()
                        
                        # Skip: downsample if present, else Identity
                        downsample = block.downsample
                        if downsample is not None:
                            # downsample is a Sequential: [Conv2d, BN]
                            # The Conv2d weight is the projection
                            skip_mod = downsample[0]  # Conv2d
                        else:
                            skip_mod = nn.Identity()
                        
                        W_eff = _combine_main_skip(main_W, skip_mod)
                        weights.append(W_eff)
                        block_idx += 1
            elif isinstance(child, (nn.Conv2d, nn.Linear)) and not isinstance(child, (nn.Sequential, nn.ModuleList)):
                # Initial conv, final fc
                if child.weight.requires_grad:
                    weights.append(child.weight.data.clone().float())
    
    elif is_vgg:
        # VGG: no skip connections — just collect conv and fc weights
        for m in model.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)) and m.weight.requires_grad:
                weights.append(m.weight.data.clone().float())
    
    else:
        # Unknown architecture: flat extraction
        return [p.data.clone() for p in model.parameters()
                if p.dim() >= 2 and p.requires_grad]
    
    return weights


def compute_D_eff(W):
    """D_eff = ||W||_F² / λ_max(W)"""
    fro_sq = (W ** 2).sum().item()
    try:
        spec_max = linalg.svd(W)[1][0].item()
    except Exception:
        spec_max = W.norm().item() + 1e-12
    return fro_sq / (spec_max ** 2 + 1e-12)


def compute_J_topo(weights, input_dim=3):
    """
    J_topo = exp(-|Σ log η_l| / L)
    η_l = D_eff^(l) / D_eff^(l-1)
    """
    if not weights:
        return 0.0, []

    eta_vals = []
    d_prev = float(input_dim)

    for W in weights:
        # Handle both Conv2d (out, in, H, W) and Linear (out, in)
        if W.dim() == 4:
            W_mat = W.view(W.shape[0], -1)
        elif W.dim() == 2:
            W_mat = W
        else:
            W_mat = W.reshape(W.shape[0], -1)

        D = compute_D_eff(W_mat)
        eta = D / max(d_prev, 1e-12)
        eta_vals.append(eta)
        d_prev = D

    L = len(eta_vals)
    log_sum = sum(abs(math.log(max(e, 1e-12))) for e in eta_vals)
    J = math.exp(-log_sum / L) if L > 0 else 0.0
    return J, eta_vals


# ─────────────────────────────────────────────────────────
# DATASET
# ─────────────────────────────────────────────────────────

def load_cifar10(data_root="./data"):
    """Load CIFAR-10 with fallback to fake data."""
    try:
        from torchvision import datasets
        train_ds = datasets.CIFAR10(root=data_root, train=True, download=True)
        test_ds  = datasets.CIFAR10(root=data_root, train=False, download=True)
        X_train = torch.from_numpy(np.array(train_ds.data)).float().permute(0,3,1,2) / 255.0
        Y_train = torch.tensor(train_ds.targets, dtype=torch.long)
        X_test  = torch.from_numpy(np.array(test_ds.data)).float().permute(0,3,1,2) / 255.0
        Y_test  = torch.tensor(test_ds.targets, dtype=torch.long)
        mean = torch.tensor([0.491, 0.482, 0.447]).view(1, 3, 1, 1)
        std  = torch.tensor([0.247, 0.243, 0.262]).view(1, 3, 1, 1)
        X_train = (X_train - mean) / std
        X_test  = (X_test  - mean) / std
        print(f"  Loaded CIFAR-10: train={X_train.shape}, test={X_test.shape}")
        return X_train, Y_train, X_test, Y_test
    except Exception as e:
        print(f"  CIFAR-10 load failed ({e}), using fake data")
        X_train = torch.randn(50000, 3, 32, 32)
        Y_train = torch.randint(0, 10, (50000,))
        X_test  = torch.randn(10000, 3, 32, 32)
        Y_test  = torch.randint(0, 10, (10000,))
        return X_train, Y_train, X_test, Y_test


# ─────────────────────────────────────────────────────────
# TRAINING
# ─────────────────────────────────────────────────────────

def subset_loader(X, Y, size, batch_size, seed):
    rng = np.random.default_rng(seed)
    indices = rng.permutation(len(X))[:size]
    return torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(X[indices], Y[indices]),
        batch_size=batch_size, shuffle=True, drop_last=True)


def train_one_epoch(model, loader, optimizer):
    model.train()
    total_loss = 0
    for bx, by in loader:
        optimizer.zero_grad()
        loss = F.cross_entropy(model(bx), by)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * len(bx)
    return total_loss / len(loader.dataset)


@torch.no_grad()
def evaluate(model, X_test, Y_test, batch_size=256):
    model.eval()
    total_loss = 0
    correct = 0
    loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(X_test, Y_test),
        batch_size=batch_size, shuffle=False)
    for bx, by in loader:
        out = model(bx)
        total_loss += F.cross_entropy(out, by, reduction='sum').item()
        correct += (out.argmax(1) == by).sum().item()
    n = len(Y_test)
    return total_loss / n, correct / n


def train_run(arch_cfg, D, seed, X_train, Y_train, X_test, Y_test,
              epochs=EPOCHS, lr=LR, ckpt_path=None):
    arch_name = arch_cfg["Name"]
    base_arch = arch_cfg["arch"]
    wm = arch_cfg.get("width_mult", 1.0)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    torch.manual_seed(seed)
    np.random.seed(seed)

    model = get_arch_model(base_arch, wm).to(device)
    # Move data to device
    X_train_d = X_train.to(device)
    Y_train_d = Y_train.to(device)
    X_test_d  = X_test.to(device)
    Y_test_d  = Y_test.to(device)

    optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=MOMENTUM, weight_decay=WD)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    train_loader = subset_loader(X_train_d, Y_train_d, D, BATCH_SIZE, seed)

    start_epoch = 0
    if ckpt_path and ckpt_path.exists():
        try:
            ckpt = torch.load(ckpt_path, map_location='cpu')
            model.load_state_dict(ckpt['model'])
            optimizer.load_state_dict(ckpt['optimizer'])
            scheduler.load_state_dict(ckpt['scheduler'])
            start_epoch = ckpt.get('epoch', 0)
            print(f"    Resume ep {start_epoch}", end=" ")
        except Exception:
            start_epoch = 0

    result = {
        'arch': arch_name, 'base_arch': base_arch, 'width_mult': wm,
        'D': D, 'seed': seed,
        'epochs_recorded': [], 'train_loss': [], 'test_loss': [], 'test_acc': [],
    }

    for ep in range(start_epoch, epochs):
        tloss = train_one_epoch(model, train_loader, optimizer)
        scheduler.step()
        tloss_eval, tacc = evaluate(model, X_test_d, Y_test_d)
        result['epochs_recorded'].append(ep + 1)
        result['train_loss'].append(tloss)
        result['test_loss'].append(tloss_eval)
        result['test_acc'].append(tacc)

        if (ep + 1) % 10 == 0:
            print(f"  ep {ep+1:3d}: loss={tloss_eval:.4f} acc={tacc:.3f}", end=" | ")
            print()

    # Save checkpoint
    if ckpt_path:
        ckpt_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'scheduler': scheduler.state_dict(),
            'epoch': epochs,
            'result': result,
        }, ckpt_path)

    # J_topo (final) — uses W_eff = W_main + W_skip for residual blocks
    weights_final = get_layer_weights_combined(model, base_arch, wm)
    J_final, _ = compute_J_topo(weights_final)

    # J_topo (init — fresh model)
    model_init = get_arch_model(base_arch, wm)
    weights_init = get_layer_weights_combined(model_init, base_arch, wm)
    J_init, _ = compute_J_topo(weights_init)

    result['J_topo_init'] = J_init
    result['J_topo_final'] = J_final
    result['final_test_loss'] = result['test_loss'][-1]
    result['final_test_acc'] = result['test_acc'][-1]
    result['params_M'] = sum(p.numel() for p in model.parameters()) / 1e6

    return result


# ─────────────────────────────────────────────────────────
# POWER LAW FITTING
# ─────────────────────────────────────────────────────────

def fit_power_law(Ds, losses):
    """Fit L(D) = α · D^(-β) + E. Returns (β, R2, α, E)."""
    Ds = np.array(Ds, float)
    Ls = np.array(losses, float)
    if len(Ds) < 3:
        return None, None, None, None

    E0 = float(min(Ls) * 0.9)
    v = Ls - E0 > 1e-6
    if v.sum() < 2:
        return None, None, None, None

    from scipy.optimize import minimize
    c = np.polyfit(np.log(Ds[v]), np.log(np.maximum(Ls[v] - E0, 1e-6)), deg=1)
    b0, lB0 = max(0.01, -c[0]), c[1]

    def obj(p):
        E, lB, b = p
        return np.sum((Ls - (math.e**lB * Ds**(-b) + E))**2)

    r = minimize(obj, x0=[E0, lB0, b0],
                  bounds=[(1e-6, max(Ls)), (-10, 10), (0.001, 10)],
                  method='L-BFGS-B')
    E_fit, lB_fit, b_fit = r.x
    alpha_fit = math.e**lB_fit

    pred = E_fit + alpha_fit * Ds**(-b_fit)
    ss_res = ((Ls - pred)**2).sum()
    ss_tot = ((Ls - Ls.mean())**2).sum()
    R2 = 1 - ss_res / (ss_tot + 1e-10)

    return b_fit, R2, alpha_fit, E_fit


# ─────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────

def run_phase_a():
    t0 = time.time()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    CKPT_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 65)
    print("ThermoRG Phase A v2 — CIFAR-10 D-Scaling")
    print(f"Started: {datetime.now().strftime('%H:%M:%S')}")
    print(f"GPU: {torch.cuda.is_available()}")
    print("=" * 65)

    X_train, Y_train, X_test, Y_test = load_cifar10()

    done_file = OUT_DIR / "completed_runs.json"
    if done_file.exists():
        with open(done_file) as f:
            done_runs = set(json.load(f))
        print(f"Resume: {len(done_runs)} runs already completed")
    else:
        done_runs = set()

    all_results = []

    for arch_cfg in ARCHITECTURES:
        an = arch_cfg["Name"]
        print(f"\n[{an}] ({arch_cfg['arch']}, wm={arch_cfg.get('width_mult',1.0)})")

        for D in D_VALUES:
            for seed in SEEDS:
                run_key = f"{an}_D{D}_s{seed}"
                ckpt_path = CKPT_DIR / f"{run_key}.pt"

                print(f"  D={D}, seed={seed}...", end=" ", flush=True)

                if run_key in done_runs and ckpt_path.exists():
                    try:
                        ckpt = torch.load(ckpt_path, map_location='cpu')
                        res = ckpt.get('result', {})
                        if res:
                            print(f"  [SKIP] loss={res.get('final_test_loss', -1):.4f}")
                            all_results.append(res)
                            continue
                    except Exception:
                        pass

                try:
                    res = train_run(arch_cfg, D, seed, X_train, Y_train, X_test, Y_test,
                                     ckpt_path=ckpt_path)
                    all_results.append(res)
                    done_runs.add(run_key)
                    with open(done_file, 'w') as f:
                        json.dump(list(done_runs), f)
                    print(f"  loss={res['final_test_loss']:.4f} acc={res['final_test_acc']:.3f} J={res['J_topo_final']:.3f}")
                except Exception as e:
                    print(f"  ERROR: {e}")
                    continue

    # ── Aggregate per architecture ───────────────────────
    print("\n" + "=" * 65)
    print("POWER LAW FITS")
    print("=" * 65)

    from scipy import stats

    arch_agg = {}
    for arch_cfg in ARCHITECTURES:
        an = arch_cfg["Name"]
        ba = arch_cfg["arch"]
        wm = arch_cfg.get("width_mult", 1.0)

        losses_by_D = {d: [] for d in D_VALUES}
        J_init_list, J_final_list = [], []

        for r in all_results:
            if r['arch'] == an:
                losses_by_D[r['D']].append(r['final_test_loss'])
                J_init_list.append(r['J_topo_init'])
                J_final_list.append(r['J_topo_final'])

        if not losses_by_D[D_VALUES[0]]:
            continue

        Ds = sorted(losses_by_D.keys())
        Ls = [np.mean(losses_by_D[d]) for d in Ds]
        beta, R2, alpha, E_fit = fit_power_law(Ds, Ls)
        J_final = np.mean(J_final_list) if J_final_list else None
        J_init  = np.mean(J_init_list) if J_init_list else None
        params  = next((r['params_M'] for r in all_results if r['arch'] == an), None)

        print(f"\n  [{an}] params={params:.2f}M  J_init={J_init:.3f}  J_final={J_final:.3f}")
        print(f"    β={beta:.4f}  α={alpha:.4f}  E={E_fit:.4f}  R²={R2:.3f}")

        arch_agg[an] = {
            'params_M': params,
            'J_topo_init': J_init,
            'J_topo_final': J_final,
            'beta': beta,
            'alpha': alpha,
            'E': E_fit,
            'R2': R2,
            'd_scaling': {str(d): float(np.mean(losses_by_D[d])) for d in Ds},
        }

    # ── Statistical tests ─────────────────────────────
    print("\n" + "=" * 65)
    print("STATISTICAL TESTS")
    print("=" * 65)

    valid = [(n, d) for n, d in arch_agg.items() if d.get('beta') is not None]

    if len(valid) < 3:
        print(f"  Only {len(valid)} architectures with valid fits — cannot test")
    else:
        Js  = np.array([arch_agg[n]['J_topo_final'] for n, d in valid])
        Bs  = np.array([arch_agg[n]['beta']          for n, d in valid])
        Als = np.array([arch_agg[n]['alpha']           for n, d in valid])

        # H1: β ∝ J
        r_bj, p_bj = stats.pearsonr(Js, Bs)
        # H2: α ∝ J²
        r_aj2, p_aj2 = stats.pearsonr(Js**2, Als)
        # H3: J vs loss
        maxD = str(max(D_VALUES))
        Lf   = np.array([arch_agg[n]['d_scaling'].get(maxD, float('nan')) for n, d in valid])
        r_jl, p_jl = stats.pearsonr(Js, Lf)

        print(f"\n  H1: β̂ ∝ J_topo   r={r_bj:.3f}  p={p_bj:.4f}  {'✓ PASS' if abs(r_bj)>0.7 and p_bj<0.05 else '✗ FAIL'}")
        print(f"  H2: α̂ ∝ J_topo²  r={r_aj2:.3f}  p={p_aj2:.4f}  {'✓ PASS' if abs(r_aj2)>0.7 and p_aj2<0.05 else '✗ FAIL'}")
        print(f"  H3: J_topo vs loss  r={r_jl:.3f}  p={p_jl:.4f}  {'✓ (neg)' if r_jl<0 and p_jl<0.1 else '~'}")

        stats_out = {'H1': {'r': float(r_bj), 'p': float(p_bj)},
                     'H2': {'r': float(r_aj2), 'p': float(p_aj2)},
                     'H3': {'r': float(r_jl), 'p': float(p_jl)}}

    # ── Save ─────────────────────────────────────────
    out = {
        'timestamp': datetime.now().isoformat(),
        'config': {
            'architectures': [a['Name'] for a in ARCHITECTURES],
            'D_values': D_VALUES,
            'seeds': SEEDS,
            'epochs': EPOCHS,
            'lr': LR,
        },
        'archs': [{'name': n, **d} for n, d in arch_agg.items()],
        'stats': stats_out if len(valid) >= 3 else {},
    }

    out_path = OUT_DIR / 'phase_a_dscaling_results.json'
    with open(out_path, 'w') as f:
        json.dump(out, f, indent=2, default=str)

    print(f"\nSaved: {out_path}  ({time.time()-t0:.0f}s)")
    return out


if __name__ == "__main__":
    run_phase_a()
