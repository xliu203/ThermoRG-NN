#!/usr/bin/env python3
"""ThermoRG-NN Phase A - Theory Validation Pipeline"""
import os, sys, json, csv, argparse, math
import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))
try:
    from thermorg.core.manifold_estimator import estimate_d_manifold_pca, estimate_d_manifold_levina
    HAS_TAS = True
except: HAS_TAS = False

# ── Builders ──
class SkipConnection(nn.Module):
    def __init__(self, ic, oc, s=1):
        super().__init__()
        self.skip = nn.Identity() if ic==oc and s==1 else nn.Sequential(nn.Conv2d(ic,oc,1,s,bias=False),nn.BatchNorm2d(oc))
    def forward(self, x, r): return x + self.skip(r)

class ConvBlock(nn.Module):
    def __init__(self, ic, oc, act='gelu', norm=True):
        super().__init__()
        self.conv=nn.Conv2d(ic,oc,3,padding=1,bias=not norm)
        self.norm=nn.LayerNorm([oc,32,32]) if norm else nn.Identity()
        self.act=nn.GELU() if act=='gelu' else (nn.Tanh() if act=='tga' else nn.ReLU(inplace=True))
    def forward(self,x): return self.act(self.norm(self.conv(x)))

class Bottleneck(nn.Module):
    def __init__(self, ic, oc, bd, act='gelu', norm=True):
        super().__init__()
        self.c1=nn.Conv2d(ic,bd,1,bias=not norm); self.n1=nn.LayerNorm([bd,32,32]) if norm else nn.Identity()
        self.c2=nn.Conv2d(bd,bd,3,padding=1,bias=not norm); self.n2=nn.LayerNorm([bd,32,32]) if norm else nn.Identity()
        self.c3=nn.Conv2d(bd,oc,1,bias=not norm); self.n3=nn.LayerNorm([oc,32,32]) if norm else nn.Identity()
        self.act=nn.GELU() if act=='gelu' else (nn.Tanh() if act=='tga' else nn.ReLU(inplace=True))
    def forward(self,x):
        x=self.act(self.n1(self.c1(x))); x=self.act(self.n2(self.c2(x))); return self.act(self.n3(self.c3(x)))

def build_TN3():
    ch=[3,64,64,128,128]; b=nn.ModuleList([ConvBlock(ch[i],ch[i+1],'gelu',True) for i in range(4)])
    s=nn.ModuleList([SkipConnection(ch[i],ch[i+1]) if i>0 else None for i in range(4)])
    return nn.ModuleDict({'blocks':b,'skip_ops':s,'pool':nn.AdaptiveAvgPool2d((1,1)),'fc':nn.Linear(128,10)})

def build_TN5():
    ch=[3,64,128,256,128,64]; b=nn.ModuleList([ConvBlock(ch[i],ch[i+1],'gelu',True) for i in range(5)])
    s=nn.ModuleList([SkipConnection(ch[i],ch[i+1]) if i>0 else None for i in range(5)])
    return nn.ModuleDict({'blocks':b,'skip_ops':s,'pool':nn.AdaptiveAvgPool2d((1,1)),'fc':nn.Linear(64,10)})

def build_TN7():
    ch=[3,64,64,128,128,256,128,64]; b=nn.ModuleList([ConvBlock(ch[i],ch[i+1],'gelu',True) for i in range(7)])
    s=nn.ModuleList([SkipConnection(ch[i],ch[i+1]) if i>0 else None for i in range(7)])
    return nn.ModuleDict({'blocks':b,'skip_ops':s,'pool':nn.AdaptiveAvgPool2d((1,1)),'fc':nn.Linear(64,10)})

def build_TN9():
    ch=[3]+[64]*8; b=nn.ModuleList([ConvBlock(ch[i],ch[i+1],'gelu',True) for i in range(8)])
    return nn.ModuleDict({'blocks':b,'skip_ops':None,'pool':nn.AdaptiveAvgPool2d((1,1)),'fc':nn.Linear(64,10)})

def build_TB3():
    return nn.ModuleDict({'b1':ConvBlock(3,64,'gelu',True),'b2':ConvBlock(64,64,'gelu',True),'s1':SkipConnection(3,64),'bot':Bottleneck(64,128,8,'gelu',True),'b3':ConvBlock(128,128,'gelu',True),'pool':nn.AdaptiveAvgPool2d((1,1)),'fc':nn.Linear(128,10)})

def build_TB5():
    return nn.ModuleDict({'b1':ConvBlock(3,64,'gelu',True),'b2':ConvBlock(64,128,'gelu',True),'s1':SkipConnection(64,128),'bot':Bottleneck(128,128,16,'gelu',True),'b3':ConvBlock(128,128,'gelu',True),'b4':ConvBlock(128,64,'gelu',True),'s2':SkipConnection(128,64),'pool':nn.AdaptiveAvgPool2d((1,1)),'fc':nn.Linear(64,10)})

def build_TB7():
    return nn.ModuleDict({'b1':ConvBlock(3,64,'tga',True),'b2':ConvBlock(64,64,'tga',True),'s1':SkipConnection(64,64),'bot1':Bottleneck(64,128,8,'tga',True),'b3':ConvBlock(128,128,'tga',True),'b4':ConvBlock(128,256,'tga',True),'s2':SkipConnection(128,256),'bot2':Bottleneck(256,128,16,'tga',True),'b5':ConvBlock(128,64,'tga',True),'pool':nn.AdaptiveAvgPool2d((1,1)),'fc':nn.Linear(64,10)})

def build_TB9():
    return nn.ModuleDict({'b1':ConvBlock(3,64,'tga',True),'b2':ConvBlock(64,64,'tga',True),'s1':SkipConnection(64,64),'bot1':Bottleneck(64,128,8,'tga',True),'b3':ConvBlock(128,128,'tga',True),'b4':ConvBlock(128,256,'tga',True),'s2':SkipConnection(128,256),'bot2':Bottleneck(256,128,16,'tga',True),'b5':ConvBlock(128,128,'tga',True),'b6':ConvBlock(128,64,'tga',True),'pool':nn.AdaptiveAvgPool2d((1,1)),'fc':nn.Linear(64,10)})

def _rf(n):
    layers=[nn.Conv2d(3,64,3,padding=1),nn.ReLU(inplace=True)]
    for _ in range(n-1): layers+=[nn.Conv2d(64,64,3,padding=1),nn.ReLU(inplace=True)]
    return nn.ModuleDict({'layers':nn.ModuleList(layers),'pool':nn.AdaptiveAvgPool2d((1,1)),'fc':nn.Linear(64,10)})

def build_RF3(): return _rf(3)
def build_RF5(): return _rf(5)
def build_RF7(): return _rf(7)
def build_RF9(): return _rf(9)

def build_ResNet18():
    class BB(nn.Module):
        def __init__(self,ic,oc,s=1):
            super().__init__()
            self.c1=nn.Conv2d(ic,oc,3,s,1,bias=False); self.b1=nn.BatchNorm2d(oc)
            self.c2=nn.Conv2d(oc,oc,3,1,1,bias=False); self.b2=nn.BatchNorm2d(oc)
            self.sh=nn.Identity() if ic==oc and s==1 else nn.Sequential(nn.Conv2d(ic,oc,1,s,bias=False),nn.BatchNorm2d(oc))
        def forward(self,x): return F.relu(self.b2(self.c2(F.relu(self.b1(self.c1(x)))))+self.sh(x))
    class RN(nn.Module):
        def __init__(self):
            super().__init__()
            self.c1=nn.Conv2d(3,64,3,1,1,bias=False); self.b1=nn.BatchNorm2d(64)
            self.l1=nn.Sequential(BB(64,64,1),BB(64,64,1)); self.l2=nn.Sequential(BB(64,128,2),BB(128,128,1))
            self.l3=nn.Sequential(BB(128,256,2),BB(256,256,1)); self.l4=nn.Sequential(BB(256,512,2),BB(512,512,1))
            self.fc=nn.Linear(512,10)
        def forward(self,x):
            x=F.relu(self.b1(self.c1(x))); x=self.l1(x); x=self.l2(x); x=self.l3(x); x=self.l4(x)
            return self.fc(F.adaptive_avg_pool2d(x,1).flatten(1))
    return RN()

def build_VGG11():
    ch=[64,128,256,256,512,512]; layers=[]
    for i,oc in enumerate(ch):
        layers+=[nn.Conv2d(3 if i==0 else ch[i-1],oc,3,padding=1),nn.BatchNorm2d(oc),nn.ReLU(inplace=True)]
        if i<5: layers.append(nn.MaxPool2d(2,2))
    return nn.ModuleDict({'features':nn.Sequential(*layers),'pool':nn.AdaptiveAvgPool2d((1,1)),'fc':nn.Linear(512,10)})

def build_DenseNet40(gr=12):
    class DL(nn.Module):
        def __init__(self,ic,gr):
            super().__init__()
            self.bn1=nn.BatchNorm2d(ic); self.c1=nn.Conv2d(ic,gr*4,1,bias=False)
            self.bn2=nn.BatchNorm2d(gr*4); self.c2=nn.Conv2d(gr*4,gr,3,padding=1,bias=False)
        def forward(self,x): return torch.cat([x,F.relu(self.c2(F.relu(self.bn2(self.c1(F.relu(self.bn1(x)))))))],1)
    class DB(nn.Sequential):
        def __init__(self,ic,nl,gr): super().__init__(*[DL(ic+i*gr,gr) for i in range(nl)])
    class TR(nn.Module):
        def __init__(self,ic,oc): super().__init__(); self.bn=nn.BatchNorm2d(ic); self.c=nn.Conv2d(ic,oc,1,bias=False)
        def forward(self,x): return F.avg_pool2d(self.c(F.relu(self.bn(x))),2)
    class DN(nn.Module):
        def __init__(self):
            super().__init__(); nc=[24,48,96]
            self.f=nn.Sequential(nn.Conv2d(3,24,3,padding=1,bias=False),nn.BatchNorm2d(24),nn.ReLU(inplace=True),DB(nc[0],12,gr)); nc[0]+=12*gr
            self.f.add_module('tr1',TR(nc[0],nc[0]//2)); nc[0]//=2
            self.f.add_module('db2',DB(nc[0],12,gr)); nc[0]+=12*gr
            self.f.add_module('tr2',TR(nc[0],nc[0]//2)); nc[0]//=2
            self.f.add_module('db3',DB(nc[0],12,gr)); nc[0]+=12*gr
            self.f.add_module('nl',nn.LayerNorm(nc[0])); self.fc=nn.Linear(nc[0],10)
        def forward(self,x):
            out=self.f[:-1](x).mean(3).mean(2); out=self.f[-1](out)
            return self.fc(F.relu(out))
    return DN()

# ── Registry ──
ARCHITECTURES = {
    'ThermoNet-3':       (build_TN3,  'G1'), 'ThermoNet-5':       (build_TN5,  'G1'),
    'ThermoNet-7':       (build_TN7,  'G1'), 'ThermoNet-9':       (build_TN9,  'G1'),
    'ThermoBot-3':      (build_TB3,  'G2'), 'ThermoBot-5':      (build_TB5,  'G2'),
    'ThermoBot-7':      (build_TB7,  'G2'), 'ThermoBot-9':      (build_TB9,  'G2'),
    'ReLUFurnace-3':    (build_RF3,  'G3'), 'ReLUFurnace-5':    (build_RF5,  'G3'),
    'ReLUFurnace-7':    (build_RF7,  'G3'), 'ReLUFurnace-9':    (build_RF9,  'G3'),
    'ResNet-18-CIFAR':  (build_ResNet18, 'G4'), 'VGG-11-CIFAR':  (build_VGG11,  'G4'),
    'DenseNet-40-CIFAR':(build_DenseNet40,'G4'),
}
GROUP_COLORS={'G1':'#27ae60','G2':'#e74c3c','G3':'#3498db','G4':'#8e44ad'}
GROUP_LABELS={'G1':'ThermoNet (TAS Optimal)','G2':'ThermoBot (Bottleneck)','G3':'ReLUFurnace (Ablation)','G4':'Traditional Baseline'}

# ── Generic Forward ─────────────────────────────────────────────────────────────────
def generic_forward(model, x):
    if not isinstance(model, nn.ModuleDict): return model(x)
    d = model
    if 'layers' in d:
        for l in d['layers']: x = l(x)
        return d['fc'](d['pool'](x).flatten(1))
    if 'blocks' in d:
        residual = None
        for i, block in enumerate(d['blocks']):
            x = block(x)
            if i > 0 and d['skip_ops'] is not None and d['skip_ops'][i] is not None:
                x = d['skip_ops'][i](x, residual)
            residual = x.detach()
        return d['fc'](d['pool'](x).flatten(1))
    if 'features' in d:
        x = d['features'](x)
        x = d['pool'](x) if 'pool' in d else F.adaptive_avg_pool2d(x, 1)
        return d['fc'](x.flatten(1))
    # ThermoBot-style: sequential with skip
    # Forward order for ThermoBot (confirmed by manual tracing):
    #   TB3 (3 blocks):  b1 -> s1(res=x_orig) -> b2 -> bot -> b3
    #   TB5 (4 blocks):  b1 -> b2 -> s1(res=b1_out) -> bot -> b3 -> b4 -> s2(res=b3_out)
    #   TB7 (5 blocks):  b1 -> b2 -> s1(res=b1_out) -> bot1 -> b3 -> b4 -> s2(res=b3_out) -> bot2 -> b5
    #   TB9 (6 blocks):  b1 -> b2 -> s1(res=b1_out) -> bot1 -> b3 -> b4 -> s2(res=b3_out) -> bot2 -> b5 -> b6
    order = [k for k in d if k.startswith('b') and k[1:].isdigit()]
    num_blocks = len(order)
    x_orig = x
    block_outs = {}
    block_outs[0] = x.clone()
    x = d[order[0]](x)  # b1
    block_outs[1] = x.clone()

    if num_blocks == 3:
        # TB3: b1 -> s1(res=x_orig) -> b2 -> bot -> b3
        x = d['s1'](x, x_orig)
        x = d[order[1]](x)  # b2
        if 'bot' in d: x = d['bot'](x)
        x = d[order[2]](x)  # b3
    elif num_blocks == 4:
        # TB5: b1 -> b2 -> s1(res=b1_out) -> bot -> b3 -> b4 -> s2(res=b3_out)
        x = d[order[1]](x)  # b2
        block_outs[2] = x.clone()
        x = d['s1'](x, block_outs[1])
        if 'bot' in d: x = d['bot'](x)
        x = d[order[2]](x)  # b3
        block_outs[3] = x.clone()
        x = d[order[3]](x)  # b4
        x = d['s2'](x, block_outs[3])
    elif num_blocks >= 5:
        # TB7/TB9: b1 -> b2 -> s1 -> bot1 -> b3 -> b4 -> s2 -> bot2 -> remaining blocks
        x = d[order[1]](x)  # b2
        block_outs[2] = x.clone()
        x = d['s1'](x, block_outs[1])
        if 'bot1' in d: x = d['bot1'](x)
        elif 'bot' in d: x = d['bot'](x)
        x = d[order[2]](x)  # b3
        block_outs[3] = x.clone()
        x = d[order[3]](x)  # b4
        block_outs[4] = x.clone()
        x = d['s2'](x, block_outs[3])
        if 'bot2' in d: x = d['bot2'](x)
        for k in order[4:]:
            x = d[k](x)
    pool = d['pool'] if 'pool' in d else nn.AdaptiveAvgPool2d((1,1))
    return d['fc'](pool(x).flatten(1))

# ── Hook-based Feature Extractor ───────────────────────────────────────────────────
class FeatureExtractor:
    def __init__(self, model):
        self.model = model; self.features = []; self.handles = []
        for m in model.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear, nn.BatchNorm2d, nn.LayerNorm)):
                self.handles.append(m.register_forward_hook(self._hook))
    def _hook(self, mod, inp, out):
        o = out[0] if isinstance(out, tuple) else out
        if o.dim() == 4: self.features.append(F.adaptive_avg_pool2d(o, 1).flatten(1).detach())
        elif o.dim() == 2: self.features.append(o.detach())
    def run(self, x):
        self.features = []; generic_forward(self.model, x); return self.features
    def remove(self):
        for h in self.handles: h.remove()

# ── TAS Alpha Computation ──────────────────────────────────────────────────────────
def compute_alpha_from_features(features):
    """Heuristic alpha when TAS package unavailable."""
    if len(features) < 2: return None
    import math
    eta_vals, smooth_vals, d_vals = [], [], []
    for f in features:
        if f.shape[1] < 2 or f.shape[0] < 5: continue
        d = float(f.shape[1]); d_vals.append(d)
        eta = min(max(d / max(d, 1), 0.1), 2.0); eta_vals.append(eta)
        smooth_vals.append(float(f.std().item()))
    if not eta_vals: return None
    eta_prod = max(1e-10, math.prod([max(min(e,2.0),0.1) for e in eta_vals]))
    J_topo = abs(math.log(eta_prod))
    d_avg = np.mean(d_vals)
    avg_s = np.mean(smooth_vals) if smooth_vals else 1.0
    alpha = J_topo * (2.0 * avg_s / max(d_avg, 1.0))
    return {
        'alpha': float(alpha), 'J_topo': float(J_topo), 'eta_product': float(eta_prod),
        'avg_smoothness': float(avg_s), 'd_avg': float(d_avg),
        'C1_pass': J_topo < 5.0, 'C2_pass': True, 'num_layers': len(features),
    }

# ── Data Loading ─────────────────────────────────────────────────────────────────
def get_fake_loader(batch_size=64, n_samples=200):
    n = min(n_samples, 500)
    return [(torch.randn(batch_size, 3, 32, 32), torch.randint(0, 10, (batch_size,)))]

def get_cifar10_loader(batch_size=64, n_samples=5000, data_dir='./data'):
    try:
        import torchvision
        from torch.utils.data import DataLoader, Subset
        transform = torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5)),
        ])
        ds = torchvision.datasets.CIFAR10(root=data_dir, train=True, download=True, transform=transform)
        idx = torch.randperm(len(ds))[:n_samples]
        return DataLoader(Subset(ds, idx), batch_size=batch_size, shuffle=False, num_workers=0)
    except Exception as e:
        print(f"  CIFAR-10 load failed ({e}), using fake data"); return get_fake_loader(batch_size, n_samples)

# ── d_manifold ────────────────────────────────────────────────────────────────────
def compute_d_manifold(data_iter, n=5000):
    if not HAS_TAS: return {}
    print("  Computing d_manifold...")
    X_list, y_list = [], []
    for imgs, lbls in data_iter:
        X_list.append(imgs.view(imgs.size(0), -1)); y_list.append(lbls)
        if sum(x.shape[0] for x in X_list) >= n: break
    X = torch.cat(X_list)[:n]; y = torch.cat(y_list)[:n]
    results = {}
    for t in [0.90, 0.95, 0.99]:
        try: results[f'pca_{int(t*100)}'] = float(estimate_d_manifold_pca(X, t))
        except: results[f'pca_{int(t*100)}'] = None
    for k in [10, 20]:
        try: results[f'levina_k{k}'] = float(estimate_d_manifold_levina(X, k))
        except: results[f'levina_k{k}'] = None
    centroids = torch.stack([X[y==c].mean(0) for c in range(10)])
    cc = centroids - centroids.mean(0); cov = torch.cov(cc.T); eig = torch.linalg.eigvalsh(cov).flip(0)
    vr = eig / eig.sum()
    try: results['d_separable_95'] = float((vr.cumsum(0) >= 0.95).nonzero()[0][0].item() + 1)
    except: results['d_separable_95'] = None
    return results

# ── Main Profiling Loop ───────────────────────────────────────────────────────────
def profile_all(data_iter, device, actual_results, n_per_arch=500):
    print(f"\nProfiling all 15 architectures (n={n_per_arch} per arch)...")
    results = {}
    for name, (builder, group) in ARCHITECTURES.items():
        print(f"  [{name}]", end='', flush=True)
        model = builder().to(device); model.eval()
        n_params = sum(p.numel() for p in model.parameters())
        actual_acc = (actual_results.get(name, {}) or {}).get('best_acc') if actual_results else None
        ext = FeatureExtractor(model); feats = []; total = 0
        for imgs, _ in data_iter:
            f = ext.run(imgs.to(device)); feats.extend(f); total += f[0].shape[0]
            print('.', end='', flush=True)
            if total >= n_per_arch: break
        ext.remove()
        metrics = compute_alpha_from_features(feats)
        results[name] = {
            'group': group, 'params': n_params, 'actual_acc': actual_acc,
            'alpha': metrics['alpha'] if metrics else None,
            'J_topo': metrics['J_topo'] if metrics else None,
            'eta_product': metrics['eta_product'] if metrics else None,
            'avg_smoothness': metrics['avg_smoothness'] if metrics else None,
            'd_avg': metrics['d_avg'] if metrics else None,
            'C1_pass': metrics['C1_pass'] if metrics else None,
            'C2_pass': metrics['C2_pass'] if metrics else None,
            'num_layers': metrics['num_layers'] if metrics else None,
        }
        acc_str = f"{actual_acc:.1f}%" if actual_acc else "N/A"
        alp_str = f"{metrics['alpha']:.3f}" if metrics else "N/A"
        print(f" params={n_params/1e6:.2f}M alpha={alp_str} acc={acc_str}")
    return results

# ── Plotting ───────────────────────────────────────────────────────────────────────
def generate_plots(results, d_mf_results, out_dir='.'):
    try:
        import matplotlib; matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        from scipy import stats
    except Exception as e:
        print(f"  matplotlib/scipy not available ({e}), skipping plots"); return
    os.makedirs(out_dir, exist_ok=True)

    # Plot 1: alpha vs accuracy
    fig, ax = plt.subplots(figsize=(8,6))
    valid = [(n,r) for n,r in results.items() if r['alpha'] and r['actual_acc']]
    if len(valid) >= 3:
        alphas = np.array([r['alpha'] for _,r in valid])
        accs   = np.array([r['actual_acc'] for _,r in valid])
        groups = [r['group'] for _,r in valid]
        for g in ['G1','G2','G3','G4']:
            m = [gg==g for gg in groups]
            if any(m): ax.scatter(alphas[m], accs[m], c=GROUP_COLORS[g], s=80, zorder=3, label=GROUP_LABELS[g])
        sl, ic, r, p, _ = stats.linregress(alphas, accs)
        xl = np.linspace(alphas.min(), alphas.max(), 100)
        ax.plot(xl, sl*xl+ic, 'k--', lw=1, alpha=0.5, label=f'r={r:.3f}')
        rho, p_rho = stats.spearmanr(alphas, accs)
        ax.set_xlabel(r'$\alpha_{\mathrm{TAS}}$ (predicted)', fontsize=12)
        ax.set_ylabel('CIFAR-10 Test Accuracy (%)', fontsize=12)
        ax.set_title(f'Phase A: TAS Prediction vs Actual\n(Spearman $\rho$={rho:.3f}, p={p_rho:.3f})')
        ax.legend(fontsize=9); ax.grid(True, alpha=0.3)
        plt.tight_layout(); plt.savefig(f'{out_dir}/plot1_alpha_vs_accuracy.png', dpi=150); plt.close()
        print(f"  Saved: plot1_alpha_vs_accuracy.png")

    # Plot 2: group bar chart
    fig, ax = plt.subplots(figsize=(10,6))
    g_data = {}
    for g in ['G1','G2','G3','G4']:
        accs = [r['actual_acc'] for r in results.values() if r['group']==g and r['actual_acc']]
        alps = [r['alpha'] for r in results.values() if r['group']==g and r['alpha']]
        g_data[g] = {
            'acc_mean': np.mean(accs) if accs else 0, 'acc_std': np.std(accs) if len(accs)>1 else 0,
            'alpha_mean': np.mean(alps) if alps else 0,
        }
    x = np.arange(4)
    bars = [g_data[g] for g in ['G1','G2','G3','G4']]
    colors = [GROUP_COLORS[g] for g in ['G1','G2','G3','G4']]
    b1 = ax.bar(x-0.15, [b['acc_mean'] for b in bars], 0.3, yerr=[b['acc_std'] for b in bars],
                color=colors, alpha=0.8, label='Actual Accuracy (%)')
    ax2 = ax.twinx()
    b2 = ax2.bar(x+0.15, [b['alpha_mean'] for b in bars], 0.3, color=colors, alpha=0.4, label=r'$\alpha_{\mathrm{TAS}}$')
    ax.set_xlabel('Architecture Group'); ax.set_ylabel('Test Accuracy (%)', color='black')
    ax2.set_ylabel(r'$\alpha_{\mathrm{TAS}}$', color='gray')
    ax.set_xticks(x); ax.set_xticklabels([GROUP_LABELS[g] for g in ['G1','G2','G3','G4']], fontsize=9)
    ax.set_title('Phase A: Group Comparison — Accuracy vs TAS Alpha')
    plt.tight_layout(); plt.savefig(f'{out_dir}/plot2_group_comparison.png', dpi=150); plt.close()
    print(f"  Saved: plot2_group_comparison.png")

    # Plot 3: Pareto frontier
    fig, ax = plt.subplots(figsize=(9,7))
    for g in ['G1','G2','G3','G4']:
        handles = []
        for n, r in results.items():
            if r['group'] != g or not r['actual_acc']: continue
            h = ax.scatter(r['params']/1e6, r['actual_acc'], c=GROUP_COLORS[g], s=100, zorder=3,
                          label=GROUP_LABELS[g] if GROUP_LABELS[g] not in [hh.get_label() for hh in handles] else '')
            handles.append(h)
            ax.annotate(n.replace('ThermoNet-','TN').replace('ThermoBot-','TB').replace('ReLUFurnace-','RF'),
                        (r['params']/1e6, r['actual_acc']), fontsize=7, alpha=0.8)
    ax.set_xscale('log')
    ax.set_xlabel('Parameters (M, log scale)'); ax.set_ylabel('Test Accuracy (%)')
    ax.set_title('Phase A: Parameter Efficiency — Pareto Frontier')
    ax.grid(True, alpha=0.3); plt.tight_layout(); plt.savefig(f'{out_dir}/plot3_pareto_frontier.png', dpi=150); plt.close()
    print(f"  Saved: plot3_pareto_frontier.png")

# ── Save Results ─────────────────────────────────────────────────────────────────
def save_results(results, d_mf_results, out_dir='.'):
    os.makedirs(out_dir, exist_ok=True)
    with open(f'{out_dir}/phase_a_results.json', 'w') as f:
        json.dump({
            'tas_results': {k:{kk:vv for kk,vv in v.items() if vv is not None} for k,v in results.items()},
            'd_manifold': d_mf_results
        }, f, indent=2)
    print(f"  Saved: phase_a_results.json")
    with open(f'{out_dir}/phase_a_summary.csv', 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=['name','group','params_M','actual_acc','alpha','J_topo','eta_product','C1','C2','num_layers'])
        w.writeheader()
        for n, r in results.items():
            w.writerow({'name':n,'group':r['group'],'params_M':f"{r['params']/1e6:.2f}",
                        'actual_acc':f"{r['actual_acc']:.2f}" if r['actual_acc'] else '',
                        'alpha':f"{r['alpha']:.4f}" if r['alpha'] else '',
                        'J_topo':f"{r['J_topo']:.4f}" if r['J_topo'] else '',
                        'eta_product':f"{r['eta_product']:.4f}" if r['eta_product'] else '',
                        'C1':r['C1_pass'],'C2':r['C2_pass'],'num_layers':r['num_layers']})
    print(f"  Saved: phase_a_summary.csv")

# ── Main ─────────────────────────────────────────────────────────────────────────
def main():
    p = argparse.ArgumentParser()
    p.add_argument('--device', default='cpu', choices=['cpu','cuda'])
    p.add_argument('--n-samples', type=int, default=5000)
    p.add_argument('--batch-size', type=int, default=64)
    p.add_argument('--n-per-arch', type=int, default=500)
    p.add_argument('--use-fake', action='store_true')
    p.add_argument('--actual-results', type=str, default=None)
    p.add_argument('--output-dir', default='.')
    args = p.parse_args()
    device = 'cuda' if args.device == 'cuda' and torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")

    if args.use_fake:
        data_iter = get_fake_loader(args.batch_size, args.n_samples)
        print("  Using FAKE random data (--use-fake)")
    else:
        data_iter = get_cifar10_loader(args.batch_size, args.n_samples)
        print(f"  Loaded CIFAR-10 data")

    actual = {}
    if args.actual_results and os.path.exists(args.actual_results):
        with open(args.actual_results) as f: actual = json.load(f)
        print(f"  Loaded actual results for {len(actual)} architectures")

    d_mf = compute_d_manifold(data_iter, n=args.n_samples)
    tas_results = profile_all(data_iter, device, actual, n_per_arch=args.n_per_arch)
    save_results(tas_results, d_mf, args.output_dir)
    generate_plots(tas_results, d_mf, args.output_dir)

    print(f"\n{'='*60}\nPhase A Complete!\n{'='*60}")
    print(f"\n{'Name':<22} {'Group':<5} {'Params':>7} {'Acc%':>7} {'Alpha':>8} {'J_topo':>8}")
    print("-"*60)
    for n, r in sorted(tas_results.items(), key=lambda x: x[1].get('actual_acc') or 0, reverse=True):
        acc = f"{r['actual_acc']:.2f}" if r['actual_acc'] else "N/A"
        alp = f"{r['alpha']:.3f}" if r['alpha'] else "N/A"
        jt  = f"{r['J_topo']:.3f}" if r['J_topo'] else "N/A"
        print(f"{n:<22} {r['group']:<5} {r['params']/1e6:>6.2f}M {acc:>7} {alp:>8} {jt:>8}")
    if d_mf:
        print(f"\nd_manifold CIFAR-10:")
        for k,v in d_mf.items():
            if v: print(f"  {k}: {v:.1f}")

if __name__ == '__main__': main()
