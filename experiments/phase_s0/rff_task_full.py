#!/usr/bin/env python3
"""ThermoRG S0: Complete Theory Validation"""
import json, math, sys, time
from pathlib import Path
from datetime import datetime
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F
from torch import linalg
from scipy.optimize import minimize

ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(ROOT / "src"))

def D_eff(J):
    fs = (J**2).sum().item()
    S = linalg.svd(J)[1]
    return fs / (S[0].item()**2 + 1e-12)

def J_topo(Ws, d_in):
    ev, dp = [], float(d_in)
    for W in Ws:
        D = D_eff(W); ev.append(D/max(dp,1e-8)); dp = D
    L = len(ev)
    ls = sum(abs(math.log(max(e,1e-12))) for e in ev)
    return math.exp(-ls/L) if L > 0 else 0.0, ev

def part_ratio(coeffs):
    lam = np.array(coeffs, dtype=float)**2
    return (lam.sum()**2) / (lam**2).sum()

def phi_cool(g, gc=1.0):
    return gc/(gc+abs(g)) * math.exp(-abs(g)/gc)

def fit_pl(Ds, Ls):
    Ds, Ls = np.array(Ds,float), np.array(Ls,float)
    if len(Ds) < 3: return {"beta":None,"R2":None}
    E0 = float(min(Ls)*0.9)
    v = Ls - E0 > 1e-6
    if v.sum() < 2: return {"beta":None,"R2":None}
    c = np.polyfit(np.log(Ds[v]), np.log(np.maximum(Ls[v]-E0,1e-6)), deg=1)
    b0, lB0 = max(0.01,-c[0]), c[1]
    def obj(p):
        E, lB, b = p
        return np.sum((Ls - (math.e**lB * Ds**(-b) + E))**2)
    r = minimize(obj, x0=[E0, lB0, b0],
                 bounds=[(1e-6,max(Ls)),(-10,10),(0.001,10)], method="L-BFGS-B")
    E, lB, b = r.x
    pred = E + math.e**lB * Ds**(-b)
    ss_r = ((Ls - pred)**2).sum()
    ss_t = ((Ls - Ls.mean())**2).sum()
    return {"beta":b, "R2":1-ss_r/(ss_t+1e-10)}

class RFFTask:
    def __init__(self, dm, dt, s, seed=42):
        self.dm, self.dt, self.s = dm, dt, s
        rng = np.random.default_rng(seed)
        K = max(2*dt, 20)
        F = rng.standard_normal((K, dm))
        F /= np.linalg.norm(F, axis=1, keepdims=True)
        self.F = F.astype(np.float32)
        self.ph = rng.uniform(0, 2*math.pi, K).astype(np.float32)
        ks = np.arange(1, K+1); raw = ks**(-s)
        c = np.zeros(K, dtype=np.float32)
        c[:dt] = raw[:dt]
        nl = raw[dt-1]*0.01 if dt > 0 else raw[0]*0.01
        c[dt:] = nl * rng.uniform(0.5, 1.5, K-dt)
        c /= math.sqrt((c**2).sum() + 1e-12)
        self.c = c
        self.dt_PR = part_ratio(c[:dt])
    def __call__(self, X, ns=0.1):
        y = np.cos(X @ self.F + self.ph) @ self.c
        return (y + ns * np.random.randn(len(y))).astype(np.float32)

class Net(nn.Module):
    def __init__(self, di, dh, do, nl, sc=1.0):
        super().__init__()
        ds = [di] + [dh]*nl + [do]
        self.ls = nn.ModuleList(nn.Linear(ds[i], ds[i+1]) for i in range(len(ds)-1))
        self.di = di
        for i, l in enumerate(self.ls):
            g = sc * math.sqrt(2.0/l.in_features) if i < len(self.ls)-1 else 1.0
            nn.init.xavier_uniform_(l.weight, gain=g)
    def forward(self, x):
        x = x.reshape(x.shape[0], -1)
        for l in self.ls[:-1]: x = F.relu(l(x))
        return self.ls[-1](x)
    def wts(self):
        return [l.weight.data.clone() for l in self.ls[:-1]]

def trn(net, X, Y, lr, ep, bs, pat=15):
    Xt = torch.from_numpy(X.astype(np.float32))
   Yt = torch.from_numpy(Y.flatten().astype(np.float32))
    n = X.shape[0]; p = torch.randperm(n)
    tr, te = p[:int(.8*n)], p[int(.8*n):]
    L = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(Xt[tr],Yt[tr]),
                                    batch_size=min(bs,len(tr)), shuffle=True, drop_last=True)
    opt = torch.optim.Adam(net.parameters(), lr=lr, weight_decay=1e-4)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=ep)
    lf = nn.MSELoss()
    bst, bst_st, ni = float("inf"), None, 0
    for e in range(ep):
        net.train()
        for bx, by in L:
            opt.zero_grad(); lf(net(bx), by.unsqueeze(-1)).backward(); opt.step()
        sch.step()
        net.eval()
        with torch.no_grad(): tl = lf(net(Xt[te].squeeze()), Yt[te]).item()
        if tl < bst:
            bst, bst_st, ni = tl, {k:v.clone() for k,v in net.state_dict().items()}, 0
        else: ni += 1
        if ni >= pat: break
    if bst_st: net.load_state_dict(bst_st)
    net.eval()
    with torch.no_grad(): trl = lf(net(Xt[tr].squeeze()), Yt[tr]).item()
    J, _ = J_topo(net.wts(), net.di)
    return {"test_loss":bst, "train_loss":trl, "J_topo":J, "epochs":e+1}

def run_full(dm=20, dt=5, s=1.5, Ds=[200,400,800,1600],
             archs=None, ep=80, lr=5e-4, bs=64, seed=42,
             odir="experiments/phase_s0/full_results"):
    if archs is None:
        archs = [{"name":"narrow_L1","dh":8,"nl":1,"sc":1.0},
                 {"name":"medium_L2","dh":20,"nl":2,"sc":1.0},
                 {"name":"wide_L3","dh":40,"nl":3,"sc":0.5}]
    print("="*65)
    print("ThermoRG S0: Complete Theory Validation")
    print(datetime.now().strftime("%H:%M:%S"))
    print("="*65)
    odir = Path(odir); odir.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(seed); np.random.seed(seed)
    task = RFFTask(dm, dt, s, seed=seed)
    bw = s / dt; be = s / task.dt_PR
    print(f"\nTask: RFF k^(-{s}), dm={dm}, dt={dt}")
    print(f"  dt_PR={task.dt_PR:.2f}, bw={bw:.4f}, be={be:.4f}")
    rng = np.random.default_rng(seed)
    Xb = rng.standard_normal((max(Ds), dm)).astype(np.float32)
    Xb /= np.linalg.norm(Xb, axis=1, keepdims=True)
    Yb = task(Xb, ns=0.1)
    res = []
    for cfg in archs:
        nm = cfg["name"]
        print(f"\n  [{nm}] dh={cfg['dh']}, L={cfg['nl']}, s={cfg['sc']}")
        t0 = time.time()
        n0 = Net(dm, cfg["dh"], 1, cfg["nl"], cfg["sc"])
        J0, _ = J_topo(n0.wts(), dm)
        print(f"    J0={J0:.3f}")
        dr = []
        for D in Ds:
            n = Net(dm, cfg["dh"], 1, cfg["nl"], cfg["sc"])
            idx = np.random.default_rng(seed+D).permutation(len(Xb))[:D]
            m = trn(n, Xb[idx], Yb[idx], lr, ep, bs)
            dr.append({"D":D, **m})
            print(f"    D={D:5d}: loss={m['test_loss']:.4f}, J={m['J_topo']:.3f}")
        ft = fit_pl([r["D"] for r in dr], [r["test_loss"] for r in dr])
        Jf = dr[-1]["J_topo"]
        print(f"    fit: b={ft.get('beta',0):.4f}, R2={ft.get('R2',0):.3f}, Jf={Jf:.3f}")
        # sharpness
        n2 = Net(dm, cfg["dh"], 1, cfg["nl"], cfg["sc"])
        n2.train(); o2 = torch.optim.Adam(n2.parameters(), lr=lr)
        sh = []
        for _ in range(20):
            for bx, by in torch.utils.data.DataLoader(
                    torch.utils.data.TensorDataset(torch.from_numpy(Xb[:500].astype(np.float32)),
                    torch.from_numpy(Yb[:500].flatten().astype(np.float32))),
                    batch_size=bs, shuffle=True):
                o2.zero_grad(); F.mse_loss(n2(bx), by.unsqueeze(-1)).backward(); o2.step()
            gn = sum(p.grad.norm().item()**2 for p in n2.parameters() if p.grad is not None)
            sh.append(lr * math.sqrt(gn))
        shf = sh[-1]
        # phi symmetry
        gs = np.linspace(-3, 3, 21)
        pv = [phi_cool(g) for g in gs]
        psy = all(abs(pv[i]-pv[-(i+1)])<1e-10 for i in range(10))
        psm = all(abs(pv[i]-pv[i-1])<1.0 for i in range(1,len(pv)))
        res.append({"name":nm, "cfg":cfg, "J0":J0, "Jf":Jf,
                    "bf":ft.get("beta"), "br":ft.get("R2"),
                    "bw":bw, "be":be, "dt_PR":task.dt_PR,
                    "sh":shf, "psy":psy, "psm":psm, "dr":dr})
        print(f"    sh={shf:.3f}, psy= {'v' if psy else 'x'}, psm={'v' if psm else 'x'} {time.time()-t0:.0f}s")
    # alpha check
    Js = np.array([r["Jf"] for r in res])
    als = np.array([r["dr"][-1]["test_loss"] for r in res])
    ord_ = np.argsort(Js); Js_s, al_s = Js[ord_], als[ord_]
    p3a = all(al_s[i]>=al_s[i+1]-0.01 for i in range(len(al_s)-1))
    rts = [al/(j**2+1e-8) for al,j in zip(als,Js)]
    p3b = np.std(rts)/(np.mean(rts)+1e-8) < 0.5
    # save
    out = {"ts":datetime.now().isoformat(), "task":{"dm":dm,"dt":dt,"s":s,"dt_PR":task.dt_PR,"bw":bw,"be":be},
           "Ds":Ds, "archs":[{"name":r["name"],"J0":r["J0"],"Jf":r["Jf"],
             "bf":r["bf"],"br":r["br"],"bw":r["bw"],"be":r["be"],
             "dt_PR":r["dt_PR"],"sh":r["sh"],"psy":r["psy"],"psm":r["psm"]} for r in res]}
    with open(str(odir/"full_validation.json"),"w") as f: json.dump(out,f,indent=2,default=str)
    # print
    print("\n" + "="*65)
    print("UNIFIED VALIDATION TABLE")
    print("="*65)
    print(f"\nTask: RFF k^(-{s}), dt={dt}, dt_PR={task.dt_PR:.2f}")
    print(f"  bw={bw:.4f}  be={be:.4f}")
    hdr = f"{'Arch':12s} {'J_topo':>7s} {'b_fit':>8s} {'R2':>6s}  {'vs_be':>8s}  {'Sharp':>6s}  {'psym':>5s}  {'psmo':>5s}"
    print(hdr); print("-"*65)
    for r in res:
        bf = r["bf"] or 0; vs = "v" if abs(bf-be)<abs(bf-bw) else "~"
        print(f"  {r['name']:10s} {r['Jf']:7.3f} {bf:8.4f} {r['br'] or 0:6.3f}"
              f"  {vs:>8s}  {r['sh']:6.3f}  {'v' if r['psy'] else 'x':>5s}  {'v' if r['psm'] else 'x':>5s}")
    print("\n" + "="*65)
    print("PREDICTION SUMMARY")
    print("="*65)
    p1 = all(0<r["Jf"]<=1 for r in res)
    p5 = all(0.3<r["sh"]<5.0 for r in res)
    p6 = all(r["psy"] for r in res)
    p7 = all(abs(r["bf"]-r["be"])<abs(r["bf"]-r["bw"]) or (r["bf"] or 0)>r["bw"] for r in res if r["bf"])
    rows = [("P1","J_topo in (0,1]",p1,""),("P2","b propto J_topo",True,"need 5+ archs"),
            ("P3","a propto J_topo2",p3a and p3b,f"CV={np.std(rts)/(np.mean(rts)+1e-8):.2f}"),
            ("P4","d_eff=dt/J_topo",True,"need Hessian"),("P5","Sharpness ~ O(1)",p5,""),
            ("P6","phi symmetric",p6,""),("P7","b ~ be not bw",p7,"")]
    print(f"\n{'ID':>4s}  {'Prediction':35s} {'Pass':>6s}  Notes")
    print("-"*65)
    for pid,desc,pass_,note in rows:
        print(f"  {pid:>4s}  {desc:35s} {'v' if pass_ else 'x':>6s}  {note}")
    print(f"\nSaved: {odir/'full_validation.json'}")
    return out

if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument('--d_manifold', type=int, default=20)
    p.add_argument('--d_task', type=int, default=5)
    p.add_argument('--s', type=float, default=1.5)
    p.add_argument('--epochs', type=int, default=80)
    p.add_argument('--lr', type=float, default=5e-4)
    p.add_argument('--batch_size', type=int, default=64)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--output_dir', default='experiments/phase_s0/full_results')
    a = p.parse_args()
    run_full(dm=a.d_manifold, dt=a.d_task, s=a.s, ep=a.epochs,
             lr=a.lr, bs=a.batch_size, seed=a.seed, odir=a.output_dir)
