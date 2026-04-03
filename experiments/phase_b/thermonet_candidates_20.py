"""
ThermoNet candidate architectures for Phase B experiment.
20 configurations covering wide J_topo range.
Latin hypercube-style sampling across depth × width × skip.
"""

CANDIDATES_20 = [
    # ID, depth, width_mult, use_skip, expected_J_range, note
    # High J_topo region (wide, no skip) — Theory predicts HIGH E_floor
    {"id": "T01", "depth": 3,  "width_mult": 2.0, "skip": False, "region": "high_J"},
    {"id": "T02", "depth": 5,  "width_mult": 1.5, "skip": False, "region": "high_J"},
    {"id": "T03", "depth": 7,  "width_mult": 1.0, "skip": False, "region": "high_J"},
    {"id": "T04", "depth": 9,  "width_mult": 0.75,"skip": False, "region": "high_J"},
    {"id": "T05", "depth": 12, "width_mult": 0.5, "skip": False, "region": "high_J"},
    
    # Medium J_topo region
    {"id": "T06", "depth": 3,  "width_mult": 1.5, "skip": False, "region": "mid_J"},
    {"id": "T07", "depth": 5,  "width_mult": 2.0, "skip": False, "region": "mid_J"},
    {"id": "T08", "depth": 7,  "width_mult": 0.75,"skip": False, "region": "mid_J"},
    {"id": "T09", "depth": 9,  "width_mult": 1.0, "skip": False, "region": "mid_J"},
    {"id": "T10", "depth": 12, "width_mult": 1.5, "skip": False, "region": "mid_J"},
    
    # Low J_topo region (skip connections help reduce J_topo) — Theory predicts LOW E_floor
    {"id": "T11", "depth": 3,  "width_mult": 1.0, "skip": True,  "region": "low_J"},
    {"id": "T12", "depth": 5,  "width_mult": 0.75,"skip": True,  "region": "low_J"},
    {"id": "T13", "depth": 7,  "width_mult": 0.5, "skip": True,  "region": "low_J"},
    {"id": "T14", "depth": 9,  "width_mult": 1.5, "skip": True,  "region": "low_J"},
    {"id": "T15", "depth": 12, "width_mult": 2.0, "skip": True,  "region": "low_J"},
    
    # Additional diverse configs
    {"id": "T16", "depth": 5,  "width_mult": 1.0, "skip": True,  "region": "low_J"},
    {"id": "T17", "depth": 7,  "width_mult": 1.5, "skip": True,  "region": "low_J"},
    {"id": "T18", "depth": 9,  "width_mult": 0.5, "skip": True,  "region": "low_J"},
    {"id": "T19", "depth": 15, "width_mult": 1.0, "skip": True,  "region": "low_J"},
    {"id": "T20", "depth": 15, "width_mult": 0.75,"skip": False, "region": "mid_J"},
]

# Reference baselines (from literature, NOT trained by us)
BASELINES = {
    "ResNet-18":       {"source": "He et al. 2016", "accuracy": 94.8, "params_M": 11.7},
    "WideResNet-40-2": {"source": "Zagoruyko et al. 2016", "accuracy": 94.4, "params_M": 2.7},
    "DenseNet-40":     {"source": "Huang et al. 2017", "accuracy": 94.3, "params_M": 1.0},
}

if __name__ == "__main__":
    print("=== ThermoNet 20 Candidates ===")
    print(f"{'ID':<5} {'Depth':<6} {'Width':<7} {'Skip':<5} {'Region':<8}")
    print("-" * 35)
    for c in CANDIDATES_20:
        print(f"{c['id']:<5} {c['depth']:<6} {c['width_mult']:<7} {str(c['skip']):<5} {c['region']:<8}")
    
    print(f"\nTotal candidates: {len(CANDIDATES_20)}")
    print(f"High J region:  {sum(1 for c in CANDIDATES_20 if c['region']=='high_J')}")
    print(f"Medium J region: {sum(1 for c in CANDIDATES_20 if c['region']=='mid_J')}")
    print(f"Low J region:  {sum(1 for c in CANDIDATES_20 if c['region']=='low_J')}")
    
    print("\n=== Reference Baselines ===")
    for name, info in BASELINES.items():
        print(f"{name}: {info['accuracy']}% ({info['source']})")
