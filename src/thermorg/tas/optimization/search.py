# SPDX-License-Identifier: Apache-2.0

"""Architecture search with constrained optimization.

Phase 6: Constrained Optimization
    Maximize α subject to:
        Params ≤ P_max
        FLOPs ≤ F_max
        C1: J_topo = |∑log η_l| ≤ ε_topo (topological isometry)
        C2: T̃_eff = T_eff · ε_coupling ≤ ξ_opt · T_c (thermal safety)
"""

import numpy as np
from numpy.typing import NDArray
from typing import Optional, List, Dict, Any, Callable, Tuple, TYPE_CHECKING
from dataclasses import dataclass, field
from concurrent.futures import ThreadPoolExecutor
import itertools

from ...utils.math import safe_log, product_log

if TYPE_CHECKING:
    from ..predictor import OptimalityResult


@dataclass
class ArchitectureConfig:
    """Configuration for a neural network architecture."""
    name: str
    layer_widths: List[int]
    layer_types: List[str]
    activation: str = 'relu'
    use_batch_norm: bool = False
    dropout_rate: float = 0.0
    
    @property
    def n_layers(self) -> int:
        return len(self.layer_widths)
    
    def to_dict(self) -> dict:
        return {
            'name': self.name,
            'layer_widths': self.layer_widths,
            'layer_types': self.layer_types,
            'activation': self.activation,
            'use_batch_norm': self.use_batch_norm,
            'dropout_rate': self.dropout_rate,
        }


@dataclass
class ConstraintBounds:
    """Constraint bounds for architecture search."""
    params_max: float = float('inf')
    flops_max: float = float('inf')
    latency_max_ms: Optional[float] = None
    memory_max_mb: Optional[float] = None


@dataclass
class SearchResult:
    """Result from architecture search with optimality verification.
    
    Attributes:
        architecture: The best architecture found
        alpha: Scaling exponent α
        optimality: OptimalityResult with C1/C2 verification
        metrics: Resource usage metrics (params, FLOPs)
        feasible: Whether architecture satisfies all constraints
    """
    architecture: ArchitectureConfig
    alpha: float
    optimality: 'OptimalityResult'
    metrics: Dict[str, float] = field(default_factory=dict)
    feasible: bool = True
    
    def summary(self) -> str:
        """Return human-readable summary."""
        feas = "FEASIBLE" if self.feasible else "INFEASIBLE"
        opt_status = "✓" if self.optimality.is_feasible else "✗"
        return (
            f"SearchResult: {self.architecture.name} ({feas})\n"
            f"  α = {self.alpha:.4f}\n"
            f"  Optimality: {opt_status}\n"
            f"  Params: {self.metrics.get('params', 0):,}\n"
            f"  FLOPs: {self.metrics.get('flops', 0):,}"
        )


class ArchitectureSearcher:
    """Searches for optimal architectures under constraints.
    
    Phase 6:
        Maximize α subject to:
            Params ≤ P_max
            FLOPs ≤ F_max
            (and other optional constraints)
    """
    
    def __init__(
        self,
        constraints: Optional[ConstraintBounds] = None,
        search_strategy: str = 'grid',
        max_candidates: int = 100,
    ):
        """Initialize ArchitectureSearcher.
        
        Args:
            constraints: Constraint bounds
            search_strategy: Search strategy ('grid', 'random', 'genetic')
            max_candidates: Maximum number of candidates to evaluate
        """
        self.constraints = constraints or ConstraintBounds()
        self.search_strategy = search_strategy
        self.max_candidates = max_candidates
        self._best_architecture: Optional[ArchitectureConfig] = None
        self._best_alpha: Optional[float] = None
        
    def estimate_params(self, arch: ArchitectureConfig, input_dim: int) -> int:
        """Estimate number of parameters in architecture.
        
        Args:
            arch: Architecture configuration
            input_dim: Input dimension
            
        Returns:
            Total number of parameters
        """
        params = 0
        prev_dim = input_dim
        
        for width, ltype in zip(arch.layer_widths, arch.layer_types):
            if ltype == 'linear':
                params += prev_dim * width + width  # weights + biases
            elif 'conv' in ltype:
                # Rough estimate for conv layers
                kernel_size = 3 if 'conv2d' in ltype else 3
                channels_in = prev_dim if isinstance(prev_dim, int) else prev_dim
                params += kernel_size * kernel_size * channels_in * width + width
            prev_dim = width
        
        # Add output layer estimate
        params += prev_dim * 10 + 10  # Assume 10 classes
        
        return params
    
    def estimate_flops(self, arch: ArchitectureConfig, input_dim: int, seq_length: int = 1) -> int:
        """Estimate FLOPs for one forward pass.
        
        Args:
            arch: Architecture configuration
            input_dim: Input dimension
            seq_length: Sequence length (for recurrent models)
            
        Returns:
            Estimated FLOPs
        """
        flops = 0
        prev_dim = input_dim
        batch_size = 32  # Assume standard batch size
        
        for width, ltype in zip(arch.layer_widths, arch.layer_types):
            if ltype == 'linear':
                # Mul-adds for matrix multiplication
                flops += batch_size * prev_dim * width
                # Bias addition
                flops += batch_size * width
            elif 'conv' in ltype:
                # Convolution FLOPs: 2 * kernel_size^2 * channels_in * channels_out * output_size
                kernel_size = 3
                output_size = batch_size * seq_length * width
                flops += 2 * kernel_size * kernel_size * prev_dim * width * output_size
            prev_dim = width
        
        return flops
    
    def check_constraints(
        self,
        arch: ArchitectureConfig,
        input_dim: int,
    ) -> Tuple[bool, Dict[str, float]]:
        """Check if architecture satisfies constraints.
        
        Args:
            arch: Architecture configuration
            input_dim: Input dimension
            
        Returns:
            Tuple of (is_valid, metrics_dict)
        """
        params = self.estimate_params(arch, input_dim)
        flops = self.estimate_flops(arch, input_dim)
        
        metrics = {
            'params': params,
            'flops': flops,
            'params_ratio': params / self.constraints.params_max if self.constraints.params_max < float('inf') else 0,
            'flops_ratio': flops / self.constraints.flops_max if self.constraints.flops_max < float('inf') else 0,
        }
        
        valid = (
            (self.constraints.params_max >= float('inf') or params <= self.constraints.params_max) and
            (self.constraints.flops_max >= float('inf') or flops <= self.constraints.flops_max)
        )
        
        return valid, metrics
    
    def generate_candidates(
        self,
        input_dim: int,
        output_dim: int = 10,
        width_range: Tuple[int, int] = (32, 512),
        depth_range: Tuple[int, int] = (2, 6),
    ) -> List[ArchitectureConfig]:
        """Generate candidate architectures.
        
        Args:
            input_dim: Input dimension
            output_dim: Output dimension
            width_range: Min and max layer widths
            depth_range: Min and max number of layers
            
        Returns:
            List of candidate architectures
        """
        candidates = []
        widths = list(range(width_range[0], width_range[1] + 1, 32))
        depths = list(range(depth_range[0], depth_range[1] + 1))
        
        # Generate combinations
        for depth in depths:
            for _ in range(10):  # 10 random width combinations per depth
                layer_widths = np.random.choice(widths, depth).tolist()
                arch = ArchitectureConfig(
                    name=f"arch_d{depth}_w{layer_widths[0]}",
                    layer_widths=layer_widths,
                    layer_types=['linear'] * depth,
                )
                
                valid, _ = self.check_constraints(arch, input_dim)
                if valid:
                    candidates.append(arch)
                    
                    if len(candidates) >= self.max_candidates:
                        return candidates
        
        return candidates
    
    def search(
        self,
        alpha_predictor: Callable[[ArchitectureConfig, Dict[str, Any]], float],
        input_dim: int,
        **predictor_kwargs
    ) -> Tuple[ArchitectureConfig, float, Dict[str, Any]]:
        """Search for optimal architecture.
        
        Args:
            alpha_predictor: Function that predicts alpha for an architecture
            input_dim: Input dimension
            **predictor_kwargs: Additional arguments for alpha predictor
            
        Returns:
            Tuple of (best_architecture, best_alpha, metadata)
        """
        candidates = self.generate_candidates(input_dim)
        
        best_arch = None
        best_alpha = -float('inf')
        results = []
        
        for arch in candidates:
            try:
                alpha = alpha_predictor(arch, **predictor_kwargs)
                valid, metrics = self.check_constraints(arch, input_dim)
                
                results.append({
                    'arch': arch,
                    'alpha': alpha,
                    'valid': valid,
                    'metrics': metrics,
                })
                
                if valid and alpha > best_alpha:
                    best_alpha = alpha
                    best_arch = arch
                    
            except Exception as e:
                continue
        
        self._best_architecture = best_arch
        self._best_alpha = best_alpha
        
        metadata = {
            'total_candidates': len(candidates),
            'valid_candidates': sum(1 for r in results if r['valid']),
            'all_results': results,
        }
        
        return best_arch, best_alpha, metadata
    
    def search_grid(
        self,
        width_options: List[int],
        depth_options: List[int],
        alpha_predictor: Callable[[ArchitectureConfig, Dict[str, Any]], float],
        input_dim: int,
        **predictor_kwargs
    ) -> Tuple[ArchitectureConfig, float, Dict[str, Any]]:
        """Grid search over architecture options.
        
        Args:
            width_options: List of layer width options
            depth_options: List of depth options
            alpha_predictor: Function that predicts alpha
            input_dim: Input dimension
            **predictor_kwargs: Additional arguments for predictor
            
        Returns:
            Tuple of (best_architecture, best_alpha, metadata)
        """
        candidates = []
        
        for depth in depth_options:
            for widths in itertools.product(width_options, repeat=depth):
                arch = ArchitectureConfig(
                    name=f"arch_d{depth}_w{widths[0]}",
                    layer_widths=list(widths),
                    layer_types=['linear'] * depth,
                )
                
                valid, _ = self.check_constraints(arch, input_dim)
                if valid:
                    candidates.append(arch)
        
        # Evaluate candidates
        results = []
        for arch in candidates:
            try:
                alpha = alpha_predictor(arch, **predictor_kwargs)
                results.append({'arch': arch, 'alpha': alpha})
            except Exception:
                continue
        
        if not results:
            return ArchitectureConfig(name='default', layer_widths=[64], layer_types=['linear']), 0.0, {}
        
        # Sort by alpha
        results.sort(key=lambda x: x['alpha'], reverse=True)
        
        best = results[0]
        self._best_architecture = best['arch']
        self._best_alpha = best['alpha']
        
        return best['arch'], best['alpha'], {'candidates': len(candidates), 'results': results}
    
    def optimize_continuously(
        self,
        alpha_predictor: Callable[[ArchitectureConfig, Dict[str, Any]], float],
        initial_arch: ArchitectureConfig,
        input_dim: int,
        n_iterations: int = 50,
        **predictor_kwargs
    ) -> Tuple[ArchitectureConfig, float]:
        """Continuously optimize architecture using local search.
        
        Args:
            alpha_predictor: Function that predicts alpha
            initial_arch: Starting architecture
            input_dim: Input dimension
            n_iterations: Number of optimization iterations
            **predictor_kwargs: Additional arguments for predictor
            
        Returns:
            Tuple of (optimized_architecture, best_alpha)
        """
        current_arch = initial_arch
        current_alpha = alpha_predictor(current_arch, **predictor_kwargs)
        
        for _ in range(n_iterations):
            # Generate neighbors by width perturbation
            candidates = []
            
            for i, width in enumerate(current_arch.layer_widths):
                for delta in [-32, 32]:
                    new_width = max(32, min(512, width + delta))
                    if new_width != width:
                        new_widths = current_arch.layer_widths.copy()
                        new_widths[i] = new_width
                        
                        new_arch = ArchitectureConfig(
                            name=f"{current_arch.name}_w{i}{delta}",
                            layer_widths=new_widths,
                            layer_types=current_arch.layer_types.copy(),
                        )
                        
                        valid, _ = self.check_constraints(new_arch, input_dim)
                        if valid:
                            candidates.append(new_arch)
            
            # Evaluate candidates
            best_candidate = None
            best_candidate_alpha = -float('inf')
            
            for arch in candidates:
                try:
                    alpha = alpha_predictor(arch, **predictor_kwargs)
                    if alpha > best_candidate_alpha:
                        best_candidate_alpha = alpha
                        best_candidate = arch
                except Exception:
                    continue
            
            # Accept move if improvement
            if best_candidate and best_candidate_alpha > current_alpha:
                current_arch = best_candidate
                current_alpha = best_candidate_alpha
        
        self._best_architecture = current_arch
        self._best_alpha = current_alpha
        
        return current_arch, current_alpha
    
    def search_with_optimality(
        self,
        alpha_predictor: Callable[[ArchitectureConfig, Dict[str, Any]], float],
        optimality_verifier: Callable[[ArchitectureConfig, Dict[str, Any]], 'OptimalityResult'],
        input_dim: int,
        **predictor_kwargs
    ) -> SearchResult:
        """Search for optimal architecture with thermogeometric feasibility filtering.
        
        Phase 6:
            1. Filter architectures by C1 and C2 feasibility
            2. Only consider feasible architectures for optimization
            3. Return feasibility status along with optimal architecture
        
        Args:
            alpha_predictor: Function that predicts alpha for an architecture
            optimality_verifier: Function that returns OptimalityResult for an architecture
            input_dim: Input dimension
            **predictor_kwargs: Additional arguments for predictors
            
        Returns:
            SearchResult with best architecture, alpha, and optimality status
        """
        candidates = self.generate_candidates(input_dim)
        
        best_arch: Optional[ArchitectureConfig] = None
        best_alpha = -float('inf')
        best_optimality: Optional[OptimalityResult] = None
        best_metrics: Dict[str, float] = {}
        
        feasible_candidates = 0
        infeasible_candidates = 0
        
        for arch in candidates:
            try:
                # Check resource constraints first
                valid, metrics = self.check_constraints(arch, input_dim)
                if not valid:
                    continue
                
                # Verify thermogeometric optimality (C1 and C2)
                optimality = optimality_verifier(arch, **predictor_kwargs)
                
                if not optimality.is_feasible:
                    infeasible_candidates += 1
                    continue
                
                # This candidate is feasible - evaluate alpha
                alpha = alpha_predictor(arch, **predictor_kwargs)
                feasible_candidates += 1
                
                if alpha > best_alpha:
                    best_alpha = alpha
                    best_arch = arch
                    best_optimality = optimality
                    best_metrics = metrics
                    
            except Exception as e:
                continue
        
        if best_arch is None:
            # Return default if no feasible architecture found
            # Import here to avoid circular import at module level
            from ..predictor import OptimalityResult
            default_arch = ArchitectureConfig(
                name='default', layer_widths=[64], layer_types=['linear']
            )
            return SearchResult(
                architecture=default_arch,
                alpha=0.0,
                optimality=OptimalityResult(
                    is_feasible=False,
                    c1_satisfied=False,
                    c2_satisfied=False,
                    alpha=0.0,
                ),
                metrics={},
                feasible=False,
            )
        
        self._best_architecture = best_arch
        self._best_alpha = best_alpha
        
        return SearchResult(
            architecture=best_arch,
            alpha=best_alpha,
            optimality=best_optimality,
            metrics=best_metrics,
            feasible=True,
        )
    
    def filter_by_optimality(
        self,
        architectures: List[ArchitectureConfig],
        optimality_verifier: Callable[[ArchitectureConfig, Dict[str, Any]], 'OptimalityResult'],
        input_dim: int,
        **predictor_kwargs
    ) -> Tuple[List[ArchitectureConfig], List['OptimalityResult']]:
        """Filter a list of architectures by thermogeometric feasibility.
        
        Args:
            architectures: List of architectures to filter
            optimality_verifier: Function that returns OptimalityResult
            input_dim: Input dimension
            **predictor_kwargs: Additional arguments for verifier
            
        Returns:
            Tuple of (feasible_architectures, optimality_results)
        """
        feasible_archs = []
        optimality_results = []
        
        for arch in architectures:
            try:
                valid, _ = self.check_constraints(arch, input_dim)
                if not valid:
                    continue
                    
                optimality = optimality_verifier(arch, **predictor_kwargs)
                
                if optimality.is_feasible:
                    feasible_archs.append(arch)
                    optimality_results.append(optimality)
            except Exception:
                continue
        
        return feasible_archs, optimality_results
    
    @property
    def best_architecture(self) -> Optional[ArchitectureConfig]:
        """Return best architecture found."""
        return self._best_architecture
    
    @property
    def best_alpha(self) -> Optional[float]:
        """Return best alpha found."""
        return self._best_alpha
