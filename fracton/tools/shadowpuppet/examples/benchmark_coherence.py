"""
ShadowPuppet Coherence Benchmark Suite - Experiment 1B

Systematically measures whether ShadowPuppet's evolutionary loop actually
improves code fitness across generations.

Captures per-run:
  - Per-generation fitness trajectories (mean, max, min coherence)
  - Birth/death counts per generation
  - Crossover vs mutation birth counts
  - Refinement success rates (improved vs failed)
  - Time per generation
  - Population diversity (variance of birth coherence scores)

Configurations tested (all using MockGenerator for reproducibility):
  - baseline      : mutation only (crossover=False, refinement=False)
  - +crossover    : mutation + crossover (crossover=True, refinement=False)
  - +refinement   : mutation + refinement (crossover=False, refinement=True)
  - full          : mutation + crossover + refinement (both True)
  - high_threshold: full with coherence_threshold=0.80
  - low_threshold : full with coherence_threshold=0.55

Usage:
    python -m fracton.tools.shadowpuppet.examples.benchmark_coherence
"""

import io
import math
import time
import contextlib
import statistics
from typing import Any, Dict, List, Optional

from fracton.tools.shadowpuppet import (
    EvolutionConfig,
    GrowthGap,
    SoftwareEvolution,
    TestSuite,
)
from fracton.tools.shadowpuppet.evolution import GenerationStats
from fracton.tools.shadowpuppet.generators.mock import MockGenerator
from fracton.tools.shadowpuppet.protocols import ComponentOrganism
from fracton.tools.shadowpuppet.examples.task_manager_seed import (
    DOMAIN_TYPES,
    build_specs,
    test_manager_status_transitions,
    test_manager_validation,
    test_renderer_colors,
    test_renderer_list,
    test_store_crud,
    test_store_persistence,
)


# ============================================================================
# BENCHMARK CALLBACKS
# ============================================================================


class BenchmarkCallbacks:
    """
    Implements the EvolutionCallbacks protocol (structural subtyping).

    Captures per-generation metrics from evolution events:
      - on_generation_start / on_generation_end  -> timing + stats
      - on_birth                                  -> fitness + generation method
      - on_death                                  -> (counted via GenerationStats)
      - on_refinement                             -> improved vs failed count
    """

    def __init__(self) -> None:
        self.trajectories: List[Dict[str, Any]] = []
        # Per-generation accumulators (reset each generation)
        self._gen_start_time: float = 0.0
        self._gen_birth_fitnesses: List[float] = []
        self._gen_crossover_births: int = 0
        self._gen_mutation_births: int = 0
        self._gen_refinement_improved: int = 0
        self._gen_refinement_failed: int = 0

    # ------------------------------------------------------------------
    # Callback handlers
    # ------------------------------------------------------------------

    def on_generation_start(self, generation: int, population: int) -> None:
        """Record wall-clock start and reset per-generation accumulators."""
        self._gen_start_time = time.time()
        self._gen_birth_fitnesses = []
        self._gen_crossover_births = 0
        self._gen_mutation_births = 0
        self._gen_refinement_improved = 0
        self._gen_refinement_failed = 0

    def on_birth(self, component: ComponentOrganism, fitness: float) -> None:
        """Capture birth fitness and whether it was via crossover or mutation."""
        self._gen_birth_fitnesses.append(fitness)
        if "+crossover" in component.generator_used:
            self._gen_crossover_births += 1
        else:
            self._gen_mutation_births += 1

    def on_death(self, component: ComponentOrganism, reason: str) -> None:
        """Deaths are already tracked in GenerationStats; no extra work here."""
        pass

    def on_refinement(
        self,
        component: ComponentOrganism,
        old_fitness: float,
        new_fitness: float,
    ) -> None:
        """Track refinement outcome."""
        if new_fitness > old_fitness:
            self._gen_refinement_improved += 1
        else:
            self._gen_refinement_failed += 1

    def on_generation_end(self, generation: int, stats: GenerationStats) -> None:
        """Commit all per-generation metrics to the trajectory log."""
        elapsed = time.time() - self._gen_start_time
        births = self._gen_birth_fitnesses

        # Diversity = variance of birth fitness scores for this generation
        diversity = statistics.variance(births) if len(births) > 1 else 0.0

        self.trajectories.append(
            {
                "generation": generation,
                "mean_coherence": stats.mean_coherence,
                "max_coherence": stats.max_coherence,
                "min_coherence": min(births) if births else 0.0,
                "population": stats.population,
                "births": stats.births,
                "deaths": stats.deaths,
                "crossover_births": self._gen_crossover_births,
                "mutation_births": self._gen_mutation_births,
                "refinement_improved": self._gen_refinement_improved,
                "refinement_failed": self._gen_refinement_failed,
                "time_seconds": elapsed,
                "diversity": diversity,
            }
        )

    def reset(self) -> None:
        """Clear all data for reuse across runs."""
        self.__init__()


# ============================================================================
# BENCHMARK RUNNER
# ============================================================================


class BenchmarkRunner:
    """
    Orchestrates evolution benchmarks across configurations and repeated runs.

    Methods:
        run_single      - single evolution run; returns trajectory + summary
        run_suite       - multiple configs x multiple runs
        compare_configs - statistical comparison across configurations
    """

    def __init__(self, quiet: bool = True) -> None:
        """
        Args:
            quiet: If True, suppress SoftwareEvolution print output during runs.
        """
        self.quiet = quiet

    # ------------------------------------------------------------------
    # Core runner
    # ------------------------------------------------------------------

    def run_single(
        self,
        config: EvolutionConfig,
        gaps: List[GrowthGap],
        config_name: str = "unnamed",
        run_idx: int = 0,
    ) -> Dict[str, Any]:
        """
        Single evolution run with the given config.

        Args:
            config:      EvolutionConfig (with save_checkpoints=False for speed)
            gaps:        Growth gaps to evolve
            config_name: Label for this configuration
            run_idx:     Run index within a suite

        Returns:
            Dict with 'trajectories' list and aggregated summary fields.
        """
        callbacks = BenchmarkCallbacks()
        evolution = SoftwareEvolution(
            generator=MockGenerator(),
            config=config,
            callbacks=callbacks,
        )

        # Optionally suppress the evolution engine's print output
        start = time.time()
        if self.quiet:
            with contextlib.redirect_stdout(io.StringIO()):
                results = evolution.grow(gaps)
        else:
            results = evolution.grow(gaps)
        total_time = time.time() - start

        trajectories = callbacks.trajectories

        # ---- Derived metrics ----------------------------------------

        # Delta coherence: improvement from generation 0 -> last
        if len(trajectories) >= 2:
            delta_mean = trajectories[-1]["mean_coherence"] - trajectories[0]["mean_coherence"]
            delta_max = trajectories[-1]["max_coherence"] - trajectories[0]["max_coherence"]
        else:
            delta_mean = delta_max = 0.0

        # Convergence: first generation where min birth fitness >= threshold
        convergence_gen: Optional[int] = None
        for t in trajectories:
            if t["population"] > 0 and t["min_coherence"] >= config.coherence_threshold:
                convergence_gen = t["generation"]
                break

        # Refinement success rate
        total_refs = sum(
            t["refinement_improved"] + t["refinement_failed"] for t in trajectories
        )
        improved_refs = sum(t["refinement_improved"] for t in trajectories)
        refinement_rate: Optional[float] = (
            improved_refs / total_refs if total_refs > 0 else None
        )

        # Birth/death tallies
        total_births = sum(t["births"] for t in trajectories)
        total_deaths = sum(t["deaths"] for t in trajectories)
        total_crossover = sum(t["crossover_births"] for t in trajectories)
        total_mutation = sum(t["mutation_births"] for t in trajectories)

        # Mean population diversity across generations
        mean_diversity = (
            statistics.mean(t["diversity"] for t in trajectories)
            if trajectories
            else 0.0
        )

        final = trajectories[-1] if trajectories else {}

        return {
            "config_name": config_name,
            "run_idx": run_idx,
            "trajectories": trajectories,
            "success": results["success"],
            "generations_run": results["generations"],
            "final_population": results["final_population"],
            "final_mean_coherence": final.get("mean_coherence", 0.0),
            "final_max_coherence": final.get("max_coherence", 0.0),
            "convergence_generation": convergence_gen,
            "delta_mean_coherence": delta_mean,
            "delta_max_coherence": delta_max,
            "total_births": total_births,
            "total_deaths": total_deaths,
            "total_crossover_births": total_crossover,
            "total_mutation_births": total_mutation,
            "refinement_success_rate": refinement_rate,
            "total_time_seconds": total_time,
            "mean_diversity": mean_diversity,
        }

    def run_suite(
        self,
        configs: Dict[str, EvolutionConfig],
        gaps: List[GrowthGap],
        n_runs: int = 3,
    ) -> Dict[str, List[Dict[str, Any]]]:
        """
        Run every config n_runs times.

        Args:
            configs: Mapping of config_name -> EvolutionConfig
            gaps:    Growth gaps (shared across all configs)
            n_runs:  Number of independent runs per config

        Returns:
            Mapping of config_name -> list of run result dicts
        """
        all_results: Dict[str, List[Dict[str, Any]]] = {}
        n_configs = len(configs)
        config_num = 0

        for config_name, config in configs.items():
            config_num += 1
            print(f"\n[{config_num}/{n_configs}] Config: '{config_name}' — {n_runs} runs")

            all_results[config_name] = []

            for run_idx in range(n_runs):
                print(f"  Run {run_idx + 1}/{n_runs}...", end="", flush=True)
                run_result = self.run_single(config, gaps, config_name, run_idx)
                all_results[config_name].append(run_result)

                status = "OK" if run_result["success"] else "EXTINCT"
                gens = run_result["generations_run"]
                fmc = run_result["final_mean_coherence"]
                print(f" [{status}]  gens={gens}  final_mean={fmc:.4f}")

        return all_results

    def compare_configs(
        self,
        results: Dict[str, List[Dict[str, Any]]],
    ) -> Dict[str, Dict[str, Any]]:
        """
        Compute summary statistics across runs for each configuration.

        Args:
            results: Output from run_suite()

        Returns:
            Mapping of config_name -> summary stat dict
        """
        comparison: Dict[str, Dict[str, Any]] = {}

        for config_name, runs in results.items():
            if not runs:
                continue

            def _safe_mean(values: list) -> float:
                vals = [v for v in values if v is not None]
                return statistics.mean(vals) if vals else 0.0

            def _safe_stdev(values: list) -> float:
                vals = [v for v in values if v is not None]
                return statistics.stdev(vals) if len(vals) > 1 else 0.0

            final_means = [r["final_mean_coherence"] for r in runs]
            final_maxes = [r["final_max_coherence"] for r in runs]
            deltas_mean = [r["delta_mean_coherence"] for r in runs]
            deltas_max = [r["delta_max_coherence"] for r in runs]
            conv_gens = [
                r["convergence_generation"]
                for r in runs
                if r["convergence_generation"] is not None
            ]
            ref_rates = [
                r["refinement_success_rate"]
                for r in runs
                if r["refinement_success_rate"] is not None
            ]
            diversities = [r["mean_diversity"] for r in runs]
            times = [r["total_time_seconds"] for r in runs]
            success_flags = [r["success"] for r in runs]

            comparison[config_name] = {
                # Final fitness
                "mean_final_coherence": _safe_mean(final_means),
                "std_final_coherence": _safe_stdev(final_means),
                "max_final_coherence": max(final_maxes) if final_maxes else 0.0,
                # Improvement (gen0 -> last)
                "mean_delta_mean": _safe_mean(deltas_mean),
                "std_delta_mean": _safe_stdev(deltas_mean),
                "mean_delta_max": _safe_mean(deltas_max),
                # Convergence
                "convergence_rate": len(conv_gens) / len(runs),
                "mean_convergence_gen": _safe_mean(conv_gens),
                # Success (any survivors at end)
                "success_rate": sum(success_flags) / len(success_flags),
                # Refinement
                "mean_refinement_success_rate": _safe_mean(ref_rates) if ref_rates else None,
                # Diversity
                "mean_diversity": _safe_mean(diversities),
                # Timing
                "mean_time_seconds": _safe_mean(times),
                # Metadata
                "n_runs": len(runs),
            }

        return comparison


# ============================================================================
# BENCHMARK CONFIGURATIONS
# ============================================================================


def build_benchmark_configs(max_generations: int = 5) -> Dict[str, EvolutionConfig]:
    """
    Build the six benchmark configurations.

    All share the same base parameters; only crossover/refinement
    flags and the coherence threshold differ.

    Args:
        max_generations: Cap on generations per run (default 5 for validation)

    Returns:
        Dict of config_name -> EvolutionConfig
    """
    base: Dict[str, Any] = dict(
        coherence_threshold=0.65,
        candidates_per_gap=3,
        max_generations=max_generations,
        save_checkpoints=False,
        output_dir=None,
    )

    return {
        "baseline": EvolutionConfig(
            **base,
            enable_crossover=False,
            enable_refinement=False,
        ),
        "+crossover": EvolutionConfig(
            **base,
            enable_crossover=True,
            enable_refinement=False,
            crossover_rate=0.3,
        ),
        "+refinement": EvolutionConfig(
            **base,
            enable_crossover=False,
            enable_refinement=True,
            refinement_threshold=0.5,
            max_refinement_attempts=2,
        ),
        "full": EvolutionConfig(
            **base,
            enable_crossover=True,
            enable_refinement=True,
            crossover_rate=0.3,
            refinement_threshold=0.5,
            max_refinement_attempts=2,
        ),
        "high_threshold": EvolutionConfig(
            **{**base, "coherence_threshold": 0.80},
            enable_crossover=True,
            enable_refinement=True,
            crossover_rate=0.3,
            refinement_threshold=0.5,
            max_refinement_attempts=2,
        ),
        "low_threshold": EvolutionConfig(
            **{**base, "coherence_threshold": 0.55},
            enable_crossover=True,
            enable_refinement=True,
            crossover_rate=0.3,
            refinement_threshold=0.5,
            max_refinement_attempts=2,
        ),
    }


# ============================================================================
# OUTPUT FORMATTING
# ============================================================================


def print_trajectory_table(
    config_name: str, runs: List[Dict[str, Any]]
) -> None:
    """Print a per-generation averaged trajectory table for one config."""
    if not runs:
        return

    print(f"\n  Trajectory — '{config_name}' (averaged over {len(runs)} run(s)):")
    header = (
        f"  {'Gen':>4} | {'Mean':>6} | {'Max':>6} | {'Min':>6} | "
        f"{'Diversity':>9} | {'Births':>6} | {'Deaths':>6} | "
        f"{'XoverB':>6} | {'RefOK':>5} | {'Time(s)':>7}"
    )
    print(header)
    print("  " + "-" * (len(header) - 2))

    max_gens = max(len(r["trajectories"]) for r in runs)

    for gen_idx in range(max_gens):
        gen_data = [
            r["trajectories"][gen_idx]
            for r in runs
            if gen_idx < len(r["trajectories"])
        ]
        if not gen_data:
            continue

        def avg(key: str) -> float:
            vals = [d[key] for d in gen_data if d.get(key) is not None]
            return statistics.mean(vals) if vals else 0.0

        print(
            f"  {gen_idx:>4} | "
            f"{avg('mean_coherence'):>6.3f} | "
            f"{avg('max_coherence'):>6.3f} | "
            f"{avg('min_coherence'):>6.3f} | "
            f"{avg('diversity'):>9.5f} | "
            f"{avg('births'):>6.1f} | "
            f"{avg('deaths'):>6.1f} | "
            f"{avg('crossover_births'):>6.1f} | "
            f"{avg('refinement_improved'):>5.1f} | "
            f"{avg('time_seconds'):>7.3f}"
        )


def print_comparison_table(
    comparison: Dict[str, Dict[str, Any]]
) -> None:
    """Print the main cross-configuration comparison table."""
    divider = "=" * 108
    print(f"\n{divider}")
    print("  BENCHMARK RESULTS — Configuration Comparison")
    print(divider)

    col_hdr = (
        f"  {'Config':<18} | "
        f"{'MeanFinal':>9} | "
        f"{'+-Std':>6} | "
        f"{'MaxFinal':>8} | "
        f"{'Dmean':>7} | "
        f"{'Dmax':>7} | "
        f"{'Conv%':>6} | "
        f"{'Succ%':>6} | "
        f"{'Ref%':>6} | "
        f"{'Diversity':>9} | "
        f"{'Time(s)':>7}"
    )
    print(col_hdr)
    print("  " + "-" * (len(col_hdr) - 2))

    for config_name, stats in comparison.items():
        ref_rate = stats.get("mean_refinement_success_rate")
        ref_str = f"{ref_rate * 100:>5.0f}%" if ref_rate is not None else "  N/A "

        delta_mean_sign = "+" if stats["mean_delta_mean"] >= 0 else ""
        delta_max_sign = "+" if stats["mean_delta_max"] >= 0 else ""

        print(
            f"  {config_name:<18} | "
            f"{stats['mean_final_coherence']:>9.4f} | "
            f"{stats['std_final_coherence']:>6.4f} | "
            f"{stats['max_final_coherence']:>8.4f} | "
            f"{delta_mean_sign}{stats['mean_delta_mean']:>6.4f} | "
            f"{delta_max_sign}{stats['mean_delta_max']:>6.4f} | "
            f"{stats['convergence_rate'] * 100:>5.0f}% | "
            f"{stats['success_rate'] * 100:>5.0f}% | "
            f"{ref_str} | "
            f"{stats['mean_diversity']:>9.5f} | "
            f"{stats['mean_time_seconds']:>7.2f}"
        )

    print(divider)
    print(
        "  Columns: MeanFinal=mean coherence at last gen, +-Std=run variance, "
        "MaxFinal=best seen,\n"
        "           Dmean/Dmax=improvement gen0->last, Conv%=runs that converged,\n"
        "           Succ%=runs with survivors, Ref%=refinement success rate, "
        "Diversity=coherence var."
    )
    print(divider)


def print_summary_insights(comparison: Dict[str, Dict[str, Any]]) -> None:
    """Print a brief qualitative summary of key findings."""
    print("\n  KEY INSIGHTS")
    print("  " + "-" * 60)

    baseline = comparison.get("baseline", {})
    full = comparison.get("full", {})

    # Best config by mean final coherence
    best_name = max(comparison, key=lambda k: comparison[k]["mean_final_coherence"])
    best_val = comparison[best_name]["mean_final_coherence"]
    print(f"  Best mean final coherence : '{best_name}'  ({best_val:.4f})")

    # Crossover effect
    xover = comparison.get("+crossover", {})
    if baseline and xover:
        delta = xover["mean_final_coherence"] - baseline["mean_final_coherence"]
        arrow = "UP" if delta > 0 else "DOWN"
        print(f"  Crossover effect (vs baseline)  : {arrow} {abs(delta):.4f}")

    # Refinement effect
    ref = comparison.get("+refinement", {})
    if baseline and ref:
        delta = ref["mean_final_coherence"] - baseline["mean_final_coherence"]
        arrow = "UP" if delta > 0 else "DOWN"
        print(f"  Refinement effect (vs baseline) : {arrow} {abs(delta):.4f}")

    # Full vs baseline
    if baseline and full:
        delta = full["mean_final_coherence"] - baseline["mean_final_coherence"]
        arrow = "UP" if delta > 0 else "DOWN"
        print(f"  Full (both) vs baseline         : {arrow} {abs(delta):.4f}")

    # Threshold comparison
    hi = comparison.get("high_threshold", {})
    lo = comparison.get("low_threshold", {})
    if hi and lo:
        print(
            f"  Threshold 0.80 vs 0.55 final   : "
            f"{hi['mean_final_coherence']:.4f} vs {lo['mean_final_coherence']:.4f}"
        )
        print(
            f"  Success rate (hi vs lo thresh)  : "
            f"{hi['success_rate'] * 100:.0f}% vs {lo['success_rate'] * 100:.0f}%"
        )

    print()


# ============================================================================
# GAP BUILDER
# ============================================================================


def build_gaps() -> List[GrowthGap]:
    """
    Build GrowthGap objects from the task_manager_seed protocols.

    Uses the canonical build_specs() function from task_manager_seed
    and wires in the test suites and domain types.
    """
    task_store_spec, task_manager_spec, cli_renderer_spec, task_app_spec = build_specs()

    return [
        GrowthGap(
            protocol=task_store_spec,
            test_suite=TestSuite(unit=[test_store_persistence, test_store_crud]),
            domain_types=DOMAIN_TYPES,
        ),
        GrowthGap(
            protocol=task_manager_spec,
            test_suite=TestSuite(
                unit=[test_manager_status_transitions, test_manager_validation]
            ),
            domain_types=DOMAIN_TYPES,
        ),
        GrowthGap(
            protocol=cli_renderer_spec,
            test_suite=TestSuite(unit=[test_renderer_colors, test_renderer_list]),
            domain_types=DOMAIN_TYPES,
        ),
        GrowthGap(
            protocol=task_app_spec,
            domain_types=DOMAIN_TYPES,
        ),
    ]


# ============================================================================
# MAIN ENTRY POINT
# ============================================================================


def main(max_generations: int = 5, n_runs: int = 3) -> None:
    """
    Run the full benchmark suite and print results.

    Args:
        max_generations: Generations per run (default 5 for validation)
        n_runs:          Runs per configuration (default 3 for validation)
    """
    print("\n" + "=" * 60)
    print("  ShadowPuppet Coherence Benchmark — Experiment 1B")
    print("=" * 60)
    print(f"  Generator    : MockGenerator (deterministic templates)")
    print(f"  Seed         : task_manager_seed (4 protocols)")
    print(f"  Max gens     : {max_generations}")
    print(f"  Runs/config  : {n_runs}")
    print(f"  Configs      : baseline, +crossover, +refinement, full,")
    print(f"                 high_threshold, low_threshold")
    print("=" * 60)

    gaps = build_gaps()
    configs = build_benchmark_configs(max_generations=max_generations)
    runner = BenchmarkRunner(quiet=True)

    # Run the full suite
    suite_start = time.time()
    all_results = runner.run_suite(configs, gaps, n_runs=n_runs)
    suite_elapsed = time.time() - suite_start

    print(f"\n[*] Suite complete in {suite_elapsed:.1f}s")

    # Per-config trajectory tables
    print("\n" + "=" * 60)
    print("  PER-GENERATION TRAJECTORIES (averaged over runs)")
    print("=" * 60)
    for config_name, runs in all_results.items():
        print_trajectory_table(config_name, runs)

    # Cross-configuration comparison
    comparison = runner.compare_configs(all_results)
    print_comparison_table(comparison)
    print_summary_insights(comparison)


if __name__ == "__main__":
    main(max_generations=5, n_runs=3)
