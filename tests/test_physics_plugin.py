"""Tests for PhysicsPlugin protocol and concrete plugin implementations."""

import pytest

from fracton.core.physics_plugin import PhysicsPlugin
from fracton.core.recursive_engine import RecursiveExecutor, ExecutionContext
from fracton.plugins.cascade import CascadePlugin
from fracton.plugins.correction import CorrectionPlugin
from fracton.plugins.sec import SECPlugin


# ---------------------------------------------------------------------------
# Protocol conformance
# ---------------------------------------------------------------------------

class TestProtocol:
    def test_cascade_is_physics_plugin(self):
        assert isinstance(CascadePlugin(), PhysicsPlugin)

    def test_correction_is_physics_plugin(self):
        assert isinstance(CorrectionPlugin(), PhysicsPlugin)

    def test_sec_is_physics_plugin(self):
        assert isinstance(SECPlugin(), PhysicsPlugin)

    def test_names_are_unique(self):
        names = {CascadePlugin().name, CorrectionPlugin().name, SECPlugin().name}
        assert len(names) == 3


# ---------------------------------------------------------------------------
# CascadePlugin
# ---------------------------------------------------------------------------

class TestCascadePlugin:
    def test_scale_at_depth_zero_is_one(self):
        p = CascadePlugin()
        ctx = ExecutionContext(entropy=0.5, depth=0)
        p.on_recurse(ctx, 0)
        assert p._last_scale == pytest.approx(1.0)

    def test_scale_decreases_with_depth(self):
        p = CascadePlugin()
        ctx = ExecutionContext(entropy=0.5)
        p.on_recurse(ctx, 5)
        assert p._last_scale < 1.0
        assert p._last_scale == pytest.approx(p.base_decay ** 5)

    def test_crystallize_scales_numeric_result(self):
        p = CascadePlugin()
        ctx = ExecutionContext(entropy=0.5)
        p.on_recurse(ctx, 3)
        result = p.on_crystallize(ctx, 100.0)
        assert result == pytest.approx(100.0 * p.base_decay ** 3)

    def test_crystallize_passes_non_numeric(self):
        p = CascadePlugin()
        ctx = ExecutionContext(entropy=0.5)
        p.on_recurse(ctx, 1)
        result = p.on_crystallize(ctx, {"key": "value"})
        assert result == {"key": "value"}

    def test_validate_true_at_reasonable_depth(self):
        p = CascadePlugin()
        ctx = ExecutionContext(entropy=0.5)
        p.on_recurse(ctx, 10)
        assert p.validate(ctx) is True

    def test_validate_false_at_extreme_depth(self):
        p = CascadePlugin(min_energy=0.01)
        ctx = ExecutionContext(entropy=0.5)
        p.on_recurse(ctx, 100)  # φ⁻¹⁰⁰ ≈ 1.4e-21 << 0.01
        assert p.validate(ctx) is False

    def test_cascade_scale_stored_in_metadata(self):
        p = CascadePlugin()
        ctx = ExecutionContext(entropy=0.5, depth=0)
        p.on_recurse(ctx, 3)
        assert "cascade_scale" in ctx.metadata
        assert ctx.metadata["cascade_scale"] == pytest.approx(p.base_decay ** 3)


# ---------------------------------------------------------------------------
# CorrectionPlugin
# ---------------------------------------------------------------------------

class TestCorrectionPlugin:
    def test_default_is_em_correction(self):
        p = CorrectionPlugin()
        # EM correction: 1 + F3/(1·π·F4²) = 1 + 2/(π·9) ≈ 1.0707
        assert p._template.factor == pytest.approx(1.0707, abs=0.01)

    def test_crystallize_applies_factor(self):
        p = CorrectionPlugin()
        ctx = ExecutionContext(entropy=0.5)
        result = p.on_crystallize(ctx, 1.0)
        assert result == pytest.approx(p._template.factor)

    def test_crystallize_non_numeric_passthrough(self):
        p = CorrectionPlugin()
        ctx = ExecutionContext(entropy=0.5)
        assert p.on_crystallize(ctx, [1, 2, 3]) == [1, 2, 3]

    def test_validate_always_true(self):
        p = CorrectionPlugin()
        ctx = ExecutionContext(entropy=0.0)
        assert p.validate(ctx) is True


# ---------------------------------------------------------------------------
# SECPlugin
# ---------------------------------------------------------------------------

class TestSECPlugin:
    def test_validate_above_threshold(self):
        p = SECPlugin()
        ctx = ExecutionContext(entropy=0.5)
        p.on_recurse(ctx, 0)
        assert p.validate(ctx) is True

    def test_validate_below_threshold(self):
        p = SECPlugin()
        ctx = ExecutionContext(entropy=0.01)  # below XI_SEC ≈ 0.0618
        p.on_recurse(ctx, 0)
        assert p.validate(ctx) is False

    def test_validate_at_threshold(self):
        p = SECPlugin()
        ctx = ExecutionContext(entropy=p.threshold)
        p.on_recurse(ctx, 0)
        # At threshold exactly → not > threshold → False
        assert p.validate(ctx) is False

    def test_crystallize_passthrough(self):
        p = SECPlugin()
        ctx = ExecutionContext(entropy=0.5)
        assert p.on_crystallize(ctx, 42) == 42

    def test_reads_entropy_from_dict_context(self):
        p = SECPlugin()
        p.on_recurse({"entropy": 0.001}, 0)
        assert p.validate({}) is False


# ---------------------------------------------------------------------------
# RecursiveExecutor integration
# ---------------------------------------------------------------------------

class TestExecutorPluginIntegration:
    def test_executor_accepts_plugins(self):
        plugins = [CascadePlugin(), SECPlugin()]
        executor = RecursiveExecutor(pac_regulation=False, plugins=plugins)
        assert len(executor._plugins) == 2

    def test_add_plugin_method(self):
        executor = RecursiveExecutor(pac_regulation=False)
        executor.add_plugin(CascadePlugin())
        assert len(executor._plugins) == 1

    def test_plugin_on_recurse_called(self):
        """Verify plugins receive on_recurse during execute()."""
        calls = []

        class TrackingPlugin:
            name = "tracker"
            def on_recurse(self, ctx, depth):
                calls.append(("recurse", depth))
            def on_crystallize(self, ctx, result):
                calls.append(("crystallize", result))
                return result
            def validate(self, ctx):
                return True

        executor = RecursiveExecutor(pac_regulation=False, plugins=[TrackingPlugin()])
        ctx = ExecutionContext(entropy=0.5, depth=0)
        result = executor.execute(lambda mem, ctx: 42, None, ctx)
        assert result == 42
        assert ("recurse", 0) in calls
        assert ("crystallize", 42) in calls

    def test_plugin_validation_halts_recursion(self):
        """When a plugin's validate() returns False, execute returns None."""

        class BlockerPlugin:
            name = "blocker"
            def on_recurse(self, ctx, depth):
                pass
            def on_crystallize(self, ctx, result):
                return result
            def validate(self, ctx):
                return False

        executor = RecursiveExecutor(pac_regulation=False, plugins=[BlockerPlugin()])
        ctx = ExecutionContext(entropy=0.5, depth=0)
        result = executor.execute(lambda mem, ctx: 42, None, ctx)
        assert result is None

    def test_plugin_crystallize_modifies_result(self):
        """CascadePlugin should scale the numeric result."""
        cascade = CascadePlugin()
        executor = RecursiveExecutor(pac_regulation=False, plugins=[cascade])
        ctx = ExecutionContext(entropy=0.5, depth=3)
        result = executor.execute(lambda mem, ctx: 100.0, None, ctx)
        # depth in ctx is 3, so cascade scales by φ⁻³
        assert result == pytest.approx(100.0 * cascade.base_decay ** 3)
