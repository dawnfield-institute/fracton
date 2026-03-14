"""
Tests for fracton.lang.physics_decorators module.
"""

import math
import pytest


class TestPACConserved:
    """Test @pac_conserved decorator."""

    def test_valid_conservation(self):
        from fracton.lang.physics_decorators import pac_conserved

        @pac_conserved()
        def split(total):
            return {
                "potential": total,
                "actualized": total * 0.6,
                "xi": total * 0.3,
                "theta": total * 0.1,
            }

        result = split(100)
        assert result["potential"] == 100

    def test_violation_raises(self):
        from fracton.lang.physics_decorators import pac_conserved

        @pac_conserved(tolerance=1e-10)
        def bad_split(total):
            return {
                "potential": total,
                "actualized": total * 0.5,
                "xi": total * 0.3,
                "theta": total * 0.1,  # Only 0.9, not 1.0
            }

        with pytest.raises(ValueError, match="PAC conservation violated"):
            bad_split(100)

    def test_custom_components(self):
        from fracton.lang.physics_decorators import pac_conserved

        @pac_conserved(components=("P", "A", "x", "t"))
        def split(total):
            return {"P": total, "A": total * 0.5, "x": total * 0.3, "t": total * 0.2}

        result = split(100)
        assert result["P"] == 100

    def test_non_dict_passthrough(self):
        from fracton.lang.physics_decorators import pac_conserved

        @pac_conserved()
        def scalar():
            return 42

        assert scalar() == 42  # No validation on non-dict

    def test_metadata(self):
        from fracton.lang.physics_decorators import pac_conserved

        @pac_conserved(tolerance=0.01)
        def f():
            return {}

        assert f._fracton_pac_conserved is True
        assert f._fracton_pac_tolerance == 0.01


class TestFibonacciCascade:
    """Test @fibonacci_cascade decorator."""

    def test_basic_cascade(self):
        from fracton.lang.physics_decorators import fibonacci_cascade

        @fibonacci_cascade(depth=5)
        def energies(cascade_weights):
            return [100 * w for w in cascade_weights]

        result = energies()
        assert len(result) == 5
        assert result[0] == 100 * 1.0  # k=0: weight=1
        assert result[1] < result[0]   # Decaying

    def test_weights_are_phi_inv(self):
        from fracton.lang.physics_decorators import fibonacci_cascade
        from fracton.constants.mathematical import PHI_INV

        @fibonacci_cascade(depth=3)
        def get_weights(cascade_weights):
            return cascade_weights

        weights = get_weights()
        assert abs(weights[0] - 1.0) < 1e-15
        assert abs(weights[1] - PHI_INV) < 1e-15
        assert abs(weights[2] - PHI_INV ** 2) < 1e-15

    def test_metadata(self):
        from fracton.lang.physics_decorators import fibonacci_cascade

        @fibonacci_cascade(depth=7)
        def f(cascade_weights):
            return []

        assert f._fracton_fibonacci_cascade is True
        assert f._fracton_cascade_depth == 7
        assert len(f._fracton_cascade_weights) == 7


class TestCorrectionTemplate:
    """Test @correction_template decorator."""

    def test_em_correction(self):
        from fracton.lang.physics_decorators import correction_template
        from fracton.constants.mathematical import PHI
        from fracton.fibonacci import fib
        from fracton.constants import ALPHA_EM_PAC

        @correction_template(force="em")
        def alpha_base():
            return fib(3) / (fib(4) * PHI * fib(10))

        result = alpha_base()
        assert abs(result - ALPHA_EM_PAC) < 1e-15

    def test_gravity_correction(self):
        from fracton.lang.physics_decorators import correction_template

        @correction_template(force="gravity")
        def base():
            return 1.0

        result = base()
        assert result > 2.0  # K ≈ 2.159

    def test_custom_params(self):
        from fracton.lang.physics_decorators import correction_template

        @correction_template(force="custom", custom_a=5, custom_b=3, custom_n=2, custom_sign=+1)
        def base():
            return 1.0

        result = base()
        assert result > 1.0  # Anti-screening

    def test_unknown_force_raises(self):
        from fracton.lang.physics_decorators import correction_template
        with pytest.raises(ValueError, match="Unknown force"):
            @correction_template(force="antimatter")
            def f():
                return 1.0

    def test_metadata(self):
        from fracton.lang.physics_decorators import correction_template

        @correction_template(force="em")
        def f():
            return 1.0

        assert f._fracton_correction_template is True
        assert f._fracton_correction_force == "em"
        assert f._fracton_correction_params == (10, 7, 4, -1)


class TestMEDBounded:
    """Test @med_bounded decorator."""

    def test_within_bound(self):
        from fracton.lang.physics_decorators import med_bounded

        @med_bounded(max_depth=2)
        def emerge(depth=0):
            return f"emerged at depth {depth}"

        assert emerge(depth=1) == "emerged at depth 1"
        assert emerge(depth=2) == "emerged at depth 2"

    def test_exceeds_bound(self):
        from fracton.lang.physics_decorators import med_bounded

        @med_bounded(max_depth=2)
        def emerge(depth=0):
            return "ok"

        with pytest.raises(ValueError, match="MED constraint violated"):
            emerge(depth=3)

    def test_no_depth_arg(self):
        from fracton.lang.physics_decorators import med_bounded

        @med_bounded()
        def f():
            return "ok"

        assert f() == "ok"  # depth defaults to 0


class TestSECPhase:
    """Test @sec_phase decorator."""

    def test_above_threshold(self):
        from fracton.lang.physics_decorators import sec_phase

        @sec_phase(threshold="xi")
        def actualize(entropy=None):
            return "actualized"

        result = actualize(entropy=1.1)
        assert result == "actualized"

    def test_below_threshold(self):
        from fracton.lang.physics_decorators import sec_phase

        @sec_phase(threshold="xi")
        def actualize(entropy=None):
            return "actualized"

        result = actualize(entropy=0.5)
        assert result is None  # Potential phase

    def test_xi_floor_threshold(self):
        from fracton.lang.physics_decorators import sec_phase
        from fracton.constants.mathematical import LN2_SQUARED

        @sec_phase(threshold="xi_floor")
        def f(entropy=None):
            return "ok"

        # xi_floor = 1 - ln^2(2) ≈ 0.5199
        assert f(entropy=0.6) == "ok"
        assert f(entropy=0.4) is None

    def test_custom_threshold(self):
        from fracton.lang.physics_decorators import sec_phase

        @sec_phase(threshold="custom", custom_threshold=0.75)
        def f(entropy=None):
            return "ok"

        assert f(entropy=0.8) == "ok"
        assert f(entropy=0.5) is None

    def test_no_entropy_passthrough(self):
        from fracton.lang.physics_decorators import sec_phase

        @sec_phase(threshold="xi")
        def f(entropy=None):
            return "ok"

        assert f() == "ok"  # No entropy = no gating

    def test_metadata(self):
        from fracton.lang.physics_decorators import sec_phase

        @sec_phase(threshold="xi")
        def f(entropy=None):
            pass

        assert f._fracton_sec_phase is True
        assert abs(f._fracton_sec_threshold - 1.0584) < 0.001
