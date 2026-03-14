"""
Tests for fracton.experiment toolkit.
"""

import json
import os
import tempfile
from pathlib import Path

import pytest


class TestExperimentResult:
    """Test the ExperimentResult and PartResult classes."""

    def test_part_result_basic(self):
        from fracton.experiment import PartResult
        part = PartResult("A", "Test cascade density")
        part.add_row({"r": 1.5, "rho": 0.667})
        part.add_row({"r": 2.0, "rho": 0.500})
        part.finding = "Density follows 1/r"
        part.passed = True

        d = part.to_dict()
        assert d["description"] == "Test cascade density"
        assert d["finding"] == "Density follows 1/r"
        assert d["passed"] is True
        assert len(d["rows"]) == 2

    def test_part_result_minimal(self):
        from fracton.experiment import PartResult
        part = PartResult("B", "Minimal part")
        d = part.to_dict()
        assert "finding" not in d
        assert "passed" not in d
        assert "rows" not in d

    def test_experiment_result_full_workflow(self):
        from fracton.experiment import ExperimentResult
        result = ExperimentResult("exp_42_test")

        part_a = result.add_part("A", "Cascade density")
        part_a.passed = True
        part_a.finding = "Matches prediction"

        part_b = result.add_part("B", "Metric components")
        part_b.passed = True

        synth = result.synthesize("CONFIRMED", "All parts pass")
        assert synth["status"] == "CONFIRMED"
        assert result.overall_passed is True

        d = result.to_dict()
        assert d["experiment"] == "exp_42_test"
        assert "A" in d["parts"]
        assert "B" in d["parts"]
        assert d["synthesis"]["status"] == "CONFIRMED"

    def test_experiment_result_partial(self):
        from fracton.experiment import ExperimentResult
        result = ExperimentResult("exp_43_partial")
        result.add_part("A", "Pass part").passed = True
        result.add_part("B", "Fail part").passed = False

        assert result.overall_passed is False

    def test_save_and_load(self):
        from fracton.experiment import ExperimentResult
        result = ExperimentResult("exp_44_save")
        result.add_part("A", "Test save").passed = True
        result.synthesize("CONFIRMED", "Save works")

        with tempfile.TemporaryDirectory() as tmpdir:
            path = result.save(tmpdir, filename="test_output.json")
            assert path.exists()

            with open(path) as f:
                loaded = json.load(f)
            assert loaded["experiment"] == "exp_44_save"
            assert loaded["synthesis"]["status"] == "CONFIRMED"

    def test_save_results_legacy(self):
        from fracton.experiment import save_results
        data = {"experiment": "legacy", "value": 42}

        with tempfile.TemporaryDirectory() as tmpdir:
            path = save_results(data, tmpdir, "legacy_test")
            assert path.exists()
            with open(path) as f:
                loaded = json.load(f)
            assert loaded["value"] == 42


class TestFormatting:
    """Test formatting utilities."""

    def test_print_header(self, capsys):
        from fracton.experiment import print_header
        print_header("Test Title", "subtitle")
        captured = capsys.readouterr()
        assert "Test Title" in captured.out
        assert "subtitle" in captured.out
        assert "=" * 72 in captured.out

    def test_print_table(self, capsys):
        from fracton.experiment.formatting import print_table
        print_table(
            ["Name", "Value"],
            [("alpha", 0.007297), ("theta", 0.23121)],
        )
        captured = capsys.readouterr()
        assert "Name" in captured.out
        assert "alpha" in captured.out

    def test_print_result_ppm(self, capsys):
        from fracton.experiment.formatting import print_result
        print_result("alpha_EM", 0.007297311, 0.007297353, error_type="ppm")
        captured = capsys.readouterr()
        assert "ppm" in captured.out
        assert "alpha_EM" in captured.out

    def test_print_result_pct(self, capsys):
        from fracton.experiment.formatting import print_result
        print_result("G", 6.662e-11, 6.674e-11, error_type="pct")
        captured = capsys.readouterr()
        assert "%" in captured.out


class TestMetadata:
    """Test metadata utilities."""

    def test_experiment_header(self, capsys):
        from fracton.experiment import experiment_header
        meta = experiment_header(
            "exp_42", "Test cascade",
            paper="PACSeries Paper 6",
            milestone="milestone4",
        )
        captured = capsys.readouterr()
        assert "exp_42" in captured.out
        assert "PACSeries Paper 6" in captured.out
        assert meta["experiment"] == "exp_42"
        assert meta["paper"] == "PACSeries Paper 6"
        assert "timestamp" in meta

    def test_experiment_docstring(self):
        from fracton.experiment.metadata import experiment_docstring
        doc = experiment_docstring(
            purpose="Test something",
            hypothesis=["H1", "H2"],
            design=["Part A: test", "Part B: verify"],
            corpus_context=["exp_01: base case"],
            output="results/test.json",
        )
        assert "PURPOSE: Test something" in doc
        assert "1. H1" in doc
        assert "Part A: test" in doc
        assert "exp_01: base case" in doc


class TestTimer:
    """Test timing utilities."""

    def test_timer_measures_time(self):
        import time
        from fracton.experiment import timer
        with timer() as t:
            time.sleep(0.05)
        assert t.elapsed >= 0.04  # allow some slack

    def test_timer_factory(self):
        from fracton.experiment import timer, Timer
        t = timer()
        assert isinstance(t, Timer)
