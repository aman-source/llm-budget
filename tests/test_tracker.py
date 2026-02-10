"""Tests for the tracker module."""
from __future__ import annotations

import tempfile
import threading
from pathlib import Path

import pytest

from llm_budget.tracker import Tracker


@pytest.fixture
def tracker(tmp_path):
    """Create a tracker with a temporary database."""
    db_path = str(tmp_path / "test_spend.db")
    t = Tracker(db_path=db_path)
    yield t
    t.close()


class TestRecord:
    def test_record_and_retrieve(self, tracker):
        tracker.record(
            model="gpt-4o",
            input_tokens=100,
            output_tokens=50,
            cost_usd=0.005,
        )
        history = tracker.get_history(last_n=1)
        assert len(history) == 1
        assert history[0].model == "gpt-4o"
        assert history[0].input_tokens == 100
        assert history[0].output_tokens == 50
        assert history[0].cost_usd == 0.005

    def test_record_with_metadata(self, tracker):
        tracker.record(
            model="gpt-4o",
            input_tokens=100,
            output_tokens=50,
            cost_usd=0.005,
            metadata={"session": "test"},
        )
        history = tracker.get_history(last_n=1)
        assert '"session"' in history[0].metadata


class TestGetSpend:
    def test_total_spend(self, tracker):
        tracker.record(model="gpt-4o", input_tokens=100, output_tokens=50, cost_usd=0.01)
        tracker.record(model="gpt-4o", input_tokens=200, output_tokens=100, cost_usd=0.02)
        total = tracker.get_spend(period="total")
        assert abs(total - 0.03) < 1e-9

    def test_empty_db(self, tracker):
        assert tracker.get_spend(period="total") == 0.0

    def test_model_filter(self, tracker):
        tracker.record(model="gpt-4o", input_tokens=100, output_tokens=50, cost_usd=0.01)
        tracker.record(model="gpt-4o-mini", input_tokens=100, output_tokens=50, cost_usd=0.001)
        assert abs(tracker.get_spend(period="total", model="gpt-4o") - 0.01) < 1e-9
        assert abs(tracker.get_spend(period="total", model="gpt-4o-mini") - 0.001) < 1e-9

    def test_period_daily(self, tracker):
        tracker.record(model="gpt-4o", input_tokens=100, output_tokens=50, cost_usd=0.01)
        spend = tracker.get_spend(period="daily")
        assert spend == 0.01  # Just recorded, should be within the day


class TestSpendBreakdown:
    def test_breakdown(self, tracker):
        tracker.record(model="gpt-4o", input_tokens=100, output_tokens=50, cost_usd=0.01)
        tracker.record(model="gpt-4o-mini", input_tokens=100, output_tokens=50, cost_usd=0.001)
        tracker.record(model="gpt-4o", input_tokens=200, output_tokens=100, cost_usd=0.02)
        breakdown = tracker.get_spend_breakdown(period="total")
        assert "gpt-4o" in breakdown
        assert "gpt-4o-mini" in breakdown
        assert abs(breakdown["gpt-4o"] - 0.03) < 1e-9
        assert abs(breakdown["gpt-4o-mini"] - 0.001) < 1e-9


class TestHistory:
    def test_limit(self, tracker):
        for i in range(10):
            tracker.record(model="gpt-4o", input_tokens=i, output_tokens=i, cost_usd=0.001)
        history = tracker.get_history(last_n=5)
        assert len(history) == 5

    def test_model_filter(self, tracker):
        tracker.record(model="gpt-4o", input_tokens=100, output_tokens=50, cost_usd=0.01)
        tracker.record(model="gpt-4o-mini", input_tokens=100, output_tokens=50, cost_usd=0.001)
        history = tracker.get_history(model="gpt-4o")
        assert len(history) == 1
        assert history[0].model == "gpt-4o"


class TestOutputRatio:
    def test_returns_none_when_insufficient_data(self, tracker):
        # Only 2 records, but min_samples=5 by default
        tracker.record(model="gpt-4o", input_tokens=100, output_tokens=50, cost_usd=0.01)
        tracker.record(model="gpt-4o", input_tokens=200, output_tokens=100, cost_usd=0.02)
        assert tracker.get_output_ratio("gpt-4o") is None

    def test_returns_none_for_unknown_model(self, tracker):
        assert tracker.get_output_ratio("nonexistent-model") is None

    def test_learns_ratio_from_history(self, tracker):
        # 10 records all with ratio = 0.5 (output/input)
        for _ in range(10):
            tracker.record(model="gpt-4o", input_tokens=100, output_tokens=50, cost_usd=0.01)
        ratio = tracker.get_output_ratio("gpt-4o")
        assert ratio is not None
        assert abs(ratio - 0.5) < 1e-9

    def test_median_not_mean(self, tracker):
        # 4 records at ratio=0.5, 1 outlier at ratio=5.0
        # Median should be 0.5, not dragged up by the outlier
        for _ in range(4):
            tracker.record(model="gpt-4o", input_tokens=100, output_tokens=50, cost_usd=0.01)
        tracker.record(model="gpt-4o", input_tokens=100, output_tokens=500, cost_usd=0.05)
        ratio = tracker.get_output_ratio("gpt-4o", min_samples=5)
        assert ratio is not None
        assert abs(ratio - 0.5) < 1e-9

    def test_model_isolation(self, tracker):
        # gpt-4o: ratio=0.5, gpt-4o-mini: ratio=2.0
        for _ in range(5):
            tracker.record(model="gpt-4o", input_tokens=100, output_tokens=50, cost_usd=0.01)
        for _ in range(5):
            tracker.record(model="gpt-4o-mini", input_tokens=100, output_tokens=200, cost_usd=0.01)
        assert abs(tracker.get_output_ratio("gpt-4o") - 0.5) < 1e-9
        assert abs(tracker.get_output_ratio("gpt-4o-mini") - 2.0) < 1e-9

    def test_custom_min_samples(self, tracker):
        for _ in range(3):
            tracker.record(model="gpt-4o", input_tokens=100, output_tokens=50, cost_usd=0.01)
        # Default min_samples=5 → None
        assert tracker.get_output_ratio("gpt-4o") is None
        # Lower threshold → returns value
        assert tracker.get_output_ratio("gpt-4o", min_samples=3) is not None

    def test_skips_zero_token_records(self, tracker):
        # Records with input_tokens=0 or output_tokens=0 should be excluded
        for _ in range(5):
            tracker.record(model="gpt-4o", input_tokens=0, output_tokens=0, cost_usd=0.0)
        for _ in range(3):
            tracker.record(model="gpt-4o", input_tokens=100, output_tokens=50, cost_usd=0.01)
        # Only 3 valid records, default min_samples=5 → None
        assert tracker.get_output_ratio("gpt-4o") is None


class TestValidation:
    def test_negative_cost_rejected(self, tracker):
        with pytest.raises(ValueError, match="negative"):
            tracker.record(model="gpt-4o", input_tokens=100, output_tokens=50, cost_usd=-1.0)

    def test_negative_input_tokens_rejected(self, tracker):
        with pytest.raises(ValueError, match="negative"):
            tracker.record(model="gpt-4o", input_tokens=-1, output_tokens=50, cost_usd=0.01)

    def test_negative_output_tokens_rejected(self, tracker):
        with pytest.raises(ValueError, match="negative"):
            tracker.record(model="gpt-4o", input_tokens=100, output_tokens=-1, cost_usd=0.01)

    def test_zero_cost_accepted(self, tracker):
        record = tracker.record(model="gpt-4o", input_tokens=0, output_tokens=0, cost_usd=0.0)
        assert record.cost_usd == 0.0


class TestContextManager:
    def test_context_manager(self, tmp_path):
        db_path = str(tmp_path / "ctx_test.db")
        with Tracker(db_path=db_path) as t:
            t.record(model="gpt-4o", input_tokens=10, output_tokens=5, cost_usd=0.001)
            assert t.get_spend(period="total") == 0.001
        # After exit, connection is closed
        with pytest.raises(Exception):
            t.record(model="gpt-4o", input_tokens=10, output_tokens=5, cost_usd=0.001)


class TestThreadSafety:
    def test_concurrent_writes(self, tracker):
        errors = []

        def write_records():
            try:
                for _ in range(20):
                    tracker.record(
                        model="gpt-4o", input_tokens=10, output_tokens=5, cost_usd=0.001
                    )
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=write_records) for _ in range(5)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert not errors
        total = tracker.get_spend(period="total")
        assert abs(total - 0.1) < 1e-9  # 5 threads * 20 records * 0.001
