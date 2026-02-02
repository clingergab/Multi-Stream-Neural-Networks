"""
Comprehensive tests for HyperOptSearch and HistoricalMedianStoppingRule warm-start.

Tests cover:
- HyperOptSearch save/restore roundtrip
- A→B→A scenario (switching search spaces, correct history loaded for each)
- A→B→A→B accumulation (4 sessions, each config accumulates independently)
- Backward compatibility (old-format checkpoint migration)
- HistoricalMedianStoppingRule hash filtering (A→B→A)
- HistoricalMedianStoppingRule with numeric hash dtype in CSV
- HistoricalMedianStoppingRule epoch CSV accumulation (append, not overwrite)
- _setup_hyperopt monkey-patch (trials preserved after set_search_properties)
- Monkey-patch cleanup before save (no stale closures pickled)
- Error handling (corrupt checkpoint, missing file)
- Trial accumulation across sessions (never lose history)
- Combined integration test (both components, full A→B→A→B flow)
- Cell 26 epoch CSV save logic (dedup, hash tagging, merge)
- NaN/mixed-type hash columns in CSV
"""

import hashlib
import json
import os
import shutil
import tempfile

import pandas as pd
import pytest
from ray.tune.search.hyperopt import HyperOptSearch
from ray.tune.search.sample import Domain

from src.training.historical_median_stopping import HistoricalMedianStoppingRule


# ---------------------------------------------------------------------------
# Helpers — replicate notebook logic exactly
# ---------------------------------------------------------------------------

def get_search_space_hash(search_space: dict) -> str:
    """Replicate the notebook's hash function (Cell 23)."""
    def _serialize_value(v):
        if isinstance(v, Domain):
            if hasattr(v, "categories"):
                return sorted([repr(c) for c in v.categories])
            return repr(v.domain_str) if hasattr(v, "domain_str") else type(v).__name__
        return repr(v)

    space_repr = {k: _serialize_value(v) for k, v in sorted(search_space.items())}
    space_str = json.dumps(space_repr, sort_keys=True)
    return hashlib.md5(space_str.encode()).hexdigest()[:8]


def _make_searcher_with_trials(space, n_trials=5, accuracy_base=0.5, accuracy_step=0.01):
    """Create a HyperOptSearch, feed it n_trials observations, return it."""
    searcher = HyperOptSearch(metric="accuracy", mode="max")
    searcher.set_search_properties(
        metric="accuracy",
        mode="max",
        config=space,
        spec=None,
    )
    for i in range(n_trials):
        trial_id = f"trial_{i}"
        searcher.suggest(trial_id)
        searcher.on_trial_complete(
            trial_id, {"accuracy": accuracy_base + i * accuracy_step}
        )
    return searcher


def _restore_searcher(checkpoint_path, space):
    """Restore a HyperOptSearch from checkpoint, applying the full Cell 24 logic."""
    searcher = HyperOptSearch(metric="accuracy", mode="max")
    searcher.restore(checkpoint_path)

    # Clear stale trial mappings
    searcher._live_trial_mapping = {}

    # Remove any instance-level _setup_hyperopt from pickle
    if '_setup_hyperopt' in searcher.__dict__:
        del searcher.__dict__['_setup_hyperopt']

    # Wrap _setup_hyperopt to preserve restored trials
    restored_trials = searcher._hpopt_trials
    original_setup = searcher._setup_hyperopt

    def _patched_setup():
        original_setup()
        searcher._hpopt_trials = restored_trials

    searcher._setup_hyperopt = _patched_setup

    # Clear saved search space so set_search_properties works
    searcher._space = None
    searcher.domain = None
    searcher._points_to_evaluate = None

    return searcher


def _save_searcher(searcher, checkpoint_path):
    """Save a HyperOptSearch, applying the full Cell 24 post-fit + Cell 26 logic."""
    # Remove monkey-patched _setup_hyperopt so save() doesn't pickle the closure
    if '_setup_hyperopt' in searcher.__dict__:
        del searcher.__dict__['_setup_hyperopt']
    searcher.save(checkpoint_path)


def _simulate_session(space, checkpoint_path, n_new_trials, trial_prefix="trial"):
    """Simulate a full notebook session: restore (if exists) → suggest/complete → save.

    Returns the searcher after save, with total trial count.
    """
    if os.path.exists(checkpoint_path):
        searcher = _restore_searcher(checkpoint_path, space)
        searcher.set_search_properties(
            metric="accuracy", mode="max", config=space, spec=None
        )
    else:
        searcher = HyperOptSearch(metric="accuracy", mode="max")
        searcher.set_search_properties(
            metric="accuracy", mode="max", config=space, spec=None
        )

    n_before = len(searcher._hpopt_trials.trials)
    for i in range(n_new_trials):
        tid = f"{trial_prefix}_{i}"
        searcher.suggest(tid)
        searcher.on_trial_complete(tid, {"accuracy": 0.5 + i * 0.01})

    _save_searcher(searcher, checkpoint_path)
    return searcher, n_before


def _simulate_epoch_csv_save(epoch_csv_path, trial_data, search_space_hash):
    """Simulate Cell 26 epoch CSV saving logic.

    trial_data: list of (trial_id, [(epoch, accuracy), ...])
    Appends to existing CSV if present, deduplicates.
    """
    rows = []
    for trial_id, epochs in trial_data:
        for epoch, acc in epochs:
            rows.append({
                "trial_id": trial_id,
                "training_iteration": epoch,
                "accuracy": acc,
                "search_space_hash": search_space_hash,
            })
    new_df = pd.DataFrame(rows)

    if os.path.exists(epoch_csv_path):
        previous_df = pd.read_csv(epoch_csv_path)
        combined = pd.concat([previous_df, new_df], ignore_index=True)
        combined = combined.drop_duplicates(
            subset=["trial_id", "training_iteration"], keep="last"
        )
        combined.to_csv(epoch_csv_path, index=False)
    else:
        new_df.to_csv(epoch_csv_path, index=False)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def tmp_dir():
    d = tempfile.mkdtemp()
    yield d
    shutil.rmtree(d)


@pytest.fixture
def space_a():
    from ray import tune
    return {
        "lr": tune.choice([1e-3, 2e-3, 3e-3]),
        "batch_size": tune.choice([32, 64]),
    }


@pytest.fixture
def space_b():
    from ray import tune
    return {
        "lr": tune.choice([1e-4, 2e-4]),
        "batch_size": tune.choice([16, 32, 64]),
    }


@pytest.fixture
def space_c():
    from ray import tune
    return {
        "lr": tune.choice([5e-3, 6e-3]),
        "wd": tune.choice([1e-4]),
    }


# ===========================================================================
# HyperOptSearch Tests
# ===========================================================================

class TestHyperOptSearchBasics:
    """Basic save/restore functionality."""

    def test_save_restore_roundtrip(self, tmp_dir, space_a):
        """Save → restore preserves exact trial count."""
        searcher = _make_searcher_with_trials(space_a, n_trials=5)
        assert len(searcher._hpopt_trials.trials) == 5

        path = os.path.join(tmp_dir, "searcher.pkl")
        searcher.save(path)
        assert os.path.exists(path)

        restored = _restore_searcher(path, space_a)
        assert len(restored._hpopt_trials.trials) == 5

    def test_restore_then_set_search_properties_preserves_trials(self, tmp_dir, space_a):
        """set_search_properties (what Tuner.fit calls) doesn't destroy restored trials."""
        searcher = _make_searcher_with_trials(space_a, n_trials=7)
        path = os.path.join(tmp_dir, "searcher.pkl")
        searcher.save(path)

        restored = _restore_searcher(path, space_a)
        result = restored.set_search_properties(
            metric="accuracy", mode="max", config=space_a, spec=None
        )
        assert result is True
        assert len(restored._hpopt_trials.trials) == 7

    def test_suggest_after_restore(self, tmp_dir, space_a):
        """Restored searcher produces valid suggestions."""
        searcher = _make_searcher_with_trials(space_a, n_trials=5)
        path = os.path.join(tmp_dir, "searcher.pkl")
        searcher.save(path)

        restored = _restore_searcher(path, space_a)
        restored.set_search_properties(
            metric="accuracy", mode="max", config=space_a, spec=None
        )
        suggestion = restored.suggest("new_trial_0")
        assert suggestion is not None
        assert "lr" in suggestion
        assert "batch_size" in suggestion

    def test_suggest_produces_valid_values(self, tmp_dir, space_a):
        """Suggestions after restore contain values from the search space."""
        searcher = _make_searcher_with_trials(space_a, n_trials=10)
        path = os.path.join(tmp_dir, "searcher.pkl")
        searcher.save(path)

        restored = _restore_searcher(path, space_a)
        restored.set_search_properties(
            metric="accuracy", mode="max", config=space_a, spec=None
        )

        for i in range(5):
            suggestion = restored.suggest(f"check_{i}")
            assert suggestion["lr"] in [1e-3, 2e-3, 3e-3]
            assert suggestion["batch_size"] in [32, 64]
            restored.on_trial_complete(f"check_{i}", {"accuracy": 0.6})


class TestHyperOptSearchMonkeyPatch:
    """Monkey-patch lifecycle: applied on restore, removed before save."""

    def test_monkey_patch_not_in_fresh_searcher(self, space_a):
        """Fresh searcher has no instance-level _setup_hyperopt."""
        searcher = _make_searcher_with_trials(space_a, n_trials=3)
        assert '_setup_hyperopt' not in searcher.__dict__

    def test_monkey_patch_not_pickled(self, tmp_dir, space_a):
        """After cleanup + save, no instance-level _setup_hyperopt in pickle."""
        searcher = _make_searcher_with_trials(space_a, n_trials=3)
        path = os.path.join(tmp_dir, "searcher.pkl")
        searcher.save(path)

        restored = HyperOptSearch(metric="accuracy", mode="max")
        restored.restore(path)
        assert '_setup_hyperopt' not in restored.__dict__

    def test_cleanup_before_save_removes_patch(self, tmp_dir, space_a):
        """_save_searcher helper removes the monkey-patch before saving."""
        searcher = _make_searcher_with_trials(space_a, n_trials=3)
        path = os.path.join(tmp_dir, "searcher.pkl")
        searcher.save(path)

        restored = _restore_searcher(path, space_a)
        # After restore, monkey-patch IS in __dict__
        assert '_setup_hyperopt' in restored.__dict__

        # After _save_searcher, it's removed
        path2 = os.path.join(tmp_dir, "searcher2.pkl")
        _save_searcher(restored, path2)
        assert '_setup_hyperopt' not in restored.__dict__

    def test_stale_monkey_patch_removed_on_restore(self, tmp_dir, space_a):
        """If a previous session's monkey-patch was pickled, restore removes it."""
        searcher = _make_searcher_with_trials(space_a, n_trials=3)
        path = os.path.join(tmp_dir, "searcher.pkl")

        # Deliberately inject a fake monkey-patch into __dict__ before save
        # (simulating a bug where cleanup didn't happen)
        searcher._setup_hyperopt = lambda: None  # instance-level override
        searcher.save(path)

        # Verify it was pickled
        raw_restored = HyperOptSearch(metric="accuracy", mode="max")
        raw_restored.restore(path)
        assert '_setup_hyperopt' in raw_restored.__dict__

        # Our _restore_searcher should clean it up
        proper_restored = _restore_searcher(path, space_a)
        # It will have a NEW monkey-patch (our wrapper), not the stale one
        assert '_setup_hyperopt' in proper_restored.__dict__
        # Verify it works — the wrapper should preserve trials
        proper_restored.set_search_properties(
            metric="accuracy", mode="max", config=space_a, spec=None
        )
        assert len(proper_restored._hpopt_trials.trials) == 3


class TestHyperOptSearchAccumulation:
    """Trials accumulate across sessions, never overwritten."""

    def test_two_session_accumulation(self, tmp_dir, space_a):
        """Session 1: 5 trials, Session 2: 5 more → 10 total."""
        hash_a = get_search_space_hash(space_a)
        path = os.path.join(tmp_dir, f"hyperopt_searcher_{hash_a}.pkl")

        s1, _ = _simulate_session(space_a, path, 5, "s1")
        assert len(s1._hpopt_trials.trials) == 5

        s2, n_before = _simulate_session(space_a, path, 5, "s2")
        assert n_before == 5
        assert len(s2._hpopt_trials.trials) == 10

    def test_three_session_accumulation(self, tmp_dir, space_a):
        """5 → 10 → 15 across 3 sessions."""
        hash_a = get_search_space_hash(space_a)
        path = os.path.join(tmp_dir, f"hyperopt_searcher_{hash_a}.pkl")

        _simulate_session(space_a, path, 5, "s1")
        _simulate_session(space_a, path, 5, "s2")
        s3, n_before = _simulate_session(space_a, path, 5, "s3")
        assert n_before == 10
        assert len(s3._hpopt_trials.trials) == 15

        # Verify final checkpoint has 15
        final = _restore_searcher(path, space_a)
        assert len(final._hpopt_trials.trials) == 15

    def test_accumulation_persists_after_reread(self, tmp_dir, space_a):
        """After 3 sessions (15 trials), re-reading checkpoint still shows 15."""
        hash_a = get_search_space_hash(space_a)
        path = os.path.join(tmp_dir, f"hyperopt_searcher_{hash_a}.pkl")

        _simulate_session(space_a, path, 5, "s1")
        _simulate_session(space_a, path, 5, "s2")
        _simulate_session(space_a, path, 5, "s3")

        # Read it multiple times — should always be 15
        for _ in range(3):
            r = _restore_searcher(path, space_a)
            assert len(r._hpopt_trials.trials) == 15


class TestHyperOptSearchSpaceSwitching:
    """Switching between search spaces preserves each independently."""

    def test_a_b_a_scenario(self, tmp_dir, space_a, space_b):
        """A(5) → B(3) → A again: A still has 5, B still has 3."""
        hash_a = get_search_space_hash(space_a)
        hash_b = get_search_space_hash(space_b)
        assert hash_a != hash_b

        path_a = os.path.join(tmp_dir, f"hyperopt_searcher_{hash_a}.pkl")
        path_b = os.path.join(tmp_dir, f"hyperopt_searcher_{hash_b}.pkl")

        _simulate_session(space_a, path_a, 5, "a1")
        _simulate_session(space_b, path_b, 3, "b1")

        # Restore A — should still be 5
        restored_a = _restore_searcher(path_a, space_a)
        assert len(restored_a._hpopt_trials.trials) == 5

        # Restore B — should still be 3
        restored_b = _restore_searcher(path_b, space_b)
        assert len(restored_b._hpopt_trials.trials) == 3

    def test_a_b_a_b_accumulation(self, tmp_dir, space_a, space_b):
        """A(5) → B(3) → A(5 more=10) → B(3 more=6): each accumulates."""
        hash_a = get_search_space_hash(space_a)
        hash_b = get_search_space_hash(space_b)
        path_a = os.path.join(tmp_dir, f"hyperopt_searcher_{hash_a}.pkl")
        path_b = os.path.join(tmp_dir, f"hyperopt_searcher_{hash_b}.pkl")

        # Session 1: A gets 5
        _simulate_session(space_a, path_a, 5, "a1")
        # Session 2: B gets 3
        _simulate_session(space_b, path_b, 3, "b1")
        # Session 3: A gets 5 more → 10 total
        s3, n_before = _simulate_session(space_a, path_a, 5, "a2")
        assert n_before == 5
        assert len(s3._hpopt_trials.trials) == 10
        # Session 4: B gets 3 more → 6 total
        s4, n_before = _simulate_session(space_b, path_b, 3, "b2")
        assert n_before == 3
        assert len(s4._hpopt_trials.trials) == 6

        # Final verification
        final_a = _restore_searcher(path_a, space_a)
        assert len(final_a._hpopt_trials.trials) == 10
        final_b = _restore_searcher(path_b, space_b)
        assert len(final_b._hpopt_trials.trials) == 6

    def test_three_spaces_independent(self, tmp_dir, space_a, space_b, space_c):
        """Three different search spaces, each maintains independent history."""
        hash_a = get_search_space_hash(space_a)
        hash_b = get_search_space_hash(space_b)
        hash_c = get_search_space_hash(space_c)
        assert len({hash_a, hash_b, hash_c}) == 3, "Need 3 distinct hashes"

        path_a = os.path.join(tmp_dir, f"hyperopt_searcher_{hash_a}.pkl")
        path_b = os.path.join(tmp_dir, f"hyperopt_searcher_{hash_b}.pkl")
        path_c = os.path.join(tmp_dir, f"hyperopt_searcher_{hash_c}.pkl")

        _simulate_session(space_a, path_a, 4, "a")
        _simulate_session(space_b, path_b, 6, "b")
        _simulate_session(space_c, path_c, 2, "c")

        assert len(_restore_searcher(path_a, space_a)._hpopt_trials.trials) == 4
        assert len(_restore_searcher(path_b, space_b)._hpopt_trials.trials) == 6
        assert len(_restore_searcher(path_c, space_c)._hpopt_trials.trials) == 2

    def test_space_switch_doesnt_corrupt_other(self, tmp_dir, space_a, space_b):
        """Running B doesn't corrupt A's checkpoint file."""
        hash_a = get_search_space_hash(space_a)
        path_a = os.path.join(tmp_dir, f"hyperopt_searcher_{hash_a}.pkl")

        _simulate_session(space_a, path_a, 5, "a1")

        # Get file size/mtime of A's checkpoint
        stat_before = os.stat(path_a)

        # Run B (different file)
        hash_b = get_search_space_hash(space_b)
        path_b = os.path.join(tmp_dir, f"hyperopt_searcher_{hash_b}.pkl")
        _simulate_session(space_b, path_b, 3, "b1")

        # A's checkpoint should be untouched
        stat_after = os.stat(path_a)
        assert stat_before.st_size == stat_after.st_size
        assert stat_before.st_mtime == stat_after.st_mtime


class TestHyperOptSearchBackwardCompatibility:
    """Old-format checkpoint migration."""

    def test_old_format_migration_hash_match(self, tmp_dir, space_a):
        """Old hyperopt_searcher.pkl + .hash → renamed to hyperopt_searcher_{hash}.pkl."""
        hash_a = get_search_space_hash(space_a)
        old_path = os.path.join(tmp_dir, "hyperopt_searcher.pkl")
        old_hash_path = old_path + ".hash"
        new_path = os.path.join(tmp_dir, f"hyperopt_searcher_{hash_a}.pkl")

        searcher = _make_searcher_with_trials(space_a, n_trials=4)
        searcher.save(old_path)
        with open(old_hash_path, "w") as f:
            f.write(hash_a)

        # Simulate Cell 24 migration logic
        assert not os.path.exists(new_path)
        assert os.path.exists(old_path)
        assert os.path.exists(old_hash_path)

        with open(old_hash_path, "r") as f:
            saved_hash = f.read().strip()
        if saved_hash == hash_a:
            os.rename(old_path, new_path)
            os.remove(old_hash_path)

        assert os.path.exists(new_path)
        assert not os.path.exists(old_path)
        assert not os.path.exists(old_hash_path)

        restored = _restore_searcher(new_path, space_a)
        assert len(restored._hpopt_trials.trials) == 4

    def test_old_format_migration_hash_mismatch(self, tmp_dir, space_a, space_b):
        """Old checkpoint with different hash is NOT migrated."""
        hash_a = get_search_space_hash(space_a)
        hash_b = get_search_space_hash(space_b)

        old_path = os.path.join(tmp_dir, "hyperopt_searcher.pkl")
        old_hash_path = old_path + ".hash"

        searcher = _make_searcher_with_trials(space_a, n_trials=4)
        searcher.save(old_path)
        with open(old_hash_path, "w") as f:
            f.write(hash_a)

        # Trying to load for space B
        new_path_b = os.path.join(tmp_dir, f"hyperopt_searcher_{hash_b}.pkl")
        with open(old_hash_path, "r") as f:
            saved_hash = f.read().strip()

        # Migration condition not met
        assert saved_hash != hash_b
        # Old files should remain
        assert os.path.exists(old_path)
        assert os.path.exists(old_hash_path)
        assert not os.path.exists(new_path_b)

    def test_old_format_migration_then_accumulate(self, tmp_dir, space_a):
        """After migration, new sessions still accumulate trials."""
        hash_a = get_search_space_hash(space_a)
        old_path = os.path.join(tmp_dir, "hyperopt_searcher.pkl")
        old_hash_path = old_path + ".hash"
        new_path = os.path.join(tmp_dir, f"hyperopt_searcher_{hash_a}.pkl")

        # Create old-format with 4 trials
        searcher = _make_searcher_with_trials(space_a, n_trials=4)
        searcher.save(old_path)
        with open(old_hash_path, "w") as f:
            f.write(hash_a)

        # Migrate
        os.rename(old_path, new_path)
        os.remove(old_hash_path)

        # Session 2: add 3 more
        s2, n_before = _simulate_session(space_a, new_path, 3, "s2")
        assert n_before == 4
        assert len(s2._hpopt_trials.trials) == 7


class TestHyperOptSearchErrorHandling:
    """Graceful handling of missing/corrupt checkpoints."""

    def test_missing_checkpoint_no_crash(self, tmp_dir, space_a):
        """Missing checkpoint → os.path.exists returns False → no restore."""
        path = os.path.join(tmp_dir, "nonexistent.pkl")
        assert not os.path.exists(path)
        # Notebook checks exists before restore, so just verify the check works

    def test_corrupt_checkpoint_raises(self, tmp_dir):
        """Corrupt pickle file raises an exception (caught by notebook try/except)."""
        path = os.path.join(tmp_dir, "corrupt.pkl")
        with open(path, "wb") as f:
            f.write(b"this is not a valid pickle")

        searcher = HyperOptSearch(metric="accuracy", mode="max")
        with pytest.raises(Exception):
            searcher.restore(path)

    def test_corrupt_checkpoint_doesnt_affect_fresh_start(self, tmp_dir, space_a):
        """After corrupt restore fails, a fresh searcher still works normally."""
        path = os.path.join(tmp_dir, "corrupt.pkl")
        with open(path, "wb") as f:
            f.write(b"corrupted data")

        searcher = HyperOptSearch(metric="accuracy", mode="max")
        try:
            searcher.restore(path)
        except Exception:
            pass

        # Fresh searcher should work fine
        fresh = HyperOptSearch(metric="accuracy", mode="max")
        fresh.set_search_properties(
            metric="accuracy", mode="max", config=space_a, spec=None
        )
        suggestion = fresh.suggest("trial_0")
        assert suggestion is not None


# ===========================================================================
# HistoricalMedianStoppingRule Tests
# ===========================================================================

class TestHistoricalMedianStoppingRuleBasics:
    """Basic loading and filtering."""

    def _make_epoch_csv(self, path, trials_data, search_space_hash=None):
        """Create epoch history CSV.
        trials_data: list of (trial_id, [(epoch, accuracy), ...])
        """
        rows = []
        for trial_id, epochs in trials_data:
            for epoch, acc in epochs:
                row = {
                    "trial_id": trial_id,
                    "training_iteration": epoch,
                    "accuracy": acc,
                }
                if search_space_hash is not None:
                    row["search_space_hash"] = search_space_hash
                rows.append(row)
        pd.DataFrame(rows).to_csv(path, index=False)

    def test_loads_history_with_hash_filter(self, tmp_dir):
        """Only trials matching the hash are loaded."""
        csv_path = os.path.join(tmp_dir, "epoch_history.csv")

        self._make_epoch_csv(csv_path, [
            ("trial_A1", [(1, 0.5), (2, 0.6), (3, 0.7)]),
            ("trial_A2", [(1, 0.4), (2, 0.55), (3, 0.65)]),
            ("trial_B1", [(1, 0.3), (2, 0.4), (3, 0.5)]),
        ], search_space_hash=None)

        df = pd.read_csv(csv_path)
        df.loc[df["trial_id"].isin(["trial_A1", "trial_A2"]), "search_space_hash"] = "hash_aaa"
        df.loc[df["trial_id"] == "trial_B1", "search_space_hash"] = "hash_bbb"
        df.to_csv(csv_path, index=False)

        scheduler = HistoricalMedianStoppingRule(
            historical_csv_path=csv_path,
            search_space_hash="hash_aaa",
            metric="accuracy",
            mode="max",
            time_attr="training_iteration",
            grace_period=2,
            min_historical_epochs=2,
        )
        assert scheduler._num_historical == 2
        assert "historical_trial_A1" in scheduler._results
        assert "historical_trial_A2" in scheduler._results
        assert "historical_trial_B1" not in scheduler._results

    def test_hash_mismatch_loads_nothing(self, tmp_dir):
        """Wrong hash → 0 trials loaded."""
        csv_path = os.path.join(tmp_dir, "epoch_history.csv")
        self._make_epoch_csv(csv_path, [
            ("trial_1", [(1, 0.5), (2, 0.6), (3, 0.7)]),
        ], search_space_hash="hash_aaa")

        scheduler = HistoricalMedianStoppingRule(
            historical_csv_path=csv_path,
            search_space_hash="hash_zzz",
            metric="accuracy",
            mode="max",
            time_attr="training_iteration",
            grace_period=2,
            min_historical_epochs=2,
        )
        assert scheduler._num_historical == 0

    def test_no_hash_column_loads_all(self, tmp_dir):
        """CSV without search_space_hash column → all trials loaded."""
        csv_path = os.path.join(tmp_dir, "epoch_history.csv")
        rows = []
        for i in range(3):
            for epoch in range(1, 6):
                rows.append({
                    "trial_id": f"trial_{i}",
                    "training_iteration": epoch,
                    "accuracy": 0.5 + epoch * 0.02,
                })
        pd.DataFrame(rows).to_csv(csv_path, index=False)

        scheduler = HistoricalMedianStoppingRule(
            historical_csv_path=csv_path,
            search_space_hash="any_hash",
            metric="accuracy",
            mode="max",
            time_attr="training_iteration",
            grace_period=3,
            min_historical_epochs=3,
        )
        assert scheduler._num_historical == 3

    def test_min_historical_epochs_filters_short_trials(self, tmp_dir):
        """Trials shorter than min_historical_epochs are excluded."""
        csv_path = os.path.join(tmp_dir, "epoch_history.csv")
        self._make_epoch_csv(csv_path, [
            ("long_trial", [(1, 0.5), (2, 0.6), (3, 0.7), (4, 0.75), (5, 0.8)]),
            ("short_trial", [(1, 0.3), (2, 0.4)]),
        ])

        scheduler = HistoricalMedianStoppingRule(
            historical_csv_path=csv_path,
            metric="accuracy",
            mode="max",
            time_attr="training_iteration",
            grace_period=3,
            min_historical_epochs=3,
        )
        assert scheduler._num_historical == 1
        assert "historical_long_trial" in scheduler._results
        assert "historical_short_trial" not in scheduler._results

    def test_deduplication(self, tmp_dir):
        """Duplicate (trial_id, epoch) rows are deduplicated, keeping last."""
        csv_path = os.path.join(tmp_dir, "epoch_history.csv")
        rows = [
            {"trial_id": "t1", "training_iteration": 1, "accuracy": 0.5},
            {"trial_id": "t1", "training_iteration": 1, "accuracy": 0.55},
            {"trial_id": "t1", "training_iteration": 2, "accuracy": 0.6},
            {"trial_id": "t1", "training_iteration": 3, "accuracy": 0.7},
        ]
        pd.DataFrame(rows).to_csv(csv_path, index=False)

        scheduler = HistoricalMedianStoppingRule(
            historical_csv_path=csv_path,
            metric="accuracy",
            mode="max",
            time_attr="training_iteration",
            grace_period=2,
            min_historical_epochs=2,
        )
        assert scheduler._num_historical == 1
        assert len(scheduler._results["historical_t1"]) == 3

    def test_print_message(self, tmp_dir, capsys):
        """Print message displayed when history loads."""
        csv_path = os.path.join(tmp_dir, "epoch_history.csv")
        self._make_epoch_csv(csv_path, [
            ("trial_1", [(1, 0.5), (2, 0.6), (3, 0.7)]),
        ])

        HistoricalMedianStoppingRule(
            historical_csv_path=csv_path,
            metric="accuracy",
            mode="max",
            time_attr="training_iteration",
            grace_period=2,
            min_historical_epochs=2,
        )
        captured = capsys.readouterr()
        assert "HistoricalMedianStoppingRule: loaded 1 historical trials" in captured.out

    def test_results_contain_correct_metric_values(self, tmp_dir):
        """Loaded results have correct accuracy values at each epoch."""
        csv_path = os.path.join(tmp_dir, "epoch_history.csv")
        self._make_epoch_csv(csv_path, [
            ("t1", [(1, 0.10), (2, 0.20), (3, 0.30)]),
        ])

        scheduler = HistoricalMedianStoppingRule(
            historical_csv_path=csv_path,
            metric="accuracy",
            mode="max",
            time_attr="training_iteration",
            grace_period=2,
            min_historical_epochs=2,
        )
        results = scheduler._results["historical_t1"]
        assert len(results) == 3
        assert results[0] == {"training_iteration": 1, "accuracy": 0.10}
        assert results[1] == {"training_iteration": 2, "accuracy": 0.20}
        assert results[2] == {"training_iteration": 3, "accuracy": 0.30}


class TestHistoricalMedianStoppingRuleDtype:
    """Hash column dtype edge cases."""

    def test_numeric_hash_float(self, tmp_dir):
        """Float-typed hash column (12055273.0) matches string '12055273'."""
        csv_path = os.path.join(tmp_dir, "epoch_history.csv")

        # Create CSV where hash column will be read as float
        rows = []
        for epoch in range(1, 4):
            rows.append({
                "trial_id": "trial_1",
                "training_iteration": epoch,
                "accuracy": 0.5 + epoch * 0.1,
                "search_space_hash": 12055273,  # int, becomes float in CSV with NaN
            })
        for epoch in range(1, 4):
            rows.append({
                "trial_id": "trial_2",
                "training_iteration": epoch,
                "accuracy": 0.4 + epoch * 0.1,
                "search_space_hash": 99887766,
            })
        pd.DataFrame(rows).to_csv(csv_path, index=False)

        scheduler = HistoricalMedianStoppingRule(
            historical_csv_path=csv_path,
            search_space_hash="12055273",
            metric="accuracy",
            mode="max",
            time_attr="training_iteration",
            grace_period=2,
            min_historical_epochs=2,
        )
        assert scheduler._num_historical == 1
        assert "historical_trial_1" in scheduler._results

    def test_numeric_hash_with_nan_rows(self, tmp_dir):
        """Mixed NaN + numeric hash column still works."""
        csv_path = os.path.join(tmp_dir, "epoch_history.csv")

        rows = [
            {"trial_id": "t1", "training_iteration": 1, "accuracy": 0.5, "search_space_hash": 12345678},
            {"trial_id": "t1", "training_iteration": 2, "accuracy": 0.6, "search_space_hash": 12345678},
            {"trial_id": "t1", "training_iteration": 3, "accuracy": 0.7, "search_space_hash": 12345678},
            # Row with NaN hash (could happen from partial writes)
            {"trial_id": "t_nan", "training_iteration": 1, "accuracy": 0.3},
        ]
        pd.DataFrame(rows).to_csv(csv_path, index=False)

        # Verify the NaN causes float dtype
        verify = pd.read_csv(csv_path)
        assert verify["search_space_hash"].dtype == float

        scheduler = HistoricalMedianStoppingRule(
            historical_csv_path=csv_path,
            search_space_hash="12345678",
            metric="accuracy",
            mode="max",
            time_attr="training_iteration",
            grace_period=2,
            min_historical_epochs=2,
        )
        assert scheduler._num_historical == 1
        assert "historical_t1" in scheduler._results

    def test_string_hash_works_normally(self, tmp_dir):
        """Normal hex string hashes (e.g., 'a1b2c3d4') work fine."""
        csv_path = os.path.join(tmp_dir, "epoch_history.csv")

        rows = []
        for epoch in range(1, 4):
            rows.append({
                "trial_id": "trial_1",
                "training_iteration": epoch,
                "accuracy": 0.5 + epoch * 0.1,
                "search_space_hash": "a1b2c3d4",
            })
        pd.DataFrame(rows).to_csv(csv_path, index=False)

        scheduler = HistoricalMedianStoppingRule(
            historical_csv_path=csv_path,
            search_space_hash="a1b2c3d4",
            metric="accuracy",
            mode="max",
            time_attr="training_iteration",
            grace_period=2,
            min_historical_epochs=2,
        )
        assert scheduler._num_historical == 1


class TestHistoricalMedianStoppingRuleSpaceSwitching:
    """A→B→A scenario for HistoricalMedianStoppingRule."""

    def test_a_b_a_loads_correct_trials(self, tmp_dir):
        """A→B→A: each load gets only the matching trials."""
        csv_path = os.path.join(tmp_dir, "epoch_history.csv")

        rows = []
        for i in range(5):
            for epoch in range(1, 11):
                rows.append({
                    "trial_id": f"A_trial_{i}",
                    "training_iteration": epoch,
                    "accuracy": 0.5 + epoch * 0.02,
                    "search_space_hash": "hash_aaa",
                })
        for i in range(3):
            for epoch in range(1, 11):
                rows.append({
                    "trial_id": f"B_trial_{i}",
                    "training_iteration": epoch,
                    "accuracy": 0.4 + epoch * 0.03,
                    "search_space_hash": "hash_bbb",
                })
        pd.DataFrame(rows).to_csv(csv_path, index=False)

        sched_a = HistoricalMedianStoppingRule(
            historical_csv_path=csv_path, search_space_hash="hash_aaa",
            metric="accuracy", mode="max", time_attr="training_iteration",
            grace_period=5, min_historical_epochs=5,
        )
        assert sched_a._num_historical == 5

        sched_b = HistoricalMedianStoppingRule(
            historical_csv_path=csv_path, search_space_hash="hash_bbb",
            metric="accuracy", mode="max", time_attr="training_iteration",
            grace_period=5, min_historical_epochs=5,
        )
        assert sched_b._num_historical == 3

        # A again
        sched_a2 = HistoricalMedianStoppingRule(
            historical_csv_path=csv_path, search_space_hash="hash_aaa",
            metric="accuracy", mode="max", time_attr="training_iteration",
            grace_period=5, min_historical_epochs=5,
        )
        assert sched_a2._num_historical == 5

    def test_no_cross_contamination(self, tmp_dir):
        """Loading for hash A never includes hash B trials."""
        csv_path = os.path.join(tmp_dir, "epoch_history.csv")

        rows = []
        for i in range(3):
            for epoch in range(1, 6):
                rows.append({
                    "trial_id": f"A_{i}", "training_iteration": epoch,
                    "accuracy": 0.5, "search_space_hash": "aaa",
                })
                rows.append({
                    "trial_id": f"B_{i}", "training_iteration": epoch,
                    "accuracy": 0.5, "search_space_hash": "bbb",
                })
        pd.DataFrame(rows).to_csv(csv_path, index=False)

        sched = HistoricalMedianStoppingRule(
            historical_csv_path=csv_path, search_space_hash="aaa",
            metric="accuracy", mode="max", time_attr="training_iteration",
            grace_period=3, min_historical_epochs=3,
        )
        for key in sched._results:
            assert "B_" not in key, f"Cross-contamination: {key}"


class TestHistoricalMedianStoppingRuleAccumulation:
    """Epoch CSV accumulation across sessions (Cell 26 logic)."""

    def test_epoch_csv_accumulates(self, tmp_dir):
        """Session 1 saves 2 trials, session 2 appends 3 more → 5 total."""
        csv_path = os.path.join(tmp_dir, "epoch_history.csv")

        # Session 1: 2 trials
        _simulate_epoch_csv_save(csv_path, [
            ("s1_t0", [(1, 0.5), (2, 0.6), (3, 0.7)]),
            ("s1_t1", [(1, 0.4), (2, 0.55), (3, 0.65)]),
        ], search_space_hash="hash_aaa")

        df1 = pd.read_csv(csv_path)
        assert len(df1["trial_id"].unique()) == 2

        # Session 2: 3 more trials
        _simulate_epoch_csv_save(csv_path, [
            ("s2_t0", [(1, 0.6), (2, 0.7), (3, 0.8)]),
            ("s2_t1", [(1, 0.5), (2, 0.65), (3, 0.75)]),
            ("s2_t2", [(1, 0.55), (2, 0.68), (3, 0.78)]),
        ], search_space_hash="hash_aaa")

        df2 = pd.read_csv(csv_path)
        assert len(df2["trial_id"].unique()) == 5

        # Scheduler should see all 5
        scheduler = HistoricalMedianStoppingRule(
            historical_csv_path=csv_path, search_space_hash="hash_aaa",
            metric="accuracy", mode="max", time_attr="training_iteration",
            grace_period=2, min_historical_epochs=2,
        )
        assert scheduler._num_historical == 5

    def test_epoch_csv_accumulates_across_spaces(self, tmp_dir):
        """A and B epoch data coexist in same CSV, each loads its own."""
        csv_path = os.path.join(tmp_dir, "epoch_history.csv")

        # Session 1: space A
        _simulate_epoch_csv_save(csv_path, [
            ("a_t0", [(1, 0.5), (2, 0.6), (3, 0.7)]),
            ("a_t1", [(1, 0.4), (2, 0.55), (3, 0.65)]),
        ], search_space_hash="hash_aaa")

        # Session 2: space B
        _simulate_epoch_csv_save(csv_path, [
            ("b_t0", [(1, 0.3), (2, 0.45), (3, 0.55)]),
        ], search_space_hash="hash_bbb")

        # Session 3: more space A
        _simulate_epoch_csv_save(csv_path, [
            ("a_t2", [(1, 0.6), (2, 0.7), (3, 0.8)]),
        ], search_space_hash="hash_aaa")

        # A should see 3 trials, B should see 1
        sched_a = HistoricalMedianStoppingRule(
            historical_csv_path=csv_path, search_space_hash="hash_aaa",
            metric="accuracy", mode="max", time_attr="training_iteration",
            grace_period=2, min_historical_epochs=2,
        )
        assert sched_a._num_historical == 3

        sched_b = HistoricalMedianStoppingRule(
            historical_csv_path=csv_path, search_space_hash="hash_bbb",
            metric="accuracy", mode="max", time_attr="training_iteration",
            grace_period=2, min_historical_epochs=2,
        )
        assert sched_b._num_historical == 1

    def test_epoch_csv_dedup_on_rerun(self, tmp_dir):
        """Re-saving same trial_id+epoch updates accuracy, doesn't duplicate."""
        csv_path = os.path.join(tmp_dir, "epoch_history.csv")

        _simulate_epoch_csv_save(csv_path, [
            ("t0", [(1, 0.5), (2, 0.6)]),
        ], search_space_hash="hash_aaa")

        # "Re-run" same trial with different accuracy
        _simulate_epoch_csv_save(csv_path, [
            ("t0", [(1, 0.55), (2, 0.65)]),  # Updated values
        ], search_space_hash="hash_aaa")

        df = pd.read_csv(csv_path)
        # Should have 2 rows (deduplicated), not 4
        assert len(df) == 2
        # Should have the LAST (updated) values
        assert df[df["training_iteration"] == 1]["accuracy"].values[0] == 0.55
        assert df[df["training_iteration"] == 2]["accuracy"].values[0] == 0.65


class TestHistoricalMedianStoppingRuleErrorHandling:
    """Graceful handling of bad input."""

    def test_missing_csv_no_crash(self, tmp_dir):
        """Missing CSV → 0 historical, no exception."""
        scheduler = HistoricalMedianStoppingRule(
            historical_csv_path=os.path.join(tmp_dir, "nonexistent.csv"),
            search_space_hash="abc",
            metric="accuracy", mode="max", time_attr="training_iteration",
            grace_period=5,
        )
        assert scheduler._num_historical == 0

    def test_empty_csv_no_crash(self, tmp_dir):
        """Empty CSV → 0 historical, no exception."""
        csv_path = os.path.join(tmp_dir, "empty.csv")
        pd.DataFrame().to_csv(csv_path, index=False)

        scheduler = HistoricalMedianStoppingRule(
            historical_csv_path=csv_path,
            metric="accuracy", mode="max", time_attr="training_iteration",
            grace_period=5,
        )
        assert scheduler._num_historical == 0

    def test_missing_columns_no_crash(self, tmp_dir):
        """CSV missing required columns → 0 historical, no exception."""
        csv_path = os.path.join(tmp_dir, "bad.csv")
        pd.DataFrame({"wrong_col": [1, 2, 3]}).to_csv(csv_path, index=False)

        scheduler = HistoricalMedianStoppingRule(
            historical_csv_path=csv_path,
            metric="accuracy", mode="max", time_attr="training_iteration",
            grace_period=5,
        )
        assert scheduler._num_historical == 0

    def test_none_csv_path(self):
        """None csv_path → 0 historical, no exception."""
        scheduler = HistoricalMedianStoppingRule(
            historical_csv_path=None,
            metric="accuracy", mode="max", time_attr="training_iteration",
            grace_period=5,
        )
        assert scheduler._num_historical == 0


# ===========================================================================
# Combined Integration Tests
# ===========================================================================

class TestCombinedIntegration:
    """Both HyperOptSearch and HistoricalMedianStoppingRule together."""

    def test_full_a_b_a_scenario(self, tmp_dir, space_a, space_b):
        """Full A→B→A: both components load correct history."""
        hash_a = get_search_space_hash(space_a)
        hash_b = get_search_space_hash(space_b)

        hyperopt_path_a = os.path.join(tmp_dir, f"hyperopt_searcher_{hash_a}.pkl")
        hyperopt_path_b = os.path.join(tmp_dir, f"hyperopt_searcher_{hash_b}.pkl")
        epoch_csv = os.path.join(tmp_dir, "epoch_history.csv")

        # Session 1: Space A — 5 trials
        _simulate_session(space_a, hyperopt_path_a, 5, "a1")
        _simulate_epoch_csv_save(epoch_csv, [
            (f"A_trial_{i}", [(e, 0.5 + e * 0.02) for e in range(1, 11)])
            for i in range(5)
        ], search_space_hash=hash_a)

        # Session 2: Space B — 3 trials
        _simulate_session(space_b, hyperopt_path_b, 3, "b1")
        _simulate_epoch_csv_save(epoch_csv, [
            (f"B_trial_{i}", [(e, 0.4 + e * 0.03) for e in range(1, 11)])
            for i in range(3)
        ], search_space_hash=hash_b)

        # Session 3: Space A again
        # HyperOptSearch: 5 trials
        restored = _restore_searcher(hyperopt_path_a, space_a)
        restored.set_search_properties(
            metric="accuracy", mode="max", config=space_a, spec=None
        )
        assert len(restored._hpopt_trials.trials) == 5

        # HistoricalMedianStoppingRule: 5 space-A trials
        scheduler = HistoricalMedianStoppingRule(
            historical_csv_path=epoch_csv, search_space_hash=hash_a,
            metric="accuracy", mode="max", time_attr="training_iteration",
            grace_period=5, min_historical_epochs=5,
        )
        assert scheduler._num_historical == 5
        for key in scheduler._results:
            assert "B_trial" not in key

    def test_full_a_b_a_b_accumulation(self, tmp_dir, space_a, space_b):
        """A(5)→B(3)→A(5 more)→B(3 more): both accumulate."""
        hash_a = get_search_space_hash(space_a)
        hash_b = get_search_space_hash(space_b)

        hp_a = os.path.join(tmp_dir, f"hyperopt_searcher_{hash_a}.pkl")
        hp_b = os.path.join(tmp_dir, f"hyperopt_searcher_{hash_b}.pkl")
        epoch_csv = os.path.join(tmp_dir, "epoch_history.csv")

        # Session 1: A(5)
        _simulate_session(space_a, hp_a, 5, "a1")
        _simulate_epoch_csv_save(epoch_csv, [
            (f"a1_t{i}", [(e, 0.5 + e * 0.02) for e in range(1, 6)])
            for i in range(5)
        ], search_space_hash=hash_a)

        # Session 2: B(3)
        _simulate_session(space_b, hp_b, 3, "b1")
        _simulate_epoch_csv_save(epoch_csv, [
            (f"b1_t{i}", [(e, 0.4 + e * 0.03) for e in range(1, 6)])
            for i in range(3)
        ], search_space_hash=hash_b)

        # Session 3: A(5 more → 10)
        s3, n_before = _simulate_session(space_a, hp_a, 5, "a2")
        assert n_before == 5
        assert len(s3._hpopt_trials.trials) == 10
        _simulate_epoch_csv_save(epoch_csv, [
            (f"a2_t{i}", [(e, 0.55 + e * 0.02) for e in range(1, 6)])
            for i in range(5)
        ], search_space_hash=hash_a)

        # Session 4: B(3 more → 6)
        s4, n_before = _simulate_session(space_b, hp_b, 3, "b2")
        assert n_before == 3
        assert len(s4._hpopt_trials.trials) == 6
        _simulate_epoch_csv_save(epoch_csv, [
            (f"b2_t{i}", [(e, 0.45 + e * 0.03) for e in range(1, 6)])
            for i in range(3)
        ], search_space_hash=hash_b)

        # Final verification — HyperOptSearch
        final_a = _restore_searcher(hp_a, space_a)
        assert len(final_a._hpopt_trials.trials) == 10
        final_b = _restore_searcher(hp_b, space_b)
        assert len(final_b._hpopt_trials.trials) == 6

        # Final verification — HistoricalMedianStoppingRule
        sched_a = HistoricalMedianStoppingRule(
            historical_csv_path=epoch_csv, search_space_hash=hash_a,
            metric="accuracy", mode="max", time_attr="training_iteration",
            grace_period=3, min_historical_epochs=3,
        )
        assert sched_a._num_historical == 10  # 5 from session 1 + 5 from session 3

        sched_b = HistoricalMedianStoppingRule(
            historical_csv_path=epoch_csv, search_space_hash=hash_b,
            metric="accuracy", mode="max", time_attr="training_iteration",
            grace_period=3, min_historical_epochs=3,
        )
        assert sched_b._num_historical == 6  # 3 from session 2 + 3 from session 4


# ===========================================================================
# Edge Cases
# ===========================================================================

class TestEdgeCases:

    def test_search_space_hash_deterministic(self, space_a):
        """Same space → same hash across calls."""
        h1 = get_search_space_hash(space_a)
        h2 = get_search_space_hash(space_a)
        assert h1 == h2

    def test_different_spaces_different_hashes(self, space_a, space_b):
        """Different spaces → different hashes."""
        assert get_search_space_hash(space_a) != get_search_space_hash(space_b)

    def test_hash_is_8_chars(self, space_a):
        """Hash is exactly 8 hex characters."""
        h = get_search_space_hash(space_a)
        assert len(h) == 8
        assert all(c in "0123456789abcdef" for c in h)

    def test_warm_start_disabled_fresh_searcher(self, space_a):
        """WARM_START_ENABLED=False → fresh searcher works fine."""
        searcher = HyperOptSearch(metric="accuracy", mode="max")
        searcher.set_search_properties(
            metric="accuracy", mode="max", config=space_a, spec=None
        )
        suggestion = searcher.suggest("trial_0")
        assert suggestion is not None

    def test_single_trial_roundtrip(self, tmp_dir, space_a):
        """Even a single trial is preserved across save/restore."""
        searcher = _make_searcher_with_trials(space_a, n_trials=1)
        path = os.path.join(tmp_dir, "searcher.pkl")
        searcher.save(path)

        restored = _restore_searcher(path, space_a)
        assert len(restored._hpopt_trials.trials) == 1
