"""
Historical Median Stopping Rule for Ray Tune.

Extends MedianStoppingRule to accept historical per-epoch trial data from
previous Colab sessions, enabling informed early stopping decisions from
the very first trial of a new session.
"""

import logging
import os

import pandas as pd
from ray.tune.schedulers import MedianStoppingRule

logger = logging.getLogger(__name__)


class HistoricalMedianStoppingRule(MedianStoppingRule):
    """MedianStoppingRule pre-populated with historical per-epoch trial data.

    Subclasses MedianStoppingRule and injects historical trial metric curves
    into ``self._results`` during construction. This gives the scheduler a
    rich median baseline so that stop/continue decisions at the grace period
    boundary are informed immediately, rather than waiting for
    ``min_samples_required`` current-session trials to reach that epoch.

    Historical data does NOT bypass ``grace_period``. No trial is stopped
    before ``grace_period`` epochs regardless of historical data. The benefit
    is that once a trial reaches ``grace_period``, the scheduler already has
    enough comparison data to make a decision.

    Args:
        historical_csv_path: Path to CSV with per-epoch metrics from prior
            sessions. Must contain ``time_attr`` and ``metric`` columns, plus
            ``trial_id_col`` to distinguish trials. If ``None``, file doesn't
            exist, or CSV is malformed, behaves like vanilla MedianStoppingRule.
        trial_id_col: Column in CSV identifying distinct trials.
        search_space_hash: If provided, only load rows where the
            ``search_space_hash`` column matches this value.
        min_historical_epochs: Only include historical trials that reached
            at least this many epochs. Defaults to ``grace_period`` since
            shorter trials can't inform stopping decisions anyway.
        **kwargs: Passed to ``MedianStoppingRule.__init__``.
    """

    def __init__(
        self,
        historical_csv_path=None,
        trial_id_col="trial_id",
        search_space_hash=None,
        min_historical_epochs=None,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self._num_historical = 0

        if min_historical_epochs is None:
            min_historical_epochs = self._grace_period

        if historical_csv_path is not None:
            self._load_historical(
                historical_csv_path,
                trial_id_col,
                search_space_hash,
                min_historical_epochs,
            )

    def _load_historical(
        self,
        csv_path,
        trial_id_col,
        search_space_hash,
        min_historical_epochs,
    ):
        if not os.path.exists(csv_path):
            logger.warning(
                "Historical CSV not found at %s — running without historical data.",
                csv_path,
            )
            return

        try:
            df = pd.read_csv(csv_path)
        except Exception as e:
            logger.warning(
                "Failed to read historical CSV at %s: %s — running without historical data.",
                csv_path,
                e,
            )
            return

        if df.empty:
            logger.warning("Historical CSV at %s is empty — running without historical data.", csv_path)
            return

        # Validate required columns
        required_cols = {self._time_attr, self._metric, trial_id_col}
        missing = required_cols - set(df.columns)
        if missing:
            logger.warning(
                "Historical CSV missing columns %s — running without historical data.",
                missing,
            )
            return

        # Filter by search space hash if provided.
        # Pandas may read all-digit hashes as int64 or float64 (e.g., 12055273.0
        # instead of "12055273"), so we normalize the column: convert to str, then
        # strip any trailing ".0" that float→str conversion produces.
        if search_space_hash is not None and "search_space_hash" in df.columns:
            hash_col = (
                df["search_space_hash"]
                .astype(str)
                .str.replace(r"\.0$", "", regex=True)
            )
            target = str(search_space_hash).strip()
            df = df[hash_col == target]
            if df.empty:
                logger.warning(
                    "No historical data matches search_space_hash=%s — running without historical data.",
                    search_space_hash,
                )
                return

        # Deduplicate on (trial_id, training_iteration) keeping last occurrence
        df = df.drop_duplicates(
            subset=[trial_id_col, self._time_attr], keep="last"
        )

        # Filter to trials that reached min_historical_epochs
        max_epochs_per_trial = df.groupby(trial_id_col)[self._time_attr].max()
        valid_trials = max_epochs_per_trial[
            max_epochs_per_trial >= min_historical_epochs
        ].index
        df = df[df[trial_id_col].isin(valid_trials)]

        if df.empty:
            logger.warning(
                "No historical trials reached %d epochs — running without historical data.",
                min_historical_epochs,
            )
            return

        # Inject into self._results with string keys
        grouped = df.groupby(trial_id_col)
        for trial_id, group in grouped:
            key = f"historical_{trial_id}"
            results_list = []
            for _, row in group.sort_values(self._time_attr).iterrows():
                results_list.append(
                    {
                        self._time_attr: row[self._time_attr],
                        self._metric: row[self._metric],
                    }
                )
            self._results[key] = results_list

        self._num_historical = len(grouped)
        logger.info(
            "Loaded %d historical trials from %s", self._num_historical, csv_path
        )
        print(
            f"✅ HistoricalMedianStoppingRule: loaded {self._num_historical} "
            f"historical trials from {csv_path}"
        )

    def debug_string(self):
        base = super().debug_string()
        return f"{base} | historical_trials={self._num_historical}"
