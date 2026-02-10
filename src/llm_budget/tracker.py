"""SQLite-backed spend tracking."""
from __future__ import annotations

import json
import logging
import sqlite3
import threading
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

DEFAULT_DB_PATH = Path.home() / ".llm-budget" / "spend.db"

_CREATE_TABLE = """
CREATE TABLE IF NOT EXISTS spend_log (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp TEXT NOT NULL DEFAULT (datetime('now')),
    model TEXT NOT NULL,
    input_tokens INTEGER NOT NULL,
    output_tokens INTEGER NOT NULL,
    cost_usd REAL NOT NULL,
    metadata TEXT
)
"""

_CREATE_INDEX = """
CREATE INDEX IF NOT EXISTS idx_spend_timestamp_model
ON spend_log(timestamp, model)
"""


@dataclass
class SpendRecord:
    """A single recorded API call."""

    id: Optional[int]
    timestamp: str
    model: str
    input_tokens: int
    output_tokens: int
    cost_usd: float
    metadata: Optional[str]


class Tracker:
    """Thread-safe SQLite-backed spend tracker."""

    def __init__(self, db_path: Optional[str] = None) -> None:
        resolved = Path(db_path).expanduser() if db_path else DEFAULT_DB_PATH
        resolved.parent.mkdir(parents=True, exist_ok=True)
        self._db_path = resolved
        self._lock = threading.Lock()
        self._conn = sqlite3.connect(
            str(self._db_path), check_same_thread=False
        )
        self._conn.row_factory = sqlite3.Row
        self._create_tables()

    def _create_tables(self) -> None:
        with self._lock:
            self._conn.execute(_CREATE_TABLE)
            self._conn.execute(_CREATE_INDEX)
            self._conn.commit()

    def __enter__(self) -> "Tracker":
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    def record(
        self,
        model: str,
        input_tokens: int,
        output_tokens: int,
        cost_usd: float,
        metadata: Optional[dict] = None,
    ) -> SpendRecord:
        """Record a completed API call."""
        if cost_usd < 0:
            raise ValueError(f"cost_usd cannot be negative, got {cost_usd}")
        if input_tokens < 0 or output_tokens < 0:
            raise ValueError("Token counts cannot be negative")
        ts = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
        meta_str = json.dumps(metadata) if metadata else None
        with self._lock:
            cursor = self._conn.execute(
                "INSERT INTO spend_log (timestamp, model, input_tokens, "
                "output_tokens, cost_usd, metadata) VALUES (?, ?, ?, ?, ?, ?)",
                (ts, model, input_tokens, output_tokens, cost_usd, meta_str),
            )
            self._conn.commit()
        return SpendRecord(
            id=cursor.lastrowid,
            timestamp=ts,
            model=model,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cost_usd=cost_usd,
            metadata=meta_str,
        )

    def get_spend(
        self,
        period: str = "today",
        model: Optional[str] = None,
    ) -> float:
        """Get total spend for a time period.

        Args:
            period: One of 'today', 'this_week', 'this_month', 'total',
                or 'hourly', 'daily', 'weekly', 'monthly'.
            model: Optional model filter.

        Returns:
            Total cost as float.
        """
        cutoff = self._period_to_cutoff(period)
        query = "SELECT COALESCE(SUM(cost_usd), 0) FROM spend_log"
        params: list = []
        clauses: list[str] = []

        if cutoff:
            clauses.append("timestamp >= ?")
            params.append(cutoff)
        if model:
            clauses.append("model = ?")
            params.append(model)

        if clauses:
            query += " WHERE " + " AND ".join(clauses)

        with self._lock:
            row = self._conn.execute(query, params).fetchone()
        return float(row[0])

    def get_spend_breakdown(
        self,
        period: str = "this_month",
    ) -> dict[str, float]:
        """Get spend broken down by model for a period."""
        cutoff = self._period_to_cutoff(period)
        query = "SELECT model, COALESCE(SUM(cost_usd), 0) FROM spend_log"
        params: list = []

        if cutoff:
            query += " WHERE timestamp >= ?"
            params.append(cutoff)

        query += " GROUP BY model ORDER BY SUM(cost_usd) DESC"

        with self._lock:
            rows = self._conn.execute(query, params).fetchall()
        return {row[0]: float(row[1]) for row in rows}

    def get_call_count_breakdown(
        self,
        period: str = "this_month",
    ) -> dict[str, int]:
        """Get call count broken down by model for a period."""
        cutoff = self._period_to_cutoff(period)
        query = "SELECT model, COUNT(*) FROM spend_log"
        params: list = []

        if cutoff:
            query += " WHERE timestamp >= ?"
            params.append(cutoff)

        query += " GROUP BY model ORDER BY COUNT(*) DESC"

        with self._lock:
            rows = self._conn.execute(query, params).fetchall()
        return {row[0]: int(row[1]) for row in rows}

    def get_total_call_count(
        self,
        period: str = "this_month",
    ) -> int:
        """Get total call count for a period."""
        cutoff = self._period_to_cutoff(period)
        query = "SELECT COUNT(*) FROM spend_log"
        params: list = []

        if cutoff:
            query += " WHERE timestamp >= ?"
            params.append(cutoff)

        with self._lock:
            row = self._conn.execute(query, params).fetchone()
        return int(row[0])

    def get_history(
        self,
        last_n: int = 50,
        model: Optional[str] = None,
    ) -> list[SpendRecord]:
        """Get recent spend records."""
        query = "SELECT * FROM spend_log"
        params: list = []

        if model:
            query += " WHERE model = ?"
            params.append(model)

        query += " ORDER BY timestamp DESC LIMIT ?"
        params.append(last_n)

        with self._lock:
            rows = self._conn.execute(query, params).fetchall()

        return [
            SpendRecord(
                id=row["id"],
                timestamp=row["timestamp"],
                model=row["model"],
                input_tokens=row["input_tokens"],
                output_tokens=row["output_tokens"],
                cost_usd=row["cost_usd"],
                metadata=row["metadata"],
            )
            for row in rows
        ]

    def get_output_ratio(
        self,
        model: str,
        last_n: int = 50,
        min_samples: int = 5,
    ) -> Optional[float]:
        """Learn the output/input token ratio from historical data.

        Queries the last N calls for a given model and returns the median
        output/input ratio. Returns None if fewer than min_samples records
        exist (cold start → caller should fall back to static heuristic).

        Args:
            model: Model name to query.
            last_n: Number of recent records to consider.
            min_samples: Minimum records needed to return a ratio.

        Returns:
            Median output/input ratio, or None if insufficient data.
        """
        query = (
            "SELECT input_tokens, output_tokens FROM spend_log "
            "WHERE model = ? AND input_tokens > 0 AND output_tokens > 0 "
            "ORDER BY timestamp DESC LIMIT ?"
        )
        with self._lock:
            rows = self._conn.execute(query, (model, last_n)).fetchall()

        if len(rows) < min_samples:
            return None

        ratios = sorted(r["output_tokens"] / r["input_tokens"] for r in rows)
        mid = len(ratios) // 2
        if len(ratios) % 2 == 0:
            return (ratios[mid - 1] + ratios[mid]) / 2
        return ratios[mid]

    def _period_to_cutoff(self, period: str) -> Optional[str]:
        """Convert period string to cutoff datetime string."""
        now = datetime.now(timezone.utc)
        today_midnight = now.replace(hour=0, minute=0, second=0, microsecond=0)
        mapping: dict[str, Optional[datetime]] = {
            "hourly": now - timedelta(hours=1),
            "daily": today_midnight,
            "today": today_midnight,
            "weekly": now - timedelta(weeks=1),
            "this_week": now - timedelta(weeks=1),
            "monthly": now - timedelta(days=30),
            "this_month": now - timedelta(days=30),
            "total": None,
        }
        if period not in mapping:
            logger.warning("Unknown period '%s', returning total spend", period)
            return None
        cutoff_dt = mapping[period]
        if cutoff_dt is None:
            return None
        return cutoff_dt.strftime("%Y-%m-%d %H:%M:%S")

    def close(self) -> None:
        """Close the database connection and clear singleton if applicable."""
        global _default_tracker
        self._conn.close()
        if _default_tracker is self:
            _default_tracker = None


_default_tracker: Optional[Tracker] = None


def get_tracker() -> Tracker:
    """Get or create the default tracker (lazy singleton)."""
    global _default_tracker
    if _default_tracker is None:
        _default_tracker = Tracker()
    return _default_tracker
