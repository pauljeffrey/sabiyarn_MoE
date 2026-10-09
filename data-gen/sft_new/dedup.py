"""Exact deduplication of finished rows, by content hash, on disk (a few bytes per row, so tens of millions of rows stay light).

`Seen.add(row_id)` is True the first time an id is seen. The id is the 64-bit blake2b of (task, languages, input, instruction,
response) computed in compose.finish, so two datasets that contain the same pair, or the same sentence reached through the
forward and the backward direction of different sources, collapse to one row. Near-duplicates are not touched here.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path


class Seen:
    def __init__(self, path: str | Path):
        self.db = sqlite3.connect(str(path))
        self.db.execute("PRAGMA journal_mode=OFF")
        self.db.execute("PRAGMA synchronous=OFF")
        self.db.execute("CREATE TABLE IF NOT EXISTS s (h INTEGER PRIMARY KEY, c TEXT)")
        self.pending = 0

    def add(self, hex_id: str, tag: str = "") -> bool:
        """True the first time. `tag` names the build step that saw it, so a step that has to be re-run can first `forget` its own ids."""
        cur = self.db.execute("INSERT OR IGNORE INTO s VALUES (?, ?)", (int(hex_id, 16) - (1 << 63), tag))
        self.pending += 1
        if self.pending >= 20_000:
            self.db.commit()
            self.pending = 0
        return cur.rowcount == 1

    def forget(self, tag: str) -> int:
        self.db.commit()
        n = self.db.execute("DELETE FROM s WHERE c = ?", (tag,)).rowcount
        self.db.commit()
        return n

    def __len__(self):
        return self.db.execute("SELECT COUNT(*) FROM s").fetchone()[0]

    def close(self):
        self.db.commit()
        self.db.close()
