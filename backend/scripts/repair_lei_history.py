"""CLI wrapper attorno a `app.services.admin.data_repair.repair_lei_history`.

Uso (dal repo root del backend):
    "d:/Antigravity/.venv/Scripts/python.exe" scripts/repair_lei_history.py
    "d:/Antigravity/.venv/Scripts/python.exe" scripts/repair_lei_history.py --dry-run

Vedi `app/services/admin/data_repair.py` per la business logic.
Esposto anche come endpoint `POST /api/v1/admin/repair-lei`.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

_THIS = Path(__file__).resolve()
sys.path.insert(0, str(_THIS.parents[1]))

from app.database import SessionLocal
from app.services.admin.data_repair import repair_lei_history


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true", help="Anteprima senza commit")
    args = parser.parse_args()

    db = SessionLocal()
    try:
        report = repair_lei_history(db, dry_run=args.dry_run, sample_size=20)
    finally:
        db.close()

    for s in report.sample_changes:
        print(
            f"  {s['date']} lei {s['lei_roc_before']:+.2f} -> {s['lei_roc_after']:+.3f}  "
            f"regime {s['regime_before']} -> {s['regime_after']}  "
            f"conf {s['confidence_before']:.2f} -> {s['confidence_after']:.2f}"
        )

    print()
    print(
        f"Summary: total={report.total_records} corrupted={report.corrupted} "
        f"repaired={report.repaired} skipped={report.skipped} dry_run={report.dry_run}"
    )


if __name__ == "__main__":
    main()
