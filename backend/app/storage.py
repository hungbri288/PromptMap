import json
from pathlib import Path

from backend.app.schemas import RunRecord


class RunStore:
    def __init__(self, root: Path):
        self.root = root
        self.root.mkdir(parents=True, exist_ok=True)

    def save(self, record: RunRecord) -> None:
        path = self._path(record.id)
        path.write_text(record.model_dump_json(indent=2), encoding="utf-8")

    def load(self, run_id: str) -> RunRecord:
        path = self._path(run_id)
        if not path.exists():
            raise FileNotFoundError(run_id)
        return RunRecord.model_validate(json.loads(path.read_text(encoding="utf-8")))

    def _path(self, run_id: str) -> Path:
        return self.root / f"{run_id}.json"
