"""Read recorded decisions without importing or contacting a model backend."""
from dataclasses import dataclass
import json
from pathlib import Path

from .protocol import parse_decision


@dataclass(frozen=True)
class ReplayLog:
    path: Path
    records: tuple

    @classmethod
    def load(cls, path):
        path = Path(path).expanduser().resolve()
        records = []
        for line_number, line in enumerate(path.read_text(encoding="utf-8-sig").splitlines(), 1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
                if not isinstance(record, dict) or not isinstance(record.get("kind"), str):
                    raise ValueError("记录需要 kind 字符串")
                if not isinstance(record.get("data"), dict):
                    raise ValueError("记录需要 data 对象")
                turn = record.get("turn_id")
                if turn is not None and (isinstance(turn, bool) or not isinstance(turn, int)):
                    raise ValueError("turn_id 需要整数或 null")
                if record["kind"] == "tool_call":
                    decision = parse_decision(json.dumps(record["data"]))
                    if "tool" not in decision or turn is None:
                        raise ValueError("tool_call 需要工具指令和 turn_id")
                if record["kind"] == "decision" and "final" in record["data"]:
                    parse_decision(json.dumps(record["data"]))
                records.append(record)
            except (ValueError, TypeError) as exc:
                raise ValueError(f"日志第 {line_number} 行：{exc}") from exc
        if not records:
            raise ValueError("日志为空")
        return cls(path, tuple(records))

    def expected_result(self, index):
        """Find only the result belonging to this tool, never a later call."""
        turn = self.records[index].get("turn_id")
        for record in self.records[index + 1:]:
            if record.get("turn_id") != turn or record["kind"] == "tool_call":
                break
            if record["kind"] == "tool_result":
                return record["data"]
        return None

    @property
    def summary(self):
        return {
            "path": str(self.path),
            "events": len(self.records),
            "turns": len({r["turn_id"] for r in self.records if r.get("turn_id") is not None}),
            "tools": sum(r["kind"] == "tool_call" for r in self.records),
        }
