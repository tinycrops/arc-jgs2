from __future__ import annotations

import csv
import html
import json
from collections import Counter
from pathlib import Path

from .orient import OrientationRecord, to_jsonable


def write_reports(records: list[OrientationRecord], out_dir: str | Path) -> None:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    with (out / "orientation.jsonl").open("w") as f:
        for record in records:
            f.write(json.dumps(to_jsonable(record), sort_keys=True) + "\n")

    with (out / "orientation.csv").open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "task_id",
                "train_count",
                "test_count",
                "top_family",
                "top_score",
                "tags",
                "train_input_dims",
                "train_output_dims",
                "test_input_dims",
                "rejected_overshoots",
                "notes",
                "source",
            ],
        )
        writer.writeheader()
        for record in records:
            writer.writerow(
                {
                    "task_id": record.task_id,
                    "train_count": record.train_count,
                    "test_count": record.test_count,
                    "top_family": record.top_family,
                    "top_score": f"{record.top_score:.3f}",
                    "tags": " ".join(record.global_tags),
                    "train_input_dims": repr(record.train_input_dims),
                    "train_output_dims": repr(record.train_output_dims),
                    "test_input_dims": repr(record.test_input_dims),
                    "rejected_overshoots": " ".join(record.rejected_overshoots),
                    "notes": " | ".join(record.notes),
                    "source": record.source,
                }
            )

    summary = {
        "task_count": len(records),
        "top_family_counts": dict(Counter(record.top_family for record in records).most_common()),
        "tag_counts": dict(Counter(tag for record in records for tag in record.global_tags).most_common()),
        "overshoot_counts": dict(Counter(item for record in records for item in record.rejected_overshoots).most_common()),
        "fully_supported_top_family": sum(1 for record in records if record.top_score == 1.0),
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    (out / "index.html").write_text(_html(records, summary))


def _html(records: list[OrientationRecord], summary: dict) -> str:
    rows = []
    for record in records:
        tags = " ".join(f"<span>{html.escape(tag)}</span>" for tag in record.global_tags)
        evidence = "<br>".join(
            html.escape(f"{item.name}: {item.score:.2f} ({item.support}/{item.total}) {item.detail}")
            for item in record.primitive_evidence[:5]
        )
        rows.append(
            "<tr>"
            f"<td>{html.escape(record.task_id)}</td>"
            f"<td>{html.escape(record.top_family)}</td>"
            f"<td>{record.top_score:.2f}</td>"
            f"<td>{tags}</td>"
            f"<td>{html.escape(repr(record.train_input_dims))} -> {html.escape(repr(record.train_output_dims))}</td>"
            f"<td>{evidence}</td>"
            f"<td>{html.escape(' '.join(record.rejected_overshoots))}</td>"
            f"<td>{html.escape(' | '.join(record.notes))}</td>"
            "</tr>"
        )

    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>ARC-JGS2 Orientation</title>
  <style>
    body {{ font-family: system-ui, sans-serif; margin: 24px; color: #1d252b; background: #f7f4ee; }}
    h1 {{ margin-bottom: 4px; }}
    .summary {{ display: grid; grid-template-columns: repeat(3, minmax(0, 1fr)); gap: 12px; margin: 18px 0; }}
    .metric {{ background: #fff; border: 1px solid #d7d0c4; border-radius: 6px; padding: 12px; }}
    .metric b {{ display: block; font-size: 22px; }}
    table {{ width: 100%; border-collapse: collapse; background: #fff; border: 1px solid #d7d0c4; }}
    th, td {{ text-align: left; vertical-align: top; padding: 8px; border-bottom: 1px solid #e5ded3; font-size: 13px; }}
    th {{ background: #ece5d9; position: sticky; top: 0; }}
    span {{ display: inline-block; background: #d9e7e2; border-radius: 4px; padding: 2px 5px; margin: 1px; }}
    code {{ background: #eee6da; padding: 1px 4px; border-radius: 4px; }}
  </style>
</head>
<body>
  <h1>ARC-JGS2 Orientation</h1>
  <p>Corpus-level task world states for local perturbation and overshoot analysis.</p>
  <div class="summary">
    <div class="metric"><b>{summary["task_count"]}</b>tasks oriented</div>
    <div class="metric"><b>{summary["fully_supported_top_family"]}</b>single-family fully supported</div>
    <div class="metric"><b>{html.escape(repr(summary["top_family_counts"]))}</b>top families</div>
  </div>
  <table>
    <thead>
      <tr>
        <th>Task</th><th>Top Family</th><th>Score</th><th>Tags</th><th>Dims</th><th>Evidence</th><th>Overshoots</th><th>Notes</th>
      </tr>
    </thead>
    <tbody>
      {''.join(rows)}
    </tbody>
  </table>
</body>
</html>
"""
