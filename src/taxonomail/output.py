"""Write a result: ``taxonomy.yaml``, ``assignments.jsonl`` and ``clusters.json``."""

from __future__ import annotations

import json
from pathlib import Path

import yaml

from taxonomail.pipeline import Result


def write_result(result: Result, out: Path) -> None:
    """Write the taxonomy, the folder of every mail and the named clusters to ``out``."""
    out.mkdir(parents=True, exist_ok=True)
    taxonomy = {
        "folders": [
            {"id": f.id, "name": f.name, "description": f.description, "mails": f.size}
            for f in result.folders
        ]
    }
    (out / "taxonomy.yaml").write_text(
        yaml.safe_dump(taxonomy, sort_keys=False, allow_unicode=True, width=100), "utf-8"
    )
    with (out / "assignments.jsonl").open("w", encoding="utf-8") as handle:
        for a in result.assignments:
            row = {
                "message_id": a.mail.id,
                "from": a.mail.sender,
                "subject": a.mail.subject,
                "folder": a.folder,
                "method": a.method,
            }
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    clusters = [
        {
            "cluster": cluster,
            "name": name.name,
            "description": name.description,
            "folder": result.cluster_folders[cluster],
        }
        for cluster, name in sorted(result.cluster_names.items())
    ]
    (out / "clusters.json").write_text(
        json.dumps(clusters, indent=2, ensure_ascii=False) + "\n", "utf-8"
    )
