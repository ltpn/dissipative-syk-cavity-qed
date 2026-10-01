#!/usr/bin/env python3
"""Copy the light canonical inputs; normalize paths and audit numerical values with Julia."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--generation-root", type=Path)
    parser.add_argument("--physical-generation-root", type=Path)
    parser.add_argument("--julia", default=os.environ.get("JULIA", "julia"))
    args = parser.parse_args()
    source, output = args.source_root.resolve(), args.output_root.resolve()
    if output.exists():
        parser.error("output root already exists; choose a new directory")
    manifest = json.loads((source / "DATASET_MANIFEST.json").read_text())
    selected = manifest["selected_analysis"]
    output.mkdir(parents=True)
    copies = []

    def copy(path):
        destination = output / path.relative_to(source)
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, destination)
        copies.append(path.relative_to(source).as_posix())

    for name, analysis in selected.items():
        copy(source / "runs" / name / "config.toml")
        for path in (source / analysis).rglob("*"):
            if path.is_file() and path.suffix in (".jld2", ".toml"):
                copy(path)
        for filename in ("selection.toml", "base_manifest.toml"):
            path = source / "runs" / name / "recovery" / filename
            if path.exists():
                copy(path)
    for directory in ("data", "derived/sigma_sff", "figures"):
        for path in (source / directory).rglob("*"):
            if path.is_file() and (path.suffix == ".toml" or
                    (path.suffix == ".jld2" and (directory != "figures" or "plotdata" in path.name))):
                copy(path)
    validation = source / "derived/figure3_syk4_reference_validation.toml"
    copy(validation)

    lines = ["schema_version = 1", 'dataset = "canonical_n10_q4_r16"',
             "n_orb = 10", "filling = 4", "n_seeds = 16", "", "[selected_analysis]"]
    lines += [f"{key} = {json.dumps(value)}" for key, value in sorted(selected.items())]
    (output / "dataset.toml").write_text("\n".join(lines) + "\n")
    repo = Path(__file__).resolve().parents[3]
    with tempfile.TemporaryDirectory(prefix="syk-cavity-qed-bundle-") as temporary:
        config = Path(temporary) / "bundle.toml"
        mappings = {str(source): ".", **manifest.get("historical_path_mapping", {})}
        lines = [f"source_root = {json.dumps(str(source))}",
                 f"output_root = {json.dumps(str(output))}",
                 "files = " + json.dumps(copies), "", "[path_mapping]"]
        lines += [f"{json.dumps(k)} = {json.dumps(v)}" for k, v in mappings.items()]
        if args.generation_root:
            lines += ["", "[generation_roots]"]
            for name in selected:
                root = (args.physical_generation_root if name == "physical_n10_q4"
                        and args.physical_generation_root else args.generation_root / "runs" / name)
                lines.append(f"{name} = {json.dumps(str(root.resolve()))}")
        config.write_text("\n".join(lines) + "\n")
        subprocess.run([args.julia, "--project=" + str(repo / "src/environment"),
                        str(repo / "src/model/scripts/_normalize_reproduction_bundle.jl"),
                        str(config)], check=True)
    inventory = []
    for path in sorted(output.rglob("*")):
        if path.is_file():
            inventory.append({"path": path.relative_to(output).as_posix(),
                              "bytes": path.stat().st_size,
                              "sha256": hashlib.sha256(path.read_bytes()).hexdigest()})
    (output / "inventory.json").write_text(json.dumps({"schema_version": 1,
        "source_code_revisions": manifest.get("source_code_revisions", []),
        "files": inventory}, indent=2) + "\n")
    print(f"Bundled {len(inventory)} files, {sum(f['bytes'] for f in inventory)/2**20:.1f} MiB")


if __name__ == "__main__":
    main()
