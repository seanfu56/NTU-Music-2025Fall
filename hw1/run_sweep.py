import argparse
import subprocess
import sys
from pathlib import Path

import yaml

try:
    import wandb  # type: ignore
except Exception as e:
    raise SystemExit("wandb is required. Please install with `pip install wandb`.\n" + str(e))


def main():
    parser = argparse.ArgumentParser(description="Create and run a Weights & Biases sweep")
    parser.add_argument("--yaml", type=str, default=str(Path(__file__).parent / "sweep.yaml"))
    parser.add_argument("--project", type=str, default="ntu-music-singer-cnn")
    parser.add_argument("--entity", type=str, default=None)
    parser.add_argument("--count", type=int, default=20)
    args = parser.parse_args()

    yaml_path = Path(args.yaml)
    if not yaml_path.exists():
        raise SystemExit(f"Sweep YAML not found: {yaml_path}")

    with open(yaml_path, "r", encoding="utf-8") as f:
        sweep_cfg = yaml.safe_load(f)

    # Prefer CLI path so we can parse the Sweep URL to recover entity when unset
    cmd = ["wandb", "sweep", "--project", args.project, str(yaml_path)]
    if args.entity:
        cmd = ["wandb", "sweep", "--entity", args.entity, "--project", args.project, str(yaml_path)]
    print("Create sweep via CLI:", " ".join(cmd))
    proc = subprocess.run(cmd, capture_output=True, text=True)
    out = (proc.stdout or "") + (proc.stderr or "")
    sys.stdout.write(out)
    if proc.returncode != 0:
        raise SystemExit(f"wandb sweep failed with code {proc.returncode}")

    # Parse Sweep URL: https://wandb.ai/<entity>/<project>/sweeps/<id>
    agent_target = None
    for line in out.splitlines():
        if "Sweep URL:" in line and "wandb.ai" in line:
            try:
                path = line.split("wandb.ai/", 1)[1].strip()
                parts = path.split("/")
                entity = parts[0]
                project = parts[1]
                sweep_id = parts[3].split()[0]
                agent_target = f"{entity}/{project}/{sweep_id}"
                break
            except Exception:
                continue
        elif "Run sweep agent with:" in line:
            try:
                agent_target = line.split("wandb agent", 1)[1].strip()
                break
            except Exception:
                continue
    if not agent_target:
        raise SystemExit("Could not parse Sweep URL to determine agent target. Pass --entity explicitly.")

    print(f"Launching agent for {args.count} runs on {agent_target} ...")
    subprocess.run(["wandb", "agent", "--count", str(args.count), agent_target], check=True)


if __name__ == "__main__":
    main()
