"""Run paired G1 root-tracking evaluations with a fixed motion set and seed.

Run with the SONIC training Python and Isaac Sim environment. A second pass
applies the same world-frame velocity impulse to both controllers.
"""

import argparse
import json
from pathlib import Path
import subprocess
import sys


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--adapted", type=Path, required=True)
    parser.add_argument("--motion-file", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--num-envs", type=int, default=16)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-motions", type=int, default=512)
    parser.add_argument("--push-step", type=int, default=50)
    args = parser.parse_args()
    repo = Path(__file__).resolve().parents[2]
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    summary = {"seed": args.seed, "motion_file": str(args.motion_file.resolve()), "scenarios": {}}
    for scenario, push_step in (("nominal", None), ("push", args.push_step)):
        paired = {}
        for label, checkpoint in (("baseline", args.baseline), ("adapted", args.adapted)):
            directory = output / scenario / label
            directory.mkdir(parents=True, exist_ok=True)
            command = [
                sys.executable, str(repo / "gear_sonic/eval_agent_trl.py"),
                f"checkpoint={checkpoint.resolve()}",
                f"++seed={args.seed}", f"++num_envs={args.num_envs}",
                "++headless=true", "++use_wandb=false",
                "++eval_callbacks=im_eval", "++run_eval_loop=false",
                f"++eval_output_dir={directory}",
                "++callbacks.im_eval.root_tracking_metrics=true",
                f"++callbacks.im_eval.root_tracking_push_step={push_step if push_step is not None else 'null'}",
                "++callbacks.im_eval.root_tracking_push_velocity=[0.0,0.3,0.0]",
                f"++manager_env.commands.motion.motion_lib_cfg.motion_file={args.motion_file.resolve()}",
                f"++manager_env.commands.motion.motion_lib_cfg.max_unique_motions={args.max_motions}",
                "++manager_env.commands.motion.motion_lib_cfg.smpl_motion_file=dummy",
                "++manager_env.commands.motion.encoder_sample_probs.g1=1.0",
                "++manager_env.commands.motion.encoder_sample_probs.teleop=0.0",
                "++manager_env.commands.motion.encoder_sample_probs.smpl=0.0",
                "+manager_env/terminations=tracking/eval",
            ]
            print(f"Evaluating {scenario}/{label}; log: {directory / 'eval.log'}", flush=True)
            with (directory / "eval.log").open("w") as log:
                subprocess.run(command, cwd=repo, stdout=log, stderr=subprocess.STDOUT, check=True)
            paired[label] = json.loads((directory / "metrics_eval.json").read_text())
        assert (
            paired["baseline"]["eval/all_metrics_dict"]["motion_keys"]
            == paired["adapted"]["eval/all_metrics_dict"]["motion_keys"]
        )
        names = [name for name in paired["baseline"] if name.startswith("eval/root_tracking/")]
        summary["scenarios"][scenario] = {
            name: {label: metrics[name] for label, metrics in paired.items()} for name in names
        }
        (output / "comparison.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
