"""CLI entry points. Parsing arguments does not import model dependencies."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from selectvp.config import load_config
from selectvp.data import load_samples, load_truth, normalize_annotations, validate_samples
from selectvp.utils.io import read_json, write_json


def parser():
    root = argparse.ArgumentParser(prog="selectvp", description="SelectVP inference and evaluation tools")
    commands = root.add_subparsers(dest="command", required=True)
    for name in ("prepare-data", "preprocess", "infer", "evaluate"):
        child = commands.add_parser(name)
        child.add_argument("--config", required=True)
        child.add_argument("--set", action="append", default=[], metavar="KEY=VALUE")
        group = child.add_mutually_exclusive_group()
        group.add_argument("--sample-ids", help="JSON list of typed sample IDs")
        group.add_argument("--full", action="store_true", help="Explicitly process the whole split")
        if name == "prepare-data":
            child.add_argument("--source")
            child.add_argument("--durations", help="Charades CSV or video-to-duration JSON")
            child.add_argument("--annotations-only", action="store_true", help="Convert annotations before videos are available")
            child.add_argument("--check", action="store_true", help="Validate already-converted inputs")
        elif name == "preprocess":
            child.add_argument("--stages", default="all", help="Comma-separated: videos,blip2,events,targets,detection,keyframes,prompts")
        elif name == "infer":
            child.add_argument("--branch", choices=("bl", "es", "od", "km", "all"), default="all")
            child.add_argument("--retry-failed", action="store_true")
        else:
            child.add_argument("--predictions", help="Final predictions; default: paths.output/predictions.json")
            child.add_argument("--allow-missing", action="store_true")
    return root


def main(argv=None):
    args = parser().parse_args(argv)
    try:
        config = load_config(args.config, args.set)
        if args.command == "prepare-data":
            if not args.check:
                samples, truth = normalize_annotations(args.source or config["data"]["source"], config["dataset"], args.durations)
                write_json(config["data"]["annotations"], samples)
                if truth:
                    write_json(config["data"]["ground_truth"], truth)
            samples = load_samples(config, args.full, args.sample_ids, check_videos=not args.annotations_only)
            if args.check:
                load_truth(config, samples)
            print(f"Validated {len(samples)} selected queries for {config['dataset']}")
            return 0
        samples = load_samples(config, args.full, args.sample_ids,
                               check_videos=args.command == "infer" or (args.command == "preprocess" and "videos" not in args.stages and args.stages != "all"))
        if args.command == "evaluate":
            from selectvp.evaluation.metrics import evaluate
            predictions = read_json(args.predictions or Path(config["paths"]["output"]) / "predictions.json")
            metrics = evaluate(predictions, load_truth(config, samples), config["evaluation"], args.allow_missing)
            write_json(Path(config["paths"]["output"]) / "metrics.json", metrics)
            print(json.dumps(metrics, ensure_ascii=False, indent=2))
            return 0
        universe = read_json(config["data"]["annotations"])
        if args.command == "preprocess":
            from selectvp.preprocessing.runner import Preprocessor, STAGES
            stages = STAGES if args.stages == "all" else args.stages.split(",")
            Preprocessor(config, samples, universe).run(stages)
            return 0
        from selectvp.pipeline import infer
        from selectvp.selection.selector import BRANCHES
        success = infer(config, samples, universe, BRANCHES if args.branch == "all" else (args.branch,), args.retry_failed)
        return 0 if success else 1
    except (ValueError, FileNotFoundError, KeyError, RuntimeError) as exc:
        print(f"selectvp: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
