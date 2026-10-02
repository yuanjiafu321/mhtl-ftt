from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .config import load_config
from .synthetic import generate_synthetic


def parser():
    cli = argparse.ArgumentParser(prog="mhtl-ftt", description="Multi-task FT-Transformer pipeline")
    commands = cli.add_subparsers(dest="command", required=True)
    for name in ("synthetic", "demo"):
        command = commands.add_parser(name)
        command.add_argument("--output", required=True, type=Path)
        command.add_argument("--config", type=Path)
        command.add_argument("--synthetic-seed", type=int, default=42)
        command.add_argument("--counts", type=int, nargs=3, default=[120, 80, 50], metavar=("LANDSLIDE", "COLLAPSE", "DEBRIS"))
        command.add_argument("--pool-size", type=int, default=1200)
    for name in ("prepare", "run"):
        command = commands.add_parser(name)
        command.add_argument("--data", required=True, type=Path)
        command.add_argument("--output", required=True, type=Path)
        command.add_argument("--config", type=Path)
    for name in ("cv", "fit", "evaluate"):
        command = commands.add_parser(name)
        command.add_argument("--run-dir", required=True, type=Path)
    command = commands.add_parser("predict")
    command.add_argument("--run-dir", required=True, type=Path)
    command.add_argument("--input", required=True, type=Path)
    command.add_argument("--output", required=True, type=Path)
    command.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    command.add_argument("--fold", type=int, help="Predict with one fold; otherwise average all fold models")
    return cli


def main(argv=None):
    args = parser().parse_args(argv)
    try:
        config = load_config(getattr(args, "config", None))
        if args.command == "synthetic":
            generate_synthetic(args.output, args.synthetic_seed, args.counts, args.pool_size, config.data)
            print(f"Synthetic inputs saved to {args.output}")
            return 0
        from . import pipeline
        if args.command == "demo":
            if args.output.exists() and any(args.output.iterdir()):
                raise FileExistsError("Choose an empty output directory for the demo")
            inputs = generate_synthetic(args.output / "inputs", args.synthetic_seed, args.counts, args.pool_size, config.data)
            pipeline.run(inputs, args.output, config)
        elif args.command == "prepare":
            pipeline.prepare(args.data, args.output, config)
        elif args.command == "run":
            pipeline.run(args.data, args.output, config)
        elif args.command == "cv":
            pipeline.cross_validate(args.run_dir)
        elif args.command == "fit":
            pipeline.fit(args.run_dir)
        elif args.command == "evaluate":
            pipeline.evaluate(args.run_dir)
        elif args.command == "predict":
            pipeline.predict(args.run_dir, args.input, args.output, args.device, args.fold)
        print("Completed.")
        return 0
    except (FileNotFoundError, FileExistsError, ValueError, RuntimeError) as error:
        print(f"Error: {error}", file=sys.stderr)
        raise SystemExit(1) from error


if __name__ == "__main__":
    main()

