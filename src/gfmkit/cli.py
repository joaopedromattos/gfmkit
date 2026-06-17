from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from .api import create_model, list_models


def _parse_value(raw: str) -> Any:
    lower = raw.lower()
    if lower in {"true", "false"}:
        return lower == "true"
    if lower in {"none", "null"}:
        return None
    try:
        return int(raw)
    except ValueError:
        pass
    try:
        return float(raw)
    except ValueError:
        return raw


def _parse_sets(items: list[str] | None) -> dict[str, Any]:
    params = {}
    for item in items or []:
        if "=" not in item:
            raise SystemExit(f"--set values must use key=value syntax, got {item!r}")
        key, raw = item.split("=", 1)
        params[key] = _parse_value(raw)
    return params


def _add_common(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--model", required=True, help="Model name, e.g. GNN, GPF, GraphPrompt, GFT, GILT.")
    parser.add_argument("--datasets", default=None, help="Dataset preset/name/list. Defaults depend on command.")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--data-root", type=Path, default=None)
    parser.add_argument("--output-root", type=Path, default=None)
    parser.add_argument("--checkpoint-root", type=Path, default=None)
    parser.add_argument("--set", action="append", default=[], help="Adapter hyperparameter override as key=value.")


def _model_from_args(args):
    return create_model(
        args.model,
        seed=args.seed,
        device=args.device,
        data_root=args.data_root,
        output_root=args.output_root,
        checkpoint_root=args.checkpoint_root,
        **_parse_sets(args.set),
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="gfmkit")
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("list-models", help="List registered GFM adapters.")

    for command in ("pretrain", "finetune", "infer", "run-lp"):
        cmd = sub.add_parser(command)
        _add_common(cmd)
        if command in {"infer", "run-lp"}:
            cmd.add_argument("--save-metadata", action="store_true", help="Include metadata fields in saved .pt outputs.")

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command == "list-models":
        for name in list_models():
            print(name)
        return 0

    model = _model_from_args(args)
    if args.command == "pretrain":
        model.pretrain(args.datasets or "link1")
    elif args.command == "finetune":
        model.finetune(args.datasets or "link2")
    elif args.command == "infer":
        outputs = model.infer(args.datasets or "link2")
        if getattr(args, "save_metadata", False):
            model.save_outputs(outputs, include_metadata=True)
    elif args.command == "run-lp":
        model.pretrain("link1")
        model.finetune(args.datasets or "link2")
        outputs = model.infer(args.datasets or "link2")
        if getattr(args, "save_metadata", False):
            model.save_outputs(outputs, include_metadata=True)
    else:
        parser.error(f"Unknown command: {args.command}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

