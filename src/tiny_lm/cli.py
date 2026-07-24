"""Command-line interface for training, evaluation, and generation."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from dataclasses import replace
from pathlib import Path

from tiny_lm.checkpoint import load_checkpoint, model_from_checkpoint
from tiny_lm.config import Config
from tiny_lm.data import EncodedDataset, encode_dataset, load_data
from tiny_lm.evaluation import dataset_metrics
from tiny_lm.generation import generate
from tiny_lm.model import MiniGPT
from tiny_lm.tokenizer import BPETokenizer
from tiny_lm.training import initialize_runtime, train_model

CONFIG_ARGUMENTS = {
    "data_file": ("--data-file", Path),
    "vocab_size": ("--vocab-size", int),
    "seq_len": ("--seq-len", int),
    "d_model": ("--d-model", int),
    "num_heads": ("--num-heads", int),
    "d_ff": ("--d-ff", int),
    "num_layers": ("--num-layers", int),
    "batch_size": ("--batch-size", int),
    "max_steps": ("--max-steps", int),
    "eval_interval": ("--eval-interval", int),
    "eval_batches": ("--eval-batches", int),
    "learning_rate": ("--learning-rate", float),
    "dropout": ("--dropout", float),
    "max_token_length": ("--max-token-length", int),
    "seed": ("--seed", int),
}


def add_device_argument(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--device",
        choices=("auto", "cpu", "cuda", "mps"),
        default="auto",
        help="execution device (default: auto)",
    )


def add_config_arguments(
    parser: argparse.ArgumentParser,
    fields: Sequence[str] | None = None,
) -> None:
    selected = fields or tuple(CONFIG_ARGUMENTS)
    for field in selected:
        option, value_type = CONFIG_ARGUMENTS[field]
        parser.add_argument(option, dest=field, type=value_type, default=None)
    parser.add_argument(
        "--compile",
        dest="compile_model",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="enable or disable torch.compile",
    )


def config_with_cli_overrides(config: Config, args: argparse.Namespace) -> Config:
    updates = {}
    for field in (*CONFIG_ARGUMENTS, "compile_model"):
        if hasattr(args, field):
            value = getattr(args, field)
            if value is not None:
                updates[field] = value
    updated = replace(config, **updates)
    updated.validate()
    return updated


def run_train(args: argparse.Namespace) -> None:
    config = config_with_cli_overrides(Config.from_env(), args)
    train_samples, validation_samples = load_data(config.data_file)
    device = initialize_runtime(config.seed, args.device)
    tokenizer = BPETokenizer.train(
        train_samples,
        config.vocab_size,
        config.max_token_length,
    )
    train_data = encode_dataset(train_samples, tokenizer, config.seq_len, "training")
    validation_data = encode_dataset(validation_samples, tokenizer, config.seq_len, "validation")

    model = MiniGPT(config, tokenizer.vocab_size).to(device)
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    print(f"Transformer parameters: {parameter_count:,}")
    train_model(
        model,
        tokenizer,
        train_data,
        validation_data,
        config,
        device,
        checkpoint_path=args.checkpoint,
        checkpoint_interval=args.checkpoint_interval,
    )


def run_resume(args: argparse.Namespace) -> None:
    checkpoint = load_checkpoint(args.checkpoint)
    checkpoint_config = Config.from_dict(checkpoint["config"])
    config = config_with_cli_overrides(checkpoint_config, args)
    completed_step = int(checkpoint["step"])
    if config.max_steps <= completed_step:
        raise ValueError(
            f"--max-steps must exceed checkpoint step {completed_step}; "
            f"received {config.max_steps}."
        )

    device = initialize_runtime(config.seed, args.device)
    tokenizer = BPETokenizer.from_dict(
        checkpoint["tokenizer"],
        checkpoint.get("tokenizer_merges"),
    )
    train_samples, validation_samples = load_data(config.data_file)
    train_data = encode_dataset(train_samples, tokenizer, config.seq_len, "training")
    validation_data = encode_dataset(validation_samples, tokenizer, config.seq_len, "validation")
    model = MiniGPT(config, tokenizer.vocab_size).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    train_model(
        model,
        tokenizer,
        train_data,
        validation_data,
        config,
        device,
        checkpoint_path=args.output_checkpoint or args.checkpoint,
        checkpoint_interval=args.checkpoint_interval,
        resume_state=checkpoint,
    )


def run_generate(args: argparse.Namespace) -> None:
    checkpoint = load_checkpoint(args.checkpoint)
    config = Config.from_dict(checkpoint["config"])
    seed = config.seed if args.seed is None else args.seed
    device = initialize_runtime(seed, args.device)
    model, tokenizer, _ = model_from_checkpoint(checkpoint, device)
    generated_ids = generate(
        model,
        tokenizer.encode(args.prompt),
        tokenizer.stop_id,
        temperature=args.temperature,
        top_k=args.top_k,
        max_new_tokens=args.max_new_tokens,
        stop_on_eot=not args.continue_after_eot,
    )
    stop_replacement = "\n" if args.continue_after_eot else ""
    print(tokenizer.decode(generated_ids, stop_replacement=stop_replacement))


def run_evaluate(args: argparse.Namespace) -> None:
    checkpoint = load_checkpoint(args.checkpoint)
    checkpoint_config = Config.from_dict(checkpoint["config"])
    data_file = args.data_file or checkpoint_config.data_file
    device = initialize_runtime(checkpoint_config.seed, args.device)
    model, tokenizer, _ = model_from_checkpoint(checkpoint, device)
    train_samples, validation_samples = load_data(data_file)
    samples = train_samples if args.split == "train" else validation_samples
    dataset = encode_dataset(samples, tokenizer, checkpoint_config.seq_len, args.split)
    if args.max_samples is not None:
        if args.max_samples < 1:
            raise ValueError("--max-samples must be positive.")
        dataset = EncodedDataset(
            dataset.tokens[: args.max_samples],
            dataset.valid_lens[: args.max_samples],
        )
    batch_size = args.batch_size or checkpoint_config.batch_size
    if batch_size < 1:
        raise ValueError("--batch-size must be positive.")
    loss, result = dataset_metrics(model, dataset, batch_size, device)
    print(f"{args.split}_loss={loss:.6f}")
    print(f"{args.split}_perplexity={result:.6f}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train and use the tiny BPE Transformer.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    train_parser = subparsers.add_parser("train", help="train a new tokenizer and model")
    add_device_argument(train_parser)
    add_config_arguments(train_parser)
    train_parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path("checkpoints/latest.pt"),
        help="checkpoint output path (default: checkpoints/latest.pt)",
    )
    train_parser.add_argument(
        "--checkpoint-interval",
        type=int,
        default=1000,
        help="save every N completed steps (default: 1000)",
    )
    train_parser.set_defaults(handler=run_train)

    resume_parser = subparsers.add_parser("resume", help="resume training from a checkpoint")
    resume_parser.add_argument("checkpoint", type=Path)
    add_device_argument(resume_parser)
    add_config_arguments(
        resume_parser,
        fields=(
            "data_file",
            "batch_size",
            "max_steps",
            "eval_interval",
            "eval_batches",
        ),
    )
    resume_parser.add_argument(
        "--output-checkpoint",
        type=Path,
        default=None,
        help="write to a new checkpoint instead of replacing the input",
    )
    resume_parser.add_argument(
        "--checkpoint-interval",
        type=int,
        default=1000,
        help="save every N completed steps (default: 1000)",
    )
    resume_parser.set_defaults(handler=run_resume)

    generate_parser = subparsers.add_parser("generate", help="generate text from a checkpoint")
    generate_parser.add_argument("checkpoint", type=Path)
    generate_parser.add_argument("--prompt", required=True)
    generate_parser.add_argument("--temperature", type=float, default=1.0)
    generate_parser.add_argument("--top-k", type=int, default=50)
    generate_parser.add_argument("--max-new-tokens", type=int, default=100)
    generate_parser.add_argument("--seed", type=int, default=None)
    generate_parser.add_argument(
        "--continue-after-eot",
        action="store_true",
        help="continue across end-of-text boundaries and render them as newlines",
    )
    add_device_argument(generate_parser)
    generate_parser.set_defaults(handler=run_generate)

    evaluate_parser = subparsers.add_parser(
        "evaluate", help="evaluate a checkpoint on the configured dataset"
    )
    evaluate_parser.add_argument("checkpoint", type=Path)
    evaluate_parser.add_argument("--data-file", type=Path, default=None)
    evaluate_parser.add_argument(
        "--split",
        choices=("train", "validation"),
        default="validation",
    )
    evaluate_parser.add_argument("--batch-size", type=int, default=None)
    evaluate_parser.add_argument("--max-samples", type=int, default=None)
    add_device_argument(evaluate_parser)
    evaluate_parser.set_defaults(handler=run_evaluate)
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        args.handler(args)
    except (OSError, UnicodeError, ValueError, RuntimeError) as exc:
        parser.error(str(exc))


if __name__ == "__main__":
    main()
