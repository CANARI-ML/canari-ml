# import typer
# from canari_ml.download import main as download_main

# app = typer.Typer()

# @app.command()
# def download():
#     """Download dataset."""
#     download_main()


# if __name__ == "__main__":
#     app()


import argparse
import sys
import os

from canari_ml.cli import train, predict
from canari_ml.download import era5
from canari_ml.preprocess import preprocess


def main():
    prog_name = os.path.basename(sys.argv[0])

    parser = argparse.ArgumentParser(prog=prog_name, add_help=True)
    subparsers = parser.add_subparsers(dest="command", help="Available subcommands")

    # Add subcommands without help (let Hydra handle it)
    subparsers.add_parser("download", add_help=False)

    # Pre-processing commands
    preprocess_parser = subparsers.add_parser("preprocess", add_help=True)
    preprocess_subparsers = preprocess_parser.add_subparsers(dest="subcommand")
    preprocess_subparsers.add_parser("train", add_help=False)
    preprocess_subparsers.add_parser("predict", add_help=False)

    # Train/predict commands
    subparsers.add_parser("train", add_help=False)
    subparsers.add_parser("predict", add_help=False)

    # Let argparse only parse known args
    args, unknown_args = parser.parse_known_args()

    # Reconstruct `sys.argv` for Hydra (removing the command/subcommand parts)
    sys.argv = [prog_name] + unknown_args
    if args.command == "download":
        era5.download()
    elif args.command == "preprocess":
        # Takes in `args.subcommand` of `train` or `predict`
        preprocess.main(preprocess_type=args.subcommand)
    elif args.command == "train":
        train.main()
    elif args.command == "predict":
        predict.main()
    # elif args.command == "preprocess":
    #     if args.subcommand in ["translate", "rotate"]:
    #         getattr(preprocess, f"{args.subcommand}_main")()
    #     else:
    #         print(f"Usage: {prog_name} preprocess {{translate,rotate}} [HYDRA_ARGS]")
    #         sys.exit(1)
    else:
        commands = ",".join(list(subparsers.choices.keys()))
        print(f"Usage: {prog_name} {{commands}} [HYDRA_ARGS]")
        sys.exit(1)


if __name__ == "__main__":
    main()
