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

from canari_ml.cli import download, preprocess


def main():
    prog_name = os.path.basename(sys.argv[0])

    parser = argparse.ArgumentParser(prog=prog_name, add_help=True)
    subparsers = parser.add_subparsers(dest="command", help="Available subcommands")

    # Add subcommands without help (let Hydra handle it)
    subparsers.add_parser("download", add_help=False)
    preprocess_parser = subparsers.add_parser("preprocess", add_help=False)
    # preprocess_subparsers = preprocess_parser.add_subparsers(dest="subcommand")
    # preprocess_subparsers.add_parser("translate", add_help=False)
    # preprocess_subparsers.add_parser("rotate", add_help=False)

    # Let argparse only parse known args
    args, unknown_args = parser.parse_known_args()

    # Reconstruct `sys.argv` for Hydra (removing the command/subcommand parts)
    sys.argv = [prog_name] + unknown_args
    if args.command == "download":
        download.main()
    elif args.command == "preprocess":
        preprocess.main()
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
