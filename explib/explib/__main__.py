import json
import os

from explib import cli_helper
from .experiment import Experiment

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Experiment runner")
    parser.add_argument(
        "experiment_file",
        type=str,
        help="Experiment file",
        default=None,
    )
    parser.add_argument(
        "--disable_wandb",
        action="store_true",
        help="Debug mode, won't create wandb logs",
        default=False,
    )
    parser.add_argument("--gpu", nargs="?", type=str, default="cuda", help="GPU name")
    parser.add_argument(
        "--dummy_run",
        action="store_true",
        help="Enable dummy run for tests - only runs one iteration per epoch",
        default=False,
    )
    cli_helper.add_dotenv_option(parser)

    args = parser.parse_args()

    cli_helper.load_dotenv_if_required(args)

    if args.experiment_file is None:
        raise ValueError

    with open(args.experiment_file, "r") as fp:
        exp_dict = json.load(fp)
        filename = os.path.split(args.experiment_file)[1]
        basename, ext = os.path.splitext(filename)
        slug, uuid = basename.rsplit("_", 1)
        exp = Experiment(
            exp_dict,
            slug,
            uuid,
            args.disable_wandb,
            args.gpu,
            args.dummy_run,
        )
        exp.run()
