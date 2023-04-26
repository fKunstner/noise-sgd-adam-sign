import json
import os
import sys

import explib.results.wandb_cleanups
from explib.results import data
from explib import cli_helper

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Tools to download results")
    parser.add_argument(
        "--download",
        action="store_true",
        help="Download all the data from wandb",
        default=False,
    )
    parser.add_argument(
        "--download-new",
        action="store_true",
        help="Download all the data from wandb",
        default=False,
    )
    parser.add_argument(
        "--summary",
        action="store_true",
        help="Download the summary of all runs from wandb",
        default=False,
    )
    parser.add_argument(
        "--concat",
        action="store_true",
        help="Concatenate all runs as one file for faster loading",
        default=False,
    )
    parser.add_argument(
        "--download-group",
        default=None,
        type=str,
        help="Download all the data from wandb for a specific group only",
    )
    parser.add_argument(
        "--checkup",
        action="store_true",
        help="Checks if the runs have finished successfully and flag in wandb",
        default=False,
    )
    parser.add_argument(
        "--checkup-group",
        default=None,
        type=str,
        help="Checks if the runs have finished successfully and flag in wandb (for a group only)",
    )
    cli_helper.add_dotenv_option(parser)

    if len(sys.argv) == 1:
        parser.print_help(sys.stderr)

    args = parser.parse_args()
    cli_helper.load_dotenv_if_required(args)

    if args.summary:
        data.download_summary(download_runs=False)
    elif args.download:
        data.download_summary(download_runs=True)
    elif args.download_new:
        data.download_summary(download_runs=True, only_new=True)
    elif args.download_group is not None:
        data.download_summary(download_runs=True, group=args.download_all_group)
    elif args.concat:
        data.concatenate_all_runs()
    elif args.checkup:
        explib.results.wandb_cleanups.checkup()
    elif args.checkup_group is not None:
        explib.results.wandb_cleanups.checkup(group=args.checkup_group)
