import logging
import os
from pathlib import Path
import wandb
from wandb.util import generate_id
from dotenv import load_dotenv
import sys
import datetime

from explib import config

base_logger = None
wandb_is_enabled = True


def log_data(dict, commit=True):
    if wandb_is_enabled:
        wandb.log(dict, commit=commit)
    base_logger.info(dict)


def init_logging_stdout(level=None):
    global base_logger

    logging.basicConfig(
        level=config.get_console_logging_level() if level is None else level,
        format="%(asctime)s %(filename)s[line:%(lineno)d] %(levelname)s %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%S",
    )
    base_logger = logging.getLogger(__name__)

    base_logger.info("Explib env/configuration: {}".format(config.get_all()))


def init_logging_for_exp(
    slug, exp_uuid, exp_dict, disable_wandb, additional_config=None
):
    """Initialize the logging"""

    load_dotenv()

    logs_path = os.path.join(config.get_workspace(), slug, "logs")
    Path(logs_path).mkdir(parents=True, exist_ok=True)

    timestring = datetime.datetime.utcnow().strftime("%Y-%m-%dT%H-%M-%S%z")
    log_filename = f"{slug}_{exp_uuid}_{timestring}.log"
    file_path = os.path.join(logs_path, log_filename)
    logging.basicConfig(
        level=logging.DEBUG,
        format="%(asctime)s %(filename)s[line:%(lineno)d] %(levelname)s %(message)s",
        datefmt="%a, %d %b %Y %H:%M:%S",
        filename=file_path,
        filemode="a+",
    )

    global base_logger
    base_logger = logging.getLogger(__name__)
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    formatter = logging.Formatter(
        "%(asctime)s %(filename)s[line:%(lineno)d] %(levelname)s %(message)s"
    )
    console.setFormatter(formatter)
    base_logger.addHandler(console)

    global wandb_is_enabled
    if disable_wandb:
        wandb_is_enabled = False
    else:
        if additional_config is None:
            additional_config = {}
        wandb.init(
            project=config.get_wandb_project(),
            id=generate_id(16),
            entity=config.get_wandb_entity(),
            dir=logs_path,
            config={**exp_dict, **additional_config},
            group=slug,
            force=True,
        )
        wandb_is_enabled = True

    def error_handler(exctype, value, tb):
        base_logger.error("Uncaught exception", exc_info=(exctype, value, tb))

    sys.excepthook = error_handler

    return log_data


def info(*args, **kwargs):
    base_logger.info(*args, **kwargs)


def warn(*args, **kwargs):
    base_logger.warn(*args, **kwargs)


def debug(*args, **kwargs):
    base_logger.debug(*args, **kwargs)


def full_stack():
    """https://stackoverflow.com/a/16589622"""
    import traceback, sys

    exc = sys.exc_info()[0]
    stack = traceback.extract_stack()[:-1]  # last one would be full_stack()
    if exc is not None:  # i.e. an exception is present
        del stack[-1]  # remove call of full_stack, the printed exception
        # will contain the caught exception caller instead
    trc = "Traceback (most recent call last):\n"
    stackstr = trc + "".join(traceback.format_list(stack))
    if exc is not None:
        stackstr += "  " + traceback.format_exc().lstrip(trc)
    return stackstr
