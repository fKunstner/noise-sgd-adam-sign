import os
from dotenv import load_dotenv


def load_dotenv_file(path=None):
    """Load a dotenv file from path (defaults to cwd if None)"""
    if path is None:
        load_dotenv(verbose=True, override=True)
    else:
        load_dotenv_file(path, verbose=True, override=True)


def get_workspace():
    return os.path.realpath(os.environ["EXPLIB_WORKSPACE"])


def get_wandb_project():
    return os.environ["EXPLIB_WANDB_PROJECT"]


def get_notification_email():
    return os.environ.get("EXPLIB_NOTIFICATION_EMAIL", None)


def get_wandb_entity():
    return os.environ["EXPLIB_WANDB_ENTITY"]


def get_slurm_partition():
    return os.environ.get("EXPLIB_SLURM_PARTITION", None)


def get_conda_env():
    return os.environ.get("EXPLIB_LOAD_CONDA", None)


def get_env_file_to_source():
    return os.environ.get("EXPLIB_ENVFILE", None)


def get_slurm_account():
    return os.environ.get("EXPLIB_SLURM_ACCOUNT", "def-schmidtm")


def get_console_logging_level():
    return os.environ.get("EXPLIB_CONSOLE_LOGGING_LEVEL", "INFO")


def get_all():
    return {
        "workspace": get_workspace(),
        "wandb_project": get_wandb_project(),
        "wandb_entity": get_wandb_entity(),
        "slurm_account": get_slurm_account(),
        "notification_email": get_notification_email(),
        "console_logging_level": get_console_logging_level(),
    }
