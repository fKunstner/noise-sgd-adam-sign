from explib import config


def add_dotenv_option(parser):
    parser.add_argument(
        "--dotenv",
        type=str,
        help=".env file to override local environment variables (including workspace)",
        default=None,
    )
    return parser


def load_dotenv_if_required(args):
    if getattr(args, "dotenv", None) is not None:
        config.load_dotenv_file(args["dotenv"])
