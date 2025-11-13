import argparse
import os
import subprocess
from importlib.metadata import version
try:
    PACKAGE_VERSION = version("fairxai")
except Exception:
    # Fallback/caso di sviluppo se il pacchetto non è ancora installato
    PACKAGE_VERSION = "0.0.0 (Not Installed)"

def main():
    #########################################
    # create the top-level parser
    #########################################
    parser = argparse.ArgumentParser(prog="fairxai", description="FAIRXAI command-line interface")
    parser.add_argument(
        "-v", "--version",
        help="Show installed FAIRXAI version",
        action="version",
        version=f"%(prog)s {PACKAGE_VERSION}"
    )
    subparsers = parser.add_subparsers(title='[sub-commands]', dest='command')

    #########################################
    # $ fairxai app
    #########################################
    parser_app = subparsers.add_parser('app', help='Launch the FAIRXAI Streamlit WebApp')

    def app_fn(parser, args):
        app_path = os.path.join(os.path.dirname(__file__), "app", "app.py")
        subprocess.run(["streamlit", "run", app_path])

    parser_app.set_defaults(func=app_fn)


    #########################################
    # Parse and execute
    #########################################
    args = parser.parse_args()
    if not args.command:
        parser.print_help()
        return
    args.func(parser, args)

if __name__ == "__main__":
    main()
