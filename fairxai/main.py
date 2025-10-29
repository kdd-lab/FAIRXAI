import argparse
from setup import version

def main():
    #########################################
    # create the top-level parser
    #########################################
    parser = argparse.ArgumentParser(prog='fairxai', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("-v", "--version", help="Installed gemini version",
                        action="version", version="%(prog)s " + version)
    subparsers = parser.add_subparsers(title='[sub-commands]', dest='command')

    #########################################
    # $ fairxai browser
    #########################################
    parser_browser = subparsers.add_parser('browser',help='Browser interface to FAIRXAI')
    parser_browser.add_argument('--host', metavar='host', default='localhost',help='Hostname, default: localhost.')
    parser_browser.add_argument('--port', metavar='port', default='8088',help='Port, default: 8088.')

    def browser_fn(parser, args):
        from fairxai.app import app
        browser.browser_main(parser, args)

    parser_browser.set_defaults(func=browser_fn)

if __name__ == "__main__":
    main()
