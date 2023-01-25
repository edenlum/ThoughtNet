import logging
import argparse
import torch
import json

def cli():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "cfg",
        help="configuration json file", type=str)
    parser.add_argument(
        '-d', '--debug',
        help="Print lots of debugging statements",
        action="store_const", dest="loglevel", const=logging.DEBUG,
        default=logging.CRITICAL,
    )
    parser.add_argument(
        '-v', '--verbose',
        help="Be verbose",
        action="store_const", dest="loglevel", const=logging.INFO,
    )
    args = parser.parse_args()    
    logging.basicConfig(level=args.loglevel)
    return args 


def main():
    args = cli()
    logging.debug(args)
    with open(args.cfg) as f:
        config = json.load(f)
    # load network from args.cfg
    net = torch.load(config['net'])



if __name__ == '__main__':
    main()