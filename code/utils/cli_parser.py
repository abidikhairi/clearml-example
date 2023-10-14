import argparse


def get_default_parser() -> argparse.ArgumentParser:
    """Create a parser object

    Returns:
        argparse.ArgumentParser
    """
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--data_root", default='data', help="Root data dir")
    parser.add_argument('--num_epochs', type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--log_interval", type=int, default=100)

    return parser