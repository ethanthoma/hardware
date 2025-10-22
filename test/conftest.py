import argparse


def pytest_addoption(parser):
    parser.addoption(
        "--vcd",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="whether to produce vcd files",
    )
