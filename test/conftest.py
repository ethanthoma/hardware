import argparse

import pytest


def pytest_addoption(parser):
    parser.addoption(
        "--vcd",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="whether to produce vcd files",
    )
