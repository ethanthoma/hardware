import argparse

import pytest

timing_results = {}


def pytest_addoption(parser):
    parser.addoption(
        "--vcd",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="whether to produce vcd files",
    )


def pytest_terminal_summary(terminalreporter, exitstatus, config):
    if timing_results:
        terminalreporter.section("Component Timing")
        for name, info in timing_results.items():
            terminalreporter.write_line(f"  {name}: {info}")


@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_call(item):
    yield
    if hasattr(item, "timing_info"):
        test_name = item.name.replace("test_", "").replace("_timing", "")
        timing_results[test_name] = item.timing_info
