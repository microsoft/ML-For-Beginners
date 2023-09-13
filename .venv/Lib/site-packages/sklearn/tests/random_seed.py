"""global_random_seed fixture

The goal of this fixture is to prevent tests that use it to be sensitive
to a specific seed value while still being deterministic by default.

See the documentation for the SKLEARN_TESTS_GLOBAL_RANDOM_SEED
variable for insrtuctions on how to use this fixture.

https://scikit-learn.org/dev/computing/parallelism.html#sklearn-tests-global-random-seed
"""
from os import environ
from random import Random

import pytest


# Passes the main worker's random seeds to workers
class XDistHooks:
    def pytest_configure_node(self, node) -> None:
        random_seeds = node.config.getoption("random_seeds")
        node.workerinput["random_seeds"] = random_seeds


def pytest_configure(config):
    if config.pluginmanager.hasplugin("xdist"):
        config.pluginmanager.register(XDistHooks())

    RANDOM_SEED_RANGE = list(range(100))  # All seeds in [0, 99] should be valid.
    random_seed_var = environ.get("SKLEARN_TESTS_GLOBAL_RANDOM_SEED")
    if hasattr(config, "workerinput") and "random_seeds" in config.workerinput:
        # Set worker random seed from seed generated from main process
        random_seeds = config.workerinput["random_seeds"]
    elif random_seed_var is None:
        # This is the way.
        random_seeds = [42]
    elif random_seed_var == "any":
        # Pick-up one seed at random in the range of admissible random seeds.
        random_seeds = [Random().choice(RANDOM_SEED_RANGE)]
    elif random_seed_var == "all":
        random_seeds = RANDOM_SEED_RANGE
    else:
        if "-" in random_seed_var:
            start, stop = random_seed_var.split("-")
            random_seeds = list(range(int(start), int(stop) + 1))
        else:
            random_seeds = [int(random_seed_var)]

        if min(random_seeds) < 0 or max(random_seeds) > 99:
            raise ValueError(
                "The value(s) of the environment variable "
                "SKLEARN_TESTS_GLOBAL_RANDOM_SEED must be in the range [0, 99] "
                f"(or 'any' or 'all'), got: {random_seed_var}"
            )
    config.option.random_seeds = random_seeds

    class GlobalRandomSeedPlugin:
        @pytest.fixture(params=random_seeds)
        def global_random_seed(self, request):
            """Fixture to ask for a random yet controllable random seed.

            All tests that use this fixture accept the contract that they should
            deterministically pass for any seed value from 0 to 99 included.

            See the documentation for the SKLEARN_TESTS_GLOBAL_RANDOM_SEED
            variable for insrtuctions on how to use this fixture.

            https://scikit-learn.org/dev/computing/parallelism.html#sklearn-tests-global-random-seed
            """
            yield request.param

    config.pluginmanager.register(GlobalRandomSeedPlugin())


def pytest_report_header(config):
    random_seed_var = environ.get("SKLEARN_TESTS_GLOBAL_RANDOM_SEED")
    if random_seed_var == "any":
        return [
            "To reproduce this test run, set the following environment variable:",
            f'    SKLEARN_TESTS_GLOBAL_RANDOM_SEED="{config.option.random_seeds[0]}"',
            (
                "See: https://scikit-learn.org/dev/computing/parallelism.html"
                "#sklearn-tests-global-random-seed"
            ),
        ]
