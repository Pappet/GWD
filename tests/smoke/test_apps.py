import pytest

from .conftest import assert_ran_cleanly, run_stage
from .entrypoints import APPS

pytestmark = pytest.mark.apps


@pytest.mark.parametrize("name", APPS)
def test_app_starts_without_interactive_backend(name, app_home):
    assert_ran_cleanly(name, run_stage(APPS[name], app_home))
