"""Stop an app with a hint when matplotlib cannot open a window (apps:APPX-2).

With a non-interactive backend such as Agg, plt.show() returns at once, and the app
used to end with exit code 0 without ever showing a window. require_interactive_backend()
prints what to install and exits with code 2 instead. GWD_HEADLESS_OK=1 skips the check,
so the smoke tests can start the apps under Agg.
"""

import os
import sys

import matplotlib
from matplotlib.backends import BackendFilter, backend_registry

HEADLESS_OK = "GWD_HEADLESS_OK"
EXIT_NO_BACKEND = 2

HINT = """\
{app}: matplotlib uses the backend '{backend}', which cannot open a window.
Install Tk for the Python that runs gwd, for example:
  Debian/Ubuntu: sudo apt install python3-tk
  Fedora:        sudo dnf install python3-tkinter
On WSL, windows need WSLg (update with 'wsl --update').
If MPLBACKEND is set to a non-interactive backend such as Agg, unset it.
'gwd env' shows the backend. {headless_ok}=1 starts the app without a window."""


def backend_status():
    """(name, interactive) of the backend pyplot uses.

    Only matplotlib's own non-interactive backends (agg, cairo, pdf, pgf, ps, svg,
    template) count as not interactive; any other backend may open a window.
    """
    name = matplotlib.get_backend()
    headless = backend_registry.list_builtin(BackendFilter.NON_INTERACTIVE)
    return name, name.lower() not in headless


def require_interactive_backend(app):
    """Exit with EXIT_NO_BACKEND and a hint on stderr if app cannot open a window."""
    if os.environ.get(HEADLESS_OK) == "1":
        return
    name, interactive = backend_status()
    if not interactive:
        print(
            HINT.format(app=app, backend=name, headless_ok=HEADLESS_OK),
            file=sys.stderr,
        )
        sys.exit(EXIT_NO_BACKEND)
