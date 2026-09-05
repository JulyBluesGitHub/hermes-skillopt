"""Say which `skillopt` the suite is about to prove something about.

A local editable checkout of upstream and an install from PyPI can carry the same
version string and ship different modules. Published 0.1.0 has no `skillopt_sleep`
at all; the checkout that calls itself 0.1.0 does. A green run that does not say
which of the two it ran against is not evidence, and that divergence has already
shipped one break to `main` (a fixed-signature wrapper that raised TypeError on
every replay under 0.2.0, while local pytest stayed green).

So the header carries the version and the path. An editable install resolves
outside site-packages, which is the tell.
"""

import importlib.metadata as metadata
import os


def pytest_report_header(config):
    try:
        version = metadata.version("skillopt")
    except metadata.PackageNotFoundError:
        return "skillopt: NOT INSTALLED"
    try:
        import skillopt_sleep
        path = os.path.dirname(skillopt_sleep.__file__)
    except ImportError:
        return f"skillopt: {version} (no skillopt_sleep module; needs >= 0.2.0)"
    editable = "site-packages" not in path.replace("\\", "/")
    return f"skillopt: {version} from {path}" + (" (editable checkout)" if editable else "")
