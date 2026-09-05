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

# Run the scorer a machine without the optional [topic] extra runs, which is also
# what CI runs. Left to itself the suite would pick the embedding scorer on any
# developer box that happens to have torch, testing a configuration CI never
# exercises and paying 15 seconds to load a transformer for tests that are about
# task mining. tests/test_topic.py covers the embedding path on its own terms.
os.environ.setdefault("HERMES_SKILLOPT_TOPIC_SCORER", "lexical")


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
