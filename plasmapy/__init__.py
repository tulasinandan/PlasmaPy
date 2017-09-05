from ._metadata import (
    name as __name__,
    version as __version__,
    description as __doc__,
    author as __author__,
)

from .classes import Plasma
from . import classes
from . import constants
from . import atomic
from . import analytic
from . import physics
from . import utils

import sys
import warnings

if sys.version_info[:2] < (3, 6):  # coveralls: ignore
    warnings.warn("PlasmaPy does not support Python 3.5 and below")
