"""
Machining Feature Recognition (MFR) Toolkit
Using pythonOCC for B-Rep Hole/Boss/Chamfer feature recognition.
"""

__version__ = "2.0.0"

try:
    from . import utils
    from . import feature_recognizer
    _OCC_AVAILABLE = True
except ImportError:
    _OCC_AVAILABLE = False

__all__ = []

if _OCC_AVAILABLE:
    __all__.extend([
        "utils",
        "feature_recognizer",
    ])
