from .__version__ import __version__

__all__ = [
    "mosaic",
    "__version__",
]


def __getattr__(name):
    if name == "mosaic":
        from .coordinator import mosaic

        return mosaic
    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")
