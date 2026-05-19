"""Optional Cesium simulator shim for legacy imports."""


class CesiumEarthImageSimulator:  # pylint: disable=too-few-public-methods
    """Placeholder class when Cesium integration is unavailable in this repo."""

    def __init__(self, *args, **kwargs):  # noqa: D401
        raise RuntimeError(
            "CesiumEarthImageSimulator is not available in this repository. "
            "Use use_cesium=False for simulated image generation."
        )
