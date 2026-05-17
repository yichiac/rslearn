"""Random temporal truncation transform for time-series inputs."""

from typing import Any

import torch

from rslearn.train.model_context import RasterImage

from .transform import Transform, read_selector, write_selector


class RandomTemporalTruncate(Transform):
    """Randomly truncate a raster input to a contiguous temporal prefix.

    Trains models to handle variable-length time series at inference, e.g. for
    in-season prediction where only a leading portion of the season is
    available. For each sample, k is sampled uniformly from
    [min_timesteps, max_timesteps] and only the first k timesteps along the T
    axis are kept. Use it only in train_config.transforms so val/test/predict
    continue to see the full sequence.
    """

    def __init__(
        self,
        selector: str,
        min_timesteps: int = 1,
        max_timesteps: int | None = None,
    ):
        """Initialize a new RandomTemporalTruncate.

        Args:
            selector: name of the raster input to truncate (e.g. "sentinel2_l2a").
                Standard rslearn selector syntax — defaults to the input dict.
            min_timesteps: minimum number of leading timesteps to keep (inclusive).
            max_timesteps: maximum number of leading timesteps to keep (inclusive).
                If None, uses the current time dim of the input. Capped to the
                runtime T if larger.
        """
        super().__init__()
        if min_timesteps < 1:
            raise ValueError("min_timesteps must be >= 1")
        if max_timesteps is not None and max_timesteps < min_timesteps:
            raise ValueError("max_timesteps must be >= min_timesteps")
        self.selector = selector
        self.min_timesteps = min_timesteps
        self.max_timesteps = max_timesteps

    def forward(
        self, input_dict: dict[str, Any], target_dict: dict[str, Any]
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        """Truncate the selected raster's time dimension to a random prefix.

        Args:
            input_dict: the input dict.
            target_dict: the target dict.

        Returns:
            tuple of (input_dict, target_dict) with the selector replaced by a
            RasterImage whose time dim has been sliced to a random prefix.
        """
        image = read_selector(input_dict, target_dict, self.selector)
        assert isinstance(image, RasterImage)

        t = image.image.shape[1]
        max_t = min(self.max_timesteps, t) if self.max_timesteps is not None else t
        min_t = min(self.min_timesteps, max_t)

        if min_t == max_t:
            k = max_t
        else:
            k = int(torch.randint(min_t, max_t + 1, (1,)).item())

        if k == t:
            return input_dict, target_dict

        new_timestamps = (
            image.timestamps[:k] if image.timestamps is not None else None
        )
        write_selector(
            input_dict,
            target_dict,
            self.selector,
            RasterImage(image.image[:, :k], timestamps=new_timestamps),
        )
        return input_dict, target_dict

