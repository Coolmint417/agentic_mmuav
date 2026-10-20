"""Metric camera-Z depth encoded as a compact, consistently scaled RGB image."""
import numpy as np
from PIL import Image

DEPTH_LIMIT_M = 10.0


def depth_image(depth_m):
    """White is near; black means >=10m or no return. No per-frame normalization."""
    values = np.asarray(depth_m)
    clipped = np.clip(
        np.nan_to_num(
            values, nan=DEPTH_LIMIT_M, posinf=DEPTH_LIMIT_M, neginf=DEPTH_LIMIT_M
        ),
        0,
        DEPTH_LIMIT_M,
    )
    intensity = np.rint(255 * (1 - clipped / DEPTH_LIMIT_M)).astype(np.uint8)
    return Image.fromarray(intensity).convert("RGB")


def depth_metadata(depth_m, model, camera_name):
    far = float(model.vis.map.zfar * model.stat.extent)
    values = np.asarray(depth_m)
    valid = np.isfinite(values) & (values > 0) & (values < far * 0.99)
    sectors = []
    for row in np.array_split(np.arange(values.shape[0]), 3):
        output = []
        for col in np.array_split(np.arange(values.shape[1]), 3):
            tile = values[np.ix_(row, col)]
            mask = valid[np.ix_(row, col)]
            output.append(round(float(tile[mask].min()), 3) if mask.any() else None)
        sectors.append(output)
    return {
        "camera": camera_name,
        "quantity": "camera_z_depth",
        "unit": "m",
        "display_range_m": [0, DEPTH_LIMIT_M],
        "intensity_to_depth": "depth_m = 10 * (1 - intensity / 255); white=near, black=>=10m or no return",
        "minimum_visible_m": round(float(values[valid].min()), 3)
        if valid.any()
        else None,
        "sector_min_m": sectors,
        "sector_order": "3x3 image grid: rows top-to-bottom, columns left-to-right; null=no return",
    }
