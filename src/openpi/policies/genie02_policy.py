import dataclasses

import einops
import numpy as np

from openpi import transforms
from openpi.models import model as _model


def make_genie02_example() -> dict:
    """Creates a random input example for the Genie02 policy."""
    return {
        "cam_high": np.random.randint(256, size=(224, 224, 3), dtype=np.uint8),
        "cam_left_wrist": np.random.randint(256, size=(224, 224, 3), dtype=np.uint8),
        "cam_right_wrist": np.random.randint(256, size=(224, 224, 3), dtype=np.uint8),
        "state": np.random.rand(24),
        "prompt": "do something",
    }


def _parse_image(image) -> np.ndarray:
    image = np.asarray(image)
    if np.issubdtype(image.dtype, np.floating):
        image = (255 * image).astype(np.uint8)
    if image.shape[0] == 3:
        image = einops.rearrange(image, "c h w -> h w c")
    return image


@dataclasses.dataclass(frozen=True)
class Genie02Inputs(transforms.DataTransformFn):
    # Kept for parity with other policy inputs and potential future model-specific image handling.
    model_type: _model.ModelType

    def __call__(self, data: dict) -> dict:
        base_image = _parse_image(data["cam_high"])
        left_wrist = _parse_image(data["cam_left_wrist"])
        right_wrist = _parse_image(data["cam_right_wrist"])

        inputs = {
            "state": data["state"],
            "image": {
                "base_0_rgb": base_image,
                "left_wrist_0_rgb": left_wrist,
                "right_wrist_0_rgb": right_wrist,
            },
            "image_mask": {
                "base_0_rgb": np.True_,
                "left_wrist_0_rgb": np.True_,
                "right_wrist_0_rgb": np.True_,
            },
        }

        if "actions" in data:
            inputs["actions"] = data["actions"]

        if "prompt" in data:
            inputs["prompt"] = data["prompt"]

        return inputs


@dataclasses.dataclass(frozen=True)
class Genie02Outputs(transforms.DataTransformFn):
    action_dim: int = 24

    def __call__(self, data: dict) -> dict:
        return {"actions": np.asarray(data["actions"][:, : self.action_dim])}
