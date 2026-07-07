"""Tests for SmolVLAAptPolicy.prepare_images() — image preprocessing pipeline.

Covers: 4D/5D input, resize+pad, [0,1]→[-1,1] normalization, camera_order sorting,
missing-camera fill, padding_mask handling, error cases, and edge conditions.
"""

import pytest
import torch

from lerobot.configs.types import FeatureType, PolicyFeature
from lerobot.policies.smolvla_apt.configuration_smolvla_apt import SmolVLAAptConfig
from lerobot.policies.smolvla_apt.modeling_smolvla_apt import (
    SmolVLAAptPolicy,
    resize_with_pad,
)


# ═══════════════════════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════════════════════

def _make_cfg(image_keys, resize=(64, 64), camera_order=None):
    """Build a SmolVLAAptConfig with the given visual input_features.

    By using a tiny resize target (64×64) we keep tensor ops fast while still
    exercising the full resize+pad pipeline.
    """
    cfg = SmolVLAAptConfig()
    cfg.input_features = {
        key: PolicyFeature(type=FeatureType.VISUAL, shape=(3, 120, 160))
        for key in image_keys
    }
    cfg.input_features["observation.state"] = PolicyFeature(
        type=FeatureType.ENV, shape=(16,)
    )
    cfg.output_features = {
        "action": PolicyFeature(type=FeatureType.ACTION, shape=(16,))
    }
    cfg.resize_imgs_with_padding = resize
    if camera_order is not None:
        cfg.camera_order = camera_order
    cfg.chunk_size = 10
    cfg.max_state_dim = 16
    cfg.max_action_dim = 16
    return cfg


class _MockPolicy:
    """Lightweight stand-in so we can call SmolVLAAptPolicy.prepare_images()
    without triggering the full VLAFlowMatching / VLM initialisation."""

    def __init__(self, config):
        self.config = config


def _make_batch(bsize, device, **images):
    """Build a minimal batch dict.

    Parameters
    ----------
    bsize : int
    device : torch.device
    **images : Tensor
        Each kwarg becomes ``batch[key]``.  Shape should be ``(B, 3, H, W)`` or
        ``(B, T, 3, H, W)``.

    Returns
    -------
    dict  with ``observation.state`` plus every image kwarg.
    """
    batch = {
        "observation.state": torch.randn(bsize, 16, device=device),
    }
    batch.update(images)
    return batch


# ═══════════════════════════════════════════════════════════════════════════════
# Fixtures
# ═══════════════════════════════════════════════════════════════════════════════

@pytest.fixture(scope="module")
def device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


@pytest.fixture
def cfg_single(device):
    return _make_cfg(["observation.images.top"])


@pytest.fixture
def cfg_multi(device):
    return _make_cfg(
        ["observation.images.top", "observation.images.wrist", "observation.images.side"]
    )


@pytest.fixture
def cfg_no_resize(device):
    cfg = _make_cfg(["observation.images.top"])
    cfg.resize_imgs_with_padding = None
    return cfg


# ═══════════════════════════════════════════════════════════════════════════════
# resize_with_pad  (standalone helper exercised by prepare_images)
# ═══════════════════════════════════════════════════════════════════════════════

class TestResizeWithPad:
    def test_square_input(self, device):
        """Square image → uniform scaling, padding only on one axis if needed."""
        img = torch.rand(2, 3, 100, 100, device=device)
        out = resize_with_pad(img, 80, 60)
        assert out.shape == (2, 3, 60, 80)

    def test_landscape_input(self, device):
        """Wider-than-tall image → height-padded after resize."""
        img = torch.rand(1, 3, 60, 120, device=device)
        out = resize_with_pad(img, 80, 80, pad_value=0)
        assert out.shape == (1, 3, 80, 80)
        # Top rows should be padding (0), bottom rows should have content
        assert (out[0, :, 0, :] == 0).all(), "top rows expected to be padding"

    def test_portrait_input(self, device):
        """Taller-than-wide image → width-padded after resize."""
        img = torch.rand(1, 3, 120, 60, device=device)
        out = resize_with_pad(img, 80, 80, pad_value=0)
        assert out.shape == (1, 3, 80, 80)
        # Left columns should be padding
        assert (out[0, :, :, 0] == 0).all(), "left cols expected to be padding"

    def test_already_fits(self, device):
        """Image already at target size → no-op (no resize, no padding)."""
        img = torch.rand(2, 3, 64, 64, device=device)
        out = resize_with_pad(img, 64, 64)
        assert out.shape == (2, 3, 64, 64)
        assert torch.allclose(out, img)

    def test_custom_pad_value(self, device):
        img = torch.rand(1, 3, 20, 40, device=device)
        out = resize_with_pad(img, 64, 64, pad_value=0.5)
        assert out.shape == (1, 3, 64, 64)
        # Padding area should equal the custom value
        assert torch.allclose(out[0, :, 0, :], torch.tensor(0.5, device=device))

    def test_errors_on_3d_input(self, device):
        img = torch.rand(3, 64, 64, device=device)
        with pytest.raises(ValueError, match="b,c,h,w"):
            resize_with_pad(img, 32, 32)


# ═══════════════════════════════════════════════════════════════════════════════
# prepare_images — core behaviour
# ═══════════════════════════════════════════════════════════════════════════════

class TestPrepareImagesBasic:
    """Single-camera, everyday-path tests."""

    def test_single_camera_4d(self, cfg_single, device):
        """4D input (B,C,H,W) → one image, one mask."""
        policy = _MockPolicy(cfg_single)
        img = torch.rand(2, 3, 120, 160, device=device)
        batch = _make_batch(2, device, **{"observation.images.top": img})

        images, masks = SmolVLAAptPolicy.prepare_images(policy, batch)

        assert len(images) == 1
        assert len(masks) == 1
        assert images[0].shape == (2, 3, 64, 64)  # resized
        assert masks[0].shape == (2,)
        assert masks[0].dtype == torch.bool

    def test_single_camera_5d_takes_last_frame(self, cfg_single, device):
        """5D input (B,T,C,H,W) → only the last time-step is used."""
        policy = _MockPolicy(cfg_single)
        # T=3: first two frames are all zeros, last frame is all ones
        img = torch.zeros(2, 3, 3, 120, 160, device=device)
        img[:, :, -1] = 1.0  # last frame = ones
        batch = _make_batch(2, device, **{"observation.images.top": img})

        images, masks = SmolVLAAptPolicy.prepare_images(policy, batch)

        assert images[0].shape == (2, 3, 64, 64)
        # After resize+pad+normalize: last-frame (ones) → content=1.0, pad=-1.0.
        # If frame 0 (zeros) were used, output would be all -1.
        # Verify the result is NOT all -1 → proves the last frame was taken.
        assert not (images[0] == -1.0).all(), (
            "output should not be all -1; last frame (ones) was expected"
        )
        # Some region should contain the normalized ones (close to 1.0).
        assert (images[0].max() > 0.9).all(), "max value should reflect last frame"

    def test_pixel_range_normalization(self, cfg_single, device):
        """Verify [0,1] → [-1,1] mapping."""
        policy = _MockPolicy(cfg_single)
        # Pure white image
        img = torch.ones(2, 3, 64, 64, device=device)
        batch = _make_batch(2, device, **{"observation.images.top": img})

        images, _ = SmolVLAAptPolicy.prepare_images(policy, batch)

        # Already at target size (64,64), so resize_with_pad is a near-no-op.
        # White [0,1] → [1.0] after *2-1
        assert torch.allclose(images[0], torch.ones_like(images[0]))

        # Pure black image
        img_black = torch.zeros(2, 3, 64, 64, device=device)
        batch_black = _make_batch(2, device, **{"observation.images.top": img_black})
        images_black, _ = SmolVLAAptPolicy.prepare_images(policy, batch_black)
        assert torch.allclose(images_black[0], -torch.ones_like(images_black[0]))

    def test_resize_is_applied(self, cfg_single, device):
        """Image larger than target gets resized down."""
        policy = _MockPolicy(cfg_single)
        img = torch.rand(2, 3, 240, 320, device=device)  # bigger than 64×64
        batch = _make_batch(2, device, **{"observation.images.top": img})

        images, _ = SmolVLAAptPolicy.prepare_images(policy, batch)
        assert images[0].shape == (2, 3, 64, 64)

    def test_with_padding_mask_in_batch(self, cfg_single, device):
        """When batch provides ``{key}_padding_mask``, that mask is used."""
        policy = _MockPolicy(cfg_single)
        img = torch.rand(2, 3, 120, 160, device=device)
        padding_mask = torch.tensor([True, False], device=device)
        batch = _make_batch(2, device, **{"observation.images.top": img})
        batch["observation.images.top_padding_mask"] = padding_mask

        images, masks = SmolVLAAptPolicy.prepare_images(policy, batch)

        assert torch.equal(masks[0], padding_mask)

    def test_without_padding_mask_defaults_to_all_true(self, cfg_single, device):
        """No padding_mask → every sample is considered valid."""
        policy = _MockPolicy(cfg_single)
        img = torch.rand(2, 3, 120, 160, device=device)
        batch = _make_batch(2, device, **{"observation.images.top": img})

        images, masks = SmolVLAAptPolicy.prepare_images(policy, batch)

        assert masks[0].all()
        assert masks[0].shape == (2,)


# ═══════════════════════════════════════════════════════════════════════════════
# prepare_images — multiple cameras & camera_order
# ═══════════════════════════════════════════════════════════════════════════════

class TestPrepareImagesMultiCamera:
    def test_all_present_maintains_order(self, cfg_multi, device):
        """Three cameras all present → output length = 3, sorted by camera_order."""
        policy = _MockPolicy(cfg_multi)
        img = torch.rand(2, 3, 120, 160, device=device)
        batch = _make_batch(
            2, device,
            **{
                "observation.images.top": img,
                "observation.images.wrist": img,
                "observation.images.side": img,
            },
        )

        images, masks = SmolVLAAptPolicy.prepare_images(policy, batch)

        assert len(images) == 3
        assert len(masks) == 3
        for im, m in zip(images, masks):
            assert im.shape == (2, 3, 64, 64)
            assert m.shape == (2,)
            assert m.all()

    def test_missing_camera_black_fill(self, cfg_multi, device):
        """Only 2 of 3 cameras present → missing one is black (-1) + mask=0."""
        policy = _MockPolicy(cfg_multi)
        img = torch.rand(2, 3, 120, 160, device=device)
        batch = _make_batch(
            2, device,
            **{
                "observation.images.top": img,
                "observation.images.wrist": img,
                # "observation.images.side" intentionally missing
            },
        )

        images, masks = SmolVLAAptPolicy.prepare_images(policy, batch)

        assert len(images) == 3  # always = len(image_features)
        assert len(masks) == 3

        # First two: valid images, mask True
        assert masks[0].all()  # top
        assert masks[1].all()  # wrist
        # Third: filled black, mask False
        assert not masks[2].any()                          # side mask = all False
        assert (images[2] == -1.0).all()                   # side image = pure -1

    def test_only_missing_cameras_present(self, cfg_multi, device):
        """Some cameras present, rest missing — mix handled correctly."""
        policy = _MockPolicy(cfg_multi)
        img = torch.rand(2, 3, 120, 160, device=device)
        batch = _make_batch(
            2, device,
            **{"observation.images.side": img},  # only side present
        )

        images, masks = SmolVLAAptPolicy.prepare_images(policy, batch)

        assert len(images) == 3
        # side is present (third in default camera_order: wrist, top, side)
        assert masks[2].all()
        assert not (images[2] == -1.0).all()
        # top and wrist are missing → black
        assert not masks[0].any()
        assert (images[0] == -1.0).all()
        assert not masks[1].any()
        assert (images[1] == -1.0).all()

    def test_camera_order_sorting(self, device):
        """Default camera_order = [wrist, top, side].  Keys are re-ordered."""
        cfg = _make_cfg(
            ["observation.images.side", "observation.images.wrist", "observation.images.top"],
            camera_order=["observation.images.wrist", "observation.images.top", "observation.images.side"],
        )
        policy = _MockPolicy(cfg)
        img = torch.rand(1, 3, 120, 160, device=device)
        # Provide images with distinguishable values so we can check order
        batch = _make_batch(
            1, device,
            **{
                "observation.images.top": torch.full((1, 3, 64, 64), 0.1, device=device),
                "observation.images.wrist": torch.full((1, 3, 64, 64), 0.5, device=device),
                "observation.images.side": torch.full((1, 3, 64, 64), 0.9, device=device),
            },
        )

        images, _ = SmolVLAAptPolicy.prepare_images(policy, batch)

        # After camera_order sort: wrist(0.5), top(0.1), side(0.9)
        # After *2-1: wrist(0.0), top(-0.8), side(0.8)
        assert torch.allclose(images[0], torch.tensor(0.0, device=device), atol=0.05), "first = wrist"
        assert torch.allclose(images[1], torch.tensor(-0.8, device=device), atol=0.05), "second = top"
        assert torch.allclose(images[2], torch.tensor(0.8, device=device), atol=0.05), "third = side"

    def test_unknown_key_in_camera_order_appended_at_end(self, device):
        """Keys not listed in camera_order appear after the listed ones."""
        cfg = _make_cfg(
            ["observation.images.extra", "observation.images.top"],
            camera_order=["observation.images.top"],
        )
        policy = _MockPolicy(cfg)
        img = torch.rand(1, 3, 64, 64, device=device)
        batch = _make_batch(
            1, device,
            **{
                "observation.images.top": torch.full((1, 3, 64, 64), 0.2, device=device),
                "observation.images.extra": torch.full((1, 3, 64, 64), 0.8, device=device),
            },
        )

        images, _ = SmolVLAAptPolicy.prepare_images(policy, batch)

        # top (0.2) first, extra (0.8) second
        assert images[0].mean() < images[1].mean(), "top should come before extra"

    def test_no_camera_order_preserves_dict_order(self, device):
        """When camera_order is empty, the original insertion order is kept."""
        cfg = _make_cfg(
            ["observation.images.z", "observation.images.a", "observation.images.m"],
            camera_order=[],
        )
        policy = _MockPolicy(cfg)
        img = torch.rand(1, 3, 64, 64, device=device)
        batch = _make_batch(
            1, device,
            **{k: img for k in cfg.image_features},
        )

        images, _ = SmolVLAAptPolicy.prepare_images(policy, batch)
        assert len(images) == 3  # all present, in dict order


# ═══════════════════════════════════════════════════════════════════════════════
# prepare_images — error & edge cases
# ═══════════════════════════════════════════════════════════════════════════════

class TestPrepareImagesErrors:
    def test_all_images_missing_raises(self, cfg_single, device):
        """When none of the configured image keys appear in the batch → ValueError."""
        policy = _MockPolicy(cfg_single)
        batch = {
            "observation.state": torch.randn(2, 16, device=device),
            # "observation.images.top" intentionally absent
        }

        with pytest.raises(ValueError, match="No image features"):
            SmolVLAAptPolicy.prepare_images(policy, batch)

    def test_no_image_key_in_batch_raises(self, cfg_single, device):
        """Config expects cameras but batch has none → ValueError."""
        policy = _MockPolicy(cfg_single)
        batch = {"observation.state": torch.randn(2, 16, device=device)}

        with pytest.raises(ValueError, match="No image features"):
            SmolVLAAptPolicy.prepare_images(policy, batch)

    def test_missing_camera_with_no_present_raises_first(self, cfg_single, device):
        """Even if camera_order lists cameras, all-missing still raises ValueError
        before attempting to fill."""
        policy = _MockPolicy(cfg_single)
        batch = {"observation.state": torch.randn(2, 16, device=device)}

        with pytest.raises(ValueError):
            SmolVLAAptPolicy.prepare_images(policy, batch)


# ═══════════════════════════════════════════════════════════════════════════════
# prepare_images — edge conditions
# ═══════════════════════════════════════════════════════════════════════════════

class TestPrepareImagesEdgeCases:
    def test_batch_size_one(self, cfg_single, device):
        """Batch size = 1 still works."""
        policy = _MockPolicy(cfg_single)
        img = torch.rand(1, 3, 120, 160, device=device)
        batch = _make_batch(1, device, **{"observation.images.top": img})

        images, masks = SmolVLAAptPolicy.prepare_images(policy, batch)
        assert images[0].shape[0] == 1
        assert masks[0].shape[0] == 1

    def test_device_consistency(self, cfg_single, device):
        """All output tensors stay on the same device as the input."""
        policy = _MockPolicy(cfg_single)
        img = torch.rand(2, 3, 120, 160, device=device)
        batch = _make_batch(2, device, **{"observation.images.top": img})

        images, masks = SmolVLAAptPolicy.prepare_images(policy, batch)
        assert images[0].device.type == device.type
        assert masks[0].device.type == device.type

    def test_no_resize_keeps_original_size(self, cfg_no_resize, device):
        """When resize_imgs_with_padding is None, the image shape is untouched."""
        policy = _MockPolicy(cfg_no_resize)
        img = torch.rand(2, 3, 120, 160, device=device)
        batch = _make_batch(2, device, **{"observation.images.top": img})

        images, masks = SmolVLAAptPolicy.prepare_images(policy, batch)
        assert images[0].shape == (2, 3, 120, 160)

    def test_5d_single_frame(self, cfg_single, device):
        """5D input with T=1 → behaves identically to 4D."""
        policy = _MockPolicy(cfg_single)
        img_4d = torch.rand(2, 3, 64, 64, device=device)
        batch_4d = _make_batch(2, device, **{"observation.images.top": img_4d})
        img_5d = img_4d.unsqueeze(1)  # (2, 1, 3, 64, 64)
        batch_5d = _make_batch(2, device, **{"observation.images.top": img_5d})

        images_4d, masks_4d = SmolVLAAptPolicy.prepare_images(policy, batch_4d)
        images_5d, masks_5d = SmolVLAAptPolicy.prepare_images(policy, batch_5d)

        assert torch.allclose(images_4d[0], images_5d[0])
        assert torch.equal(masks_4d[0], masks_5d[0])

    def test_batch_size_consistent_across_present_and_missing(self, cfg_multi, device):
        """Missing-camera placeholders have the same B as present cameras."""
        policy = _MockPolicy(cfg_multi)
        img = torch.rand(2, 3, 120, 160, device=device)
        batch = _make_batch(
            2, device,
            **{"observation.images.top": img},  # only one present
        )

        images, masks = SmolVLAAptPolicy.prepare_images(policy, batch)
        for im in images:
            assert im.shape[0] == 2
        for m in masks:
            assert m.shape[0] == 2

    def test_camera_with_different_resolutions(self, cfg_multi, device):
        """Each camera can have different native resolution; resize normalises."""
        policy = _MockPolicy(cfg_multi)
        batch = _make_batch(
            2, device,
            **{
                "observation.images.top": torch.rand(2, 3, 240, 320, device=device),
                "observation.images.wrist": torch.rand(2, 3, 480, 640, device=device),
                "observation.images.side": torch.rand(2, 3, 100, 200, device=device),
            },
        )

        images, _ = SmolVLAAptPolicy.prepare_images(policy, batch)
        for im in images:
            assert im.shape == (2, 3, 64, 64)


# ═══════════════════════════════════════════════════════════════════════════════
# prepare_images — slow integration test with real SmolVLAAptPolicy
# ═══════════════════════════════════════════════════════════════════════════════

@pytest.mark.slow
class TestPrepareImagesIntegration:
    """End-to-end test using a real SmolVLAAptPolicy instance (loads VLM)."""

    @pytest.fixture(scope="class")
    @classmethod
    def policy_with_cams(cls):
        """Real policy with 3 cameras configured.  Class-scoped → loaded once."""
        cfg = _make_cfg(
            ["observation.images.top", "observation.images.wrist", "observation.images.side"],
            resize=(64, 64),
        )
        return SmolVLAAptPolicy(cfg)

    def test_real_policy_prepare_images(self, policy_with_cams, device):
        """Full integration: real policy + real batch → images/masks returned."""
        img = torch.rand(2, 3, 120, 160, device=device)
        batch = _make_batch(
            2, device,
            **{
                "observation.images.top": img,
                "observation.images.wrist": img,
                "observation.images.side": img,
            },
        )

        images, masks = policy_with_cams.prepare_images(batch)

        assert len(images) == 3
        assert len(masks) == 3
        for im in images:
            assert im.shape == (2, 3, 64, 64)
            assert im.device.type == device.type
        for m in masks:
            assert m.shape == (2,)
            assert m.dtype == torch.bool
            assert m.all()

    def test_real_policy_missing_camera(self, policy_with_cams, device):
        """Real policy handles missing cameras with black fill."""
        img = torch.rand(1, 3, 120, 160, device=device)
        batch = _make_batch(
            1, device,
            **{"observation.images.wrist": img},  # only wrist
        )

        images, masks = policy_with_cams.prepare_images(batch)

        assert len(images) == 3
        # wrist present (idx 0 in default camera_order)
        assert masks[0].all()
        # top (idx 1) and side (idx 2) missing → black
        assert not masks[1].any()
        assert (images[1] == -1.0).all()
        assert not masks[2].any()
        assert (images[2] == -1.0).all()
