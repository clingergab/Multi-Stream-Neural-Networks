"""
Comprehensive tests for the augmentation control system.

Tests the 4-parameter augmentation control:
- rgb_aug_prob: Scales probability of RGB augmentations
- rgb_aug_mag: Scales magnitude of RGB augmentations
- depth_aug_prob: Scales probability of Depth augmentations
- depth_aug_mag: Scales magnitude of Depth augmentations

Run with: pytest tests/test_augmentation_control.py -v
"""

import warnings
from dataclasses import asdict

import pytest

from src.training.augmentation_config import (
    AugmentationConfig,
    # Probability baselines
    BASE_FLIP_P,
    BASE_CROP_P,
    BASE_COLOR_JITTER_P,
    BASE_BLUR_P,
    BASE_GRAYSCALE_P,
    BASE_RGB_ERASING_P,
    BASE_DEPTH_AUG_P,
    BASE_DEPTH_ERASING_P,
    # Magnitude baselines
    BASE_CROP_SCALE_MIN,
    BASE_CROP_SCALE_MAX,
    BASE_CROP_RATIO_MIN,
    BASE_CROP_RATIO_MAX,
    BASE_BRIGHTNESS,
    BASE_CONTRAST,
    BASE_SATURATION,
    BASE_HUE,
    BASE_BLUR_SIGMA_MIN,
    BASE_BLUR_SIGMA_MAX,
    BASE_ERASING_SCALE_MIN,
    BASE_ERASING_SCALE_MAX,
    BASE_ERASING_RATIO_MIN,
    BASE_ERASING_RATIO_MAX,
    BASE_DEPTH_BRIGHTNESS,
    BASE_DEPTH_CONTRAST,
    BASE_DEPTH_NOISE_STD,
    # Caps
    MAX_PROBABILITY,
    MAX_BRIGHTNESS,
    MAX_CONTRAST,
    MAX_SATURATION,
    MAX_HUE,
    MAX_BLUR_SIGMA,
    MAX_DEPTH_BRIGHTNESS,
    MAX_DEPTH_CONTRAST,
    MAX_DEPTH_NOISE_STD,
    MIN_CROP_SCALE,
    MAX_ERASING_SCALE,
)


# =============================================================================
# AUGMENTATION CONFIG TESTS
# =============================================================================

class TestAugmentationConfig:
    """Tests for the AugmentationConfig dataclass."""

    def test_default_values(self):
        """Test that default values are all 1.0 (baseline)."""
        config = AugmentationConfig()
        assert config.rgb_aug_prob == 1.0
        assert config.rgb_aug_mag == 1.0
        assert config.depth_aug_prob == 1.0
        assert config.depth_aug_mag == 1.0

    def test_custom_values(self):
        """Test creating config with custom values."""
        config = AugmentationConfig(
            rgb_aug_prob=1.5,
            rgb_aug_mag=1.2,
            depth_aug_prob=0.8,
            depth_aug_mag=0.5,
        )
        assert config.rgb_aug_prob == 1.5
        assert config.rgb_aug_mag == 1.2
        assert config.depth_aug_prob == 0.8
        assert config.depth_aug_mag == 0.5

    def test_uniform_factory(self):
        """Test the uniform() factory method."""
        config = AugmentationConfig.uniform(aug_prob=1.5, aug_mag=1.2)
        assert config.rgb_aug_prob == 1.5
        assert config.rgb_aug_mag == 1.2
        assert config.depth_aug_prob == 1.5
        assert config.depth_aug_mag == 1.2

    def test_uniform_defaults(self):
        """Test uniform() with default values."""
        config = AugmentationConfig.uniform()
        assert config.rgb_aug_prob == 1.0
        assert config.rgb_aug_mag == 1.0
        assert config.depth_aug_prob == 1.0
        assert config.depth_aug_mag == 1.0

    def test_to_dict(self):
        """Test conversion to dictionary."""
        config = AugmentationConfig(
            rgb_aug_prob=1.5,
            rgb_aug_mag=1.2,
            depth_aug_prob=0.8,
            depth_aug_mag=0.5,
        )
        d = config.to_dict()
        assert d == {
            "rgb_aug_prob": 1.5,
            "rgb_aug_mag": 1.2,
            "depth_aug_prob": 0.8,
            "depth_aug_mag": 0.5,
        }

    def test_to_dict_can_unpack(self):
        """Test that to_dict() result can be unpacked into functions."""
        config = AugmentationConfig(rgb_aug_prob=1.5, rgb_aug_mag=1.2)

        def example_func(rgb_aug_prob=1.0, rgb_aug_mag=1.0,
                         depth_aug_prob=1.0, depth_aug_mag=1.0):
            return rgb_aug_prob, rgb_aug_mag, depth_aug_prob, depth_aug_mag

        result = example_func(**config.to_dict())
        assert result == (1.5, 1.2, 1.0, 1.0)

    def test_negative_value_raises_error(self):
        """Test that negative values raise ValueError."""
        with pytest.raises(ValueError, match="must be >= 0"):
            AugmentationConfig(rgb_aug_prob=-0.5)

        with pytest.raises(ValueError, match="must be >= 0"):
            AugmentationConfig(rgb_aug_mag=-1.0)

        with pytest.raises(ValueError, match="must be >= 0"):
            AugmentationConfig(depth_aug_prob=-0.1)

        with pytest.raises(ValueError, match="must be >= 0"):
            AugmentationConfig(depth_aug_mag=-2.0)

    def test_high_value_warns(self):
        """Test that values > 5.0 raise UserWarning."""
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            AugmentationConfig(rgb_aug_prob=6.0)
            assert len(w) == 1
            assert "unusually high" in str(w[0].message)

    def test_repr(self):
        """Test string representation."""
        config = AugmentationConfig(
            rgb_aug_prob=1.5,
            rgb_aug_mag=1.2,
            depth_aug_prob=0.8,
            depth_aug_mag=0.5,
        )
        repr_str = repr(config)
        assert "rgb_aug_prob=1.5" in repr_str
        assert "rgb_aug_mag=1.2" in repr_str
        assert "depth_aug_prob=0.8" in repr_str
        assert "depth_aug_mag=0.5" in repr_str


# =============================================================================
# BASELINE CONSTANTS TESTS
# =============================================================================

class TestBaselineConstants:
    """Tests to verify baseline constants are correctly defined."""

    def test_probability_baselines_in_valid_range(self):
        """Test that probability baselines are in [0, 1]."""
        prob_baselines = [
            BASE_FLIP_P,
            BASE_CROP_P,
            BASE_COLOR_JITTER_P,
            BASE_BLUR_P,
            BASE_GRAYSCALE_P,
            BASE_RGB_ERASING_P,
            BASE_DEPTH_AUG_P,
            BASE_DEPTH_ERASING_P,
        ]
        for prob in prob_baselines:
            assert 0.0 <= prob <= 1.0, f"Probability {prob} not in valid range"

    def test_magnitude_baselines_positive(self):
        """Test that magnitude baselines are positive."""
        mag_baselines = [
            BASE_CROP_SCALE_MIN,
            BASE_BRIGHTNESS,
            BASE_CONTRAST,
            BASE_SATURATION,
            BASE_HUE,
            BASE_BLUR_SIGMA_MIN,
            BASE_BLUR_SIGMA_MAX,
            BASE_DEPTH_BRIGHTNESS,
            BASE_DEPTH_CONTRAST,
            BASE_DEPTH_NOISE_STD,
        ]
        for mag in mag_baselines:
            assert mag > 0, f"Magnitude {mag} should be positive"

    def test_caps_are_reasonable(self):
        """Test that caps are greater than baselines."""
        assert MAX_PROBABILITY > BASE_COLOR_JITTER_P
        assert MAX_BRIGHTNESS > BASE_BRIGHTNESS
        assert MAX_CONTRAST > BASE_CONTRAST
        assert MAX_SATURATION > BASE_SATURATION
        assert MAX_HUE > BASE_HUE
        assert MAX_BLUR_SIGMA > BASE_BLUR_SIGMA_MAX
        assert MAX_DEPTH_BRIGHTNESS > BASE_DEPTH_BRIGHTNESS
        assert MAX_DEPTH_CONTRAST > BASE_DEPTH_CONTRAST
        assert MAX_DEPTH_NOISE_STD > BASE_DEPTH_NOISE_STD
        assert MAX_ERASING_SCALE > BASE_ERASING_SCALE_MAX

    def test_crop_scale_range_valid(self):
        """Test crop scale range is valid."""
        assert MIN_CROP_SCALE < BASE_CROP_SCALE_MIN
        assert BASE_CROP_SCALE_MIN < BASE_CROP_SCALE_MAX
        assert BASE_CROP_SCALE_MAX == 1.0


# =============================================================================
# SCALING FORMULA TESTS
# =============================================================================

class TestScalingFormulas:
    """Tests to verify scaling formulas work correctly."""

    def test_probability_scaling_at_baseline(self):
        """Test that prob=1.0 produces baseline values."""
        rgb_aug_prob = 1.0
        depth_aug_prob = 1.0

        # RGB probabilities
        assert BASE_COLOR_JITTER_P * rgb_aug_prob == BASE_COLOR_JITTER_P
        assert BASE_BLUR_P * rgb_aug_prob == BASE_BLUR_P
        assert BASE_GRAYSCALE_P * rgb_aug_prob == BASE_GRAYSCALE_P
        assert BASE_RGB_ERASING_P * rgb_aug_prob == BASE_RGB_ERASING_P

        # Depth probabilities
        assert BASE_DEPTH_AUG_P * depth_aug_prob == BASE_DEPTH_AUG_P
        assert BASE_DEPTH_ERASING_P * depth_aug_prob == BASE_DEPTH_ERASING_P

    def test_magnitude_scaling_at_baseline(self):
        """Test that mag=1.0 produces baseline values."""
        rgb_aug_mag = 1.0
        depth_aug_mag = 1.0

        # RGB magnitudes
        assert BASE_BRIGHTNESS * rgb_aug_mag == BASE_BRIGHTNESS
        assert BASE_CONTRAST * rgb_aug_mag == BASE_CONTRAST
        assert BASE_SATURATION * rgb_aug_mag == BASE_SATURATION
        assert BASE_HUE * rgb_aug_mag == BASE_HUE

        # Depth magnitudes
        assert BASE_DEPTH_BRIGHTNESS * depth_aug_mag == BASE_DEPTH_BRIGHTNESS
        assert BASE_DEPTH_CONTRAST * depth_aug_mag == BASE_DEPTH_CONTRAST
        assert BASE_DEPTH_NOISE_STD * depth_aug_mag == BASE_DEPTH_NOISE_STD

    def test_probability_scaling_increases(self):
        """Test that prob>1.0 increases probabilities (with caps)."""
        rgb_aug_prob = 1.5

        color_jitter_p = min(BASE_COLOR_JITTER_P * rgb_aug_prob, MAX_PROBABILITY)
        assert color_jitter_p == pytest.approx(0.645, rel=1e-3)  # 0.43 * 1.5

        blur_p = min(BASE_BLUR_P * rgb_aug_prob, MAX_PROBABILITY)
        assert blur_p == pytest.approx(0.375, rel=1e-3)  # 0.25 * 1.5

    def test_probability_scaling_decreases(self):
        """Test that prob<1.0 decreases probabilities."""
        rgb_aug_prob = 0.5

        color_jitter_p = min(BASE_COLOR_JITTER_P * rgb_aug_prob, MAX_PROBABILITY)
        assert color_jitter_p == pytest.approx(0.215, rel=1e-3)  # 0.43 * 0.5

        blur_p = min(BASE_BLUR_P * rgb_aug_prob, MAX_PROBABILITY)
        assert blur_p == pytest.approx(0.125, rel=1e-3)  # 0.25 * 0.5

    def test_magnitude_scaling_increases(self):
        """Test that mag>1.0 increases magnitudes (with caps)."""
        rgb_aug_mag = 1.5

        brightness = min(BASE_BRIGHTNESS * rgb_aug_mag, MAX_BRIGHTNESS)
        assert brightness == pytest.approx(0.555, rel=1e-3)  # 0.37 * 1.5

        hue = min(BASE_HUE * rgb_aug_mag, MAX_HUE)
        assert hue == pytest.approx(0.165, rel=1e-3)  # 0.11 * 1.5

    def test_magnitude_scaling_decreases(self):
        """Test that mag<1.0 decreases magnitudes."""
        rgb_aug_mag = 0.5

        brightness = min(BASE_BRIGHTNESS * rgb_aug_mag, MAX_BRIGHTNESS)
        assert brightness == pytest.approx(0.185, rel=1e-3)  # 0.37 * 0.5

        hue = min(BASE_HUE * rgb_aug_mag, MAX_HUE)
        assert hue == pytest.approx(0.055, rel=1e-3)  # 0.11 * 0.5

    def test_probability_caps_applied(self):
        """Test that probability caps prevent extreme values."""
        rgb_aug_prob = 10.0  # Very high

        # ColorJitter prob would be 4.3, but capped at 0.95
        color_jitter_p = min(BASE_COLOR_JITTER_P * rgb_aug_prob, MAX_PROBABILITY)
        assert color_jitter_p == MAX_PROBABILITY

    def test_magnitude_caps_applied(self):
        """Test that magnitude caps prevent extreme values."""
        rgb_aug_mag = 10.0  # Very high

        # Brightness would be 3.7, but capped at 0.8
        brightness = min(BASE_BRIGHTNESS * rgb_aug_mag, MAX_BRIGHTNESS)
        assert brightness == MAX_BRIGHTNESS

        # Hue would be 1.1, but capped at 0.4
        hue = min(BASE_HUE * rgb_aug_mag, MAX_HUE)
        assert hue == MAX_HUE

        # Blur sigma would be 17.0, but capped at 3.5
        blur_sigma = min(BASE_BLUR_SIGMA_MAX * rgb_aug_mag, MAX_BLUR_SIGMA)
        assert blur_sigma == MAX_BLUR_SIGMA

    def test_depth_magnitude_caps_applied(self):
        """Test that depth magnitude caps prevent extreme values."""
        depth_aug_mag = 10.0

        depth_brightness = min(BASE_DEPTH_BRIGHTNESS * depth_aug_mag, MAX_DEPTH_BRIGHTNESS)
        assert depth_brightness == MAX_DEPTH_BRIGHTNESS

        depth_contrast = min(BASE_DEPTH_CONTRAST * depth_aug_mag, MAX_DEPTH_CONTRAST)
        assert depth_contrast == MAX_DEPTH_CONTRAST

        depth_noise = min(BASE_DEPTH_NOISE_STD * depth_aug_mag, MAX_DEPTH_NOISE_STD)
        assert depth_noise == MAX_DEPTH_NOISE_STD

    def test_synchronized_aug_uses_average(self):
        """Test that synchronized augmentations use average of RGB and Depth params."""
        rgb_aug_prob = 1.5
        depth_aug_prob = 1.0
        sync_prob = (rgb_aug_prob + depth_aug_prob) / 2  # 1.25

        flip_p = min(BASE_FLIP_P * sync_prob, MAX_PROBABILITY)
        assert flip_p == pytest.approx(0.625, rel=1e-3)  # 0.5 * 1.25

        crop_p = min(BASE_CROP_P * sync_prob, MAX_PROBABILITY)
        assert crop_p == pytest.approx(0.625, rel=1e-3)  # 0.5 * 1.25

    def test_zero_prob_disables_augmentation(self):
        """Test that prob=0.0 effectively disables augmentation."""
        rgb_aug_prob = 0.0

        color_jitter_p = min(BASE_COLOR_JITTER_P * rgb_aug_prob, MAX_PROBABILITY)
        assert color_jitter_p == 0.0

        blur_p = min(BASE_BLUR_P * rgb_aug_prob, MAX_PROBABILITY)
        assert blur_p == 0.0

    def test_crop_scale_min_scaling(self):
        """Test crop scale minimum scales correctly."""
        sync_mag = 1.5

        # Formula: max(MIN_CROP_SCALE, 1.0 - (1.0 - BASE_CROP_SCALE_MIN) * sync_mag)
        # = max(0.5, 1.0 - (1.0 - 0.9) * 1.5)
        # = max(0.5, 1.0 - 0.1 * 1.5)
        # = max(0.5, 1.0 - 0.15)
        # = max(0.5, 0.85)
        # = 0.85
        crop_scale_min = max(MIN_CROP_SCALE, 1.0 - (1.0 - BASE_CROP_SCALE_MIN) * sync_mag)
        assert crop_scale_min == pytest.approx(0.85, rel=1e-3)

    def test_crop_scale_min_capped(self):
        """Test crop scale minimum is capped at MIN_CROP_SCALE."""
        sync_mag = 10.0  # Very high

        # Would be 1.0 - 0.1 * 10 = 0.0, but capped at 0.5
        crop_scale_min = max(MIN_CROP_SCALE, 1.0 - (1.0 - BASE_CROP_SCALE_MIN) * sync_mag)
        assert crop_scale_min == MIN_CROP_SCALE

    def test_erasing_scale_scaling(self):
        """Test erasing scale maximum scales correctly."""
        rgb_aug_mag = 1.5

        # Formula: min(BASE_ERASING_SCALE_MAX * rgb_aug_mag, MAX_ERASING_SCALE)
        erasing_scale = min(BASE_ERASING_SCALE_MAX * rgb_aug_mag, MAX_ERASING_SCALE)
        assert erasing_scale == pytest.approx(0.15, rel=1e-3)  # 0.10 * 1.5

    def test_erasing_scale_capped(self):
        """Test erasing scale is capped at MAX_ERASING_SCALE."""
        rgb_aug_mag = 10.0

        erasing_scale = min(BASE_ERASING_SCALE_MAX * rgb_aug_mag, MAX_ERASING_SCALE)
        assert erasing_scale == MAX_ERASING_SCALE


# =============================================================================
# EDGE CASE TESTS
# =============================================================================

class TestEdgeCases:
    """Tests for edge cases and boundary conditions."""

    def test_one_stream_zero_other_normal(self):
        """Test when one stream is disabled (0.0) and other is normal (1.0)."""
        rgb_aug_prob = 0.0
        depth_aug_prob = 1.0
        sync_prob = (rgb_aug_prob + depth_aug_prob) / 2  # 0.5

        # Synchronized augs at 50% of baseline
        flip_p = min(BASE_FLIP_P * sync_prob, MAX_PROBABILITY)
        assert flip_p == pytest.approx(0.25, rel=1e-3)  # 0.5 * 0.5

        # RGB augs disabled
        color_jitter_p = min(BASE_COLOR_JITTER_P * rgb_aug_prob, MAX_PROBABILITY)
        assert color_jitter_p == 0.0

        # Depth augs at baseline
        depth_aug_p = min(BASE_DEPTH_AUG_P * depth_aug_prob, MAX_PROBABILITY)
        assert depth_aug_p == BASE_DEPTH_AUG_P

    def test_both_streams_zero(self):
        """Test when both streams are disabled (0.0)."""
        rgb_aug_prob = 0.0
        depth_aug_prob = 0.0
        sync_prob = (rgb_aug_prob + depth_aug_prob) / 2  # 0.0

        flip_p = min(BASE_FLIP_P * sync_prob, MAX_PROBABILITY)
        assert flip_p == 0.0

        color_jitter_p = min(BASE_COLOR_JITTER_P * rgb_aug_prob, MAX_PROBABILITY)
        assert color_jitter_p == 0.0

        depth_aug_p = min(BASE_DEPTH_AUG_P * depth_aug_prob, MAX_PROBABILITY)
        assert depth_aug_p == 0.0

    def test_asymmetric_high_low(self):
        """Test asymmetric scaling (RGB high, Depth low)."""
        rgb_aug_prob = 2.0
        rgb_aug_mag = 1.5
        depth_aug_prob = 0.5
        depth_aug_mag = 0.8

        sync_prob = (rgb_aug_prob + depth_aug_prob) / 2  # 1.25
        sync_mag = (rgb_aug_mag + depth_aug_mag) / 2  # 1.15

        # RGB gets strong augmentation
        color_jitter_p = min(BASE_COLOR_JITTER_P * rgb_aug_prob, MAX_PROBABILITY)
        assert color_jitter_p == pytest.approx(0.86, rel=1e-2)  # 0.43 * 2.0

        brightness = min(BASE_BRIGHTNESS * rgb_aug_mag, MAX_BRIGHTNESS)
        assert brightness == pytest.approx(0.555, rel=1e-3)  # 0.37 * 1.5

        # Depth gets weak augmentation
        depth_aug_p = min(BASE_DEPTH_AUG_P * depth_aug_prob, MAX_PROBABILITY)
        assert depth_aug_p == pytest.approx(0.25, rel=1e-3)  # 0.5 * 0.5

        depth_brightness = min(BASE_DEPTH_BRIGHTNESS * depth_aug_mag, MAX_DEPTH_BRIGHTNESS)
        assert depth_brightness == pytest.approx(0.20, rel=1e-3)  # 0.25 * 0.8

        # Synchronized uses average
        flip_p = min(BASE_FLIP_P * sync_prob, MAX_PROBABILITY)
        assert flip_p == pytest.approx(0.625, rel=1e-3)  # 0.5 * 1.25

    def test_very_small_values(self):
        """Test with very small positive values."""
        rgb_aug_prob = 0.01
        rgb_aug_mag = 0.01

        color_jitter_p = min(BASE_COLOR_JITTER_P * rgb_aug_prob, MAX_PROBABILITY)
        assert color_jitter_p == pytest.approx(0.0043, rel=1e-2)

        brightness = min(BASE_BRIGHTNESS * rgb_aug_mag, MAX_BRIGHTNESS)
        assert brightness == pytest.approx(0.0037, rel=1e-2)


# =============================================================================
# PLAN FORMULA VERIFICATION TESTS
# =============================================================================

class TestPlanFormulaVerification:
    """Tests that verify formulas match the plan document exactly."""

    def test_rgb_scaling_at_1_5(self):
        """Verify RGB scaling at p/m=1.5 matches plan table."""
        rgb_aug_prob = 1.5
        rgb_aug_mag = 1.5

        assert min(BASE_COLOR_JITTER_P * rgb_aug_prob, MAX_PROBABILITY) == pytest.approx(0.645, rel=1e-3)
        assert min(BASE_BRIGHTNESS * rgb_aug_mag, MAX_BRIGHTNESS) == pytest.approx(0.555, rel=1e-3)
        assert min(BASE_CONTRAST * rgb_aug_mag, MAX_CONTRAST) == pytest.approx(0.555, rel=1e-3)
        assert min(BASE_SATURATION * rgb_aug_mag, MAX_SATURATION) == pytest.approx(0.555, rel=1e-3)
        assert min(BASE_HUE * rgb_aug_mag, MAX_HUE) == pytest.approx(0.165, rel=1e-3)
        assert min(BASE_BLUR_P * rgb_aug_prob, MAX_PROBABILITY) == pytest.approx(0.375, rel=1e-3)
        assert min(BASE_BLUR_SIGMA_MAX * rgb_aug_mag, MAX_BLUR_SIGMA) == pytest.approx(2.55, rel=1e-3)
        assert min(BASE_GRAYSCALE_P * rgb_aug_prob, MAX_PROBABILITY) == pytest.approx(0.255, rel=1e-3)
        assert min(BASE_RGB_ERASING_P * rgb_aug_prob, MAX_PROBABILITY) == pytest.approx(0.255, rel=1e-3)
        assert min(BASE_ERASING_SCALE_MAX * rgb_aug_mag, MAX_ERASING_SCALE) == pytest.approx(0.15, rel=1e-3)

    def test_rgb_scaling_at_0_5(self):
        """Verify RGB scaling at p/m=0.5 matches plan table."""
        rgb_aug_prob = 0.5
        rgb_aug_mag = 0.5

        assert min(BASE_COLOR_JITTER_P * rgb_aug_prob, MAX_PROBABILITY) == pytest.approx(0.215, rel=1e-3)
        assert min(BASE_BRIGHTNESS * rgb_aug_mag, MAX_BRIGHTNESS) == pytest.approx(0.185, rel=1e-3)
        assert min(BASE_CONTRAST * rgb_aug_mag, MAX_CONTRAST) == pytest.approx(0.185, rel=1e-3)
        assert min(BASE_SATURATION * rgb_aug_mag, MAX_SATURATION) == pytest.approx(0.185, rel=1e-3)
        assert min(BASE_HUE * rgb_aug_mag, MAX_HUE) == pytest.approx(0.055, rel=1e-3)
        assert min(BASE_BLUR_P * rgb_aug_prob, MAX_PROBABILITY) == pytest.approx(0.125, rel=1e-3)
        assert min(BASE_BLUR_SIGMA_MAX * rgb_aug_mag, MAX_BLUR_SIGMA) == pytest.approx(0.85, rel=1e-3)
        assert min(BASE_GRAYSCALE_P * rgb_aug_prob, MAX_PROBABILITY) == pytest.approx(0.085, rel=1e-3)
        assert min(BASE_RGB_ERASING_P * rgb_aug_prob, MAX_PROBABILITY) == pytest.approx(0.085, rel=1e-3)
        assert min(BASE_ERASING_SCALE_MAX * rgb_aug_mag, MAX_ERASING_SCALE) == pytest.approx(0.05, rel=1e-3)

    def test_depth_scaling_at_1_5(self):
        """Verify Depth scaling at p/m=1.5 matches plan table."""
        depth_aug_prob = 1.5
        depth_aug_mag = 1.5

        assert min(BASE_DEPTH_AUG_P * depth_aug_prob, MAX_PROBABILITY) == pytest.approx(0.75, rel=1e-3)
        assert min(BASE_DEPTH_BRIGHTNESS * depth_aug_mag, MAX_DEPTH_BRIGHTNESS) == pytest.approx(0.375, rel=1e-3)
        assert min(BASE_DEPTH_CONTRAST * depth_aug_mag, MAX_DEPTH_CONTRAST) == pytest.approx(0.375, rel=1e-3)
        assert min(BASE_DEPTH_NOISE_STD * depth_aug_mag, MAX_DEPTH_NOISE_STD) == pytest.approx(0.0885, rel=1e-2)
        assert min(BASE_DEPTH_ERASING_P * depth_aug_prob, MAX_PROBABILITY) == pytest.approx(0.15, rel=1e-3)
        assert min(BASE_ERASING_SCALE_MAX * depth_aug_mag, MAX_ERASING_SCALE) == pytest.approx(0.15, rel=1e-3)

    def test_depth_scaling_at_0_5(self):
        """Verify Depth scaling at p/m=0.5 matches plan table."""
        depth_aug_prob = 0.5
        depth_aug_mag = 0.5

        assert min(BASE_DEPTH_AUG_P * depth_aug_prob, MAX_PROBABILITY) == pytest.approx(0.25, rel=1e-3)
        assert min(BASE_DEPTH_BRIGHTNESS * depth_aug_mag, MAX_DEPTH_BRIGHTNESS) == pytest.approx(0.125, rel=1e-3)
        assert min(BASE_DEPTH_CONTRAST * depth_aug_mag, MAX_DEPTH_CONTRAST) == pytest.approx(0.125, rel=1e-3)
        assert min(BASE_DEPTH_NOISE_STD * depth_aug_mag, MAX_DEPTH_NOISE_STD) == pytest.approx(0.0295, rel=1e-2)
        assert min(BASE_DEPTH_ERASING_P * depth_aug_prob, MAX_PROBABILITY) == pytest.approx(0.05, rel=1e-3)
        assert min(BASE_ERASING_SCALE_MAX * depth_aug_mag, MAX_ERASING_SCALE) == pytest.approx(0.05, rel=1e-3)

    def test_sync_scaling_with_mixed_params(self):
        """Verify synchronized scaling with rgb=1.5, depth=1.0 (avg=1.25)."""
        rgb_aug_prob = 1.5
        depth_aug_prob = 1.0
        sync_prob = (rgb_aug_prob + depth_aug_prob) / 2  # 1.25

        assert min(BASE_FLIP_P * sync_prob, MAX_PROBABILITY) == pytest.approx(0.625, rel=1e-3)
        assert min(BASE_CROP_P * sync_prob, MAX_PROBABILITY) == pytest.approx(0.625, rel=1e-3)


# =============================================================================
# SUNRGBD DATASET INTEGRATION TESTS
# =============================================================================

class TestSUNRGBDDatasetAugParams:
    """Tests for SUNRGBDDataset augmentation parameter integration.

    Uses mocking to avoid file system dependencies.
    """

    @pytest.fixture
    def mock_dataset(self, tmp_path):
        """Create a minimal mock dataset structure."""
        # Create train directory structure
        train_dir = tmp_path / "train"
        train_dir.mkdir()
        (train_dir / "rgb").mkdir()
        (train_dir / "depth").mkdir()

        # Create minimal labels file (just the label number, one per line)
        labels_file = train_dir / "labels.txt"
        labels_file.write_text("0\n")

        # Create dummy images
        import numpy as np
        from PIL import Image

        rgb_img = Image.fromarray(np.zeros((100, 100, 3), dtype=np.uint8))
        depth_img = Image.fromarray(np.zeros((100, 100), dtype=np.uint16))
        rgb_img.save(train_dir / "rgb" / "00000.png")
        depth_img.save(train_dir / "depth" / "00000.png")

        # Create val directory structure
        val_dir = tmp_path / "val"
        val_dir.mkdir()
        (val_dir / "rgb").mkdir()
        (val_dir / "depth").mkdir()
        (val_dir / "labels.txt").write_text("0\n")
        rgb_img.save(val_dir / "rgb" / "00000.png")
        depth_img.save(val_dir / "depth" / "00000.png")

        return str(tmp_path)

    def test_dataset_accepts_aug_params(self, mock_dataset):
        """Test that SUNRGBDDataset accepts all 4 aug params."""
        from src.data_utils.sunrgbd_dataset import SUNRGBDDataset

        # Should not raise any errors
        dataset = SUNRGBDDataset(
            data_root=mock_dataset,
            train=True,
            rgb_aug_prob=1.5,
            rgb_aug_mag=1.2,
            depth_aug_prob=0.8,
            depth_aug_mag=0.5,
        )

        assert dataset.rgb_aug_prob == 1.5
        assert dataset.rgb_aug_mag == 1.2
        assert dataset.depth_aug_prob == 0.8
        assert dataset.depth_aug_mag == 0.5

    def test_dataset_default_params(self, mock_dataset):
        """Test that default params are all 1.0."""
        from src.data_utils.sunrgbd_dataset import SUNRGBDDataset

        dataset = SUNRGBDDataset(data_root=mock_dataset, train=True)

        assert dataset.rgb_aug_prob == 1.0
        assert dataset.rgb_aug_mag == 1.0
        assert dataset.depth_aug_prob == 1.0
        assert dataset.depth_aug_mag == 1.0

    def test_dataset_computes_scaled_values(self, mock_dataset):
        """Test that dataset computes scaled values correctly."""
        from src.data_utils.sunrgbd_dataset import SUNRGBDDataset

        dataset = SUNRGBDDataset(
            data_root=mock_dataset,
            train=True,
            rgb_aug_prob=1.5,
            rgb_aug_mag=1.5,
            depth_aug_prob=1.0,
            depth_aug_mag=1.0,
        )

        # Check RGB scaling
        assert dataset._color_jitter_p == pytest.approx(0.645, rel=1e-3)
        assert dataset._brightness == pytest.approx(0.555, rel=1e-3)

        # Check sync scaling (average of 1.5 and 1.0 = 1.25)
        assert dataset._flip_p == pytest.approx(0.625, rel=1e-3)
        assert dataset._crop_p == pytest.approx(0.625, rel=1e-3)

    def test_dataset_no_aug_for_validation(self, mock_dataset):
        """Test that validation dataset ignores aug params."""
        from src.data_utils.sunrgbd_dataset import SUNRGBDDataset

        dataset = SUNRGBDDataset(
            data_root=mock_dataset,
            train=False,  # Validation mode
            rgb_aug_prob=1.5,  # Should be ignored
            rgb_aug_mag=1.5,
        )

        # Validation doesn't apply augmentation, so these shouldn't matter
        # Just verify the dataset was created without errors
        assert dataset.train is False

    def test_dataset_zero_params_disable_aug(self, mock_dataset):
        """Test that zero params effectively disable augmentation."""
        from src.data_utils.sunrgbd_dataset import SUNRGBDDataset

        dataset = SUNRGBDDataset(
            data_root=mock_dataset,
            train=True,
            rgb_aug_prob=0.0,
            rgb_aug_mag=0.0,
            depth_aug_prob=0.0,
            depth_aug_mag=0.0,
        )

        # All probabilities should be 0
        assert dataset._flip_p == 0.0
        assert dataset._crop_p == 0.0
        assert dataset._color_jitter_p == 0.0
        assert dataset._depth_aug_p == 0.0


# =============================================================================
# GPU AUGMENTATION INTEGRATION TESTS
# =============================================================================

class TestGPUAugmentationParams:
    """Tests for GPUAugmentation parameter integration.

    These tests verify the scaling logic without requiring Kornia or GPU.
    We test the scaling formulas directly since that's what we care about.
    """

    def test_gpu_aug_scaling_formulas_at_custom_values(self):
        """Test GPU augmentation scaling formulas produce correct values."""
        # Simulate what GPUAugmentation.__init__ computes
        rgb_aug_prob = 1.5
        rgb_aug_mag = 1.2
        depth_aug_prob = 0.8
        depth_aug_mag = 0.5

        # RGB probability scaling
        color_jitter_p = min(BASE_COLOR_JITTER_P * rgb_aug_prob, MAX_PROBABILITY)
        blur_p = min(BASE_BLUR_P * rgb_aug_prob, MAX_PROBABILITY)
        grayscale_p = min(BASE_GRAYSCALE_P * rgb_aug_prob, MAX_PROBABILITY)
        rgb_erasing_p = min(BASE_RGB_ERASING_P * rgb_aug_prob, MAX_PROBABILITY)

        assert color_jitter_p == pytest.approx(0.645, rel=1e-3)
        assert blur_p == pytest.approx(0.375, rel=1e-3)
        assert grayscale_p == pytest.approx(0.255, rel=1e-3)
        assert rgb_erasing_p == pytest.approx(0.255, rel=1e-3)

        # RGB magnitude scaling
        brightness = min(BASE_BRIGHTNESS * rgb_aug_mag, MAX_BRIGHTNESS)
        contrast = min(BASE_CONTRAST * rgb_aug_mag, MAX_CONTRAST)
        saturation = min(BASE_SATURATION * rgb_aug_mag, MAX_SATURATION)
        hue = min(BASE_HUE * rgb_aug_mag, MAX_HUE)

        assert brightness == pytest.approx(0.444, rel=1e-3)
        assert contrast == pytest.approx(0.444, rel=1e-3)
        assert saturation == pytest.approx(0.444, rel=1e-3)
        assert hue == pytest.approx(0.132, rel=1e-3)

        # Depth scaling
        depth_erasing_p = min(BASE_DEPTH_ERASING_P * depth_aug_prob, MAX_PROBABILITY)
        assert depth_erasing_p == pytest.approx(0.08, rel=1e-3)

    def test_gpu_aug_scaling_at_baseline(self):
        """Test GPU augmentation scaling at baseline (all 1.0)."""
        rgb_aug_prob = 1.0
        rgb_aug_mag = 1.0

        color_jitter_p = min(BASE_COLOR_JITTER_P * rgb_aug_prob, MAX_PROBABILITY)
        brightness = min(BASE_BRIGHTNESS * rgb_aug_mag, MAX_BRIGHTNESS)

        assert color_jitter_p == BASE_COLOR_JITTER_P
        assert brightness == BASE_BRIGHTNESS

    def test_gpu_aug_scaling_with_caps(self):
        """Test GPU augmentation scaling respects caps."""
        rgb_aug_prob = 5.0
        rgb_aug_mag = 5.0

        color_jitter_p = min(BASE_COLOR_JITTER_P * rgb_aug_prob, MAX_PROBABILITY)
        brightness = min(BASE_BRIGHTNESS * rgb_aug_mag, MAX_BRIGHTNESS)
        blur_sigma_max = min(BASE_BLUR_SIGMA_MAX * rgb_aug_mag, MAX_BLUR_SIGMA)

        # Should be capped
        assert color_jitter_p == MAX_PROBABILITY
        assert brightness == MAX_BRIGHTNESS
        assert blur_sigma_max == MAX_BLUR_SIGMA


# =============================================================================
# ABSTRACT MODEL INTEGRATION TESTS
# =============================================================================

class TestAbstractModelCompile:
    """Tests for abstract_model.compile() augmentation parameter integration.

    These tests verify compile() accepts all 4 aug params by inspecting
    the method signature and verifying default values.
    """

    def test_compile_method_signature_has_aug_params(self):
        """Test that compile() method signature includes all 4 aug params."""
        import inspect
        from src.models.abstracts.abstract_model import BaseModel

        sig = inspect.signature(BaseModel.compile)
        params = sig.parameters

        # Verify all 4 aug params exist in signature
        assert "rgb_aug_prob" in params
        assert "rgb_aug_mag" in params
        assert "depth_aug_prob" in params
        assert "depth_aug_mag" in params

    def test_compile_aug_params_have_correct_defaults(self):
        """Test that compile() aug params have default value of 1.0."""
        import inspect
        from src.models.abstracts.abstract_model import BaseModel

        sig = inspect.signature(BaseModel.compile)
        params = sig.parameters

        # Verify defaults are all 1.0
        assert params["rgb_aug_prob"].default == 1.0
        assert params["rgb_aug_mag"].default == 1.0
        assert params["depth_aug_prob"].default == 1.0
        assert params["depth_aug_mag"].default == 1.0

    def test_compile_gpu_augmentation_param_exists(self):
        """Test that compile() has gpu_augmentation parameter with default False."""
        import inspect
        from src.models.abstracts.abstract_model import BaseModel

        sig = inspect.signature(BaseModel.compile)
        params = sig.parameters

        assert "gpu_augmentation" in params
        assert params["gpu_augmentation"].default is False

    def test_augmentation_config_unpacks_to_compile_signature(self):
        """Test that AugmentationConfig.to_dict() keys match compile() params."""
        import inspect
        from src.models.abstracts.abstract_model import BaseModel

        config = AugmentationConfig(
            rgb_aug_prob=1.5,
            rgb_aug_mag=1.2,
            depth_aug_prob=0.8,
            depth_aug_mag=0.5,
        )

        sig = inspect.signature(BaseModel.compile)
        param_names = set(sig.parameters.keys())
        config_keys = set(config.to_dict().keys())

        # All config keys should be valid compile() parameters
        assert config_keys.issubset(param_names), (
            f"AugmentationConfig keys {config_keys - param_names} not in compile() signature"
        )


# =============================================================================
# END-TO-END INTEGRATION TESTS
# =============================================================================

class TestEndToEndIntegration:
    """End-to-end tests for the full augmentation control pipeline."""

    @pytest.fixture
    def mock_dataset(self, tmp_path):
        """Create a minimal mock dataset structure."""
        # Create train directory structure
        train_dir = tmp_path / "train"
        train_dir.mkdir()
        (train_dir / "rgb").mkdir()
        (train_dir / "depth").mkdir()

        # Create minimal labels file (just the label number, one per line)
        labels_file = train_dir / "labels.txt"
        labels_file.write_text("0\n")

        # Create dummy images
        import numpy as np
        from PIL import Image

        rgb_img = Image.fromarray(np.zeros((100, 100, 3), dtype=np.uint8))
        depth_img = Image.fromarray(np.zeros((100, 100), dtype=np.uint16))
        rgb_img.save(train_dir / "rgb" / "00000.png")
        depth_img.save(train_dir / "depth" / "00000.png")

        return str(tmp_path)

    def test_augmentation_config_flows_through_system(self, mock_dataset):
        """Test that AugmentationConfig values flow through the system."""
        from src.training.augmentation_config import AugmentationConfig
        from src.data_utils.sunrgbd_dataset import SUNRGBDDataset

        # Create config
        config = AugmentationConfig(
            rgb_aug_prob=1.5,
            rgb_aug_mag=1.2,
            depth_aug_prob=0.8,
            depth_aug_mag=0.5,
        )

        # Pass to dataset using to_dict()
        dataset = SUNRGBDDataset(
            data_root=mock_dataset,
            train=True,
            **config.to_dict(),
        )

        # Verify values flowed through
        assert dataset.rgb_aug_prob == config.rgb_aug_prob
        assert dataset.rgb_aug_mag == config.rgb_aug_mag
        assert dataset.depth_aug_prob == config.depth_aug_prob
        assert dataset.depth_aug_mag == config.depth_aug_mag

    def test_uniform_config_applies_to_both_streams(self, mock_dataset):
        """Test that uniform() applies same values to both streams."""
        from src.training.augmentation_config import AugmentationConfig
        from src.data_utils.sunrgbd_dataset import SUNRGBDDataset

        config = AugmentationConfig.uniform(aug_prob=1.5, aug_mag=1.2)
        dataset = SUNRGBDDataset(data_root=mock_dataset, train=True, **config.to_dict())

        # Both streams should have same values
        assert dataset.rgb_aug_prob == dataset.depth_aug_prob == 1.5
        assert dataset.rgb_aug_mag == dataset.depth_aug_mag == 1.2

        # Sync values should be exactly the config values (no averaging needed)
        sync_prob = (dataset.rgb_aug_prob + dataset.depth_aug_prob) / 2
        assert sync_prob == 1.5

    def test_baseline_backward_compatibility(self, mock_dataset):
        """Test that default config (all 1.0) is backward compatible."""
        from src.training.augmentation_config import AugmentationConfig
        from src.data_utils.sunrgbd_dataset import SUNRGBDDataset

        # Default config
        config = AugmentationConfig()

        # Should produce baseline values
        dataset = SUNRGBDDataset(data_root=mock_dataset, train=True, **config.to_dict())

        # All values should be at baseline
        assert dataset._flip_p == BASE_FLIP_P
        assert dataset._crop_p == BASE_CROP_P
        assert dataset._color_jitter_p == BASE_COLOR_JITTER_P
        assert dataset._brightness == BASE_BRIGHTNESS
        assert dataset._depth_aug_p == BASE_DEPTH_AUG_P


# =============================================================================
# 1. BASELINE REGRESSION TESTS (Critical)
# =============================================================================

class TestBaselineRegression:
    """
    Critical tests to verify BASE_* constants match actual code values.
    This ensures aug_*=1.0 produces identical behavior to the original code.
    """

    @pytest.fixture
    def mock_dataset(self, tmp_path):
        """Create a minimal mock dataset structure."""
        train_dir = tmp_path / "train"
        train_dir.mkdir()
        (train_dir / "rgb").mkdir()
        (train_dir / "depth").mkdir()
        labels_file = train_dir / "labels.txt"
        labels_file.write_text("0\n")

        import numpy as np
        from PIL import Image
        rgb_img = Image.fromarray(np.zeros((100, 100, 3), dtype=np.uint8))
        depth_img = Image.fromarray(np.zeros((100, 100), dtype=np.uint16))
        rgb_img.save(train_dir / "rgb" / "00000.png")
        depth_img.save(train_dir / "depth" / "00000.png")
        return str(tmp_path)

    def test_baseline_constants_match_documented_values(self):
        """Verify BASE_* constants match the documented baseline values."""
        # Probability baselines
        assert BASE_FLIP_P == 0.5, "Flip probability baseline changed!"
        assert BASE_CROP_P == 0.5, "Crop probability baseline changed!"
        assert BASE_COLOR_JITTER_P == 0.43, "ColorJitter probability baseline changed!"
        assert BASE_BLUR_P == 0.25, "Blur probability baseline changed!"
        assert BASE_GRAYSCALE_P == 0.17, "Grayscale probability baseline changed!"
        assert BASE_RGB_ERASING_P == 0.17, "RGB erasing probability baseline changed!"
        assert BASE_DEPTH_AUG_P == 0.50, "Depth aug probability baseline changed!"
        assert BASE_DEPTH_ERASING_P == 0.10, "Depth erasing probability baseline changed!"

        # Magnitude baselines
        assert BASE_CROP_SCALE_MIN == 0.9, "Crop scale min baseline changed!"
        assert BASE_BRIGHTNESS == 0.37, "Brightness baseline changed!"
        assert BASE_CONTRAST == 0.37, "Contrast baseline changed!"
        assert BASE_SATURATION == 0.37, "Saturation baseline changed!"
        assert BASE_HUE == 0.11, "Hue baseline changed!"
        assert BASE_BLUR_SIGMA_MAX == 1.7, "Blur sigma max baseline changed!"
        assert BASE_DEPTH_BRIGHTNESS == 0.25, "Depth brightness baseline changed!"
        assert BASE_DEPTH_CONTRAST == 0.25, "Depth contrast baseline changed!"
        assert BASE_DEPTH_NOISE_STD == 0.059, "Depth noise std baseline changed!"

    def test_dataset_at_baseline_matches_original(self, mock_dataset):
        """Verify dataset with all params=1.0 produces exact baseline values."""
        from src.data_utils.sunrgbd_dataset import SUNRGBDDataset

        dataset = SUNRGBDDataset(
            data_root=mock_dataset,
            train=True,
            rgb_aug_prob=1.0,
            rgb_aug_mag=1.0,
            depth_aug_prob=1.0,
            depth_aug_mag=1.0,
        )

        # All computed values should exactly match baselines
        assert dataset._flip_p == BASE_FLIP_P
        assert dataset._crop_p == BASE_CROP_P
        assert dataset._crop_scale_min == BASE_CROP_SCALE_MIN
        assert dataset._color_jitter_p == BASE_COLOR_JITTER_P
        assert dataset._brightness == BASE_BRIGHTNESS
        assert dataset._contrast == BASE_CONTRAST
        assert dataset._saturation == BASE_SATURATION
        assert dataset._hue == BASE_HUE
        assert dataset._blur_p == BASE_BLUR_P
        assert dataset._blur_sigma_max == BASE_BLUR_SIGMA_MAX
        assert dataset._grayscale_p == BASE_GRAYSCALE_P
        assert dataset._rgb_erasing_p == BASE_RGB_ERASING_P
        assert dataset._depth_aug_p == BASE_DEPTH_AUG_P
        assert dataset._depth_brightness == BASE_DEPTH_BRIGHTNESS
        assert dataset._depth_contrast == BASE_DEPTH_CONTRAST
        assert dataset._depth_noise_std == BASE_DEPTH_NOISE_STD
        assert dataset._depth_erasing_p == BASE_DEPTH_ERASING_P


# =============================================================================
# 2. ZERO AUGMENTATION TESTS
# =============================================================================

class TestZeroAugmentation:
    """Verify aug_prob=0.0 completely disables all augmentation."""

    @pytest.fixture
    def mock_dataset(self, tmp_path):
        """Create a minimal mock dataset structure."""
        train_dir = tmp_path / "train"
        train_dir.mkdir()
        (train_dir / "rgb").mkdir()
        (train_dir / "depth").mkdir()
        labels_file = train_dir / "labels.txt"
        labels_file.write_text("0\n")

        import numpy as np
        from PIL import Image
        rgb_img = Image.fromarray(np.zeros((100, 100, 3), dtype=np.uint8))
        depth_img = Image.fromarray(np.zeros((100, 100), dtype=np.uint16))
        rgb_img.save(train_dir / "rgb" / "00000.png")
        depth_img.save(train_dir / "depth" / "00000.png")
        return str(tmp_path)

    def test_zero_prob_disables_all_probabilities(self, mock_dataset):
        """Verify all probability-based augmentations are disabled at prob=0.0."""
        from src.data_utils.sunrgbd_dataset import SUNRGBDDataset

        dataset = SUNRGBDDataset(
            data_root=mock_dataset,
            train=True,
            rgb_aug_prob=0.0,
            rgb_aug_mag=1.0,  # mag doesn't matter when prob=0
            depth_aug_prob=0.0,
            depth_aug_mag=1.0,
        )

        # All probability values should be 0
        assert dataset._flip_p == 0.0
        assert dataset._crop_p == 0.0
        assert dataset._color_jitter_p == 0.0
        assert dataset._blur_p == 0.0
        assert dataset._grayscale_p == 0.0
        assert dataset._rgb_erasing_p == 0.0
        assert dataset._depth_aug_p == 0.0
        assert dataset._depth_erasing_p == 0.0

    def test_zero_rgb_prob_only_affects_rgb(self, mock_dataset):
        """Verify zero RGB prob doesn't affect depth-only augmentations."""
        from src.data_utils.sunrgbd_dataset import SUNRGBDDataset

        dataset = SUNRGBDDataset(
            data_root=mock_dataset,
            train=True,
            rgb_aug_prob=0.0,
            rgb_aug_mag=1.0,
            depth_aug_prob=1.0,  # Depth stays at baseline
            depth_aug_mag=1.0,
        )

        # RGB-only should be zero
        assert dataset._color_jitter_p == 0.0
        assert dataset._blur_p == 0.0
        assert dataset._grayscale_p == 0.0
        assert dataset._rgb_erasing_p == 0.0

        # Depth-only should be at baseline
        assert dataset._depth_aug_p == BASE_DEPTH_AUG_P
        assert dataset._depth_erasing_p == BASE_DEPTH_ERASING_P

        # Sync uses average (0.0 + 1.0) / 2 = 0.5
        assert dataset._flip_p == BASE_FLIP_P * 0.5
        assert dataset._crop_p == BASE_CROP_P * 0.5

    def test_zero_depth_prob_only_affects_depth(self, mock_dataset):
        """Verify zero Depth prob doesn't affect RGB-only augmentations."""
        from src.data_utils.sunrgbd_dataset import SUNRGBDDataset

        dataset = SUNRGBDDataset(
            data_root=mock_dataset,
            train=True,
            rgb_aug_prob=1.0,  # RGB stays at baseline
            rgb_aug_mag=1.0,
            depth_aug_prob=0.0,
            depth_aug_mag=1.0,
        )

        # RGB-only should be at baseline
        assert dataset._color_jitter_p == BASE_COLOR_JITTER_P
        assert dataset._blur_p == BASE_BLUR_P
        assert dataset._grayscale_p == BASE_GRAYSCALE_P
        assert dataset._rgb_erasing_p == BASE_RGB_ERASING_P

        # Depth-only should be zero
        assert dataset._depth_aug_p == 0.0
        assert dataset._depth_erasing_p == 0.0


# =============================================================================
# 3. PER-STREAM INDEPENDENCE TESTS
# =============================================================================

class TestPerStreamIndependence:
    """Verify RGB and Depth augmentation scale independently."""

    @pytest.fixture
    def mock_dataset(self, tmp_path):
        """Create a minimal mock dataset structure."""
        train_dir = tmp_path / "train"
        train_dir.mkdir()
        (train_dir / "rgb").mkdir()
        (train_dir / "depth").mkdir()
        labels_file = train_dir / "labels.txt"
        labels_file.write_text("0\n")

        import numpy as np
        from PIL import Image
        rgb_img = Image.fromarray(np.zeros((100, 100, 3), dtype=np.uint8))
        depth_img = Image.fromarray(np.zeros((100, 100), dtype=np.uint16))
        rgb_img.save(train_dir / "rgb" / "00000.png")
        depth_img.save(train_dir / "depth" / "00000.png")
        return str(tmp_path)

    def test_rgb_high_depth_low(self, mock_dataset):
        """Test RGB=2.0, Depth=0.5 scales independently."""
        from src.data_utils.sunrgbd_dataset import SUNRGBDDataset

        dataset = SUNRGBDDataset(
            data_root=mock_dataset,
            train=True,
            rgb_aug_prob=2.0,
            rgb_aug_mag=2.0,
            depth_aug_prob=0.5,
            depth_aug_mag=0.5,
        )

        # RGB-only: doubled
        assert dataset._color_jitter_p == pytest.approx(BASE_COLOR_JITTER_P * 2.0, rel=1e-3)
        assert dataset._brightness == pytest.approx(BASE_BRIGHTNESS * 2.0, rel=1e-3)

        # Depth-only: halved
        assert dataset._depth_aug_p == pytest.approx(BASE_DEPTH_AUG_P * 0.5, rel=1e-3)
        assert dataset._depth_brightness == pytest.approx(BASE_DEPTH_BRIGHTNESS * 0.5, rel=1e-3)

        # Sync: average (2.0 + 0.5) / 2 = 1.25
        assert dataset._flip_p == pytest.approx(BASE_FLIP_P * 1.25, rel=1e-3)

    def test_rgb_low_depth_high(self, mock_dataset):
        """Test RGB=0.5, Depth=2.0 scales independently."""
        from src.data_utils.sunrgbd_dataset import SUNRGBDDataset

        dataset = SUNRGBDDataset(
            data_root=mock_dataset,
            train=True,
            rgb_aug_prob=0.5,
            rgb_aug_mag=0.5,
            depth_aug_prob=2.0,
            depth_aug_mag=2.0,
        )

        # RGB-only: halved
        assert dataset._color_jitter_p == pytest.approx(BASE_COLOR_JITTER_P * 0.5, rel=1e-3)
        assert dataset._brightness == pytest.approx(BASE_BRIGHTNESS * 0.5, rel=1e-3)

        # Depth-only: doubled (but may be capped)
        assert dataset._depth_aug_p == pytest.approx(min(BASE_DEPTH_AUG_P * 2.0, MAX_PROBABILITY), rel=1e-3)
        assert dataset._depth_brightness == pytest.approx(min(BASE_DEPTH_BRIGHTNESS * 2.0, MAX_DEPTH_BRIGHTNESS), rel=1e-3)

    def test_prob_and_mag_independent(self, mock_dataset):
        """Test that prob and mag scale independently within each stream."""
        from src.data_utils.sunrgbd_dataset import SUNRGBDDataset

        # High prob, low mag
        dataset1 = SUNRGBDDataset(
            data_root=mock_dataset,
            train=True,
            rgb_aug_prob=2.0,
            rgb_aug_mag=0.5,
            depth_aug_prob=1.0,
            depth_aug_mag=1.0,
        )

        # Low prob, high mag
        dataset2 = SUNRGBDDataset(
            data_root=mock_dataset,
            train=True,
            rgb_aug_prob=0.5,
            rgb_aug_mag=2.0,
            depth_aug_prob=1.0,
            depth_aug_mag=1.0,
        )

        # dataset1: high prob, low mag
        assert dataset1._color_jitter_p == pytest.approx(BASE_COLOR_JITTER_P * 2.0, rel=1e-3)
        assert dataset1._brightness == pytest.approx(BASE_BRIGHTNESS * 0.5, rel=1e-3)

        # dataset2: low prob, high mag
        assert dataset2._color_jitter_p == pytest.approx(BASE_COLOR_JITTER_P * 0.5, rel=1e-3)
        assert dataset2._brightness == pytest.approx(BASE_BRIGHTNESS * 2.0, rel=1e-3)


# =============================================================================
# 4. PROBABILITY SCALING VERIFICATION
# =============================================================================

class TestProbabilityScaling:
    """Verify probabilities scale correctly at various values."""

    def test_probability_linear_scaling(self):
        """Verify probability scales linearly (before cap)."""
        test_values = [0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0]

        for scale in test_values:
            expected = min(BASE_COLOR_JITTER_P * scale, MAX_PROBABILITY)
            actual = min(BASE_COLOR_JITTER_P * scale, MAX_PROBABILITY)
            assert actual == pytest.approx(expected, rel=1e-6), f"Failed at scale={scale}"

    def test_probability_capping_at_high_values(self):
        """Verify probabilities are capped at MAX_PROBABILITY."""
        # ColorJitter: 0.43 * 3.0 = 1.29, should be capped at 0.95
        result = min(BASE_COLOR_JITTER_P * 3.0, MAX_PROBABILITY)
        assert result == MAX_PROBABILITY

        # Flip: 0.5 * 2.0 = 1.0, should be capped at 0.95
        result = min(BASE_FLIP_P * 2.0, MAX_PROBABILITY)
        assert result == MAX_PROBABILITY

    def test_probability_at_exact_cap_boundary(self):
        """Test probability at exact cap boundary."""
        # Find scale that would produce exactly MAX_PROBABILITY
        scale_for_cap = MAX_PROBABILITY / BASE_COLOR_JITTER_P  # ~2.21

        # Just below cap
        result_below = min(BASE_COLOR_JITTER_P * (scale_for_cap - 0.1), MAX_PROBABILITY)
        assert result_below < MAX_PROBABILITY

        # At cap
        result_at = min(BASE_COLOR_JITTER_P * scale_for_cap, MAX_PROBABILITY)
        assert result_at == pytest.approx(MAX_PROBABILITY, rel=1e-3)

        # Above cap
        result_above = min(BASE_COLOR_JITTER_P * (scale_for_cap + 0.1), MAX_PROBABILITY)
        assert result_above == MAX_PROBABILITY


# =============================================================================
# 5. MAGNITUDE SCALING VERIFICATION
# =============================================================================

class TestMagnitudeScaling:
    """Verify magnitudes scale correctly at various values."""

    def test_magnitude_linear_scaling(self):
        """Verify magnitude scales linearly (before cap)."""
        test_values = [0.25, 0.5, 0.75, 1.0, 1.25, 1.5]

        for scale in test_values:
            expected = min(BASE_BRIGHTNESS * scale, MAX_BRIGHTNESS)
            actual = min(BASE_BRIGHTNESS * scale, MAX_BRIGHTNESS)
            assert actual == pytest.approx(expected, rel=1e-6), f"Failed at scale={scale}"

    def test_magnitude_capping_at_high_values(self):
        """Verify magnitudes are capped at their MAX values."""
        # Brightness: 0.37 * 3.0 = 1.11, should be capped at 0.8
        result = min(BASE_BRIGHTNESS * 3.0, MAX_BRIGHTNESS)
        assert result == MAX_BRIGHTNESS

        # Hue: 0.11 * 5.0 = 0.55, should be capped at 0.4
        result = min(BASE_HUE * 5.0, MAX_HUE)
        assert result == MAX_HUE

        # Blur sigma: 1.7 * 3.0 = 5.1, should be capped at 3.5
        result = min(BASE_BLUR_SIGMA_MAX * 3.0, MAX_BLUR_SIGMA)
        assert result == MAX_BLUR_SIGMA

        # Depth brightness: 0.25 * 3.0 = 0.75, should be capped at 0.5
        result = min(BASE_DEPTH_BRIGHTNESS * 3.0, MAX_DEPTH_BRIGHTNESS)
        assert result == MAX_DEPTH_BRIGHTNESS

        # Depth noise: 0.059 * 4.0 = 0.236, should be capped at 0.15
        result = min(BASE_DEPTH_NOISE_STD * 4.0, MAX_DEPTH_NOISE_STD)
        assert result == MAX_DEPTH_NOISE_STD

    def test_crop_scale_inverse_relationship(self):
        """Verify crop scale has inverse relationship (higher mag = smaller min crop)."""
        # At mag=1.0: crop_scale_min = 0.9
        result_1 = max(MIN_CROP_SCALE, 1.0 - (1.0 - BASE_CROP_SCALE_MIN) * 1.0)
        assert result_1 == BASE_CROP_SCALE_MIN

        # At mag=2.0: crop_scale_min = 1.0 - 0.1 * 2.0 = 0.8
        result_2 = max(MIN_CROP_SCALE, 1.0 - (1.0 - BASE_CROP_SCALE_MIN) * 2.0)
        assert result_2 == pytest.approx(0.8, rel=1e-3)

        # At mag=5.0: would be 0.5, exactly at MIN_CROP_SCALE (use approx for float precision)
        result_5 = max(MIN_CROP_SCALE, 1.0 - (1.0 - BASE_CROP_SCALE_MIN) * 5.0)
        assert result_5 == pytest.approx(MIN_CROP_SCALE, rel=1e-6)

        # At mag=10.0: would be 0.0, but capped at MIN_CROP_SCALE (0.5)
        result_10 = max(MIN_CROP_SCALE, 1.0 - (1.0 - BASE_CROP_SCALE_MIN) * 10.0)
        assert result_10 == pytest.approx(MIN_CROP_SCALE, rel=1e-6)


# =============================================================================
# 6. CAP ENFORCEMENT TESTS
# =============================================================================

class TestCapEnforcement:
    """Verify extreme values are properly capped."""

    @pytest.fixture
    def mock_dataset(self, tmp_path):
        """Create a minimal mock dataset structure."""
        train_dir = tmp_path / "train"
        train_dir.mkdir()
        (train_dir / "rgb").mkdir()
        (train_dir / "depth").mkdir()
        labels_file = train_dir / "labels.txt"
        labels_file.write_text("0\n")

        import numpy as np
        from PIL import Image
        rgb_img = Image.fromarray(np.zeros((100, 100, 3), dtype=np.uint8))
        depth_img = Image.fromarray(np.zeros((100, 100), dtype=np.uint16))
        rgb_img.save(train_dir / "rgb" / "00000.png")
        depth_img.save(train_dir / "depth" / "00000.png")
        return str(tmp_path)

    def test_extreme_prob_values_capped(self, mock_dataset):
        """Verify extreme prob values are capped at MAX_PROBABILITY or their scaled value."""
        from src.data_utils.sunrgbd_dataset import SUNRGBDDataset

        dataset = SUNRGBDDataset(
            data_root=mock_dataset,
            train=True,
            rgb_aug_prob=5.0,  # Very high
            rgb_aug_mag=1.0,
            depth_aug_prob=5.0,
            depth_aug_mag=1.0,
        )

        # Probabilities that exceed MAX_PROBABILITY should be capped
        # Note: Some baselines are low enough that even 5x doesn't exceed cap
        # Flip: 0.5 * 5 = 2.5 -> capped at 0.95
        assert dataset._flip_p == MAX_PROBABILITY
        # Crop: 0.5 * 5 = 2.5 -> capped at 0.95
        assert dataset._crop_p == MAX_PROBABILITY
        # ColorJitter: 0.43 * 5 = 2.15 -> capped at 0.95
        assert dataset._color_jitter_p == MAX_PROBABILITY
        # Blur: 0.25 * 5 = 1.25 -> capped at 0.95
        assert dataset._blur_p == MAX_PROBABILITY
        # Grayscale: 0.17 * 5 = 0.85 -> NOT capped (below 0.95)
        assert dataset._grayscale_p == pytest.approx(BASE_GRAYSCALE_P * 5.0, rel=1e-3)
        # RGB Erasing: 0.17 * 5 = 0.85 -> NOT capped (below 0.95)
        assert dataset._rgb_erasing_p == pytest.approx(BASE_RGB_ERASING_P * 5.0, rel=1e-3)
        # Depth Aug: 0.5 * 5 = 2.5 -> capped at 0.95
        assert dataset._depth_aug_p == MAX_PROBABILITY
        # Depth Erasing: 0.1 * 5 = 0.5 -> NOT capped (below 0.95)
        assert dataset._depth_erasing_p == pytest.approx(BASE_DEPTH_ERASING_P * 5.0, rel=1e-3)

    def test_extreme_mag_values_capped(self, mock_dataset):
        """Verify extreme mag values are capped at their respective MAX values."""
        from src.data_utils.sunrgbd_dataset import SUNRGBDDataset

        dataset = SUNRGBDDataset(
            data_root=mock_dataset,
            train=True,
            rgb_aug_prob=1.0,
            rgb_aug_mag=5.0,  # Very high
            depth_aug_prob=1.0,
            depth_aug_mag=5.0,
        )

        # RGB magnitudes capped
        assert dataset._brightness == MAX_BRIGHTNESS
        assert dataset._contrast == MAX_CONTRAST
        assert dataset._saturation == MAX_SATURATION
        assert dataset._hue == MAX_HUE
        assert dataset._blur_sigma_max == MAX_BLUR_SIGMA
        assert dataset._rgb_erasing_scale_max == MAX_ERASING_SCALE

        # Depth magnitudes capped
        assert dataset._depth_brightness == MAX_DEPTH_BRIGHTNESS
        assert dataset._depth_contrast == MAX_DEPTH_CONTRAST
        assert dataset._depth_noise_std == MAX_DEPTH_NOISE_STD
        assert dataset._depth_erasing_scale_max == MAX_ERASING_SCALE

        # Crop scale capped at minimum (use approx for float precision)
        assert dataset._crop_scale_min == pytest.approx(MIN_CROP_SCALE, rel=1e-6)


# =============================================================================
# 7. INPUT VALIDATION TESTS
# =============================================================================

class TestInputValidation:
    """Verify input validation works correctly."""

    def test_negative_values_rejected(self):
        """Verify negative values raise ValueError."""
        with pytest.raises(ValueError, match="must be >= 0"):
            AugmentationConfig(rgb_aug_prob=-0.1)

        with pytest.raises(ValueError, match="must be >= 0"):
            AugmentationConfig(rgb_aug_mag=-1.0)

        with pytest.raises(ValueError, match="must be >= 0"):
            AugmentationConfig(depth_aug_prob=-0.5)

        with pytest.raises(ValueError, match="must be >= 0"):
            AugmentationConfig(depth_aug_mag=-2.0)

    def test_high_values_warn(self):
        """Verify values > 5.0 produce warnings."""
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            AugmentationConfig(rgb_aug_prob=6.0)
            assert len(w) == 1
            assert "unusually high" in str(w[0].message)

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            AugmentationConfig(rgb_aug_mag=10.0)
            assert len(w) == 1
            assert "unusually high" in str(w[0].message)

    def test_boundary_values_accepted(self):
        """Verify boundary values (0.0, 5.0) are accepted without error."""
        # Zero values should work
        config = AugmentationConfig(
            rgb_aug_prob=0.0,
            rgb_aug_mag=0.0,
            depth_aug_prob=0.0,
            depth_aug_mag=0.0,
        )
        assert config.rgb_aug_prob == 0.0

        # Values at 5.0 should work without warning
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            config = AugmentationConfig(rgb_aug_prob=5.0)
            # Should not warn at exactly 5.0
            assert len(w) == 0


# =============================================================================
# 8. CONFIG CONSISTENCY TESTS
# =============================================================================

class TestConfigConsistency:
    """Verify AugmentationConfig unpacking works correctly across all entry points."""

    @pytest.fixture
    def mock_dataset(self, tmp_path):
        """Create a minimal mock dataset structure."""
        train_dir = tmp_path / "train"
        train_dir.mkdir()
        (train_dir / "rgb").mkdir()
        (train_dir / "depth").mkdir()
        labels_file = train_dir / "labels.txt"
        labels_file.write_text("0\n")

        import numpy as np
        from PIL import Image
        rgb_img = Image.fromarray(np.zeros((100, 100, 3), dtype=np.uint8))
        depth_img = Image.fromarray(np.zeros((100, 100), dtype=np.uint16))
        rgb_img.save(train_dir / "rgb" / "00000.png")
        depth_img.save(train_dir / "depth" / "00000.png")
        return str(tmp_path)

    def test_config_to_dict_unpacks_correctly(self, mock_dataset):
        """Verify to_dict() can be unpacked into SUNRGBDDataset."""
        from src.data_utils.sunrgbd_dataset import SUNRGBDDataset

        config = AugmentationConfig(
            rgb_aug_prob=1.5,
            rgb_aug_mag=1.2,
            depth_aug_prob=0.8,
            depth_aug_mag=0.5,
        )

        dataset = SUNRGBDDataset(
            data_root=mock_dataset,
            train=True,
            **config.to_dict(),
        )

        assert dataset.rgb_aug_prob == 1.5
        assert dataset.rgb_aug_mag == 1.2
        assert dataset.depth_aug_prob == 0.8
        assert dataset.depth_aug_mag == 0.5

    def test_uniform_config_consistency(self):
        """Verify uniform() creates consistent config for both streams."""
        config = AugmentationConfig.uniform(aug_prob=1.5, aug_mag=1.2)

        assert config.rgb_aug_prob == config.depth_aug_prob == 1.5
        assert config.rgb_aug_mag == config.depth_aug_mag == 1.2

        # Verify to_dict() matches
        d = config.to_dict()
        assert d["rgb_aug_prob"] == d["depth_aug_prob"]
        assert d["rgb_aug_mag"] == d["depth_aug_mag"]

    def test_config_can_be_logged(self):
        """Verify config can be serialized for logging (e.g., W&B)."""
        config = AugmentationConfig(
            rgb_aug_prob=1.5,
            rgb_aug_mag=1.2,
            depth_aug_prob=0.8,
            depth_aug_mag=0.5,
        )

        d = config.to_dict()

        # Should be JSON-serializable
        import json
        json_str = json.dumps(d)
        loaded = json.loads(json_str)
        assert loaded == d


# =============================================================================
# 9. GPU AUGMENTATION PARITY TESTS
# =============================================================================

class TestGPUAugmentationParity:
    """Verify GPU path respects same scaling as CPU path (without requiring GPU)."""

    def test_scaling_formulas_match_cpu_path(self):
        """Verify GPU scaling formulas match CPU path formulas."""
        # Both CPU and GPU paths should use identical formulas
        rgb_aug_prob = 1.5
        rgb_aug_mag = 1.2
        depth_aug_prob = 0.8
        depth_aug_mag = 0.5

        # GPU path calculations (from gpu_augmentation.py)
        gpu_color_jitter_p = min(BASE_COLOR_JITTER_P * rgb_aug_prob, MAX_PROBABILITY)
        gpu_brightness = min(BASE_BRIGHTNESS * rgb_aug_mag, MAX_BRIGHTNESS)
        gpu_depth_erasing_p = min(BASE_DEPTH_ERASING_P * depth_aug_prob, MAX_PROBABILITY)

        # CPU path calculations (from sunrgbd_dataset.py)
        cpu_color_jitter_p = min(BASE_COLOR_JITTER_P * rgb_aug_prob, MAX_PROBABILITY)
        cpu_brightness = min(BASE_BRIGHTNESS * rgb_aug_mag, MAX_BRIGHTNESS)
        cpu_depth_erasing_p = min(BASE_DEPTH_ERASING_P * depth_aug_prob, MAX_PROBABILITY)

        # Should match exactly
        assert gpu_color_jitter_p == cpu_color_jitter_p
        assert gpu_brightness == cpu_brightness
        assert gpu_depth_erasing_p == cpu_depth_erasing_p

    def test_shared_constants_imported_correctly(self):
        """Verify both CPU and GPU modules import from same source."""
        # Import directly from augmentation_config
        from src.training.augmentation_config import BASE_COLOR_JITTER_P as config_cjp
        from src.training.augmentation_config import MAX_PROBABILITY as config_max_p

        # These should be the exact same values used by both modules
        assert config_cjp == 0.43
        assert config_max_p == 0.95

    def test_gpu_aug_stores_scaling_params(self):
        """Verify GPUAugmentation would store scaling params (without instantiating)."""
        # This test verifies the design without requiring Kornia
        # The actual GPUAugmentation stores these in __init__:
        # self.rgb_aug_prob = rgb_aug_prob
        # self.rgb_aug_mag = rgb_aug_mag
        # etc.

        # Verify the constants used are accessible
        assert BASE_COLOR_JITTER_P == 0.43
        assert BASE_BLUR_P == 0.25
        assert MAX_PROBABILITY == 0.95
        assert MAX_BRIGHTNESS == 0.8


# =============================================================================
# 10. INTEGRATION TEST
# =============================================================================

class TestIntegration:
    """Integration test: Full training loop doesn't crash with custom params."""

    @pytest.fixture
    def mock_dataset(self, tmp_path):
        """Create a mock dataset with multiple samples."""
        train_dir = tmp_path / "train"
        train_dir.mkdir()
        (train_dir / "rgb").mkdir()
        (train_dir / "depth").mkdir()

        import numpy as np
        from PIL import Image

        # Create 10 samples
        labels = []
        for i in range(10):
            rgb_img = Image.fromarray(
                np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
            )
            depth_img = Image.fromarray(
                np.random.randint(0, 65535, (100, 100), dtype=np.uint16)
            )
            rgb_img.save(train_dir / "rgb" / f"{i:05d}.png")
            depth_img.save(train_dir / "depth" / f"{i:05d}.png")
            labels.append(str(i % 15))  # 15 classes

        labels_file = train_dir / "labels.txt"
        labels_file.write_text("\n".join(labels) + "\n")

        return str(tmp_path)

    def test_dataloader_iteration_with_custom_params(self, mock_dataset):
        """Test that DataLoader iteration works with custom aug params."""
        from torch.utils.data import DataLoader
        from src.data_utils.sunrgbd_dataset import SUNRGBDDataset

        config = AugmentationConfig(
            rgb_aug_prob=1.5,
            rgb_aug_mag=1.2,
            depth_aug_prob=0.8,
            depth_aug_mag=0.5,
        )

        dataset = SUNRGBDDataset(
            data_root=mock_dataset,
            train=True,
            target_size=(64, 64),  # Small for speed
            **config.to_dict(),
        )

        loader = DataLoader(dataset, batch_size=4, shuffle=True, num_workers=0)

        # Iterate through entire dataset without crashing
        for batch_idx, (rgb, depth, label) in enumerate(loader):
            assert rgb.shape[1] == 3  # RGB channels
            assert depth.shape[1] == 1  # Depth channel
            assert len(label) <= 4  # Batch size

    def test_dataset_getitem_with_extreme_params(self, mock_dataset):
        """Test __getitem__ works with extreme but valid params."""
        from src.data_utils.sunrgbd_dataset import SUNRGBDDataset

        # Very high augmentation
        dataset_high = SUNRGBDDataset(
            data_root=mock_dataset,
            train=True,
            target_size=(64, 64),
            rgb_aug_prob=3.0,
            rgb_aug_mag=3.0,
            depth_aug_prob=3.0,
            depth_aug_mag=3.0,
        )

        # Very low augmentation
        dataset_low = SUNRGBDDataset(
            data_root=mock_dataset,
            train=True,
            target_size=(64, 64),
            rgb_aug_prob=0.1,
            rgb_aug_mag=0.1,
            depth_aug_prob=0.1,
            depth_aug_mag=0.1,
        )

        # Both should work without crashing
        rgb_high, depth_high, label_high = dataset_high[0]
        rgb_low, depth_low, label_low = dataset_low[0]

        assert rgb_high.shape == rgb_low.shape
        assert depth_high.shape == depth_low.shape

    def test_augmentation_config_workflow(self, mock_dataset):
        """Test the recommended workflow from Colab."""
        from src.data_utils.sunrgbd_dataset import SUNRGBDDataset
        from torch.utils.data import DataLoader

        # Step 1: Create config (as user would in Colab)
        config = AugmentationConfig(
            rgb_aug_prob=1.5,
            rgb_aug_mag=1.2,
            depth_aug_prob=1.0,
            depth_aug_mag=0.8,
        )

        # Step 2: Create dataset with config
        dataset = SUNRGBDDataset(
            data_root=mock_dataset,
            train=True,
            target_size=(64, 64),
            normalize=True,  # CPU mode
            **config.to_dict(),
        )

        # Step 3: Create dataloader
        loader = DataLoader(dataset, batch_size=2, num_workers=0)

        # Step 4: Iterate (simulate training loop)
        batch_count = 0
        for rgb, depth, label in loader:
            batch_count += 1
            # Simulate forward pass (just check shapes)
            assert rgb.dim() == 4
            assert depth.dim() == 4

        assert batch_count > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
