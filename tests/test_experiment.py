import pytest

from conversion_intelligence.experiment import two_proportion_sample_size


def test_sample_size_is_positive_and_increases_for_smaller_effect() -> None:
    large_effect = two_proportion_sample_size(0.03, 0.04)
    small_effect = two_proportion_sample_size(0.03, 0.033)
    assert large_effect > 0
    assert small_effect > large_effect


def test_sample_size_rejects_equal_rates() -> None:
    with pytest.raises(ValueError, match="must differ"):
        two_proportion_sample_size(0.03, 0.03)
