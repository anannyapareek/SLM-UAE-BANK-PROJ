import numpy as np

from src import review_budget as rb


def test_weekday_mask_matches_calendar():
    days = np.arange("2011-01-01", "2011-01-15", dtype="datetime64[D]")
    expected = np.array([d.astype(object).weekday() < 5 for d in days])
    assert np.array_equal(rb.weekday({"day": days.astype(np.int64)}), expected)


def test_capacities_include_primary_and_sensitivity():
    assert rb.capacities({"alerts_per_day": 20}) == [10, 20, 50]
