from conversion_intelligence.synthetic import generate_synthetic_conversion_data


def test_synthetic_data_matches_contract_and_target_rate() -> None:
    data = generate_synthetic_conversion_data(rows=20_000, seed=7, target_rate=0.04)
    assert list(data.columns) == [
        "country",
        "age",
        "new_user",
        "source",
        "total_pages_visited",
        "converted",
    ]
    assert abs(data["converted"].mean() - 0.04) < 0.01
