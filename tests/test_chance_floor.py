from experiments.chance_floor import measure


def test_chance_floor_is_seed_complete_and_measured():
    out = measure([42, 1337], block_size=64, n=200)
    assert out["status"] == "completed"
    assert out["seeds"] == [42, 1337]
    assert len(out["rows"]) == 2
    assert out["strongest_mean_percent"] > 40.0
    assert out["strongest_sd_percent"] >= 0.0
