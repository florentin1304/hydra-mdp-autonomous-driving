from womd_il.train import should_early_stop


def test_patience_zero_never_stops():
    stale, stop = should_early_stop(
        improved=False, stale=9, epoch=20, patience=0, min_epochs=0
    )
    assert stale == 0
    assert stop is False


def test_improvement_resets_stale():
    stale, stop = should_early_stop(
        improved=True, stale=3, epoch=10, patience=4, min_epochs=8
    )
    assert stale == 0
    assert stop is False


def test_does_not_stop_before_min_epochs():
    stale, stop = should_early_stop(
        improved=False, stale=3, epoch=5, patience=4, min_epochs=8
    )
    assert stale == 4
    assert stop is False


def test_stops_after_patience_past_min_epochs():
    stale, stop = should_early_stop(
        improved=False, stale=3, epoch=10, patience=4, min_epochs=8
    )
    assert stale == 4
    assert stop is True
