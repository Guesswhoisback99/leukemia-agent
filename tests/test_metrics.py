import pytest

from app.chain import REFUSAL
from eval.metrics import contains_expected, first_hit_rank, hit_at_k, mrr, refusal_correct


def test_first_hit_rank():
    assert first_hit_rank([[3], [16], [16, 17]], [16]) == 2
    assert first_hit_rank([[16]], [16, 40]) == 1


def test_first_hit_rank_multi_page_chunk():
    assert first_hit_rank([[10], [15, 16]], [16]) == 2


def test_first_hit_rank_no_hit():
    assert first_hit_rank([[1], [2]], [16]) is None
    assert first_hit_rank([], [16]) is None


def test_first_hit_rank_empty_expected():
    assert first_hit_rank([[1], [2]], []) is None


def test_hit_at_k():
    ranks = [1, 3, None, 6]
    assert hit_at_k(ranks, 5) == 0.5
    assert hit_at_k(ranks, 1) == 0.25
    assert hit_at_k(ranks, 10) == 0.75
    assert hit_at_k([], 5) == 0.0


def test_mrr():
    assert mrr([1, 2, None, 4]) == pytest.approx((1 + 0.5 + 0 + 0.25) / 4)
    assert mrr([None]) == 0.0
    assert mrr([]) == 0.0


def test_refusal_correct():
    assert refusal_correct(REFUSAL, "unanswerable")
    assert refusal_correct(f"  {REFUSAL}\n", "unanswerable")
    assert not refusal_correct("Canberra.", "unanswerable")
    assert refusal_correct("1500 mg/m2 [s. 16, X]", "table")
    assert not refusal_correct(REFUSAL, "prose")
    assert not refusal_correct(REFUSAL, "diagram")


def test_contains_expected():
    assert contains_expected("Dawka: 1500 mg/m 2 [s. 16]", "1500 mg/m2")
    assert contains_expected("METOTREKSAT  i.v.", "metotreksat i.v.")
    assert not contains_expected("3500 mg/m2", "1500 mg/m2")


def test_spread_sample_one_per_slice():
    import random

    from eval.generate_golden import spread_sample

    items = list(range(100))
    picked = spread_sample(items, 10, random.Random(1))
    assert len(picked) == 10
    assert [p // 10 for p in picked] == list(range(10))
    assert picked == spread_sample(items, 10, random.Random(1))
    assert spread_sample([1, 2], 5, random.Random(1)) == [1, 2]
