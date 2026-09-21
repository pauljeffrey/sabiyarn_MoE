from collections import Counter

import numpy as np
import pytest

from training.data_sampler import EpochStream, MixedBlockSampler, n_blocks_for, read_blocks


def test_every_block_is_used_exactly_once_per_epoch_across_ranks_and_steps():
    n, bs, world = 96, 4, 3  # 96 blocks / (4*3) = 8 steps per epoch
    samplers = [MixedBlockSampler(["a"], [n], seed=1, batch_size=bs, world_size=world, rank=r) for r in range(world)]
    seen = []
    for _ in range(n // (bs * world)):
        for s in samplers:
            _, ids = s.next_batch([1.0])
            seen.extend(ids.tolist())
    assert sorted(seen) == list(range(n))  # each exactly once, none skipped


def test_ranks_get_disjoint_slices_of_the_same_global_batch():
    ids_by_rank = []
    for r in range(4):
        s = MixedBlockSampler(["a"], [1000], seed=7, batch_size=8, world_size=4, rank=r)
        ids_by_rank.append(s.next_batch([1.0])[1])
    flat = np.concatenate(ids_by_rank)
    assert len(set(flat.tolist())) == 32  # 4 ranks x 8, all distinct


def test_epoch_boundary_carries_over_without_loss_or_duplication_and_reshuffles():
    st = EpochStream(10, seed=3, name="x")
    first = st.take(7).tolist()
    second = st.take(7).tolist()  # 3 finish epoch 0, 4 start epoch 1
    epoch0 = first + second[:3]
    assert sorted(epoch0) == list(range(10))
    epoch1_start = second[3:]
    assert st.epoch == 1 and len(set(epoch1_start)) == 4
    # a full second epoch is a different permutation than the first (reshuffled)
    st2 = EpochStream(50, seed=3, name="x")
    e0, e1 = st2.take(50).tolist(), st2.take(50).tolist()
    assert sorted(e0) == sorted(e1) == list(range(50)) and e0 != e1


def test_no_token_position_is_an_input_twice_and_none_is_a_target_twice_within_an_epoch():
    sl, n = 16, 12
    data = np.arange(n * sl + 1, dtype=np.uint16)  # token value == its position
    s = MixedBlockSampler(["a"], [n_blocks_for(len(data), sl)], seed=0, batch_size=3, world_size=1)
    xs, ys = [], []
    for _ in range(n // 3):
        _, ids = s.next_batch([1.0])
        x, y = read_blocks(data, ids, sl)
        xs.append(x.ravel())
        ys.append(y.ravel())
    x_all, y_all = np.concatenate(xs), np.concatenate(ys)
    assert len(set(x_all.tolist())) == len(x_all) == n * sl  # inputs: every position exactly once
    assert len(set(y_all.tolist())) == len(y_all) == n * sl  # targets: every position exactly once
    assert np.all(y_all == x_all + 1)  # y is x shifted by one token


def test_n_blocks_accounts_for_the_extra_target_token():
    assert n_blocks_for(161, 16) == 10 and n_blocks_for(160, 16) == 9 and n_blocks_for(10, 16) == 0
    with pytest.raises(ValueError):
        EpochStream(0, 0, "empty")


def test_state_dict_round_trip_continues_the_exact_same_sequence():
    s1 = MixedBlockSampler(["eng", "afr"], [500, 300], seed=5, batch_size=4, world_size=2, rank=1)
    for _ in range(37):
        s1.next_batch([0.55, 0.45])
    state = s1.state_dict()
    expected = [s1.next_batch([0.55, 0.45]) for _ in range(20)]

    s2 = MixedBlockSampler(["eng", "afr"], [500, 300], seed=5, batch_size=4, world_size=2, rank=1)
    assert s2.load_state_dict(state) == []
    got = [s2.next_batch([0.55, 0.45]) for _ in range(20)]
    assert [(b, i.tolist()) for b, i in got] == [(b, i.tolist()) for b, i in expected]


def test_state_for_a_bin_that_changed_size_is_refused_and_reported():
    s1 = MixedBlockSampler(["a", "b"], [100, 100], seed=0, batch_size=2)
    for _ in range(5):
        s1.next_batch([1, 1])
    s2 = MixedBlockSampler(["a", "b"], [100, 250], seed=0, batch_size=2)
    assert s2.load_state_dict(s1.state_dict()) == ["b"]
    assert s2.streams[1].pos == 0  # untouched


def test_mixing_tracks_the_weights_exactly_and_follows_scheduled_changes():
    s = MixedBlockSampler(["eng", "afr"], [10_000, 10_000], seed=0, batch_size=1)
    picks = Counter(s.next_batch([0.55, 0.45])[0] for _ in range(2000))
    assert abs(picks[0] - 1100) <= 1 and abs(picks[1] - 900) <= 1
    # weights shifting over time (scheduled sampling): the running share follows them
    s = MixedBlockSampler(["eng", "afr"], [10_000, 10_000], seed=0, batch_size=1)
    first = Counter(s.next_batch([0.9, 0.1])[0] for _ in range(1000))
    second = Counter(s.next_batch([0.1, 0.9])[0] for _ in range(1000))
    assert abs(first[0] - 900) <= 2 and abs(second[1] - 900) <= 2


def test_each_bin_keeps_its_own_epoch_independent_of_the_other():
    s = MixedBlockSampler(["small", "big"], [10, 1000], seed=0, batch_size=2)
    for _ in range(40):  # 20 steps of each -> 40 blocks from "small" (4 epochs), 40 from "big"
        s.next_batch([1, 1])
    e = s.epochs_done()
    assert e["small"] == pytest.approx(4.0) and e["big"] == pytest.approx(0.04)


def test_global_order_does_not_depend_on_how_it_is_split_across_ranks():
    one = MixedBlockSampler(["a"], [200], seed=9, batch_size=8, world_size=1)
    two = [MixedBlockSampler(["a"], [200], seed=9, batch_size=4, world_size=2, rank=r) for r in range(2)]
    a = np.concatenate([one.next_batch([1])[1] for _ in range(5)])
    b = np.concatenate([np.concatenate([s.next_batch([1])[1] for s in two]) for _ in range(5)])
    assert a.tolist() == b.tolist()
