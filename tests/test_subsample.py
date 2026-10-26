from __future__ import annotations

import numpy as np
import torch

from womd_il.data.subsample import assign_pdms_winners, assign_winners, class_cap_indices
from womd_il.losses.hydra import hydra_imitation_loss


def test_class_cap_keeps_every_class_and_sorts():
    winner = np.array([0, 0, 0, 1, 1, -1, 2], dtype=np.int32)
    idx = class_cap_indices(winner, cap=1, seed=0)
    assert idx.tolist() == sorted(idx.tolist())
    assert set(winner[idx].tolist()) == {0, 1, 2}
    assert len(idx) == 3
    assert np.array_equal(idx, class_cap_indices(winner, cap=1, seed=0))
    wider = class_cap_indices(winner, cap=2, seed=0)
    assert len(wider) == 5
    assert -1 not in winner[wider]
    assert class_cap_indices(winner, cap=10, seed=1).tolist() == [0, 1, 2, 3, 4, 6]


def test_assign_winners_matches_imitation_loss():
    gt = np.zeros((2, 4, 2), np.float32)
    gt[0, :, 0] = np.linspace(1, 4, 4)
    gt[1, :, 0] = np.linspace(1, 4, 4)
    gt[1, :, 1] = 3.0
    valid = np.ones((2, 4), np.float32)
    vocab = np.zeros((3, 4, 2), np.float32)
    vocab[0, :, 0] = gt[0, :, 0]
    vocab[1] = gt[1]
    vocab[2, :, 0] = 10.0
    winners = assign_winners(gt, valid, vocab, batch=2, k_chunk=2)
    assert winners.tolist() == [0, 1]
    with torch.no_grad():
        out = hydra_imitation_loss(
            torch.zeros(1, 3),
            torch.from_numpy(vocab),
            torch.from_numpy(gt[:1]),
            torch.from_numpy(valid[:1]),
            temperature=1.0,
            soft_k=2,
            front_weight=1.0,
            collision_weight=0.0,
        )
    assert int(out["winner"][0]) == 0


def test_assign_pdms_winners_picks_product_argmax():
    labels = np.zeros((3, 4, 5), np.float32)
    labels[:, :, 0] = 1.0
    labels[:, :, 1] = 1.0
    labels[:, :, 2] = 1.0
    labels[:, :, 3] = 1.0
    labels[:, :, 4] = 0.2
    labels[0, 2, 4] = 1.0
    labels[1, :, 1] = 0.0
    labels[2, 0, 4] = 0.8
    winners = assign_pdms_winners(labels, chunk=2)
    assert winners.tolist() == [2, -1, 0]
