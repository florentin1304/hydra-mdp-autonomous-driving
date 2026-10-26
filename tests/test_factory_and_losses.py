from __future__ import annotations

import torch
from omegaconf import OmegaConf

from womd_il import constants as C
from womd_il.losses.gmm import gmm_wta_nll
from womd_il.losses.hydra import hydra_imitation_loss
from womd_il.models.factory import build_policy
from womd_il.utils import load_checkpoint, save_checkpoint


def test_gmm_loss_finite():
    b, k, t = 2, 5, 10
    mu = torch.zeros(b, k, t, 2)
    mu[:, 2, :, 0] = torch.linspace(0.1, 1.0, t)
    log_std = torch.zeros(b, k, t, 2)
    logits = torch.zeros(b, k)
    gt = mu[:, 2]
    valid = torch.ones(b, t)
    stats = gmm_wta_nll(mu, log_std, logits, gt, valid)
    assert stats["winner"].tolist() == [2, 2]
    assert torch.isfinite(stats["loss"])


def test_gmm_winner_avoids_logged_overlap():
    b, k, t = 1, 2, 8
    gt = torch.zeros(b, t, 2)
    gt[0, :, 0] = torch.linspace(0.1, 0.8, t)
    mu = torch.zeros(b, k, t, 2)
    mu[0, 0] = gt[0]
    mu[0, 1, :, 0] = gt[0, :, 0]
    mu[0, 1, :, 1] = 6.0
    log_std = torch.zeros(b, k, t, 2)
    logits = torch.zeros(b, k)
    other = gt[:, None].clone()
    other_valid = torch.ones(b, 1, t)
    ego_lw = torch.tensor([[4.0, 2.0]])
    other_lw = torch.tensor([[[4.0, 2.0]]])
    kwargs = dict(
        mu=mu,
        log_std=log_std,
        logits=logits,
        gt_xy=gt,
        gt_valid=torch.ones(b, t),
        other_xy=other,
        other_valid=other_valid,
        ego_lw=ego_lw,
        other_lw=other_lw,
    )
    assert gmm_wta_nll(**kwargs, collision_weight=0.0)["winner"].item() == 0
    assert gmm_wta_nll(**kwargs, collision_weight=8.0)["winner"].item() == 1


def test_hydra_soft_target_downweights_overlap():
    t = 8
    gt = torch.zeros(1, t, 2)
    gt[0, :, 0] = torch.linspace(0.1, 0.8, t)
    vocab = torch.zeros(4, t, 2)
    vocab[0] = gt[0]
    vocab[1, :, 0] = gt[0, :, 0]
    vocab[1, :, 1] = 6.0
    vocab[2, :, 0] = 20.0
    vocab[3, :, 0] = 25.0
    logits = torch.zeros(1, 4)
    other = gt[:, None].clone()
    stats0 = hydra_imitation_loss(
        logits, vocab, gt, torch.ones(1, t), soft_k=2, collision_weight=0.0
    )
    stats1 = hydra_imitation_loss(
        logits,
        vocab,
        gt,
        torch.ones(1, t),
        soft_k=2,
        collision_weight=8.0,
        other_xy=other,
        other_valid=torch.ones(1, 1, t),
        ego_lw=torch.tensor([[4.0, 2.0]]),
        other_lw=torch.tensor([[[4.0, 2.0]]]),
    )
    assert stats0["winner"].item() == 0
    assert stats1["winner"].item() == 1
    vocab_b = vocab.unsqueeze(0)
    stats_b = hydra_imitation_loss(logits, vocab_b, gt, torch.ones(1, t), soft_k=2)
    assert stats_b["winner"].item() == stats0["winner"].item()
    assert torch.allclose(stats_b["loss"], stats0["loss"])


def test_hydra_flatten_l2_ignores_heading():
    t = 8
    gt = torch.zeros(1, t, 2)
    gt[0, :, 0] = torch.linspace(0.1, 0.8, t)
    vocab = torch.zeros(3, t, 3)
    vocab[0, :, :2] = gt[0]
    vocab[0, :, 2] = 2.5
    vocab[1, :, 0] = 20.0
    vocab[2, :, 0] = 30.0
    stats = hydra_imitation_loss(
        torch.zeros(1, 3),
        vocab,
        gt,
        torch.ones(1, t),
        temperature=1.0,
        soft_k=None,
        distance="flatten_l2",
        gt_yaw=torch.zeros(1, t),
    )
    assert int(stats["winner"].item()) == 0


def test_hydra_sse_full_chunk_includes_heading():
    """Hydra-MDP Eq. 9 / LDiP Eq. 5: full (x,y,heading) chunk, σ=0.5."""
    t = 80
    gt_xy = torch.zeros(1, t, 2)
    gt_xy[0, :, 0] = torch.linspace(0.1, 8.0, t)
    gt_yaw = torch.zeros(1, t)
    vocab = torch.zeros(3, t, 3)
    vocab[0, :, :2] = gt_xy[0]
    vocab[1, :, 0] = gt_xy[0, :, 0]
    vocab[1, :, 2] = 1.5  # same xy, wrong heading — must lose
    vocab[2, :, 0] = 20.0
    logits = torch.zeros(1, 3)
    stats = hydra_imitation_loss(
        logits,
        vocab,
        gt_xy,
        torch.ones(1, t),
        temperature=0.5,
        soft_k=None,
        distance="sse",
        gt_yaw=gt_yaw,
        imi_times=None,
    )
    assert stats["winner"].item() == 0


def test_hydra_sse_target_is_paper_softmax_over_full_set():
    """Hydra-MDP Eq. 9 / LDiP Eq. 5: mass on nearest under ||v-a||², not ADE top-k."""
    t = 8
    gt = torch.zeros(1, t, 2)
    gt[0, :, 0] = torch.linspace(0.1, 0.8, t)
    vocab = torch.zeros(5, t, 2)
    vocab[0] = gt[0]
    vocab[1, :, 0] = gt[0, :, 0] + 2.0
    vocab[2, :, 0] = 5.0
    vocab[3, :, 0] = 10.0
    vocab[4, :, 0] = 20.0
    logits = torch.zeros(1, 5)
    stats = hydra_imitation_loss(
        logits, vocab, gt, torch.ones(1, t), temperature=1.0, soft_k=None, distance="sse"
    )
    assert stats["winner"].item() == 0
    sse = (gt.unsqueeze(1) - vocab.unsqueeze(0)).pow(2).sum(dim=(-1, -2))
    target = torch.softmax(-sse, dim=-1)
    assert int(target[0].argmax().item()) == 0
    assert float(target[0, 0]) > 0.99


def test_hydra_pdms_scale_is_continuous_not_a_binary_gate():
    from womd_il.losses.hydra import hydra_imitation_loss

    t = 8
    gt = torch.zeros(1, t, 2)
    gt[0, :, 0] = torch.linspace(0.1, 0.8, t)
    vocab = torch.zeros(3, t, 2)
    vocab[0] = gt[0]
    vocab[1] = gt[0]
    vocab[2, :, 0] = 20.0
    logits = torch.zeros(1, 3)
    valid = torch.ones(1, t)
    # Same ADE, PDMS 0.2 vs 0.8 → winner is the higher PDMS, not a >0 mask.
    mid = hydra_imitation_loss(
        logits, vocab, gt, valid, soft_k=None, pdms=torch.tensor([[0.2, 0.8, 1.0]]), pdms_scale=1.0
    )
    assert mid["winner"].item() == 1
    # Vanilla ADE still picks the copy (index 0) when scale is off.
    vocab_far = vocab.clone()
    vocab_far[1, :, 1] = 6.0
    plain = hydra_imitation_loss(
        logits, vocab_far, gt, valid, soft_k=None, pdms=torch.tensor([[0.0, 1.0, 1.0]]), pdms_scale=0.0
    )
    scaled = hydra_imitation_loss(
        logits, vocab_far, gt, valid, soft_k=None, pdms=torch.tensor([[0.0, 1.0, 1.0]]), pdms_scale=1.0
    )
    assert plain["winner"].item() == 0
    assert scaled["winner"].item() == 1


def test_hydra_imitation_loss_is_kl_not_raw_ce():
    """With a diffuse soft target, CE ≈ H(q); logged loss should be KL ≈ 0 when p=q."""
    t = 4
    gt = torch.zeros(1, t, 2)
    gt[0, :, 0] = torch.linspace(0.1, 0.4, t)
    vocab = torch.zeros(8, t, 3)
    for i in range(8):
        vocab[i, :, 0] = gt[0, :, 0] + 0.05 * i
    # Build the soft target rank with zero logits, then score with matching logits.
    stats_ref = hydra_imitation_loss(
        torch.zeros(1, 8), vocab, gt, torch.ones(1, t), soft_k=None, temperature=1.0
    )
    # Recover target via ADE ranking: logits = -ade/tau matches softmax(rank).
    ade = stats_ref["ade"]
    logits_match = -ade
    stats = hydra_imitation_loss(
        logits_match, vocab, gt, torch.ones(1, t), soft_k=None, temperature=1.0
    )
    assert float(stats["loss"]) < 1e-4
    assert float(stats["loss_im_ce"]) > float(stats["loss"]) + 0.5
    assert float(stats["loss_im_H"]) > 0.5


def test_hydra_metric_bce_balances_rare_negatives():
    from womd_il.losses.hydra import hydra_metric_bce

    # Majority-safe labels: rare failures (y=0) must get upweighted, not y=1.
    logits = torch.zeros(1, 100, 1)
    labels = torch.ones(1, 100, 1)
    labels[0, 0, 0] = 0.0
    plain = hydra_metric_bce(logits, labels, class_weighted=False)
    weighted = hydra_metric_bce(logits, labels, class_weighted=True)
    assert weighted.item() > plain.item()


def test_hydra_metric_bce_uses_external_pos_rate():
    from womd_il.losses.hydra import hydra_metric_bce

    logits = torch.zeros(1, 10, 1)
    labels = torch.ones(1, 10, 1)
    labels[0, 0, 0] = 0.0
    # Force a rarer-neg prior than the batch → larger weight on the single 0.
    batch = hydra_metric_bce(logits, labels, class_weighted=True)
    prior = hydra_metric_bce(
        logits, labels, class_weighted=True, pos_rate=torch.tensor([0.99])
    )
    assert prior.item() > batch.item()


def test_hydra_pdms_loss_ranks_high_score():
    from womd_il.losses.hydra import hydra_pdms_loss

    logits = torch.zeros(1, 4)
    pdms = torch.tensor([[0.0, 0.1, 0.8, 0.2]])
    stats = hydra_pdms_loss(logits, pdms, temperature=0.1, soft_k=None, bce_weight=1.0)
    assert stats["winner"].item() == 2
    assert torch.isfinite(stats["loss"])
    logits_ok = torch.tensor([[0.0, 0.0, 4.0, 0.0]])
    logits_bad = torch.tensor([[4.0, 0.0, 0.0, 0.0]])
    good = hydra_pdms_loss(logits_ok, pdms, soft_k=None)
    bad = hydra_pdms_loss(logits_bad, pdms, soft_k=None)
    assert good["loss"].item() < bad["loss"].item()


def test_hydra_epo_loss_prefers_high_pdms():
    import math

    from womd_il.losses.hydra import hydra_epo_loss

    pdms = torch.tensor([[0.0, 0.1, 0.9, 0.2]])
    good = hydra_epo_loss(torch.tensor([[0.0, 0.0, 4.0, 0.0]]), pdms, entropy_coef=0.0)
    bad = hydra_epo_loss(torch.tensor([[4.0, 0.0, 0.0, 0.0]]), pdms, entropy_coef=0.0)
    assert good["winner"].item() == 2
    assert good["epo_regret"].item() < bad["epo_regret"].item()
    assert good["epo_adv"].item() > bad["epo_adv"].item()
    # Normalized regret ∈ [0, 1]; uniform π → high regret, peaked on best → low.
    flat = hydra_epo_loss(torch.zeros(1, 4), pdms, entropy_coef=0.0)
    assert abs(flat["epo_adv"].item()) < 0.05
    assert 0.0 <= flat["epo_regret"].item() <= 1.0 + 1e-5
    assert flat["epo_regret"].item() > good["epo_regret"].item()
    # Collapse-only squared entropy: pen iff H < ln(min_k).
    K = 64
    soft = hydra_epo_loss(
        torch.zeros(1, K), torch.rand(1, K), entropy_coef=1.0, entropy_min_k=256.0
    )
    # Uniform over 64 < floor 256 → penalty > 0
    assert soft["epo_ent_pen"].item() > 0.0
    sharp = torch.zeros(1, K)
    sharp[0, 0] = 20.0
    pen_sharp = hydra_epo_loss(
        sharp, torch.rand(1, K), entropy_coef=1.0, entropy_min_k=256.0
    )["epo_ent_pen"].item()
    assert pen_sharp > soft["epo_ent_pen"].item()
    # Uniform over 512 >= floor → zero pen
    wide = hydra_epo_loss(
        torch.zeros(1, 512), torch.rand(1, 512), entropy_coef=1.0, entropy_min_k=256.0
    )
    assert wide["epo_ent_pen"].item() == 0.0
    assert abs(wide["epo_entropy"].item() - math.log(512)) < 1e-4
    logits = torch.tensor([[8.0, 0.0, 0.0, 0.0]], requires_grad=True)
    hydra_epo_loss(
        logits, pdms, entropy_coef=0.2, temperature=1.0, entropy_min_k=256.0
    )["loss"].backward()
    assert logits.grad is not None and float(logits.grad.norm()) > 1e-4


def test_pdms_product_zeros_on_nc_or_dac():
    from womd_il.losses.hydra import pdms_from_probs

    p = torch.ones(1, 3, 5)
    p[0, 0, 0] = 0.0
    p[0, 1, 1] = 0.0
    p[0, 2, 2] = 0.2
    s = pdms_from_probs(p)
    assert s[0, 0].item() == 0.0
    assert s[0, 1].item() == 0.0
    assert 0.0 < s[0, 2].item() < 1.0
    all_ones = pdms_from_probs(torch.ones(5))
    assert abs(all_ones.item() - 1.0) < 1e-6
    p8 = torch.ones(8)
    assert abs(pdms_from_probs(p8).item() - 1.0) < 1e-6
    p8[2] = 0.0  # DDC
    assert pdms_from_probs(p8).item() == 0.0


def test_hydra_metric_labels_nc_and_progress():
    from womd_il.losses.hydra import METRIC_INDEX, METRIC_NAMES, hydra_metric_labels

    t = 40
    gt = torch.zeros(1, t, 2)
    gt[0, :, 0] = torch.linspace(0.1, 8.0, t)
    vocab = torch.zeros(1, 2, t, 2)
    vocab[0, 0] = gt[0]
    vocab[0, 1, :, 0] = gt[0, :, 0]
    vocab[0, 1, :, 1] = 8.0  # > _EP_ROUTE_LAT_M → progress 0
    other = gt[:, None].clone()
    route = torch.zeros(1, 16, 2)
    route[0, :, 0] = torch.linspace(0.0, 20.0, 16)
    batch = {
        "gt_xy": gt,
        "other_fut_xy": other,
        "other_fut_valid": torch.ones(1, 1, t),
        "ego_lw": torch.tensor([[4.0, 2.0]]),
        "other_lw": torch.tensor([[[4.0, 2.0]]]),
        "sdc_route": route,
        "sdc_route_mask": torch.ones(1, 16),
    }
    y = hydra_metric_labels(vocab, batch)
    assert y.shape == (1, 2, len(METRIC_NAMES))
    # mode 0 copies GT into the neighbor → collision; mode 1 is offset.
    assert y[0, 0, METRIC_INDEX["nc"]] < 0.5
    assert y[0, 1, METRIC_INDEX["nc"]] > 0.5
    assert y[0, 0, METRIC_INDEX["progress"]] > 0.5
    assert y[0, 1, METRIC_INDEX["progress"]] < 0.1


def test_hydra_metric_box_nc_ignores_rear_and_ttc_is_cv_probe():
    from womd_il.losses.hydra import hydra_metric_labels

    t = 40
    steps = torch.arange(1, t + 1, dtype=torch.float32)
    vocab = torch.zeros(1, 3, t, 2)
    vocab[0, :, :, 0] = steps * 1.0  # 10 m/s along +x
    yaw = torch.zeros(1, 3, t)
    other = torch.zeros(1, 1, t, 2)
    other[0, 0, :, 0] = 8.0  # parked ahead — CV 0.9 s from x=1 reaches x=10
    other_behind = other.clone()
    other_behind[0, 0, :, 0] = -2.0
    other_far = other.clone()
    other_far[0, 0, :, 0] = 80.0
    valid = torch.ones(1, 1, t)
    lw_e = torch.tensor([[4.0, 2.0]])
    lw_o = torch.tensor([[[4.0, 2.0]]])
    gt = vocab[0, 0][None]
    base = {"gt_xy": gt, "other_fut_valid": valid, "ego_lw": lw_e, "other_lw": lw_o}
    y_front = hydra_metric_labels(vocab, {**base, "other_fut_xy": other}, pred_yaw=yaw)
    y_rear = hydra_metric_labels(vocab, {**base, "other_fut_xy": other_behind}, pred_yaw=yaw)
    y_far = hydra_metric_labels(vocab, {**base, "other_fut_xy": other_far}, pred_yaw=yaw)
    from womd_il.losses.hydra import METRIC_INDEX

    assert y_front[0, 0, METRIC_INDEX["nc"]] < 0.5  # NC fail: at-fault overlap ahead
    assert y_front[0, 0, METRIC_INDEX["ttc"]] < 0.5  # TTC fail: 0.9 s CV hits the parked box
    assert y_rear[0, 0, METRIC_INDEX["nc"]] > 0.5  # NC ok: rear-end is not at-fault
    assert y_far[0, 0, METRIC_INDEX["nc"]] > 0.5 and y_far[0, 0, METRIC_INDEX["ttc"]] > 0.5


def test_hydra_metric_labels_skip_large_k_unless_forced():
    from womd_il.losses.hydra import hydra_metric_labels

    t = 8
    vocab = torch.zeros(1, 300, t, 2)
    vocab[0, :, :, 0] = torch.linspace(0.1, 1.0, t)
    other = vocab[:, :1]
    batch = {
        "gt_xy": vocab[:, 0],
        "other_fut_xy": other,
        "other_fut_valid": torch.ones(1, 1, t),
        "ego_lw": torch.tensor([[4.0, 2.0]]),
        "other_lw": torch.tensor([[[4.0, 2.0]]]),
    }
    skipped = hydra_metric_labels(vocab, batch)
    assert skipped[0, :, 0].min() > 0.5
    forced = hydra_metric_labels(vocab, batch, max_k=None)
    assert forced[0, :, 0].max() < 0.5


def _straight_lane_map(*, edge_y: float = 20.0):
    m = 20
    lane = torch.zeros(1, 1, m, 2)
    lane[0, 0, :, 0] = torch.linspace(0.0, 100.0, m)
    edge = lane.clone()
    edge[0, 0, :, 1] = edge_y
    map_xy = torch.cat([lane, edge], dim=1)
    map_pt_valid = torch.ones(1, 2, m)
    map_type = torch.tensor([[C.MAP_LANE_FREEWAY, C.MAP_ROAD_EDGE]], dtype=torch.int32)
    return {"map_xy": map_xy, "map_pt_valid": map_pt_valid, "map_type": map_type}


def test_dac_ok_when_staying_on_lane():
    from womd_il.losses.hydra import METRIC_INDEX, hydra_metric_labels

    t = 40
    vocab = torch.zeros(1, 1, t, 2)
    vocab[0, 0, :, 0] = torch.linspace(0.5, 8.0, t)
    batch = {"gt_xy": vocab[0, 0][None], **_straight_lane_map()}
    y = hydra_metric_labels(vocab, batch)
    assert y[0, 0, METRIC_INDEX["dac"]] > 0.5
    assert y[0, 0, METRIC_INDEX["lk"]] > 0.5
    assert y[0, 0, METRIC_INDEX["ddc"]] > 0.9


def test_dac_fails_when_crossing_road_edge():
    from womd_il.losses.hydra import METRIC_INDEX, hydra_metric_labels

    t = 40
    # Edge runs along x at y=5; plan crosses it from y=0 → y=10.
    vocab = torch.zeros(1, 1, t, 2)
    vocab[0, 0, :, 0] = torch.linspace(0.5, 8.0, t)
    vocab[0, 0, :, 1] = torch.linspace(0.0, 10.0, t)
    batch = {"gt_xy": vocab[0, 0][None], **_straight_lane_map(edge_y=5.0)}
    y = hydra_metric_labels(vocab, batch)
    assert y[0, 0, METRIC_INDEX["dac"]] < 0.5


def test_dac_ok_when_near_edge_but_not_crossing():
    from womd_il.losses.hydra import METRIC_INDEX, hydra_metric_labels

    t = 40
    vocab = torch.zeros(1, 1, t, 2)
    vocab[0, 0, :, 0] = torch.linspace(0.5, 8.0, t)
    vocab[0, 0, :, 1] = 3.0  # parallel to edge at y=5, within lane corridor
    batch = {"gt_xy": vocab[0, 0][None], **_straight_lane_map(edge_y=5.0)}
    y = hydra_metric_labels(vocab, batch)
    assert y[0, 0, METRIC_INDEX["dac"]] > 0.5


def test_dac_fails_when_far_from_lane_even_without_edge_cross():
    from womd_il.losses.hydra import METRIC_INDEX, hydra_metric_labels

    t = 40
    # Lane on y=0, edge far at y=40; plan at y=12 never crosses edge but leaves road.
    vocab = torch.zeros(1, 1, t, 2)
    vocab[0, 0, :, 0] = torch.linspace(0.5, 8.0, t)
    vocab[0, 0, :, 1] = 12.0
    batch = {"gt_xy": vocab[0, 0][None], **_straight_lane_map(edge_y=40.0)}
    y = hydra_metric_labels(vocab, batch)
    assert y[0, 0, METRIC_INDEX["dac"]] < 0.5


def test_progress_prefers_on_route_turn_over_straight():
    from womd_il.losses.hydra import METRIC_INDEX, hydra_metric_labels

    t = 40
    # Route goes +x then +y (left turn). GT follows route; straight mode same |disp|.
    route = torch.zeros(1, 21, 2)
    route[0, :11, 0] = torch.linspace(0.0, 20.0, 11)
    route[0, 10:, 0] = 20.0
    route[0, 10:, 1] = torch.linspace(0.0, 20.0, 11)
    route_mask = torch.ones(1, 21)
    gt = torch.zeros(1, t, 2)
    gt[0, :, 0] = torch.linspace(0.5, 20.0, t)
    gt[0, :, 1] = torch.linspace(0.0, 18.0, t)
    vocab = torch.zeros(1, 2, t, 2)
    vocab[0, 0] = gt[0]  # on-route turn
    vocab[0, 1, :, 0] = torch.linspace(0.5, 27.0, t)  # straight, similar length
    batch = {
        "gt_xy": gt,
        "sdc_route": route,
        "sdc_route_mask": route_mask,
        **_straight_lane_map(edge_y=40.0),
    }
    y = hydra_metric_labels(vocab, batch)
    assert y[0, 0, METRIC_INDEX["progress"]] > y[0, 1, METRIC_INDEX["progress"]] + 0.2


def test_ddc_penalizes_oncoming_travel():
    from womd_il.losses.hydra import METRIC_INDEX, hydra_metric_labels

    t = 40
    vocab = torch.zeros(1, 1, t, 2)
    vocab[0, 0, :, 0] = torch.linspace(40.0, 0.5, t)  # ~10 m/s against the lane
    yaw = torch.full((1, 1, t), 3.1416)
    batch = {"gt_xy": vocab[0, 0][None], **_straight_lane_map()}
    y = hydra_metric_labels(vocab, batch, pred_yaw=yaw)
    assert y[0, 0, METRIC_INDEX["ddc"]] < 0.25


def test_tlc_uses_future_red_not_just_t0():
    from womd_il.losses.hydra import METRIC_INDEX, hydra_metric_labels

    t = 40
    vocab = torch.zeros(1, 1, t, 2)
    vocab[0, 0, :, 0] = torch.linspace(1.0, 20.0, t)  # drives past stop at x=10
    # t=0 is green; later turns red — must still fail once red while past the line.
    tl_fut_state = torch.full((1, 1, t), C.TL_GO, dtype=torch.int32)
    tl_fut_state[0, 0, 10:] = C.TL_STOP
    batch = {
        "gt_xy": vocab[0, 0][None],
        "tl_xy": torch.tensor([[[10.0, 0.0]]]),
        "tl_state": torch.tensor([[C.TL_GO]], dtype=torch.int32),
        "tl_valid": torch.ones(1, 1),
        "tl_fut_state": tl_fut_state,
        "tl_fut_valid": torch.ones(1, 1, t),
    }
    y = hydra_metric_labels(vocab, batch)
    assert y[0, 0, METRIC_INDEX["tlc"]] < 0.5


def test_lk_fails_after_two_seconds_off_center():
    from womd_il.losses.hydra import METRIC_INDEX, hydra_metric_labels

    t = 40
    vocab = torch.zeros(1, 1, t, 2)
    vocab[0, 0, :, 0] = torch.linspace(0.5, 8.0, t)
    vocab[0, 0, :, 1] = 2.0
    batch = {"gt_xy": vocab[0, 0][None], **_straight_lane_map()}
    y = hydra_metric_labels(vocab, batch)
    assert y[0, 0, METRIC_INDEX["lk"]] < 0.5


def test_gather_metric_labels_uses_cache():
    from womd_il.losses.hydra import gather_or_compute_metric_labels

    vocab = torch.zeros(2, 3, 4, 2)
    cached = torch.arange(2 * 8 * 5, dtype=torch.float32).reshape(2, 8, 5)
    idx = torch.tensor([[7, 1, 0], [2, 2, 5]])
    batch = {"metric_labels": cached, "gt_xy": torch.ones(2, 4, 2)}
    y = gather_or_compute_metric_labels(vocab, batch, idx)
    assert torch.equal(y[0, 0], cached[0, 7])
    assert torch.equal(y[1, 2], cached[1, 5])


def test_format_metric_table_and_eval():
    from womd_il.losses.hydra import METRIC_NAMES, format_metric_table, hydra_metric_eval

    mdim = len(METRIC_NAMES)
    logits = torch.zeros(1, 2, mdim)
    labels = torch.zeros(1, 2, mdim)
    logits[0, 0] = 4.0
    logits[0, 0, 0] = -4.0
    logits[0, 0, -1] = -4.0
    labels[0, 0] = 1.0
    labels[0, 0, 0] = 0.0
    labels[0, 0, -1] = 0.0
    logits[0, 1] = 4.0
    labels[0, 1] = 1.0
    m = hydra_metric_eval(logits, labels)
    assert m["metric/nc/acc"] == 1.0
    assert m["metric/progress/acc"] == 1.0
    assert "metric/macro_f1" in m
    assert "metric/macro_mcc" in m
    assert "metric/nc/mcc" in m
    table = format_metric_table(m, epoch=3, lr=1e-5)
    assert "nc" in table and "BCE" in table and "F1" in table and "MCC" in table
    assert "epoch=3" in table


def test_factory_dummy_and_ckpt(tmp_path):
    model = OmegaConf.create(
        dict(
            name="wayformer",
            d_model=32,
            d_ffn=64,
            nhead=4,
            dropout=0.0,
            num_latents=8,
            encoder_layers=1,
            decoder_layers=1,
            hist_in_dim=C.HIST_IN_DIM,
            map_in_dim=C.MAP_IN_DIM,
            tl_in_dim=C.TL_IN_DIM,
            ego_in_dim=C.EGO_IN_DIM,
        )
    )
    dec = OmegaConf.create(dict(name="dummy", future_steps=80))
    policy = build_policy(model, dec)
    path = tmp_path / "p.pt"
    cfg = OmegaConf.create({"model": model, "decoder": dec})
    save_checkpoint(path, policy, cfg)
    policy2 = build_policy(model, dec)
    load_checkpoint(path, policy2)
    assert policy2.decoder_name == "dummy"


def test_load_pads_old_map_proj(tmp_path):
    old = OmegaConf.create(
        dict(
            name="wayformer",
            d_model=32,
            d_ffn=64,
            nhead=4,
            dropout=0.0,
            num_latents=8,
            encoder_layers=1,
            decoder_layers=1,
            hist_in_dim=C.HIST_IN_DIM,
            map_in_dim=20,
            tl_in_dim=C.TL_IN_DIM,
            ego_in_dim=C.EGO_IN_DIM,
        )
    )
    new = OmegaConf.create(dict(old))
    new.map_in_dim = C.MAP_IN_DIM
    dec = OmegaConf.create(dict(name="dummy", future_steps=80))
    old_policy = build_policy(old, dec)
    path = tmp_path / "old.pt"
    save_checkpoint(path, old_policy, OmegaConf.create({"model": old, "decoder": dec}))
    new_policy = build_policy(new, dec)
    load_checkpoint(path, new_policy)
    w = new_policy.encoder.map_proj.net[0].weight
    assert w.shape[1] == C.MAP_IN_DIM
    assert torch.count_nonzero(w[:, -1]) == 0
