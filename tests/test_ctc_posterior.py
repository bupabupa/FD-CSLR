import torch

from modules.prototype import (
    CTCPrototypeHead,
    _can_skip,
    _extend_targets,
    collect_batch_posteriors,
    ctc_posterior,
    ctc_viterbi_path,
)


def _brute_force_posterior(log_probs, target, blank_id=0):
    time_steps, num_classes = log_probs.shape
    if target.numel() == 0:
        expected = torch.zeros_like(log_probs)
        expected[:, blank_id] = 1.0
        return expected

    extended = _extend_targets(target, blank_id=blank_id)
    states = extended.numel()
    paths = []

    def visit(step, state, path):
        if step == time_steps - 1:
            if state == states - 1 or (
                state == states - 2 and extended[-2] != blank_id
            ):
                paths.append(path)
            return
        next_state = step + 1
        for candidate in (state, state + 1, state + 2):
            if candidate >= states:
                continue
            if candidate == state + 2 and not _can_skip(extended, state, blank_id):
                continue
            visit(next_state, candidate, path + [candidate])

    for start in (0, 1):
        if start < states:
            visit(0, start, [start])

    probabilities = []
    for path in paths:
        log_probability = log_probs[
            torch.arange(time_steps, device=log_probs.device),
            extended[torch.tensor(path, device=log_probs.device)],
        ].sum()
        probabilities.append(log_probability.exp())

    if not probabilities:
        return torch.zeros_like(log_probs)

    probabilities = torch.stack(probabilities)
    occupancy = torch.zeros_like(log_probs)
    for path, probability in zip(paths, probabilities):
        for step, state in enumerate(path):
            occupancy[step, extended[state]] += probability
    return occupancy / probabilities.sum()


def _brute_force_best_path_score(log_probs, target, blank_id=0):
    time_steps = log_probs.size(0)
    extended = _extend_targets(target, blank_id=blank_id)
    states = extended.numel()
    paths = []

    def visit(step, state, path):
        if step == time_steps - 1:
            if state == states - 1 or (
                state == states - 2 and extended[-2] != blank_id
            ):
                paths.append(path)
            return
        for candidate in (state, state + 1, state + 2):
            if candidate >= states:
                continue
            if candidate == state + 2 and not _can_skip(
                extended, state, blank_id
            ):
                continue
            visit(step + 1, candidate, path + [candidate])

    for start in (0, 1):
        if start < states:
            visit(0, start, [start])
    scores = []
    for path in paths:
        state_path = torch.tensor(path, device=log_probs.device)
        scores.append(
            log_probs[
                torch.arange(time_steps, device=log_probs.device),
                extended[state_path],
            ].sum()
        )
    return torch.stack(scores).max()


def _check_posterior(target, time_steps=5, num_classes=4, blank_id=0):
    torch.manual_seed(20261114 + target.numel() + int(target.sum()))
    logits = torch.randn(time_steps, num_classes)
    log_probs = logits.log_softmax(dim=-1)
    actual = ctc_posterior(log_probs, target, blank_id=blank_id)
    expected = _brute_force_posterior(log_probs, target, blank_id=blank_id)
    try:
        torch.testing.assert_close(actual, expected, atol=1e-6, rtol=1e-6)
    except AssertionError:
        print(f"target={target.tolist()}")
        print(f"actual={actual}")
        print(f"expected={expected}")
        print(f"actual sums={actual.sum(dim=-1)}")
        print(f"expected sums={expected.sum(dim=-1)}")
        raise
    torch.testing.assert_close(
        actual.sum(dim=-1), torch.ones(time_steps), atol=1e-6, rtol=1e-6
    )
    return actual


def test_ctc_posterior_matches_brute_force():
    for target in (
        torch.tensor([], dtype=torch.long),
        torch.tensor([1], dtype=torch.long),
        torch.tensor([1, 2], dtype=torch.long),
        torch.tensor([1, 1], dtype=torch.long),
        torch.tensor([1, 1, 2], dtype=torch.long),
    ):
        _check_posterior(target)
    _check_posterior(torch.tensor([1, 2, 3, 1, 2], dtype=torch.long))


def test_collect_batch_posteriors_respects_feature_lengths():
    torch.manual_seed(7)
    time_steps = 6
    num_classes = 5
    logits = torch.randn(time_steps, 2, num_classes)
    labels = torch.tensor([1, 2, 3], dtype=torch.long)
    label_lengths = torch.tensor([2, 1], dtype=torch.long)
    feature_lengths = torch.tensor([6, 3], dtype=torch.long)
    gammas, confidences = collect_batch_posteriors(
        logits,
        labels,
        label_lengths,
        feature_lengths=feature_lengths,
        blank_id=0,
    )

    assert gammas.shape == (time_steps, 2, num_classes)
    expected_1 = ctc_posterior(logits[:3, 1].log_softmax(-1), labels[2:3])
    torch.testing.assert_close(gammas[:3, 1], expected_1, atol=1e-6, rtol=1e-6)
    expected_margin = expected_1.topk(2, dim=-1).values
    torch.testing.assert_close(
        confidences[:3, 1],
        expected_margin[:, 0] - expected_margin[:, 1],
        atol=1e-6,
        rtol=1e-6,
    )
    assert torch.count_nonzero(gammas[3:, 1]) == 0
    assert torch.count_nonzero(confidences[3:, 1]) == 0


def test_collect_batch_posteriors_skips_impossible_samples():
    logits = torch.randn(3, 2, 5)
    labels = torch.tensor([1, 2, 2, 2], dtype=torch.long)
    label_lengths = torch.tensor([1, 3], dtype=torch.long)
    feature_lengths = torch.tensor([3, 3], dtype=torch.long)

    gammas, confidences = collect_batch_posteriors(
        logits,
        labels,
        label_lengths,
        feature_lengths=feature_lengths,
        blank_id=0,
    )

    assert torch.count_nonzero(gammas[:, 1]) == 0
    assert torch.count_nonzero(confidences[:, 1]) == 0
    assert torch.allclose(
        gammas[:, 0].sum(dim=-1),
        torch.ones(3),
        atol=1e-6,
        rtol=1e-6,
    )


def test_prototype_update_respects_confidence_gate():
    prototypes = CTCPrototypeHead(
        num_classes=3,
        feature_dim=2,
        confidence=0.5,
        min_weight=0.05,
    )
    features = torch.tensor(
        [
            [[1.0, 0.0], [8.0, 8.0]],
            [[0.0, 1.0], [9.0, 9.0]],
        ]
    )
    gamma = torch.zeros(2, 2, 3)
    gamma[:, 0, 1] = 1.0
    gamma[:, 1, 2] = 1.0
    confidence = torch.tensor([[0.9, 0.49], [0.8, 0.1]])

    prototypes.update(features, gamma, confidence)

    torch.testing.assert_close(
        prototypes.prototypes[1],
        torch.tensor([0.9 / 1.7, 0.8 / 1.7]),
    )
    torch.testing.assert_close(prototypes.prototypes[2], torch.zeros(2))
    torch.testing.assert_close(
        prototypes.support,
        torch.tensor([0.0, 1.7, 0.0]),
    )


def test_gaussian_prototype_scores_supported_classes():
    prototypes = CTCPrototypeHead(
        num_classes=3,
        feature_dim=2,
        confidence=0.5,
        min_weight=0.05,
        distribution="gaussian",
        variance_shrinkage=0.5,
    )
    features = torch.tensor([[[1.0, 0.0]]])
    gamma = torch.tensor([[[0.0, 1.0, 0.0]]])
    confidence = torch.tensor([[0.9]])

    prototypes.update(features, gamma, confidence)
    logits = prototypes(
        torch.tensor([[[1.0, 0.0]], [[-1.0, 0.0]]])
    )

    assert torch.isfinite(logits).all()
    assert logits[0, 0, 1] > logits[1, 0, 1]
    assert logits[0, 0, 2].item() == 0
    assert (prototypes.variances >= prototypes.variance_floor).all()


def test_gaussian_prototype_score_matches_diagonal_formula():
    prototypes = CTCPrototypeHead(
        num_classes=3,
        feature_dim=2,
        confidence=0.0,
        min_weight=0.0,
        distribution="gaussian",
    )
    prototypes.prototypes.copy_(torch.tensor(
        [[0.0, 0.0], [0.0, 0.0], [2.0, 0.0]]
    ))
    prototypes.variances.copy_(torch.tensor(
        [[1.0, 1.0], [1.0, 1.0], [4.0, 1.0]]
    ))
    prototypes.support.fill_(1.0)

    logits = prototypes(torch.tensor([[[1.0, 1.0]]]))
    expected = torch.tensor(
        [[
            -0.5 * (2.0 / 2.0),
            -0.5 * (2.0 / 2.0),
            -0.5 * ((1.0 / 4.0 + 1.0 + torch.log(torch.tensor(4.0))) / 2.0),
        ]]
    )
    torch.testing.assert_close(logits, expected.unsqueeze(0))


def test_prototype_alignment_masks_unavailable_target_classes():
    prototypes = CTCPrototypeHead(
        num_classes=3,
        feature_dim=2,
        confidence=0.0,
        min_weight=0.05,
    )
    prototypes.support[1] = 1.0
    prototypes.prototypes[1] = torch.tensor([1.0, 0.0])
    logits = torch.tensor([[[1.0, 2.0, 0.0], [1.0, 0.0, 2.0]]])
    gamma = torch.tensor([[[1.0, 0.0, 0.0], [0.0, 0.0, 1.0]]])
    confidence = torch.ones(1, 2)

    loss = prototypes.alignment_loss(logits, gamma, confidence)

    assert torch.isfinite(loss)
    assert loss.item() >= 0
    only_invalid = prototypes.alignment_loss(
        logits[:, 1:2],
        gamma[:, 1:2],
        confidence[:, 1:2],
    )
    assert only_invalid.item() == 0


def test_impossible_ctc_target_is_rejected():
    log_probs = torch.randn(3, 4).log_softmax(dim=-1)
    try:
        ctc_posterior(log_probs, torch.tensor([1, 1, 1]))
    except ValueError:
        return
    raise AssertionError("impossible CTC target should raise ValueError")


def test_ctc_viterbi_path_is_legal():
    torch.manual_seed(19)
    target = torch.tensor([1, 1, 2], dtype=torch.long)
    log_probs = torch.randn(8, 4).log_softmax(dim=-1)
    path = ctc_viterbi_path(log_probs, target).tolist()
    path_score = log_probs[
        torch.arange(log_probs.size(0)), torch.tensor(path)
    ].sum()
    torch.testing.assert_close(
        path_score,
        _brute_force_best_path_score(log_probs, target),
        atol=1e-6,
        rtol=1e-6,
    )
    collapsed = []
    previous = None
    for class_id in path:
        if class_id != 0 and class_id != previous:
            collapsed.append(class_id)
        previous = class_id
    assert collapsed == target.tolist()

    forced = torch.full((4, 3), -20.0)
    forced[0, 0] = 20.0
    forced[1, 1] = 20.0
    forced[2, 0] = 20.0
    forced[3, 0] = 20.0
    forced_path = ctc_viterbi_path(
        forced.log_softmax(dim=-1), torch.tensor([1])
    )
    assert forced_path.tolist() == [0, 1, 0, 0]


if __name__ == "__main__":
    test_ctc_posterior_matches_brute_force()
    test_collect_batch_posteriors_respects_feature_lengths()
    test_collect_batch_posteriors_skips_impossible_samples()
    test_prototype_update_respects_confidence_gate()
    test_gaussian_prototype_scores_supported_classes()
    test_gaussian_prototype_score_matches_diagonal_formula()
    test_prototype_alignment_masks_unavailable_target_classes()
    test_impossible_ctc_target_is_rejected()
    test_ctc_viterbi_path_is_legal()
    print("ctc posterior tests passed")
