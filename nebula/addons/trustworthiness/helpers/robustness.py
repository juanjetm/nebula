import logging
import math

import numpy as np
import torch
import torch.nn.functional as F
from art.estimators.classification import PyTorchClassifier
from art.metrics import clever_u, empirical_robustness, loss_sensitivity
from torch import nn, optim

logger = logging.getLogger(__name__)

R_L2 = 2

def _build_art_classifier(model, input_shape, nb_classes, learning_rate):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), learning_rate)

    return PyTorchClassifier(
        model=model,
        loss=criterion,
        optimizer=optimizer,
        input_shape=tuple(input_shape),
        nb_classes=nb_classes,
    )


def _validate_test_sample_tensors(test_sample):
    if not (isinstance(test_sample, (tuple, list)) and len(test_sample) >= 2):
        raise ValueError("`test_sample` must contain samples and labels.")

    samples, labels = test_sample[0], test_sample[1]
    if not (torch.is_tensor(samples) and torch.is_tensor(labels) and samples.shape[0] > 0):
        raise ValueError("`test_sample` must contain non-empty tensors for samples and labels.")

    return samples, labels


def _coerce_max_samples(max_samples, default=8):
    try:
        return max(1, int(max_samples))
    except Exception:
        return default


def get_clever_score(model, test_sample, nb_classes, learning_rate, max_samples=8):
    """
    Calculates the CLEVER score as the mean score over multiple samples.

    Args:
        model (object): The model.
        test_sample (object): A batch from the test dataloader.
        nb_classes (int): The nb_classes of the model.
        learning_rate (float): The learning rate of the model.
        max_samples (int): Maximum number of samples from the batch to evaluate.

    Returns:
        float: Mean CLEVER score across the selected samples.
    """
    samples, _ = _validate_test_sample_tensors(test_sample)

    input_shape = tuple(samples.shape[1:]) if samples.dim() >= 2 else tuple(samples.shape)

    max_samples = _coerce_max_samples(max_samples)
    n_samples = min(int(samples.shape[0]), max_samples)

    # Create the ART classifier once and reuse it for all selected samples.
    classifier = _build_art_classifier(model, input_shape, nb_classes, learning_rate)

    clever_scores = []
    for idx in range(n_samples):
        background = samples[idx].detach().cpu()
        sample_np = background.numpy()

        try:
            score_untargeted = clever_u(
                classifier,
                sample_np,
                10,
                5,
                R_L2,
                norm=2,
                pool_factor=3,
                verbose=False,
            )
            if score_untargeted is not None and not math.isnan(float(score_untargeted)):
                clever_scores.append(float(score_untargeted))
        except Exception as exc:
            logger.warning("Could not compute CLEVER score for sample index %s", idx)
            logger.warning(exc)

    if not clever_scores:
        return 0.0

    return float(np.mean(clever_scores))

def get_loss_sensitivity_score(model, test_sample, nb_classes, learning_rate, max_samples=8):

    """
    Calculates the loss sensitivity score as the mean score over multiple samples.

    Args:
        model (object): The model.
        test_sample (object): A batch from the test dataloader.
        nb_classes (int): The nb_classes of the model.
        learning_rate (float): The learning rate of the model.
        max_samples (int): Maximum number of samples from the batch to evaluate.

    Returns:
        float: Mean loss sensitivity score across the selected samples.
    """
    samples, labels = _validate_test_sample_tensors(test_sample)

    max_samples = _coerce_max_samples(max_samples)
    n_samples = min(int(samples.shape[0]), max_samples)

    # Create the ART classifier once and reuse it for all selected samples.
    classifier = _build_art_classifier(model, samples.shape[1:], nb_classes, learning_rate)

    sensitivity_scores = []
    for idx in range(n_samples):
        sample = samples[idx].detach().cpu().unsqueeze(0)
        label = labels[idx].detach().cpu().unsqueeze(0)
        label = F.one_hot(label, num_classes=nb_classes).float()

        try:
            score = loss_sensitivity(
                classifier,
                sample.numpy(),
                label.numpy(),
            )
            if score is not None and not math.isnan(float(score)):
                sensitivity_scores.append(float(score))
        except Exception as exc:
            logger.warning("Could not compute loss sensitivity for sample index %s", idx)
            logger.warning(exc)

    if not sensitivity_scores:
        return 0.0

    return float(np.mean(sensitivity_scores))


def compute_adversarial_accuracy_art(
    model,
    test_loader,
    nb_classes,
    learning_rate,
    epsilon=0.03
):
    """
    Computes adversarial accuracy using FGSM attack.

    Args:
        model (object): The model.
        test_loader (DataLoader): DataLoader providing test samples.
        nb_classes (int): The nb_classes of the model.
        learning_rate (float): The learning rate of the model.
        epsilon (float): Maximum perturbation magnitude for the attacks.

    Returns:
        float: The adversarial accuracy score.
    """

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()
    model.to(device)

    correct = 0
    total = 0

    for samples, labels in test_loader:
        samples = samples.to(device)
        labels = labels.to(device)

        x_adv = fgsm_attack(model, samples, labels, epsilon=epsilon)

        with torch.no_grad():
            outputs = model(x_adv)
            logits = outputs[0] if isinstance(outputs, (tuple, list)) else outputs
            preds = logits.argmax(dim=1)

        correct += (preds == labels).sum().item()
        total += labels.size(0)

    return correct / total if total > 0 else 0.0


def get_empirical_robustness_score(
    model,
    test_sample,
    nb_classes,
    learning_rate,
    attack_name = "fgsm",
    attack_params = None,
    max_samples = 128,
):
    """
    Calculates the Empirical Robustness score using Adversarial Robustness Toolbox (ART).

    Args:
        model (object): The model.
        test_sample (object): A batch from the test dataloader (samples, labels).
        nb_classes (int): The nb_classes of the model.
        learning_rate (float): The learning rate of the model.
        attack_name (str): Attack key supported by ART empirical_robustness.
        attack_params (dict | None): Optional attack parameters.
        max_samples (int): Max number of samples from the batch to use.

    Returns:
        float: Empirical robustness score (>= 0.0). If it cannot be computed, returns 0.0.
    """
    try:
        samples, _ = _validate_test_sample_tensors(test_sample)

        batch_size: int = int(samples.shape[0])
        n: int = int(min(max_samples, batch_size))
        x = samples[:n].detach().cpu().numpy()

        classifier = _build_art_classifier(model, samples.shape[1:], nb_classes, learning_rate)

        score = empirical_robustness(
            classifier=classifier,
            x=x,
            attack_name=attack_name,
            attack_params=attack_params,
        )

        if isinstance(score, np.ndarray):
            score = float(np.mean(score))

        if score is None or (isinstance(score, float) and math.isnan(score)):
            return 0.0

        return float(score)

    except Exception as exc:
        logger.warning("Could not compute empirical robustness (ART). Returning 0.0")
        logger.warning(exc)
        return 0.0


def _get_image_normalization_for_samples(samples):
    if not isinstance(samples, torch.Tensor) or samples.ndim < 4:
        return None

    channels = int(samples.shape[1])
    if channels == 1:
        return (0.5,), (0.5,)
    if channels == 3:
        return (0.4914, 0.4822, 0.4465), (0.2471, 0.2435, 0.2616)
    return None


def _channel_tensor(values, samples):
    shape = [1, len(values)] + [1] * max(samples.dim() - 2, 0)
    return torch.tensor(values, dtype=samples.dtype, device=samples.device).view(*shape)


def _fgsm_step_and_clamp(samples, grad, epsilon):
    normalization = _get_image_normalization_for_samples(samples)
    if normalization is None:
        return samples + epsilon * grad.sign()

    mean, std = normalization
    mean = _channel_tensor(mean, samples)
    std = _channel_tensor(std, samples)

    normalized_epsilon = float(epsilon) / std
    lower = (0.0 - mean) / std
    upper = (1.0 - mean) / std

    x_adv = samples + normalized_epsilon * grad.sign()
    x_adv = torch.max(torch.min(x_adv, samples + normalized_epsilon), samples - normalized_epsilon)
    return torch.max(torch.min(x_adv, upper), lower)


def fgsm_attack(model, samples, labels, epsilon=0.03):
    """
        Performs an FGSM (Fast Gradient Sign Method) adversarial attack on a batch of samples.

        Args:
            model (torch.nn.Module): The PyTorch model to attack.
            samples (torch.Tensor): Input samples to perturb, shape (B, ...).
            labels (torch.Tensor): True labels corresponding to the samples.
            epsilon (float, optional): Maximum perturbation magnitude for the attack. Defaults to 0.03.

        Returns:
            torch.Tensor: Adversarially perturbed samples with the same shape as `samples`.
    """
    try:
        device = next(model.parameters()).device
    except Exception:
        device = samples.device

    samples = samples.clone().detach().to(device)
    labels = labels.to(device)
    samples.requires_grad = True

    outputs = model(samples)
    logits = outputs[0] if isinstance(outputs, (tuple, list)) else outputs
    loss = nn.CrossEntropyLoss()(logits, labels)
    grad = torch.autograd.grad(loss, samples, only_inputs=True)[0]
    x_adv = _fgsm_step_and_clamp(samples, grad, epsilon)

    return x_adv.detach()


def get_confidence_score(
    model,
    test_sample,
    max_samples = 128,
    use_true_label = True,
):
    """
    Calculates the confidence score.

    Args:
        model (object): The model.
        test_sample (object): A batch from the test dataloader (samples, labels).
        max_samples (int): Max number of samples from the batch to use.
        use_true_label (bool): Whether to compute confidence with respect to the true labels. Defaults to True.

    Returns:
        float: Confidence score.
    """
    try:
        if not isinstance(model, torch.nn.Module):
            logger.warning("Model is not a torch.nn.Module")
            return 0.0

        x, y = test_sample

        if isinstance(x, torch.Tensor):
            x = x[:max_samples]
        if isinstance(y, torch.Tensor):
            y = y[:max_samples]

        try:
            device = next(model.parameters()).device
        except Exception:
            device = torch.device("cpu")

        model.eval()
        with torch.no_grad():
            x = x.to(device) if isinstance(x, torch.Tensor) else x
            out = model(x)

            logits = out[0] if isinstance(out, (tuple, list)) else out
            probs = torch.softmax(logits, dim=1)

            if use_true_label and isinstance(y, torch.Tensor):
                if y.ndim > 1:
                    y_idx = torch.argmax(y, dim=1)
                else:
                    y_idx = y
                y_idx = y_idx.to(device)

                true_probs = probs.gather(1, y_idx.view(-1, 1)).squeeze(1)
                return float(true_probs.mean().detach().cpu().item())

            msp = probs.max(dim=1).values
            return float(msp.mean().detach().cpu().item())

    except Exception as e:
        logger.warning("Could not compute confidence score")
        logger.warning(e)
        return 0.0


def attack_success_rate(model, test_sample,epsilon=0.03):
    """
    Calculates the ASR.

    Args:
        model (object): The model.
        test_sample (object): A batch from the test dataloader (samples, labels).
        epsilon (float): Maximum perturbation magnitude for the attacks.

    Returns:
        float: The ASR.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()
    model.to(device)

    images, labels = test_sample
    images = images.to(device)
    labels = labels.to(device)

    with torch.no_grad():
        outputs = model(images)
        logits = outputs[0] if isinstance(outputs, (tuple, list)) else outputs
        preds = logits.argmax(dim=1)

    correct_mask = preds.eq(labels)
    num_correct = correct_mask.sum().item()

    if num_correct == 0:
        return 0.0

    x_adv = fgsm_attack(model, images, labels, epsilon=epsilon)

    with torch.no_grad():
        outputs_adv = model(x_adv)
        logits_adv = outputs_adv[0] if isinstance(outputs_adv, (tuple, list)) else outputs_adv
        preds_adv = logits_adv.argmax(dim=1)

    successful_attacks = (correct_mask & preds_adv.ne(labels)).sum().item()

    asr = successful_attacks / num_correct

    return asr
