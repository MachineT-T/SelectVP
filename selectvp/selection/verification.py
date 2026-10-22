"""The first generated lowercase yes/no token gates the full-vocabulary score."""


def probability_difference(logits, generated_id, yes_id, no_id):
    if yes_id == no_id or min(yes_id, no_id) < 0:
        raise ValueError("Invalid yes/no token IDs")
    if generated_id not in (yes_id, no_id):
        return 0.0
    import torch
    probabilities = torch.softmax(logits, dim=-1)
    return float(probabilities[yes_id].item() - probabilities[no_id].item())


def calculate_conf(outputs, yes_id, no_id):
    if not outputs.logits:
        raise ValueError("Verification generated no tokens")
    return probability_difference(outputs.logits[0][0], int(outputs.sequences[0, -len(outputs.logits)]), yes_id, no_id)
