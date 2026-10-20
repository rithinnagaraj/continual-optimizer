def set_k(optimizer, value):
    if value <= 0:
        raise ValueError("k must be positive")
    for group in optimizer.param_groups:
        group["k"] = value
