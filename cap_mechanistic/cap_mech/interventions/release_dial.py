def set_release(optimizer, value):
    value = float("inf") if value == "infinity" else float(value)
    if value < 1:
        raise ValueError("T_m must be >= 1")
    for group in optimizer.param_groups:
        group["T_m"] = value
