def box_off(optimizer):
    for group in optimizer.param_groups:
        group["box_enabled"] = False
