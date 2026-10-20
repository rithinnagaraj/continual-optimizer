from .k_dial import set_k
from .box_toggle import box_off
from .state_reset import reset
from .importance_shuffle import shuffle, reverse
from .release_dial import set_release

def apply(optimizer, intervention):
    kind = intervention["kind"]
    if kind in ("baseline", "true_importance"):
        return
    if kind == "k":
        set_k(optimizer, intervention["value"])
    elif kind == "box_off":
        box_off(optimizer)
    elif kind == "shuffle":
        shuffle(optimizer, intervention["seed"])
    elif kind == "reverse":
        reverse(optimizer)
    elif kind == "release":
        set_release(optimizer, intervention["value"])
    else:
        reset(optimizer, kind)

def protected(optimizer):
    for group in optimizer.param_groups:
        if group["protect"]:
            for p in group["params"]:
                yield group, p, optimizer.state[p]
