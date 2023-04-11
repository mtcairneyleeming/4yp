import numpy as onp
import re

def align_right_backfill(count, num_rows, num_cols):
    """Map count plots to a num_rows * num_cols grid, filling in from the bottom right """
    assert(count <= num_cols * num_rows)
    return lambda i: i + num_cols * num_rows - count


def align_left_backfill(count, num_rows, num_cols):
    """Map count plots to a num_rows * num_cols grid, filling in the bottom rows, and the top row from the left"""

    assert(count <= num_cols * num_rows and count >= (num_rows-1) * num_cols)
    g = align_right_backfill(count, num_rows, num_cols)

    def func(i):
        if i < num_cols and i < num_cols - (num_cols * num_rows - count):
            return i
        return g(i)

    return func


def align_right_backfill_with_gp(count, num_rows, num_cols):
    """Map count plots to a num_rows * num_cols grid, filling in the bottom rows, placing the first plot in the top left, and the rest filling in from the right """
    # assume count includes the gp
    gap = num_cols * num_rows - count

    assert(count <= num_cols * num_rows and count >= (num_rows-1) * num_cols)

    def func(i):
        if gap > 0 and i == 0:  # i.e. the GP
            return 0

        if i > 0:
            return i + num_cols * num_rows - count

    return func


def calc_plot_dimensions(args, num_cols, num_rows, include_gp=False, extra_row_for_gp=False, include_standard_vae=False):
    # A = indexes over cols, B over rows - note flattening in code will
    if num_cols is None and num_rows is None:
        twoD = "Arange" in args and "Brange" in args
        num_cols = len(args["Arange"]) if twoD else 1
        num_rows = len(args["Brange"]) if twoD else len(args["loss_fn_names"])
        
    else:
        twoD = True
        assert num_cols is not None and num_rows is not None

    # add an extra row if asked, or if it won't fit in the grid
    if include_standard_vae and (num_cols * num_rows <= len(args["loss_fn_names"])):
        num_rows += 1

    # add an extra row if asked, or if it won't fit in the grid
    if include_gp and (extra_row_for_gp or num_cols * num_rows <= len(args["loss_fn_names"])):
        num_rows += 1

    return twoD, num_rows, num_cols


def clear_unused_axs(axs, mapping, twoD, total):
    if not twoD:
        return
    num_rows, num_cols = axs.shape
    used_axes = [mapping(i) for i in range(total)]
    for i in range(num_cols * num_rows):
        if i not in used_axes:
            axs[onp.unravel_index(i, (num_rows, num_cols))].remove()


def pretty_loss_fn_name(loss_fn: str):
    RCL_LATEX = r"\mathrm{RCL}"
    KLD_LATEX = r"\mathrm{KLD}"
    MMD_RBF_LATEX = r"\mathrm{MMD}_\mathrm{rbf}"
    MMD_RQK_LATEX = r"\mathrm{MMD}_\mathrm{rq}"
    parts = loss_fn.split("+")
    after = ""
    for i, part in enumerate(parts):

        if i > 0:
            after += "+"

        split = re.split(r"([a-zA-Z\s]+)", part)
        factor = split[0]
        part = "".join(split[1:])
        if factor != "":
            after += factor

        if part == "RCL":
            after += RCL_LATEX
        if part == "KLD":
            after += KLD_LATEX
        if part.startswith("mmd"):
            mmd, lss = part.split("-", 1)
            mult = lss.split(";")
            for m in mult:
                if mmd == "mmd_rbf_sum":
                    m = float(m)
                    s = str(int(m)) if m.is_integer() else f"{m:.2f}"
                    after += f"{MMD_RBF_LATEX}({s})"
                if mmd == "mmd_rqk_sum":
                    ls, a = m.split(",", 1)
                    ls = float(ls)
                    a = float(a)
                    ls_s = str(int(ls)) if ls.is_integer() else f"{ls:.2f}"
                    ls_a = str(int(a)) if a.is_integer() else f"{a:.2f}"
                    after += f"{MMD_RQK_LATEX}({ls_s},{ls_a})"

    return f"${after}$"


def pretty_label(var_name: str):

    if var_name == "n":
        return "$n$"

    if var_name == "train_num_batches":
        return "training batches"

    if var_name == "vae_scale_factor":
        return "scaling of VAE layer size"

    if var_name == "num_epochs":
        return "epochs"
    
    if var_name == "batch_size":
        return "batch size"

    return var_name
