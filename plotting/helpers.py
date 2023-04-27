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
        if gap >= 0 and i == 0:  # i.e. the GP
            return 0

        if i > 0:
            return i + num_cols * num_rows - count

    return func




def calc_plot_dimensions(args, include_gp=False):
    # A = indexes over cols, B over rows - note flattening in code will
    to_count = args["loss_fn_names"] if "loss_fn_names" in args else args["loss_fns"]

    num_lfs = len([x for x in to_count if x is not None and x != "gp"])


    twoD = "Arange" in args and "Brange" in args
    num_cols = len(args["Arange"]) if twoD else 1
    num_rows = len(args["Brange"]) if twoD else num_lfs
    

    # add an extra row if asked, or if it won't fit in the grid
    if include_gp and (num_cols * num_rows <= num_lfs):
        num_rows += 1

    return num_rows, num_cols


def clear_unused_axs(axs, mapping, total):
    if len(axs.shape) == 1:
        return 
    num_rows, num_cols = axs.shape

    used_axes = [mapping(i) for i in range(total)]
    for i in range(num_cols * num_rows):
        if i not in used_axes:
            axs.flat[i].remove()

def numstr(s:str):
    s = float(s)
    return str(int(s)) if s.is_integer() else f"{s:.2f}"

def pretty_loss_fn_name(loss_fn: str):
    RCL_LATEX = r"\mathrm{RCL}"
    KLD_LATEX = r"\mathrm{KLD}"
    MMD_RBF_LATEX = r"\mathrm{MMD}_\mathrm{rbf}"
    MMD_RQK_LATEX = r"\mathrm{MMD}_\mathrm{rq}"
    

    if "{}" in loss_fn:
        a, b = loss_fn.split("{}", 2)

        after = a + ": " if a != "" else ""    
        loss_fn = b
        
    else:
        after = ""
        



    parts = loss_fn.split("+")
    


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
            
            for i, m in enumerate(mult):
                if mmd == "mmd_rbf_sum":
                    s = numstr(m)
                    if i != 0:
                        after += "+"
                    after += f"{MMD_RBF_LATEX}({s})"
                if mmd == "mmd_rqk_sum":
                    ls, a = m.split(",", 1)
                    ls_s = numstr(ls)
                    ls_a = numstr(a) 
                    if i != 0:
                        after += "+"
                    after += f"{MMD_RQK_LATEX}({ls_s},{ls_a})"
        

    if after == "":
        return loss_fn

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

def pretty_prior(prior_choice: str, prior_args: dict):
    if prior_choice == "invgamma":
        conc = prior_args.get("concentration", 4)
        rate = prior_args.get("rate", 1)
        return f"$\mathrm{{InverseGamma}}({numstr(conc)}, {numstr(rate)})$"

    elif prior_choice == "lognormal":
        loc = prior_args.get("location", 0.0)
        scale = prior_args.get("scale", 0.1)
        return f"$\log\mathcal{{N}}({numstr(loc)}, {numstr(scale)})$"

    elif prior_choice == "normal":
        loc = prior_args.get("location", 0.0)
        scale = prior_args.get("scale", 15.0)
        return f"$\mathcal{{N}}({numstr(loc)}, {numstr(scale)})$"

    elif prior_choice == "halfnormal":
        scale = prior_args.get("scale", 15.0)
        return f"$\|\mathcal{{N}}(0, {numstr(scale)})\|$"

    elif prior_choice == "gamma":
        conc = prior_args.get("concentration", 4)
        rate = prior_args.get("rate", 1)
        return f"$\mathrm{{Gamma}}({numstr(conc)}, {numstr(rate)})$"

    elif prior_choice == "uniform":
        lower = prior_args.get("lower", 0.01)
        upper = prior_args.get("upper", 0.5)
        return f"$\mathcal{{U}}({numstr(lower)}, {numstr(upper)})$"

    raise NotImplementedError(f"Unknown prior choice {prior_choice}")