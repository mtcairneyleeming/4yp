from reusable.util import load_args, load_scores, gen_file_name
from .tables import html_table, latex_table
import pandas
from .helpers import pretty_loss_fn_name
import numpy as onp


def merge_dicts(a: dict, b: dict, expand=True):
    for key, val in b.items():
        if key in ["avg_gp_moments", "gp_moments"] and key in a:
            continue
        if (isinstance(val, list) and len(val) > 0 and isinstance(val[0], str)) or (
            isinstance(val, list)
            and len(val) > 0
            and isinstance(val[0], list)
            and len(val[0]) > 0
            and isinstance(val[0][0], str)
        ):
            new = [val] if expand else val
            if key not in a:
                a[key] = new
            else:
                a[key] = a[key] + new
        else:
            new = onp.array(val)[None] if expand else onp.array(val)
            if key not in a:
                a[key] = new
            else:
                a[key] = onp.concatenate((a[key], new), axis=0)


def get_gp_moments():
    args = load_args("16", str(1), "exp1")
    scores = load_scores(
        "16",
        gen_file_name("16", args, args["experiment"] + args["loss_fn_names"][0], "B"),
    )

    return scores["gp_moments"]


def get_loss_scores(code: int, exp_name, args_count: int, back_compat_file_names, average_subset=None):
    """Given the raw data (which is v. oddly formatted), return a dictionary of lists:
    - loss_fns
    - frobenius: items: array of length num_orders_calced
    - vae_moments: 2d array: (num_orders, n)
    - mmd: items: array of values for each kernel choice
    - mmd_kernels: list of string names
    """
    args = load_args(str(code), str(args_count), exp_name)
    return get_loss_scores_from_args(args, back_compat_file_names, average_subset=average_subset)


def get_loss_scores_from_args(args, back_compat_file_names, average_subset=None):
    if average_subset is None:
        average_subset = [0, args["n"]]
    scores = {"loss_fns": []}
    for loss_fn in args["loss_fn_names"]:
        if loss_fn == "gp":
            continue
        scores["loss_fns"].append(loss_fn)
        s = load_scores(
            args["expcode"],
            gen_file_name(
                args["expcode"],
                args,
                (args["experiment"] if "experiment" in args else "") + loss_fn,
                back_compat_file_names,
            ),
        )

        s["mmd_kernels"] = [x[0] for x in s["mmd"]]
        s["mmd"] = onp.array([x[1] for x in s["mmd"]])
        s["mmd_avg"] = onp.mean(s["mmd"])

        s["avg_vae_moments"] = [onp.mean(x[average_subset[0] : average_subset[1]]) for x in s["vae_moments"]]
        if "gp_moments" not in s:
            s["gp_moments"] = get_gp_moments()

        s["avg_gp_moments"] = [onp.mean(x[average_subset[0] : average_subset[1]]) for x in s["gp_moments"]]

        s["avg_moment_differences"] = onp.abs(onp.array(s["avg_vae_moments"]) - onp.array(s["avg_gp_moments"]))
        merge_dicts(scores, s)

    scores["avg_moments"] = onp.concatenate((scores["avg_gp_moments"], scores["avg_vae_moments"]))

    return scores


def show_score_matrix(code, exp_name, args_count, back_compat_file_names, matrix_dims, x_labels, y_labels):
    scores = get_loss_scores(code, exp_name, args_count, back_compat_file_names)
    data = onp.mean(scores["mmd"], axis=1).reshape(matrix_dims)

    print(onp.reshape(scores["loss_fns"], matrix_dims))

    html_table(data, pandas.Index(x_labels), pandas.Index(y_labels), None, False, True, False)
    latex_table(
        data,
        pandas.Index(x_labels),
        pandas.Index(y_labels),
        f"./gen_plots/{code}/tables/{code}_{exp_name}_{args_count}_mmd_mat_table.tex",
        None,
        False,
        True,
        False,
    )


def show_loss_scores(code, exp_name, args_count):

    scores = get_loss_scores(code, exp_name, args_count)
    display_loss_scores(scores, code, f"{code}_{exp_name}_{args_count}")


def show_all_loss_scores(things, backcompat_prior_names=False):
    all_scores = {}

    for code, exp_name, args_count in things:

        s = get_loss_scores(code, exp_name, args_count, backcompat_prior_names)
        merge_dicts(all_scores, s, expand=False)

    display_loss_scores(
        all_scores,
        code,
        f"{'-'.join([str(x[0]) for x in things])}_{'-'.join([str(x[1]) for x in things])}_{'-'.join([str(x[2]) for x in things])}",
    )


def display_score(
    scores,
    score_access,
    save_path,
    use_extend_row_index=False,
    colouring_data=None,
    colour_by_rank=True,
    rotate_col_headers=False,
    override_col_labels=None,
    vmin=None,
    scale_values=None,
    prettify_row_labels=True,
    override_row_index=None
):
    m = 1
    try:
        m = len(scores[score_access][0])
    except TypeError:
        pass
    r = range(1, m+1)
    HTML_COL_LABELS = {
        "frobenius": r,
        "mmd": scores["mmd_kernels"][0],
        "mmd_avg": ["Average MMD score"],
        "avg_moments": r,
    }
    LATEX_COL_LABELS = {
        "frobenius": r,
        "mmd": [pretty_loss_fn_name(x) for x in scores["mmd_kernels"][0]],
        "mmd_avg": ["Average MMD score"],
        "avg_moments": r,
    }

    

    if override_row_index is None:
        rows = (["GP"] if use_extend_row_index else []) + scores["loss_fns"]
        latex_row_index = pandas.Index([pretty_loss_fn_name(x) for x in rows] if prettify_row_labels else rows)
        html_row_index = pandas.Index(rows)
    else:
        latex_row_index = override_row_index
        html_row_index = override_row_index
        

    data = scores[score_access]

    if scale_values is not None:
        data = data * scale_values

    latex_col_index = pandas.Index(
        LATEX_COL_LABELS[score_access] if override_col_labels is None else override_col_labels, name="Loss functions"
    )
    html_col_index = pandas.Index(
        HTML_COL_LABELS[score_access] if override_col_labels is None else override_col_labels, name="Loss functions"
    )

    latex_table(
        data,
        latex_row_index,
        latex_col_index,
        save_path,
        colour_by_rank=colour_by_rank,
        show_values=True,
        rotateColHeader=rotate_col_headers,
        colouring_data=colouring_data,
        vmin=vmin,
    )
    html_table(
        data,
        html_row_index,
        html_col_index,
        show_values=True,
        colour_by_rank=colour_by_rank,
        rotateColHeader=rotate_col_headers,
        colouring_data=colouring_data,
        vmin=vmin,
    )


def display_loss_scores(scores, out_file_code, file_name):

    latex_row_index = pandas.Index([pretty_loss_fn_name(x) for x in scores["loss_fns"]])
    html_row_index = pandas.Index(scores["loss_fns"])

    extended_rows = ["GP"] + scores["loss_fns"]
    extended_latex_row_index = pandas.Index([pretty_loss_fn_name(x) for x in extended_rows])
    extended_html_row_index = pandas.Index(extended_rows)

    std_range = [1, 2, 3, 4, 5, 6, 7, 8, 9]

    plots = [
        # save_name/data access, title, html col labels, latex col labels, colouring_data
        (
            (scores["frobenius"][:, :4], "frobenius"),
            "$p$",
            [1, 2, 3, 4],
            [1, 2, 3, 4],
            html_row_index,
            latex_row_index,
        ),
        (
            "mmd",
            "MMD scores",
            scores["mmd_kernels"][0],
            [pretty_loss_fn_name(x) for x in scores["mmd_kernels"][0]],
            html_row_index,
            latex_row_index,
        ),
        (
            (onp.mean(scores["mmd"], axis=1).reshape((len(html_row_index), 1)), "mmd_avg"),
            "",
            ["Average MMD score"],
            ["Average MMD score"],
            html_row_index,
            latex_row_index,
        ),
        ("avg_moment_differences", "$p$", std_range, std_range, html_row_index, latex_row_index),
        (
            (
                onp.concatenate((scores["avg_gp_moments"], scores["avg_vae_moments"]), axis=0),
                "avg_vae_moments_ranked",
            ),
            "$p$",
            std_range,
            std_range,
            extended_html_row_index,
            extended_latex_row_index,
            onp.concatenate(
                (
                    onp.full((1, len(std_range)), -1000000),
                    onp.abs(onp.array(scores["avg_vae_moments"]) - onp.array(scores["avg_gp_moments"])),
                ),
                axis=0,
            ),
            1,
        ),
    ]

    for data_access, title, html_col_labels, latex_col_labels, html_rows, latex_rows, *rest in plots:

        if isinstance(data_access, str):
            data = scores[data_access]
        else:
            data = data_access[0]
            data_access = data_access[1]

        latex_col_index = pandas.Index(latex_col_labels, name="Loss functions $\\backslash$ " + title)
        html_col_index = pandas.Index(html_col_labels, name="Loss functions \\ " + title)
        colouring_data = rest[0] if len(rest) > 0 else None
        vmin = rest[1] if len(rest) > 1 else None

        latex_table(
            data,
            latex_rows,
            latex_col_index,
            f"./gen_plots/{out_file_code}/tables/{file_name}_{data_access}_rank_table.tex",
            colour_by_rank=True,
            show_values=True,
            rotateColHeader=title == "MMD scores",
            colouring_data=colouring_data,
            vmin=vmin,
        )

        latex_table(
            data,
            latex_rows,
            latex_col_index,
            f"./gen_plots/{out_file_code}/tables/{file_name}_{data_access}_vals_table.tex",
            colour_by_rank=False,
            show_values=True,
            rotateColHeader=title == "MMD scores",
            colouring_data=colouring_data,
            vmin=vmin,
        )

        html_table(
            data,
            html_rows,
            html_col_index,
            show_values=True,
            colour_by_rank=True,
            rotateColHeader=title == "MMD scores",
            colouring_data=colouring_data,
            vmin=vmin,
        )
