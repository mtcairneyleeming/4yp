import jax.numpy as jnp
import pandas
import matplotlib.pyplot as plt
from IPython.display import display


def rank_data_in_cols(data):
    return jnp.array(data).argsort(axis=0).argsort(axis=0)


def latex_table(
    data,
    row_index,
    col_index,
    save_path,
    colouring_data=None,
    colour_by_rank=False,
    show_values=False,
    rotateColHeader=False,
    vmin=None,
):
    df = pandas.DataFrame(
        data,
        columns=col_index,
        index=row_index,
    )
    if colouring_data is None:
        colouring_data = data
    if colour_by_rank:
        ranks = rank_data_in_cols(colouring_data)

    cmap = plt.get_cmap("Blues_r")
    cmap.set_bad(color="red")

    s = df.style
    if colour_by_rank:
        s.background_gradient(axis=None, cmap=cmap, gmap=ranks, vmin=vmin)
    else:
        s.background_gradient(axis=None, cmap=cmap, gmap=colouring_data, vmin=vmin)

    if show_values:
        s.format("{:.1e}")
    else:
        s.format(" ")
    if rotateColHeader:
        s.format_index("\\rot{{{}}}", axis=1)

    with open(save_path, "w") as f:
        s.to_latex(f, convert_css=True)


def html_table(
    data,
    row_index,
    col_index,
    colouring_data=None,
    colour_by_rank=False,
    show_values=False,
    rotateColHeader=False,
    vmin=None,
):
    df = pandas.DataFrame(
        data,
        columns=col_index,
        index=row_index,
    )
    if colouring_data is None:
        colouring_data = data
    if colour_by_rank:
        ranks = rank_data_in_cols(colouring_data)

    cmap = plt.get_cmap("Blues_r")
    cmap.set_bad(color="red")
    cmap.set_under(color="white")

    s = df.style
    if colour_by_rank:
        s.background_gradient(axis=None, cmap=cmap, gmap=ranks, vmin=vmin)
    else:
        s.background_gradient(axis=None, cmap=cmap, gmap=colouring_data, vmin=vmin)

    if show_values:
        pass #s.format("{:5.3f}")
    else:
        s.format(" ")
    if rotateColHeader:
        s.set_table_styles(
            [
                # dict(selector="th", props=[("max-width", "80px")]),
                dict(
                    selector="th.col-heading",
                    props=[
                        ("font-size", "50%"),
                        ("text-align", "center"),
                        ("transform", "translate(0%,-30%) rotate(-45deg)"),
                    ],
                ),
                # dict(selector=".row_heading, .blank", props= [('display', 'none;')])
            ]
        )

    display(s)
