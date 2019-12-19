from bokeh.plotting import figure, output_file, show, ColumnDataSource


if __name__ == "__main__":


    output_file("toolbar.html")
    fpr = [1, 2, 3, 4, 5]
    source = ColumnDataSource(data=dict(
        fpr=[1, 2, 3, 4, 5],
        tpr=[2, 4, 6, 4, 2],
        tnr = [1 - x for x in fpr],
        probs = [0.5, 0.4, 0.2, 0.1, 0.6],
    ))



    TOOLTIPS = [
        ("(spe.(tnr),se.(tpr))", "(@tnr, $y)"),
        ("threshold", "@probs"),
    ]

    p = figure(plot_width=400, plot_height=400, tooltips=TOOLTIPS,
               title="Mouse over the dots")

    p.line('fpr', 'tpr', source=source)

    show(p)