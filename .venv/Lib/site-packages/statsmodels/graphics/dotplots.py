import numpy as np

from . import utils


def dot_plot(points, intervals=None, lines=None, sections=None,
             styles=None, marker_props=None, line_props=None,
             split_names=None, section_order=None, line_order=None,
             stacked=False, styles_order=None, striped=False,
             horizontal=True, show_names="both",
             fmt_left_name=None, fmt_right_name=None,
             show_section_titles=None, ax=None):
    """
    Dot plotting (also known as forest and blobbogram).

    Produce a dotplot similar in style to those in Cleveland's
    "Visualizing Data" book ([1]_).  These are also known as "forest plots".

    Parameters
    ----------
    points : array_like
        The quantitative values to be plotted as markers.
    intervals : array_like
        The intervals to be plotted around the points.  The elements
        of `intervals` are either scalars or sequences of length 2.  A
        scalar indicates the half width of a symmetric interval.  A
        sequence of length 2 contains the left and right half-widths
        (respectively) of a nonsymmetric interval.  If None, no
        intervals are drawn.
    lines : array_like
        A grouping variable indicating which points/intervals are
        drawn on a common line.  If None, each point/interval appears
        on its own line.
    sections : array_like
        A grouping variable indicating which lines are grouped into
        sections.  If None, everything is drawn in a single section.
    styles : array_like
        A grouping label defining the plotting style of the markers
        and intervals.
    marker_props : dict
        A dictionary mapping style codes (the values in `styles`) to
        dictionaries defining key/value pairs to be passed as keyword
        arguments to `plot` when plotting markers.  Useful keyword
        arguments are "color", "marker", and "ms" (marker size).
    line_props : dict
        A dictionary mapping style codes (the values in `styles`) to
        dictionaries defining key/value pairs to be passed as keyword
        arguments to `plot` when plotting interval lines.  Useful
        keyword arguments are "color", "linestyle", "solid_capstyle",
        and "linewidth".
    split_names : str
        If not None, this is used to split the values of `lines` into
        substrings that are drawn in the left and right margins,
        respectively.  If None, the values of `lines` are drawn in the
        left margin.
    section_order : array_like
        The section labels in the order in which they appear in the
        dotplot.
    line_order : array_like
        The line labels in the order in which they appear in the
        dotplot.
    stacked : bool
        If True, when multiple points or intervals are drawn on the
        same line, they are offset from each other.
    styles_order : array_like
        If stacked=True, this is the order in which the point styles
        on a given line are drawn from top to bottom (if horizontal
        is True) or from left to right (if horizontal is False).  If
        None (default), the order is lexical.
    striped : bool
        If True, every other line is enclosed in a shaded box.
    horizontal : bool
        If True (default), the lines are drawn horizontally, otherwise
        they are drawn vertically.
    show_names : str
        Determines whether labels (names) are shown in the left and/or
        right margins (top/bottom margins if `horizontal` is True).
        If `both`, labels are drawn in both margins, if 'left', labels
        are drawn in the left or top margin.  If `right`, labels are
        drawn in the right or bottom margin.
    fmt_left_name : callable
        The left/top margin names are passed through this function
        before drawing on the plot.
    fmt_right_name : callable
        The right/bottom marginnames are passed through this function
        before drawing on the plot.
    show_section_titles : bool or None
        If None, section titles are drawn only if there is more than
        one section.  If False/True, section titles are never/always
        drawn, respectively.
    ax : matplotlib.axes
        The axes on which the dotplot is drawn.  If None, a new axes
        is created.

    Returns
    -------
    fig : Figure
        The figure given by `ax.figure` or a new instance.

    Notes
    -----
    `points`, `intervals`, `lines`, `sections`, `styles` must all have
    the same length whenever present.

    References
    ----------
    .. [1] Cleveland, William S. (1993). "Visualizing Data". Hobart Press.
    .. [2] Jacoby, William G. (2006) "The Dot Plot: A Graphical Display
       for Labeled Quantitative Values." The Political Methodologist
       14(1): 6-14.

    Examples
    --------
    This is a simple dotplot with one point per line:

    >>> dot_plot(points=point_values)

    This dotplot has labels on the lines (if elements in
    `label_values` are repeated, the corresponding points appear on
    the same line):

    >>> dot_plot(points=point_values, lines=label_values)
    """

    import matplotlib.transforms as transforms

    fig, ax = utils.create_mpl_ax(ax)

    # Convert to numpy arrays if that is not what we are given.
    points = np.asarray(points)
    asarray_or_none = lambda x : None if x is None else np.asarray(x)
    intervals = asarray_or_none(intervals)
    lines = asarray_or_none(lines)
    sections = asarray_or_none(sections)
    styles = asarray_or_none(styles)

    # Total number of points
    npoint = len(points)

    # Set default line values if needed
    if lines is None:
        lines = np.arange(npoint)

    # Set default section values if needed
    if sections is None:
        sections = np.zeros(npoint)

    # Set default style values if needed
    if styles is None:
        styles = np.zeros(npoint)

    # The vertical space (in inches) for a section title
    section_title_space = 0.5

    # The number of sections
    nsect = len(set(sections))
    if section_order is not None:
        nsect = len(set(section_order))

    # The number of section titles
    if show_section_titles is False:
        draw_section_titles = False
        nsect_title = 0
    elif show_section_titles is True:
        draw_section_titles = True
        nsect_title = nsect
    else:
        draw_section_titles = nsect > 1
        nsect_title = nsect if nsect > 1 else 0

    # The total vertical space devoted to section titles.
    section_space_total = section_title_space * nsect_title

    # Add a bit of room so that points that fall at the axis limits
    # are not cut in half.
    ax.set_xmargin(0.02)
    ax.set_ymargin(0.02)

    if section_order is None:
        lines0 = list(set(sections))
        lines0.sort()
    else:
        lines0 = section_order

    if line_order is None:
        lines1 = list(set(lines))
        lines1.sort()
    else:
        lines1 = line_order

    # A map from (section,line) codes to index positions.
    lines_map = {}
    for i in range(npoint):
        if section_order is not None and sections[i] not in section_order:
            continue
        if line_order is not None and lines[i] not in line_order:
            continue
        ky = (sections[i], lines[i])
        if ky not in lines_map:
            lines_map[ky] = []
        lines_map[ky].append(i)

    # Get the size of the axes on the parent figure in inches
    bbox = ax.get_window_extent().transformed(
        fig.dpi_scale_trans.inverted())
    awidth, aheight = bbox.width, bbox.height

    # The number of lines in the plot.
    nrows = len(lines_map)

    # The positions of the lowest and highest guideline in axes
    # coordinates (for horizontal dotplots), or the leftmost and
    # rightmost guidelines (for vertical dotplots).
    bottom, top = 0, 1

    if horizontal:
        # x coordinate is data, y coordinate is axes
        trans = transforms.blended_transform_factory(ax.transData,
                                                     ax.transAxes)
    else:
        # x coordinate is axes, y coordinate is data
        trans = transforms.blended_transform_factory(ax.transAxes,
                                                     ax.transData)

    # Space used for a section title, in axes coordinates
    title_space_axes = section_title_space / aheight

    # Space between lines
    if horizontal:
        dpos = (top - bottom - nsect_title*title_space_axes) /\
            float(nrows)
    else:
        dpos = (top - bottom) / float(nrows)

    # Determine the spacing for stacked points
    if styles_order is not None:
        style_codes = styles_order
    else:
        style_codes = list(set(styles))
        style_codes.sort()
    # Order is top to bottom for horizontal plots, so need to
    # flip.
    if horizontal:
        style_codes = style_codes[::-1]
    # nval is the maximum number of points on one line.
    nval = len(style_codes)
    if nval > 1:
        stackd = dpos / (2.5*(float(nval)-1))
    else:
        stackd = 0.

    # Map from style code to its integer position
    style_codes_map = {x: style_codes.index(x) for x in style_codes}

    # Setup default marker styles
    colors = ["r", "g", "b", "y", "k", "purple", "orange"]
    if marker_props is None:
        marker_props = {x: {} for x in style_codes}
    for j in range(nval):
        sc = style_codes[j]
        if "color" not in marker_props[sc]:
            marker_props[sc]["color"] = colors[j % len(colors)]
        if "marker" not in marker_props[sc]:
            marker_props[sc]["marker"] = "o"
        if "ms" not in marker_props[sc]:
            marker_props[sc]["ms"] = 10 if stackd == 0 else 6

    # Setup default line styles
    if line_props is None:
        line_props = {x: {} for x in style_codes}
    for j in range(nval):
        sc = style_codes[j]
        if "color" not in line_props[sc]:
            line_props[sc]["color"] = "grey"
        if "linewidth" not in line_props[sc]:
            line_props[sc]["linewidth"] = 2 if stackd > 0 else 8

    if horizontal:
        # The vertical position of the first line.
        pos = top - dpos/2 if nsect == 1 else top
    else:
        # The horizontal position of the first line.
        pos = bottom + dpos/2

    # Points that have already been labeled
    labeled = set()

    # Positions of the y axis grid lines
    ticks = []

    # Loop through the sections
    for k0 in lines0:

        # Draw a section title
        if draw_section_titles:

            if horizontal:

                y0 = pos + dpos/2 if k0 == lines0[0] else pos

                ax.fill_between((0, 1), (y0,y0),
                                (pos-0.7*title_space_axes,
                                 pos-0.7*title_space_axes),
                                color='darkgrey',
                                transform=ax.transAxes,
                                zorder=1)

                txt = ax.text(0.5, pos - 0.35*title_space_axes, k0,
                              horizontalalignment='center',
                              verticalalignment='center',
                              transform=ax.transAxes)
                txt.set_fontweight("bold")
                pos -= title_space_axes

            else:

                m = len([k for k in lines_map if k[0] == k0])

                ax.fill_between((pos-dpos/2+0.01,
                                 pos+(m-1)*dpos+dpos/2-0.01),
                                (1.01,1.01), (1.06,1.06),
                                color='darkgrey',
                                transform=ax.transAxes,
                                zorder=1, clip_on=False)

                txt = ax.text(pos + (m-1)*dpos/2, 1.02, k0,
                              horizontalalignment='center',
                              verticalalignment='bottom',
                              transform=ax.transAxes)
                txt.set_fontweight("bold")

        jrow = 0
        for k1 in lines1:

            # No data to plot
            if (k0, k1) not in lines_map:
                continue

            # Draw the guideline
            if horizontal:
                ax.axhline(pos, color='grey')
            else:
                ax.axvline(pos, color='grey')

            # Set up the labels
            if split_names is not None:
                us = k1.split(split_names)
                if len(us) >= 2:
                    left_label, right_label = us[0], us[1]
                else:
                    left_label, right_label = k1, None
            else:
                left_label, right_label = k1, None

            if fmt_left_name is not None:
                left_label = fmt_left_name(left_label)

            if fmt_right_name is not None:
                right_label = fmt_right_name(right_label)

            # Draw the stripe
            if striped and jrow % 2 == 0:
                if horizontal:
                    ax.fill_between((0, 1), (pos-dpos/2, pos-dpos/2),
                                    (pos+dpos/2, pos+dpos/2),
                                    color='lightgrey',
                                    transform=ax.transAxes,
                                    zorder=0)
                else:
                    ax.fill_between((pos-dpos/2, pos+dpos/2),
                                    (0, 0), (1, 1),
                                    color='lightgrey',
                                    transform=ax.transAxes,
                                    zorder=0)

            jrow += 1

            # Draw the left margin label
            if show_names.lower() in ("left", "both"):
                if horizontal:
                    ax.text(-0.1/awidth, pos, left_label,
                            horizontalalignment="right",
                            verticalalignment='center',
                            transform=ax.transAxes,
                            family='monospace')
                else:
                    ax.text(pos, -0.1/aheight, left_label,
                            horizontalalignment="center",
                            verticalalignment='top',
                            transform=ax.transAxes,
                            family='monospace')

            # Draw the right margin label
            if show_names.lower() in ("right", "both"):
                if right_label is not None:
                    if horizontal:
                        ax.text(1 + 0.1/awidth, pos, right_label,
                                horizontalalignment="left",
                                verticalalignment='center',
                                transform=ax.transAxes,
                                family='monospace')
                    else:
                        ax.text(pos, 1 + 0.1/aheight, right_label,
                                horizontalalignment="center",
                                verticalalignment='bottom',
                                transform=ax.transAxes,
                                family='monospace')

            # Save the vertical position so that we can place the
            # tick marks
            ticks.append(pos)

            # Loop over the points in one line
            for ji,jp in enumerate(lines_map[(k0,k1)]):

                # Calculate the vertical offset
                yo = 0
                if stacked:
                    yo = -dpos/5 + style_codes_map[styles[jp]]*stackd

                pt = points[jp]

                # Plot the interval
                if intervals is not None:

                    # Symmetric interval
                    if np.isscalar(intervals[jp]):
                        lcb, ucb = pt - intervals[jp],\
                            pt + intervals[jp]

                    # Nonsymmetric interval
                    else:
                        lcb, ucb = pt - intervals[jp][0],\
                            pt + intervals[jp][1]

                    # Draw the interval
                    if horizontal:
                        ax.plot([lcb, ucb], [pos+yo, pos+yo], '-',
                                transform=trans,
                                **line_props[styles[jp]])
                    else:
                        ax.plot([pos+yo, pos+yo], [lcb, ucb], '-',
                                transform=trans,
                                **line_props[styles[jp]])


                # Plot the point
                sl = styles[jp]
                sll = sl if sl not in labeled else None
                labeled.add(sl)
                if horizontal:
                    ax.plot([pt,], [pos+yo,], ls='None',
                            transform=trans, label=sll,
                            **marker_props[sl])
                else:
                    ax.plot([pos+yo,], [pt,], ls='None',
                            transform=trans, label=sll,
                            **marker_props[sl])

            if horizontal:
                pos -= dpos
            else:
                pos += dpos

    # Set up the axis
    if horizontal:
        ax.xaxis.set_ticks_position("bottom")
        ax.yaxis.set_ticks_position("none")
        ax.set_yticklabels([])
        ax.spines['left'].set_color('none')
        ax.spines['right'].set_color('none')
        ax.spines['top'].set_color('none')
        ax.spines['bottom'].set_position(('axes', -0.1/aheight))
        ax.set_ylim(0, 1)
        ax.yaxis.set_ticks(ticks)
        ax.autoscale_view(scaley=False, tight=True)
    else:
        ax.yaxis.set_ticks_position("left")
        ax.xaxis.set_ticks_position("none")
        ax.set_xticklabels([])
        ax.spines['bottom'].set_color('none')
        ax.spines['right'].set_color('none')
        ax.spines['top'].set_color('none')
        ax.spines['left'].set_position(('axes', -0.1/awidth))
        ax.set_xlim(0, 1)
        ax.xaxis.set_ticks(ticks)
        ax.autoscale_view(scalex=False, tight=True)

    return fig
