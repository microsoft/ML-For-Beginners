import numpy as np
import matplotlib as mpl
from seaborn.utils import _version_predates


def MarkerStyle(marker=None, fillstyle=None):
    """
    Allow MarkerStyle to accept a MarkerStyle object as parameter.

    Supports matplotlib < 3.3.0
    https://github.com/matplotlib/matplotlib/pull/16692

    """
    if isinstance(marker, mpl.markers.MarkerStyle):
        if fillstyle is None:
            return marker
        else:
            marker = marker.get_marker()
    return mpl.markers.MarkerStyle(marker, fillstyle)


def norm_from_scale(scale, norm):
    """Produce a Normalize object given a Scale and min/max domain limits."""
    # This is an internal maplotlib function that simplifies things to access
    # It is likely to become part of the matplotlib API at some point:
    # https://github.com/matplotlib/matplotlib/issues/20329
    if isinstance(norm, mpl.colors.Normalize):
        return norm

    if scale is None:
        return None

    if norm is None:
        vmin = vmax = None
    else:
        vmin, vmax = norm  # TODO more helpful error if this fails?

    class ScaledNorm(mpl.colors.Normalize):

        def __call__(self, value, clip=None):
            # From github.com/matplotlib/matplotlib/blob/v3.4.2/lib/matplotlib/colors.py
            # See github.com/matplotlib/matplotlib/tree/v3.4.2/LICENSE
            value, is_scalar = self.process_value(value)
            self.autoscale_None(value)
            if self.vmin > self.vmax:
                raise ValueError("vmin must be less or equal to vmax")
            if self.vmin == self.vmax:
                return np.full_like(value, 0)
            if clip is None:
                clip = self.clip
            if clip:
                value = np.clip(value, self.vmin, self.vmax)
            # ***** Seaborn changes start ****
            t_value = self.transform(value).reshape(np.shape(value))
            t_vmin, t_vmax = self.transform([self.vmin, self.vmax])
            # ***** Seaborn changes end *****
            if not np.isfinite([t_vmin, t_vmax]).all():
                raise ValueError("Invalid vmin or vmax")
            t_value -= t_vmin
            t_value /= (t_vmax - t_vmin)
            t_value = np.ma.masked_invalid(t_value, copy=False)
            return t_value[0] if is_scalar else t_value

    new_norm = ScaledNorm(vmin, vmax)
    new_norm.transform = scale.get_transform().transform

    return new_norm


def scale_factory(scale, axis, **kwargs):
    """
    Backwards compatability for creation of independent scales.

    Matplotlib scales require an Axis object for instantiation on < 3.4.
    But the axis is not used, aside from extraction of the axis_name in LogScale.

    """
    modify_transform = False
    if _version_predates(mpl, "3.4"):
        if axis[0] in "xy":
            modify_transform = True
            axis = axis[0]
            base = kwargs.pop("base", None)
            if base is not None:
                kwargs[f"base{axis}"] = base
            nonpos = kwargs.pop("nonpositive", None)
            if nonpos is not None:
                kwargs[f"nonpos{axis}"] = nonpos

    if isinstance(scale, str):
        class Axis:
            axis_name = axis
        axis = Axis()

    scale = mpl.scale.scale_factory(scale, axis, **kwargs)

    if modify_transform:
        transform = scale.get_transform()
        transform.base = kwargs.get("base", 10)
        if kwargs.get("nonpositive") == "mask":
            # Setting a private attribute, but we only get here
            # on an old matplotlib, so this won't break going forwards
            transform._clip = False

    return scale


def set_scale_obj(ax, axis, scale):
    """Handle backwards compatability with setting matplotlib scale."""
    if _version_predates(mpl, "3.4"):
        # The ability to pass a BaseScale instance to Axes.set_{}scale was added
        # to matplotlib in version 3.4.0: GH: matplotlib/matplotlib/pull/19089
        # Workaround: use the scale name, which is restrictive only if the user
        # wants to define a custom scale; they'll need to update the registry too.
        if scale.name is None:
            # Hack to support our custom Formatter-less CatScale
            return
        method = getattr(ax, f"set_{axis}scale")
        kws = {}
        if scale.name == "function":
            trans = scale.get_transform()
            kws["functions"] = (trans._forward, trans._inverse)
        method(scale.name, **kws)
        axis_obj = getattr(ax, f"{axis}axis")
        scale.set_default_locators_and_formatters(axis_obj)
    else:
        ax.set(**{f"{axis}scale": scale})


def get_colormap(name):
    """Handle changes to matplotlib colormap interface in 3.6."""
    try:
        return mpl.colormaps[name]
    except AttributeError:
        return mpl.cm.get_cmap(name)


def register_colormap(name, cmap):
    """Handle changes to matplotlib colormap interface in 3.6."""
    try:
        if name not in mpl.colormaps:
            mpl.colormaps.register(cmap, name=name)
    except AttributeError:
        mpl.cm.register_cmap(name, cmap)


def set_layout_engine(fig, engine):
    """Handle changes to auto layout engine interface in 3.6"""
    if hasattr(fig, "set_layout_engine"):
        fig.set_layout_engine(engine)
    else:
        # _version_predates(mpl, 3.6)
        if engine == "tight":
            fig.set_tight_layout(True)
        elif engine == "constrained":
            fig.set_constrained_layout(True)
        elif engine == "none":
            fig.set_tight_layout(False)
            fig.set_constrained_layout(False)


def share_axis(ax0, ax1, which):
    """Handle changes to post-hoc axis sharing."""
    if _version_predates(mpl, "3.5"):
        group = getattr(ax0, f"get_shared_{which}_axes")()
        group.join(ax1, ax0)
    else:
        getattr(ax1, f"share{which}")(ax0)


def get_legend_handles(legend):
    """Handle legendHandles attribute rename."""
    if _version_predates(mpl, "3.7"):
        return legend.legendHandles
    else:
        return legend.legend_handles
