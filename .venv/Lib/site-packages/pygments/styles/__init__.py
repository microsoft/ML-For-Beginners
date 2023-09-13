"""
    pygments.styles
    ~~~~~~~~~~~~~~~

    Contains built-in styles.

    :copyright: Copyright 2006-2023 by the Pygments team, see AUTHORS.
    :license: BSD, see LICENSE for details.
"""

from pygments.plugin import find_plugin_styles
from pygments.util import ClassNotFound

#: A dictionary of built-in styles, mapping style names to
#: ``'submodule::classname'`` strings.
STYLE_MAP = {
    'abap': 'abap::AbapStyle',
    'algol_nu': 'algol_nu::Algol_NuStyle',
    'algol': 'algol::AlgolStyle',
    'arduino': 'arduino::ArduinoStyle',
    'autumn': 'autumn::AutumnStyle',
    'borland': 'borland::BorlandStyle',
    'bw': 'bw::BlackWhiteStyle',
    'colorful': 'colorful::ColorfulStyle',
    'default': 'default::DefaultStyle',
    'dracula': 'dracula::DraculaStyle',
    'emacs': 'emacs::EmacsStyle',
    'friendly_grayscale': 'friendly_grayscale::FriendlyGrayscaleStyle',
    'friendly': 'friendly::FriendlyStyle',
    'fruity': 'fruity::FruityStyle',
    'github-dark': 'gh_dark::GhDarkStyle',
    'gruvbox-dark': 'gruvbox::GruvboxDarkStyle',
    'gruvbox-light': 'gruvbox::GruvboxLightStyle',
    'igor': 'igor::IgorStyle',
    'inkpot': 'inkpot::InkPotStyle',
    'lightbulb': 'lightbulb::LightbulbStyle',
    'lilypond': 'lilypond::LilyPondStyle',
    'lovelace': 'lovelace::LovelaceStyle',
    'manni': 'manni::ManniStyle',
    'material': 'material::MaterialStyle',
    'monokai': 'monokai::MonokaiStyle',
    'murphy': 'murphy::MurphyStyle',
    'native':   'native::NativeStyle',
    'nord-darker': 'nord::NordDarkerStyle',
    'nord': 'nord::NordStyle',
    'one-dark': 'onedark::OneDarkStyle',
    'paraiso-dark': 'paraiso_dark::ParaisoDarkStyle',
    'paraiso-light': 'paraiso_light::ParaisoLightStyle',
    'pastie': 'pastie::PastieStyle',
    'perldoc': 'perldoc::PerldocStyle',
    'rainbow_dash': 'rainbow_dash::RainbowDashStyle',
    'rrt': 'rrt::RrtStyle',
    'sas': 'sas::SasStyle',
    'solarized-dark': 'solarized::SolarizedDarkStyle',
    'solarized-light': 'solarized::SolarizedLightStyle',
    'staroffice': 'staroffice::StarofficeStyle',
    'stata-dark': 'stata_dark::StataDarkStyle',
    'stata-light': 'stata_light::StataLightStyle',
    'stata': 'stata_light::StataLightStyle',
    'tango': 'tango::TangoStyle',
    'trac': 'trac::TracStyle',
    'vim': 'vim::VimStyle',
    'vs': 'vs::VisualStudioStyle',
    'xcode': 'xcode::XcodeStyle',
    'zenburn': 'zenburn::ZenburnStyle'
}


def get_style_by_name(name):
    """
    Return a style class by its short name. The names of the builtin styles
    are listed in :data:`pygments.styles.STYLE_MAP`.

    Will raise :exc:`pygments.util.ClassNotFound` if no style of that name is
    found.
    """
    if name in STYLE_MAP:
        mod, cls = STYLE_MAP[name].split('::')
        builtin = "yes"
    else:
        for found_name, style in find_plugin_styles():
            if name == found_name:
                return style
        # perhaps it got dropped into our styles package
        builtin = ""
        mod = name
        cls = name.title() + "Style"

    try:
        mod = __import__('pygments.styles.' + mod, None, None, [cls])
    except ImportError:
        raise ClassNotFound("Could not find style module %r" % mod +
                         (builtin and ", though it should be builtin") + ".")
    try:
        return getattr(mod, cls)
    except AttributeError:
        raise ClassNotFound("Could not find style class %r in style module." % cls)


def get_all_styles():
    """Return a generator for all styles by name, both builtin and plugin."""
    yield from STYLE_MAP
    for name, _ in find_plugin_styles():
        yield name
