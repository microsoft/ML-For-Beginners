# This file is not meant for public use and will be removed in SciPy v2.0.0.
# Use the `scipy.constants` namespace for importing the functions
# included below.

import warnings
from . import _constants


__all__ = [  # noqa: F822
    'Avogadro', 'Boltzmann', 'Btu', 'Btu_IT', 'Btu_th', 'G',
    'Julian_year', 'N_A', 'Planck', 'R', 'Rydberg',
    'Stefan_Boltzmann', 'Wien', 'acre', 'alpha',
    'angstrom', 'arcmin', 'arcminute', 'arcsec',
    'arcsecond', 'astronomical_unit', 'atm',
    'atmosphere', 'atomic_mass', 'atto', 'au', 'bar',
    'barrel', 'bbl', 'blob', 'c', 'calorie',
    'calorie_IT', 'calorie_th', 'carat', 'centi',
    'convert_temperature', 'day', 'deci', 'degree',
    'degree_Fahrenheit', 'deka', 'dyn', 'dyne', 'e',
    'eV', 'electron_mass', 'electron_volt',
    'elementary_charge', 'epsilon_0', 'erg',
    'exa', 'exbi', 'femto', 'fermi', 'fine_structure',
    'fluid_ounce', 'fluid_ounce_US', 'fluid_ounce_imp',
    'foot', 'g', 'gallon', 'gallon_US', 'gallon_imp',
    'gas_constant', 'gibi', 'giga', 'golden', 'golden_ratio',
    'grain', 'gram', 'gravitational_constant', 'h', 'hbar',
    'hectare', 'hecto', 'horsepower', 'hour', 'hp',
    'inch', 'k', 'kgf', 'kibi', 'kilo', 'kilogram_force',
    'kmh', 'knot', 'lambda2nu', 'lb', 'lbf',
    'light_year', 'liter', 'litre', 'long_ton', 'm_e',
    'm_n', 'm_p', 'm_u', 'mach', 'mebi', 'mega',
    'metric_ton', 'micro', 'micron', 'mil', 'mile',
    'milli', 'minute', 'mmHg', 'mph', 'mu_0', 'nano',
    'nautical_mile', 'neutron_mass', 'nu2lambda',
    'ounce', 'oz', 'parsec', 'pebi', 'peta',
    'pi', 'pico', 'point', 'pound', 'pound_force',
    'proton_mass', 'psi', 'pt', 'short_ton',
    'sigma', 'slinch', 'slug', 'speed_of_light',
    'speed_of_sound', 'stone', 'survey_foot',
    'survey_mile', 'tebi', 'tera', 'ton_TNT',
    'torr', 'troy_ounce', 'troy_pound', 'u',
    'week', 'yard', 'year', 'yobi', 'yocto',
    'yotta', 'zebi', 'zepto', 'zero_Celsius', 'zetta'
]


def __dir__():
    return __all__


def __getattr__(name):
    if name not in __all__:
        raise AttributeError(
            "scipy.constants.constants is deprecated and has no attribute "
            f"{name}. Try looking in scipy.constants instead.")

    warnings.warn(f"Please use `{name}` from the `scipy.constants` namespace, "
                  "the `scipy.constants.constants` namespace is deprecated.",
                  category=DeprecationWarning, stacklevel=2)

    return getattr(_constants, name)
