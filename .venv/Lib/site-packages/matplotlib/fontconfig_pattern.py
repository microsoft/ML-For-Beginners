import re
from pyparsing import ParseException

from matplotlib._fontconfig_pattern import *  # noqa: F401, F403
from matplotlib._fontconfig_pattern import (
    parse_fontconfig_pattern, _family_punc, _value_punc)
from matplotlib import _api
_api.warn_deprecated("3.6", name=__name__, obj_type="module")


family_unescape = re.compile(r'\\([%s])' % _family_punc).sub
value_unescape = re.compile(r'\\([%s])' % _value_punc).sub
family_escape = re.compile(r'([%s])' % _family_punc).sub
value_escape = re.compile(r'([%s])' % _value_punc).sub


class FontconfigPatternParser:
    ParseException = ParseException

    def parse(self, pattern): return parse_fontconfig_pattern(pattern)
