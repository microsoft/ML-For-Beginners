from pathlib import Path
import io

import pytest

from matplotlib import ft2font
from matplotlib.testing.decorators import check_figures_equal
import matplotlib.font_manager as fm
import matplotlib.pyplot as plt


def test_fallback_errors():
    file_name = fm.findfont('DejaVu Sans')

    with pytest.raises(TypeError, match="Fallback list must be a list"):
        # failing to be a list will fail before the 0
        ft2font.FT2Font(file_name, _fallback_list=(0,))

    with pytest.raises(
            TypeError, match="Fallback fonts must be FT2Font objects."
    ):
        ft2font.FT2Font(file_name, _fallback_list=[0])


def test_ft2font_positive_hinting_factor():
    file_name = fm.findfont('DejaVu Sans')
    with pytest.raises(
            ValueError, match="hinting_factor must be greater than 0"
    ):
        ft2font.FT2Font(file_name, 0)


def test_fallback_smoke():
    fp = fm.FontProperties(family=["WenQuanYi Zen Hei"])
    if Path(fm.findfont(fp)).name != "wqy-zenhei.ttc":
        pytest.skip("Font wqy-zenhei.ttc may be missing")

    fp = fm.FontProperties(family=["Noto Sans CJK JP"])
    if Path(fm.findfont(fp)).name != "NotoSansCJK-Regular.ttc":
        pytest.skip("Noto Sans CJK JP font may be missing.")

    plt.rcParams['font.size'] = 20
    fig = plt.figure(figsize=(4.75, 1.85))
    fig.text(0.05, 0.45, "There are 几个汉字 in between!",
             family=['DejaVu Sans', "Noto Sans CJK JP"])
    fig.text(0.05, 0.25, "There are 几个汉字 in between!",
             family=['DejaVu Sans', "WenQuanYi Zen Hei"])
    fig.text(0.05, 0.65, "There are 几个汉字 in between!",
             family=["Noto Sans CJK JP"])
    fig.text(0.05, 0.85, "There are 几个汉字 in between!",
             family=["WenQuanYi Zen Hei"])

    # TODO enable fallback for other backends!
    for fmt in ['png', 'raw']:  # ["svg", "pdf", "ps"]:
        fig.savefig(io.BytesIO(), format=fmt)


@pytest.mark.parametrize('family_name, file_name',
                         [("WenQuanYi Zen Hei",  "wqy-zenhei"),
                          ("Noto Sans CJK JP", "NotoSansCJK")]
                         )
@check_figures_equal(extensions=["png", "pdf", "eps", "svg"])
def test_font_fallback_chinese(fig_test, fig_ref, family_name, file_name):
    fp = fm.FontProperties(family=[family_name])
    if file_name not in Path(fm.findfont(fp)).name:
        pytest.skip(f"Font {family_name} ({file_name}) is missing")

    text = ["There are", "几个汉字", "in between!"]

    plt.rcParams["font.size"] = 20
    test_fonts = [["DejaVu Sans", family_name]] * 3
    ref_fonts = [["DejaVu Sans"], [family_name], ["DejaVu Sans"]]

    for j, (txt, test_font, ref_font) in enumerate(
            zip(text, test_fonts, ref_fonts)
    ):
        fig_ref.text(0.05, .85 - 0.15*j, txt, family=ref_font)
        fig_test.text(0.05, .85 - 0.15*j, txt, family=test_font)


@pytest.mark.parametrize(
    "family_name, file_name",
    [
        ("WenQuanYi Zen Hei", "wqy-zenhei"),
        ("Noto Sans CJK JP", "NotoSansCJK"),
    ],
)
def test__get_fontmap(family_name, file_name):
    fp = fm.FontProperties(family=[family_name])
    found_file_name = Path(fm.findfont(fp)).name
    if file_name not in found_file_name:
        pytest.skip(f"Font {family_name} ({file_name}) is missing")

    text = "There are 几个汉字 in between!"
    ft = fm.get_font(
        fm.fontManager._find_fonts_by_props(
            fm.FontProperties(family=["DejaVu Sans", family_name])
        )
    )

    fontmap = ft._get_fontmap(text)
    for char, font in fontmap.items():
        if ord(char) > 127:
            assert Path(font.fname).name == found_file_name
        else:
            assert Path(font.fname).name == "DejaVuSans.ttf"
