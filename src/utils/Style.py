from typing import Union


class Style:
    _byte = "\33[{}m"

    def __init__(self, *styles: Union['Style', str, int]):
        self._instance = "%s"
        for style in styles:
            if isinstance(style, Style):
                style = style._instance
            else:
                style = Style._byte.format(style) + "%s"
            self._instance %= style

    def apply(self, string: str) -> str:
        return self._instance % string + Style._byte.format(0)

    def __mod__(self, other) -> Union[str, 'Style']:
        if isinstance(other, Style):
            return Style(self, other)
        if isinstance(other, str):
            return self.apply(other)
        return NotImplemented


class Styles:
    empty = Style(0)
    bold = Style(1)
    dim = Style(2)
    underlined = Style(4)
    blink = Style(5)
    reverse = Style(7)
    hidden = Style(8)

    class background:
        default = Style(49)
        black = Style(40)
        red = Style(41)
        green = Style(42)
        yellow = Style(43)
        blue = Style(44)
        magenta = Style(45)
        cyan = Style(46)
        gray = Style(100)
        light_gray = Style(47)
        light_red = Style(101)
        light_green = Style(102)
        light_yellow = Style(103)
        light_blue = Style(104)
        light_magenta = Style(105)
        light_cyan = Style(106)
        white = Style(107)

    class foreground:
        default = Style(39)
        black = Style(30)
        red = Style(31)
        green = Style(32)
        yellow = Style(33)
        blue = Style(34)
        magenta = Style(35)
        cyan = Style(36)
        gray = Style(90)
        light_gray = Style(37)
        light_red = Style(91)
        light_green = Style(92)
        light_yellow = Style(93)
        light_blue = Style(94)
        light_magenta = Style(95)
        light_cyan = Style(96)
        white = Style(97)
