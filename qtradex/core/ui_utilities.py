import os
import shutil
import sys
import time
from random import choice, sample

from qtradex.common.utilities import it


def logo(animate=False):
    """
    Display flaming QTradeX logo.
    """

    def download_text():
        r"""
           ____                                    __     __
          / __ \    ____  ____   __   ____  ____  (_ \   / _)
         / /  \ \  (_  _)(  _ \ / _\ (    \(  __)   \ \_/ /
        | |    | |   ||   )   //    \ ) D | ) _)     \   /
        | |  /\| |  (__) (__\_)\_/\_/(____/(____)    / _ \
         \ \_\ \/                                  _/ / \ \_
          \___\ \_ {}(__/   \__)
               \__)
        """
        return download_text.__doc__.replace("\n   ", "\n").strip("\n")

    def random_flame_color():
        """
        Generate random flame color for effect.
        """
        return choice(["red", "green"])

    def render_flame_line(pattern, line_length, indent=0):
        return (
            " " * indent
            + " ".join(
                [it(random_flame_color(), i) for i in sample(pattern, line_length)]
            )
            + "\n"
        )

    ev_logo = download_text().format(
        os.path.split(sys.argv[0])[1]
        .rsplit(".py", 1)[0]
        .replace("_", " ")
        .title()
        .center(31)
    )
    cols = shutil.get_terminal_size().columns
    patterns = [
        (r"              *.   ~          %             `     ,        ", 27, 5),
        (r"              *.   ~          %             `     ,        ", 27, 5),
        (r"              *.   ~          %             `     ,        ", 27, 5),
        (r"        *.()      *`()      *~()    @      *,()      *.()  ", 27, 5),
        (r"     *()/     *()/\     *()/\%   % @       *()/\     *()/ ", 28, 4),
        (r"    *()/     *()/\     *()/\%   % @       *()/\     *()/ ", 29, 3),
        (r"  ()/_ ()/\_ ()/_ ()/_ ()/_ ()/_ ()()() ()(@@)_ ()()()()()", 28, 4),
        (r"   ()/_ ()/\_ ()/_ ()/_ ()/_ ()/_ ()()() ()(@@)_ ()()()()()", 27, 5),
    ]
    idx = 0
    while animate or not idx:
        idx += 1
        msg = "\033c"
        for pattern, length, indent in patterns:
            msg += render_flame_line(pattern, length, indent)
        msg += it("cyan", ev_logo)
        msg = "\n".join(i.ljust(cols) for i in msg.split("\n"))
        print(msg)
        if animate:
            time.sleep(0.1)
        if not idx % 10:
            animate = False


def get_number(options):
    choice = "nan"
    while (not choice.isdigit() or int(choice) not in options) and choice != "":
        choice = input("? ")

    if not choice:
        choice = min(options)

    return int(choice)


def select(options):
    options = dict(enumerate(options))
    logo()
    for k, v in options.items():
        print(f"  {k}: {v}")
    print()
    return get_number(options)
