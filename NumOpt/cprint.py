from termcolor import cprint
from functools import partial


cprint_green = partial(cprint, color="green", attrs=["bold"])
cprint_magenta = partial(cprint, color="magenta", attrs=["bold"])
cprint_blue = partial(cprint, color="blue", attrs=["bold"])
cprint_red = partial(cprint, color="red", attrs=["bold"])
cprint_yellow = partial(cprint, color="yellow", attrs=["bold"])
