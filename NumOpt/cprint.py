from termcolor import cprint
from functools import partial
import contextlib
import sys,os

cprint_green = partial(cprint, color="green", attrs=["bold"])
cprint_magenta = partial(cprint, color="magenta", attrs=["bold"])
cprint_blue = partial(cprint, color="blue", attrs=["bold"])
cprint_red = partial(cprint, color="red", attrs=["bold"])
cprint_yellow = partial(cprint, color="yellow", attrs=["bold"])


@contextlib.contextmanager
def nostd():
    old_stdout=sys.stdout
    sys.stdout=open(os.devnull,"w")
    yield 
    sys.stdout.close()
    sys.stdout=old_stdout