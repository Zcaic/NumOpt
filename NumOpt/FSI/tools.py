import re
from pathlib import Path
from ..cprint import cprint_green


def deal_name(nasfile, outfile, simplify=True):
    def sub_func(match):
        old_str: list = match.group().split("\n")
        pshell_name = match.group("name")
        if simplify:
            pshell_name = pshell_name.split(":")[0]
        pshell_id = match.group("id")
        insert_str = f"$ANSA_NAME_COMMENT;{pshell_id};PSHELL;{pshell_name};"
        old_str.insert(1, insert_str)
        new_str = "\n".join(old_str)
        return new_str

    pattern = r"\$\*\s+Property:\s(?P<name>.*)\nPSHELL\s+(?P<id>\d+)\s+.*\n"
    pattern = re.compile(pattern)

    content = Path(nasfile).read_text()
    new_content = re.sub(pattern, sub_func, content)
    outfile = Path(outfile)
    if not outfile.parent.exists():
        outfile.parent.mkdir(exist_ok=True)
    outfile.write_text(new_content)
    cprint_green(f"new nastran file `{outfile.as_posix()}` is generated.")



if __name__ == "__main__":
    nasfile = "./bulk.dat"
    deal_name(nasfile, "new_bulk.dat")
