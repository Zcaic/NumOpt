import numpy as np
from pathlib import Path
import subprocess
import pandas as pd
import re


class JavaFoil:
    def __init__(self, data=None):
        self.raw_data = data

    def gen_polar(
        self, af_coords: np.ndarray, Mach: float, ReS: float, ReE: float, ReStep: float, alphaS: float, alphaE: float, alphaStep: float, AspectRatio: int, af_tag: str
    ):
        template = """
Options.Country(0);
Geometry.Open("%(af_file)s");
Options.MachNumber(%(Mach).3f);
Options.StallModel(0);
Options.TransitionModel(1);
Options.GroundEffect(0);
Options.HeightOverSpan(0.5);
Options.AspectRatio(%(AspectRatio).3f);
Options.SweepAngle(0.0);
Modify.SetPivot(25,0);
Polar.Analyze(%(Reynold_s).3f,%(Reynold_e).3f,%(Reynold_step).3f,%(alpha_s).3f,%(alpha_e).3f,%(alpha_step).3f,100,100,0,0);
Polar.Save("%(outfile)s");
JavaFoil.Exit();
"""
        af_file = Path("./runtime/af.txt")
        af_file.parent.mkdir(exist_ok=True)
        outfile = Path("./runtime/polar.txt")

        np.savetxt(af_file, af_coords, delimiter=" ", header="Untitled", comments="")

        script = template % {
            "af_file": af_file.resolve().as_posix(),
            "Mach": Mach,
            "Reynold_s": ReS,
            "Reynold_e": ReE,
            "Reynold_step": ReStep,
            "alpha_s": alphaS,
            "alpha_e": alphaE,
            "alpha_step": alphaStep,
            "AspectRatio": AspectRatio,
            "outfile": outfile.resolve().as_posix(),
        }

        script_file = Path("./runtime/polar.jfscript")
        script_file.write_text(script)

        cmd = [
            "java.exe",
            "-Dfile.encoding=UTF-8",
            "-cp",
            "D:/ProgramFiles/MHAeroTools/JavaFoil/java/mhclasses.jar",
            "-jar",
            "D:/ProgramFiles/MHAeroTools/JavaFoil/java/javafoil.jar",
            f'Script="{script_file.resolve().as_posix()}"',
        ]

        pid = subprocess.run(cmd, timeout=10, stdout=subprocess.DEVNULL)

        df = self.read_polar(outfile, af_tag)
        self.raw_data = df
        return df

    def read_polar(self, polar_file, af_tag):
        condition_pattern = re.compile(r"([\w\.]+)\s*=\s*([\d\.]+)")
        col_names = ["alpha", "CL", "CD", "CM25"]

        block_list = []
        with open(polar_file, "r") as fin:
            for line in fin:
                if line.startswith("Mach"):
                    match_res = re.findall(condition_pattern, line)
                    condition = {key: float(value) for key, value in match_res}

                    _ = [next(fin) for _ in range(3)]
                    data = []
                    for line in fin:
                        if line := line.strip():
                            line = line.split()
                            data.append(line[:4])
                        else:
                            df_block = pd.DataFrame(data=data, columns=col_names, dtype=float)
                            df_block["Mach"] = np.round(condition["Mach"], 3)
                            df_block["Re"] = np.round(condition["Re"], 3)
                            df_block["tag"] = af_tag
                            block_list.append(df_block)
                            break
        df = pd.concat(block_list, ignore_index=False, axis=0)
        return df

    def save(self, savefile):
        self.raw_data.to_csv(savefile, sep=",", index=False)

    @staticmethod
    def load(savefile):
        df = pd.read_csv(savefile, sep=",")
        return JavaFoil(data=df)


if __name__ == "__main__":
    import aerosandbox as asb

    af = asb.Airfoil("naca0012")
    coords = af.coordinates

    jf = JavaFoil()
    df = jf.gen_polar(af_coords=coords, Mach=0.3, ReS=1e6, ReE=3e6, ReStep=1e6, alphaS=-60.0, alphaE=60.0, alphaStep=1.0, AspectRatio=0, af_tag="af1")
    print(df[df["Re"]==2e6].head())
