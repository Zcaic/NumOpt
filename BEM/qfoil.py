import subprocess
import aerosandbox as asb
from typing import Union, List
from pathlib import Path
import warnings
import numpy as np
import re
import io


class Qfoil:
    def __init__(
        self,
        airfoil: asb.Airfoil,
        Re: float = 0.0,
        mach: float = 0.0,
        n_crit: float = 9.0,
        xtr_upper: float = 1.0,
        xtr_lower: float = 1.0,
        max_iter: int = 100,
        qfoil_command: str = "qfoil",
        qfoil_repanel: bool = True,
        qfoil_repanel_n_points: int = 199,
        verbose: bool = False,
        timeout: Union[float, int, None] = 30,
        working_directory: Union[Path, str] = None,
        clean_cache=False,
    ):

        if mach >= 1:
            raise ValueError("XFoil will terminate if a supersonic freestream Mach number is given.")

        self.airfoil = airfoil
        self.Re = Re
        self.mach = mach
        self.n_crit = n_crit
        self.xtr_upper = xtr_upper
        self.xtr_lower = xtr_lower
        self.max_iter = max_iter
        self.qfoil_command = qfoil_command
        self.qfoil_repanel = qfoil_repanel
        self.qfoil_repanel_n_points = qfoil_repanel_n_points
        self.verbose = verbose
        self.timeout = timeout
        self.clean_cache = clean_cache

        if working_directory is None:
            self.working_directory = Path("./runtime/")
            self.working_directory.mkdir(exist_ok=True)
        else:
            self.working_directory = Path(working_directory)

    def _default_keystrokes(
        self,
        airfoil_filename: str,
        output_filename: str,
    ) -> List[str]:
        """
        Returns a list of XFoil keystrokes that are common to all XFoil runs.

        Returns:
            A list of strings, each of which is a single XFoil keystroke to be followed by <enter>.
        """
        run_file_contents = []

        # Disable graphics
        run_file_contents += [
            "plop",
            "g",
            "w 0.05",
            "",
        ]

        # Load the airfoil
        run_file_contents += [
            f"load {airfoil_filename}",
        ]

        if self.qfoil_repanel:
            run_file_contents += [
                "ppar",
                f"n {self.qfoil_repanel_n_points}",
                "",
                "",
                "",
                # "pane",
            ]

        # Enter oper mode
        run_file_contents += [
            "oper",
        ]

        # Handle Re
        if self.Re != 0:
            run_file_contents += [
                f"Re {self.Re:.8g}",
            ]

        # Handle mach
        run_file_contents += [
            f"Mach {self.mach:.8g}",
        ]

        # Handle hinge moment
        # run_file_contents += [
        #     "hinc",
        #     f"fnew {float(self.hinge_point_x):.8g} {float(self.airfoil.local_camber(self.hinge_point_x)):.8g}",
        #     "fmom",
        # ]

        # if self.full_potential:
        #     run_file_contents += [
        #         "full",
        #         "fpar",
        #         f"i {self.max_iter}",
        #         "",
        #     ]

        # Handle iterations
        run_file_contents += [
            f"iter {self.max_iter}",
        ]

        # Handle trips and ncrit
        if not (self.xtr_upper == 1 and self.xtr_lower == 1 and self.n_crit == 9):
            run_file_contents += [
                "vpar",
                f"xtr {self.xtr_upper:.8g} {self.xtr_lower:.8g}",
                f"n {self.n_crit:.8g}",
                "",
            ]

        # Set polar accumulation
        run_file_contents += [
            "pacc",
            f"{output_filename}",
            "",
        ]

        # Include more data in polar
        run_file_contents += ["cinc"]  # include minimum Cp

        return run_file_contents

    def _run_qfoil(self, run_command: str):
        directory = self.working_directory
        output_filename = "output.txt"
        airfoil_file = "airfoil.dat"

        (directory/output_filename).unlink(missing_ok=True)

        self.airfoil.write_dat(directory / airfoil_file)

        keystrokes = self._default_keystrokes(airfoil_filename=airfoil_file, output_filename=output_filename)
        keystrokes += [run_command]
        keystrokes += ["pacc", "", "quit"]

        try:
            proc = subprocess.Popen(
                self.qfoil_command,
                cwd=directory,
                stdin=subprocess.PIPE,
                stdout=None if self.verbose else subprocess.DEVNULL,
                stderr=None if self.verbose else subprocess.DEVNULL,
                text=True,
                # shell=True,
                # timeout=self.timeout,
                # check=True
            )
            outs, errs = proc.communicate(input="\n".join(keystrokes), timeout=self.timeout)
            return_code = proc.poll()

        except subprocess.TimeoutExpired:
            proc.kill()
            outs, errs = proc.communicate()

            warnings.warn(
                "XFoil run timed out!\n" "If this was not expected, try increasing the `timeout` parameter\n" "when you create this AeroSandbox XFoil instance.",
                stacklevel=2,
            )
        except subprocess.CalledProcessError as e:
            raise e

        with open(directory / output_filename) as f:
            lines = f.readlines()
            re_match=re.compile(r".*-----\ .*")
            for idx,line in enumerate(lines):
                re_result=re.match(re_match,line)
                if re_result is not None:
                    idx_header=idx-1 
                    idx_content=idx+1 
                    break
            header=lines[idx_header]
            header=re.findall(r"\S+",header)
            raw_header_num=len(header)
            content=lines[idx_content:]
            content="".join(content)
            content=np.loadtxt(io.StringIO(content),ndmin=2)
            
            outputs={}
            uname_num=0
            for i in range(content.shape[1]):
                try:
                    outputs[header[i]]=content[:,i]
                except IndexError as e:
                    outputs[f"Uname{uname_num+1}"]=content[:,i]
                    uname_num+=1
            return outputs

    def alpha(
        self,
        alpha,
        start_at: Union[float, None] = 0,
    ):
        alphas = np.reshape(np.array(alpha), -1)
        alphas = np.sort(alphas)

        commands = []
        if len(alphas) > 1 and (start_at is not None) and (np.min(alphas) < start_at < np.max(alphas)):
            alphas_upper = alphas[alphas > start_at]
            alphas_lower = alphas[alpha <= start_at][::-1]

            for a in alphas_upper:
                commands.append(f"a {a}")

            commands.append("init")

            for a in alphas_lower:
                commands.append(f"a {a}")
        else:
            for a in alphas:
                commands.append(f"a {a}")
        
        output=self._run_qfoil("\n".join(commands))
        return output

if __name__ == "__main__":
    af = asb.Airfoil("naca2412").repanel(n_points_per_side=100)

    qf = Qfoil(
        airfoil=af,
        Re=1e6,
        qfoil_command="./qfoil.exe",
    )

    # result_at_single_alpha = xf.alpha(20)
    # print(result_at_single_alpha)

    # result_at_several_CLs = xf.cl([-0.1, 0.5, 0.7, 0.8, 0.9])

    result_at_multiple_alphas = qf.alpha(np.linspace(0,12,5))
    print(result_at_multiple_alphas["CL"])