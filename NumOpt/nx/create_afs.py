import aerosandbox as asb
from pathlib import Path
import subprocess
import json

def gen_NACA6(af_file, t, xtc, cl, a=1.0, A=1):
    template = """
Options.Country(9);
Geometry.CreateAirfoil(6,113,%(t).1f,%(xtc)d,%(cl).1f,%(a).1f,0,%(A)d,0,0,0);
Geometry.Save("%(af_file)s");
JavaFoil.Exit();
"""
    af_file = Path(af_file)
    af_file.parent.mkdir(exist_ok=True)
    script_file = Path("./runtime/af.jfscript")
    script_file.parent.mkdir(exist_ok=True)

    script = template % {"t": t, "xtc": xtc, "cl": cl, "a": a, "A": A, "af_file": af_file.resolve().as_posix()}

    script_file.write_text(script)
    cmd = [
        "java.exe",
        "-cp",
        "D:/ProgramFiles/MHAeroTools/JavaFoil/java/mhclasses.jar",
        "-jar",
        "D:/ProgramFiles/MHAeroTools/JavaFoil/java/javafoil.jar",
        f'Script="{script_file.resolve().as_posix()}"',
    ]
    pid=subprocess.run(cmd,timeout=10.0,stdout=subprocess.DEVNULL)
    return af_file

    # af=asb.Airfoil(coordinates=af_file)
    # af.draw()


def create_afs():
    templata = """
import NXOpen
import NXOpen.Features
import json

data_file = "%(data_file)s"

with open(data_file, "r") as fin:
    data = json.load(fin)

session = NXOpen.Session.GetSession()
part = session.Parts.Work
markid = session.SetUndoMark(NXOpen.Session.MarkVisibility.Visible, "create airfoils")

for name, coords in zip(data["afs_name"], data["afs"]):
    splineEX = part.Features.CreateStudioSplineBuilderEx(NXOpen.NXObject.Null)
    for i in coords:
        pt = NXOpen.Point3d(0.0, -i[0] * 1000.0, i[1] * 1000.0)
        pt = part.Points.CreatePoint(pt)
        gcons = splineEX.ConstraintManager.CreateGeometricConstraintData()
        gcons.Point = pt
        splineEX.ConstraintManager.Append(gcons)
    spline= splineEX.Commit()
    spline.SetName(name)
    splineEX.Destroy()

session.UpdateManager.DoUpdate(markid)

"""
    af0=gen_NACA6(af_file="./afs/naca64a028.dat",t=28,xtc=40,cl=0,A=1)
    af1=gen_NACA6(af_file="./afs/naca64a318.dat",t=18,xtc=40,cl=0.3,A=1)
    af2=gen_NACA6(af_file="./afs/naca64a412.dat",t=12,xtc=40,cl=0.4,A=1)
    af3=gen_NACA6(af_file="./afs/naca64a309.dat",t=9,xtc=40,cl=0.3,A=1)

    af0 = asb.Airfoil(coordinates=af0).set_TE_thickness(0.0).to_kulfan_airfoil().set_TE_thickness(2.4e-3)
    af1 = asb.Airfoil(coordinates=af1).set_TE_thickness(0.0).to_kulfan_airfoil().set_TE_thickness(2.4e-3)
    af2 = asb.Airfoil(coordinates=af2).set_TE_thickness(0.0).to_kulfan_airfoil().set_TE_thickness(2.4e-3)
    af3 = asb.Airfoil(coordinates=af3).set_TE_thickness(0.0).to_kulfan_airfoil().set_TE_thickness(2.4e-3)

    afs=[af0,af1,af2,af3]
    afs_name=["af0","af1","af2","af3"]

    data = {}

    coord_list = []

    for af in afs:
        coords = af.to_airfoil(n_coordinates_per_side=60).coordinates.tolist()
        coord_list.append(coords)

    data["afs"] = coord_list
    data["afs_name"] = afs_name

    data_file = Path("./runtime/data.json")
    script_file = Path("./runtime/nx.py")

    data_file.parent.mkdir(exist_ok=True)
    with open(data_file,"w") as fout:
        json.dump(data,fout)

    script = templata % {"data_file": data_file.resolve().as_posix()}
    script_file.write_text(script)


if __name__ == "__main__":
    create_afs()
    # gen_NACA6(af_file="./afs/naca64a312.dat",t=12,xtc=30,cl=0.4,A=1)
