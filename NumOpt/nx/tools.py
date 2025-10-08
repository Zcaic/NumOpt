import NXOpen
import NXOpen.Features
from NXOpen import ParasolidExporter

import json
import os
from pathlib import Path


def create_af(coords, session: NXOpen.Session):
    part = session.Parts.Work

    markid = session.SetUndoMark(NXOpen.Session.MarkVisibility.Visible, "create airfoil")
    splineEX = part.Features.CreateStudioSplineBuilderEx(NXOpen.NXObject.Null)
    for i in coords:
        pt = NXOpen.Point3d(i[0] * 1000.0, i[1] * 1000.0, 0.0)
        pt = part.Points.CreatePoint(pt)
        gcons = splineEX.ConstraintManager.CreateGeometricConstraintData()
        gcons.Point = pt
        splineEX.ConstraintManager.Append(gcons)
    spline = splineEX.Commit()
    splineEX.Destroy()
    session.UpdateManager.DoUpdate(markid)


def modify_af(coords, name, session: NXOpen.Session):
    part = session.Parts.Work

    markid = session.SetUndoMark(NXOpen.Session.MarkVisibility.Visible, "modify airfoil")
    spline = part.Features.FindObject(name)
    splineEX = part.Features.CreateStudioSplineBuilderEx(spline)
    for i, vi in enumerate(coords):
        gcons = splineEX.ConstraintManager.FindItem(i)
        pt = gcons.Point
        pt.SetCoordinates(NXOpen.Point3d(vi[0] * 1000.0, vi[1] * 1000.0, 0.0))
    splineEX.Evaluate()
    spline = splineEX.Commit()
    splineEX.Destroy()
    session.UpdateManager.DoUpdate(markid)


def modify_expr(expr_name: str, value: float, session: NXOpen.Session, unit: str = None, update: bool = False):
    markid = session.SetUndoMark(NXOpen.Session.MarkVisibility.Visible, f"modify {expr_name}")
    part = session.Parts.Work
    expr = part.Expressions.FindObject(expr_name)
    unit = NXOpen.Unit.Null if unit is None else part.UnitCollection.FindObject(unit)
    part.Expressions.EditExpressionWithUnits(expr, unit, f"{value:.6f}")
    if update:
        session.UpdateManager.DoUpdate(markid)


def export_solid(solid_name: list[str], outfile: str, session: NXOpen.Session):
    markid = session.SetUndoMark(NXOpen.Session.MarkVisibility.Visible, "export parasoild")

    part = session.Parts.Work
    parasolidExporter: ParasolidExporter = session.DexManager.CreateParasolidExporter()
    parasolidExporter.InputFile = part.FullPath
    parasolidExporter.OutputFile = outfile

    parasolidExporter.ExportSelectionBlock.SelectionScope = NXOpen.ObjectSelector.Scope.SelectedObjects
    for i in solid_name:
        solid = part.Bodies.FindObject(i)
        parasolidExporter.ExportSelectionBlock.SelectionComp.Add(solid)
    parasolidExporter.Commit()
    session.UpdateManager.DoUpdate(markid)


def saveAs(outfile, session: NXOpen.Session, overwrite: False):
    outfile = Path(outfile)
    if overwrite:
        if outfile.exists():
            os.unlink(outfile)
    else:
        if outfile.exists():
            raise (f"{outfile} has been existed.")

    markid = session.SetUndoMark(NXOpen.Session.MarkVisibility.Visible, f"save as")
    part = session.Parts.Work
    part.SaveAs(outfile.as_posix())
    session.UpdateManager.DoUpdate(markid)


if __name__ == "__main__":
    with open("C:/Users/Zcaic/Desktop/hxy_ppt/coords.json", "r") as fin:
        data = json.load(fin)

    session = NXOpen.Session.GetSession()
    # create_af(data["init_coords"],session=session)
    modify_af(coords=data["init_coords"], name="SPLINE(1)", session=session)
