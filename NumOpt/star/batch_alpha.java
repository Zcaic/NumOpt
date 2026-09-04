// Simcenter STAR-CCM+ macro: xxx.java
// Written by Simcenter STAR-CCM+ 21.04.008
package macro;

import java.util.*;
import java.io.FileWriter;
import java.io.PrintWriter;
import java.io.IOException;

import star.common.*;
import star.base.neo.*;
import star.base.report.*;
import star.flow.*;
import star.meshing.*;

public class batch extends StarMacro {

  @Override
  public void execute() {
    execute0();
  }

  private void execute0() {

    ArrayList<Double> alpha_list = new ArrayList<>();
    for (double i=-1.0;i<=10.0;alpha_list.add(i),i=i+1.0);

    String outfile = "C:/Users/Zcaic/Desktop/AirfoilDesign/result.csv";

    // ArrayList<Double> alpha_list = new ArrayList<>(Arrays.asList(2.0,4.0));


    Simulation simulation_0 = getActiveSimulation();

    Solution solution_0 = simulation_0.getSolution();


    MeshPipelineController meshPipelineController_0 = simulation_0.get(MeshPipelineController.class);

    meshPipelineController_0.generateVolumeMesh();

    ScalarGlobalParameter scalarGlobalParameter_0 = ((ScalarGlobalParameter) simulation_0.get(GlobalParameterManager.class).getObject("alpha"));

    Units units_0 = ((Units) simulation_0.getUnitsManager().getObject("deg"));

    solution_0.clearSolution(Solution.Clear.History, Solution.Clear.Fields, Solution.Clear.LagrangianDem);

    for (int i=0;i<alpha_list.size();++i){
      double alpha = alpha_list.get(i); 
      scalarGlobalParameter_0.getQuantity().setValueAndUnits(alpha, units_0);

      simulation_0.getSimulationIterator().run(10);

      ForceCoefficientReport rep_CL= ((ForceCoefficientReport) simulation_0.getReportManager().getReport("CL"));
      double CL = rep_CL.monitoredValue();

      ForceCoefficientReport rep_CD = ((ForceCoefficientReport) simulation_0.getReportManager().getReport("CD"));
      double CD = rep_CD.monitoredValue();

      ExpressionReport rep_CK = ((ExpressionReport) simulation_0.getReportManager().getReport("CK"));
      double CK = rep_CK.monitoredValue();

      PrintWriter pw = null;
      try{
        FileWriter fw = new FileWriter(outfile,true);
        pw = new PrintWriter(fw);
      }catch(IOException e){
        e.printStackTrace();
      }

      String header = "alpha,CL,CD,CK";
      if(i==0){
        pw.println(header);
      }

      String data = String.format("%.3f,%.6f,%.6f,%.6f",alpha,CL,CD,CK);
      pw.println(data);

      pw.close();
    }
  }
}
