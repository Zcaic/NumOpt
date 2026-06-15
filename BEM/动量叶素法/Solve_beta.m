function fm = Solve_beta(x)
%求解的beta单位为rad
global Airfoil fileDataMap pm_id Ma Re phi0 sigma theta_ele
    beta=x(1);
    gama=x(2);
    CL=x(3);
    data=fileDataMap(Airfoil{pm_id}); %翼型气动数据
    alpha=rad2deg(theta_ele-beta-phi0);
    xx_Re=data(:,1);
    yy_alpha=data(:,2);
    zz_Ma=data(:,3);
    vv_CD=data(:,4);
    vv_CL=data(:,5);
    if Ma<0.05
        Ma0=0.05;
    elseif Ma>0.9
        Ma0=0.9
    else
        Ma0=Ma;
    end
    if Re<116000
        Re0=116000;
    elseif Re>21000000
        Re0=21000000;
    else
        Re0=Re;
    end
    CD=griddata(xx_Re, yy_alpha, zz_Ma, vv_CD, Re0, alpha, Ma0, 'linear');
    fm(3)=CL-griddata(xx_Re, yy_alpha, zz_Ma, vv_CL, Re0, alpha, Ma0, 'linear');
    fm(1)=CL*sigma-4*sin(phi0+beta)*tan(beta)/(1-tan(gama)*tan(beta));
    fm(2)=gama-atan(CD/CL);
end