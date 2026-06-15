function fm = Solve_ind_V(x)
%求解轴向及切向诱导速度
global Airfoil fileDataMap pm_id Ma Re phi0 sigma theta_ele V0 Vt ...
       r PI Nb b_ele D phi1
    va=x(1);                          %轴向诱导速度
    vt=x(2);                          %切向诱导速度
    phi=atan((V0+va)/(Vt-vt));        %实际气流角
    data=fileDataMap(Airfoil{pm_id}); %翼型气动数据
    alpha=rad2deg(theta_ele-phi);
    xx_Re=data(:,1);
    yy_alpha=data(:,2);
    zz_Ma=data(:,3);
    vv_CD=data(:,4);
    vv_CL=data(:,5);
    Ma0=Ma;
    Re0=Re;
    FF1=scatteredInterpolant(xx_Re, yy_alpha, zz_Ma, vv_CD, 'linear','nearest');
    FF2=scatteredInterpolant(xx_Re, yy_alpha, zz_Ma, vv_CL, 'linear','nearest');
    CD=FF1(Re0, alpha, Ma0);
    CL=FF2(Re0, alpha, Ma0);
    f=Nb/2*(D/2-r)/(sin(phi1));
    F=2/PI*acos(exp(-f));
    fm(1)=8*PI*r*(V0+va)*va*F-((V0+va)^2+(Vt-vt)^2)*(CL*cos(phi)-CD*sin(phi))*b_ele;
    fm(2)=8*PI*r*vt*(V0+va)*F-((V0+va)^2+(Vt-vt)^2)*(CL*sin(phi)+CD*cos(phi))*b_ele;
end