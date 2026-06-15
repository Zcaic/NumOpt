function fm = Solve_phi1(x)
%求解轴向及切向诱导速度
global Airfoil fileDataMap   V0  ...
       PI Nb D b_tip Vt_tip Ma_tip Re_tip theta_tip N_pm
    va=x(1);                          %轴向诱导速度
    vt=x(2);                          %切向诱导速度
    phi=atan((V0+va)/(Vt_tip-vt));        %实际气流角
    data=fileDataMap(Airfoil{N_pm}); %翼型气动数据
    alpha=rad2deg(theta_tip-phi);
    xx_Re=data(:,1);
    yy_alpha=data(:,2);
    zz_Ma=data(:,3);
    vv_CD=data(:,4);
    vv_CL=data(:,5);
    Ma0=Ma_tip;
    Re0=Re_tip;
    r=D/2;
    FF1=scatteredInterpolant(xx_Re, yy_alpha, zz_Ma, vv_CD, 'linear','nearest');
    FF2=scatteredInterpolant(xx_Re, yy_alpha, zz_Ma, vv_CL, 'linear','nearest');
    CD=FF1(Re0, alpha, Ma0);
    CL=FF2(Re0, alpha, Ma0);
    %CD=scatteredInterpolant(xx_Re, yy_alpha, zz_Ma, vv_CD, Re0, alpha, Ma0], 'linear','linear');
    %CL=scatteredInterpolant(xx_Re, yy_alpha, zz_Ma, vv_CL, [Re0, alpha, Ma0], 'linear','linear');
    fm(1)=8*PI*r*(V0+va)*va-((V0+va)^2+(Vt_tip-vt)^2)*(CL*cos(phi)-CD*sin(phi))*b_tip;
    fm(2)=8*PI*r*vt*(V0+va)-((V0+va)^2+(Vt_tip-vt)^2)*(CL*sin(phi)+CD*cos(phi))*b_tip;
end