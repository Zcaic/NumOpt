%------基于动量定理的螺旋桨性能计算程序------------
%-----版本V0.1------------------------------------
%-----王国镔--------------------------------------
%-----2024-8-6------------------------------------
clear;
clc;
%------读取计算参数文件----------------------------
fid = fopen('旋翼设计参数.txt', 'r');
tline = fgetl(fid);
data= [];
while ischar(tline)
    
    colonIndex = strfind(tline, '：');
    if ~isempty(colonIndex)
        
        value = strtrim(tline(colonIndex+1:end));
        
        data = [data; {value}];
    end
    
    tline = fgetl(fid);
end
fclose(fid);
N_pm=str2double(data(1));
data_input=str2double(data(2:19));
%---初始化计算参数----------------------------------
V0=data_input(1);               %来流速度(m/s)
D=data_input(2);                %旋翼直径(m)
Nb=data_input(3);               % 桨叶数（个)
r_hub=data_input(4);            % 桨毂直径（m)
r_com_loc=data_input(5:8);      % 控制截面位置1~4（R)
b_com=data_input(9:12);         % 控制截面位置1~4弦长（m)
theta_com=data_input(13:16);    % 控制截面位置1~4安装角（°)
RPM_min=data_input(17);         % 最小转速（RPM）
RPM_max=data_input(18);         % 最大转速（RPM）
for i=1:N_pm
    Airfoil{i}=data{19+i};      %控制截面翼型气动力数据文件名
end
%---读取控制截面翼型气动力数据
fileDataMap = containers.Map;    % 初始化一个 Map 来存储已读取文件的内容
for i=1:N_pm
   fileName = Airfoil{i};
    
    % 检查文件是否已被读取过
    if isKey(fileDataMap, fileName)
        % 如果文件已经读取，跳过读取步骤
        disp(['文件 ', fileName, ' 已经读取，跳过。']);
        continue;
    else
        % 如果文件未读取，读取文件并存储数据
        disp(['读取文件: ', fileName]);
        fid = fopen(fileName, 'r');
        
        % 示例读取文件内容，这里假设每个文件包含单个数值
        data = fscanf(fid, '%f',[7,inf])';
        fclose(fid);
        
        % 将读取的数据存储在 Map 中
        fileDataMap(fileName) = data;
    end
end

%---基于螺旋桨性能计算--------------------------------










msgbox('OK');














