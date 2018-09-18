%% 压缩感知的正交匹配追踪算法OMP
function [ theta ] = CS_OMP( y,A,t )  
%CS_OMP Summary of this function goes here  
%Version: 1.0 written by jbb0523 @2015-04-18  
%   Detailed explanation goes here  
%   y = Phi * x  
%   x = Psi * theta  
%   y = Phi*Psi * theta  
%   令 A = Phi*Psi, 则y=A*theta  
%   现在已知y和A，求theta  
    [y_rows,y_columns] = size(y);  
    if y_rows<y_columns  
        y = y';%y should be a column vector  
    end  
    [M,N] = size(A);%传感矩阵A为M*N矩阵  
    theta = zeros(N,1);%用来存储恢复的theta(列向量)  
    At = zeros(M,t);%用来迭代过程中存储A被选择的列  
    Pos_theta = zeros(1,t);%用来迭代过程中存储A被选择的列序号  
    r_n = y;
    for ii=1:t  
        product = A'*r_n;
        [val,pos] = max(abs(product));
        At(:,ii) = A(:,pos); 
        Pos_theta(ii) = pos;
        A(:,pos) = zeros(M,1); 
        theta_ls = (At(:,1:ii)'*At(:,1:ii))^(-1)*At(:,1:ii)'*y;
        r_n = y - At(:,1:ii)*theta_ls;%更新残差          
    end  
    theta(Pos_theta)=theta_ls;%恢复出的theta  
end  