%% ѹ����֪������ƥ��׷���㷨OMP
function [ theta ] = CS_OMP( y,A,t )  
%CS_OMP Summary of this function goes here  
%Version: 1.0 written by jbb0523 @2015-04-18  
%   Detailed explanation goes here  
%   y = Phi * x  
%   x = Psi * theta  
%   y = Phi*Psi * theta  
%   �� A = Phi*Psi, ��y=A*theta  
%   ������֪y��A����theta  
    [y_rows,y_columns] = size(y);  
    if y_rows<y_columns  
        y = y';%y should be a column vector  
    end  
    [M,N] = size(A);%���о���AΪM*N����  
    theta = zeros(N,1);%�����洢�ָ���theta(������)  
    At = zeros(M,t);%�������������д洢A��ѡ�����  
    Pos_theta = zeros(1,t);%�������������д洢A��ѡ��������  
    r_n = y;
    for ii=1:t  
        product = A'*r_n;
        [val,pos] = max(abs(product));
        At(:,ii) = A(:,pos); 
        Pos_theta(ii) = pos;
        A(:,pos) = zeros(M,1); 
        theta_ls = (At(:,1:ii)'*At(:,1:ii))^(-1)*At(:,1:ii)'*y;
        r_n = y - At(:,1:ii)*theta_ls;%���²в�          
    end  
    theta(Pos_theta)=theta_ls;%�ָ�����theta  
end  