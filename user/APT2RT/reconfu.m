function [H] = reconfu(A,L)
%function [H] = reconfu(A,L)
% Description:  Reconstruction of 3D coordinates with the use local (camera
%               coordinates and the DLT coefficients for the n cameras).
% Input:        - A  file containing DLT coefficients of the n cameras
%                    [a1cam1,a1cam2...;a2cam1...]
%               - L  camera coordinates of points
%                    [xcam1,ycam1,xcam2,ycam2...;same at time 2]
% Output:       - H  global coordinates, residuals, cameras used
%                    [Xt1,Yt1,Zt1,residt1,cams_used@t1...; same for t2]
% Author:       Christoph Reinschmidt, HPL, The University of Calgary
% Date:         September, 1994
% Last change:  November 29, 1996
% Version:      1.1

n=size(A,2);
% check whether the numbers of cameras agree for A and L
if 2*n~=size(L,2); disp('the # of cameras given in A and L do not agree')
                   disp('hit any key and then "try" again'); pause; return;
end


H(size(L,1),5)=[0];         % initialize H

% ________Building L1, L2:       L1 * G (X,Y,Z) = L2________________________

for k=1:size(L,1)  %number of time points
    q=[0]; L1=[]; L2=[];  % initialize L1,L2, q(counter of 'valid' cameras)
  for  i=1:n       %number of cameras
    x=L(k,2*i-1); y=L(k,2*i);
    if ~(isnan(x) | isnan(y))  % do not construct l1,l2 if camx,y=NaN
     q=q+1;
     L1([q*2-1:q*2],:)=[A(1,i)-x*A(9,i), A(2,i)-x*A(10,i), A(3,i)-x*A(11,i); ...
                        A(5,i)-y*A(9,i), A(6,i)-y*A(10,i), A(7,i)-y*A(11,i)];
     L2([q*2-1:q*2],:)=[x-A(4,i);y-A(8,i)];
    end
  end

  if (size(L2,1)/2)>1  %check whether enough cameras available (at least 2)
   g=L1\L2; h=L1*g; DOF=(size(L2,1)-3);
   avgres=sqrt(sum([L2-h].^2)/DOF);
  else
   g=[NaN;NaN;NaN]; avgres=[NaN];
  end
   
 %find out which cameras were used for the 3d reconstruction
  b=fliplr(find(sum(reshape(isnan(L(k,:)),2,size(L(k,:),2)/2))==0));
  if size(b,2)<2; camsused=[NaN];
     else,    for w=1:size(b,2), b(1,w)=b(1,w)*10^(w-1); end    
              camsused=sum(b');
  end  

  H(k,:)=[g',avgres,camsused];
end

