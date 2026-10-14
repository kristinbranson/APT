function [R1,T1] = invertRT(R,T)
  % Invert a rigid transform X2 = R*X1 + T, returning R1, T1 such that
  % X1 = R1*X2 + T1.

  R1 = R';
  T1 = -R'*T;
end  % function
