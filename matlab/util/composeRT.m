function [R31,T31] = composeRT(R21,T21,R32,T32)
  % Compose two rigid transforms.  With the convention
  %   X_M = RMN * X_N + TMN,
  % given frame-2-from-1 and frame-3-from-2, return frame-3-from-1.

  R31 = R32*R21;
  T31 = R32*T21+T32;
end  % function
