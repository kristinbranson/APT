function xp = normalized2projectedCoords(xn,alpha_c,cc,fc,kc)
  % Map normalized camera coords to projected (pixel) coords by applying the
  % Bouguet lens-distortion model and the camera matrix.
  %
  % xn: [2xn], normalized coords
  % alpha_c, cc, fc, kc: Bouguet intrinsics (skew, principal point, focal
  %   lengths, distortion coefficients)
  %
  % xp: [2xn], projected coords

  [d,n] = size(xn);
  assert(d==2);

  r2 = sum(xn.^2,1); % [1xn], r-squared for each pt
  radlDistortFac = 1 + kc(1)*r2 + kc(2)*r2.^2 + kc(5)*r2.^3; % [1xn]
  dx = [... % tangential distortion
    2*kc(3)*xn(1,:).*xn(2,:) + kc(4)*(r2+2*xn(1,:).^2); ...
    kc(3)*(r2+2*xn(2,:).^2) + 2*kc(4)*xn(1,:).*xn(2,:)]; % [2xn]
  xd = repmat(radlDistortFac,2,1).*xn + dx; % [2xn]

  KK = [fc(1) alpha_c*fc(1) cc(1); ...
        0     fc(2)         cc(2)];
  xp = KK * [xd;ones(1,n)];
end  % function
