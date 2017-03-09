classdef OrthoCam
  methods (Static)

    function [mx,my,u0,v0,k1,k2,rvecs,t2vecs] = oFcnUnpack(p,nCalIm)
      szassert(p,[6+nCalIm*3+nCalIm*2 1]);
      mx = p(1);
      my = p(2);
      u0 = p(3);
      v0 = p(4);
      k1 = p(5);
      k2 = p(6);
      rvecs = reshape(p(7:7+nCalIm*3-1),3,nCalIm);
      t2vecs = reshape(p(7+nCalIm*3:7+nCalIm*3+nCalIm*2-1),2,nCalIm);      
    end
    function [d,dsum] = oFcn(p,nCalIm,calibWorldPtsXYZ,calibImPts)
      % Objective fcn for Orthogonal projection single-cam calib
      %
      % p = [mx; my; u0; v0; k1; k2; rvecs; t2vecs] where
      %   rvecs: [3xnCalIm] rotation vecs for calib images
      %   t2vecs: [2xnCalIm] tx ty vecs for calib images
      %   all others: scalars
      %
      % calibWorldPtsXYZ: [3xnCalPt] calibration world pts (x, y in calib pattern world frame)
      % calibImPts: [2 x nCalPt x nCalIm] x, y image pts for each cal pattern/pt      
      %
      % d: [nCalPt*nCalIm x 1] euclidean dist reproj err for each cal pt
            
      [mx,my,u0,v0,k1,k2,rvecs,t2vecs] = OrthoCam.oFcnUnpack(p,nCalIm);
      
      % compute projected pts
      nCalPt = size(calibWorldPtsXYZ,2);
      szassert(calibWorldPtsXYZ,[3 nCalPt]);
      szassert(calibImPts,[2 nCalPt nCalIm]);
      
      uvAll = nan(2,nCalPt,nCalIm);
      for iCalIm=1:nCalIm
        R = vision.internal.calibration.rodriguesVectorToMatrix(rvecs(:,iCalIm));
        R2 = R(1:2,:);
        t2 = t2vecs(:,iCalIm);
        
        X2 = R2*calibWorldPtsXYZ + t2;
        szassert(X2,[2 nCalPt]);
        r2 = sum(X2.^2,1); % [1xnCalPt]
        distort = 1 + k1*r2 + k2*r2.^2; % [1xnCalPt]
        
        uv = [ mx*X2(1,:).*distort + u0 ; ...
               my*X2(2,:).*distort + v0 ]; % [2xnCalPt]             
        uvAll(:,:,iCalIm) = uv;
      end
      
      % compute RP err/residual
      d2 = sum((uvAll-calibImPts).^2,1); % [1 nCalPt nCalIm]
      d2 = d2(:); % [nCalPt nCalIm];
      d = sqrt(d2);
      
      dsum = sum(d);
    end
    function d = oFcn1D(p,nCalIm,calibWorldPtsXYZ,calibImPts)
      d = OrthoCam.oFcn(p,nCalIm,calibWorldPtsXYZ,calibImPts);
      d = sqrt(mean(d.^2));
    end
  end
end