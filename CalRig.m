classdef CalRig < handle
  
  properties (Abstract)
    nviews
    viewNames % [nviews]. cellstr viewnames
    viewSizes % [nviews x 2]. viewSizes(iView,:) gives [nr nc]
  end
    
  methods (Abstract)
    
    % iView1: view index for anchor point
    % xy1: [2]. [x y] vector, cropped coords in iView1
    % iViewEpi: view index for target view (where EpiLine will be drawn)
    %
    % xEPL,yEPL: epipolar line, cropped coords, iViewEpi
    [xEPL,yEPL] = computeEpiPolarLine(obj,iView1,xy1,iViewEpi)
    
    % iView1: view index for anchor point
    % xy1: [2]. [x y] vector, cropped coords in iView1
    % iView2: view index for 2nd point
    % xy2: [2]. etc
    % iViewRct: view index for target view (where reconstructed point will be drawn)
    %
    % xRCT: [3] Reconstructed point spread, iViewRct, cropped coords
    % yRCT: [3] etc.
    % A "point spread" is 3 points specifying a line segment for a 
    % reconstructed point. The midpoint of the line segment (2nd point in
    % point spread) is the most likely reconstructed location. The two
    % endpoints represent extremes that lie precisely on one EPL (but not
    % necessarily the other and vice versa).
    [xRCT,yRCT] = reconstruct(obj,iView1,xy1,iView2,xy2,iViewRct)
    
  end
  
end