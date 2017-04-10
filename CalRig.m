classdef CalRig < handle
  
  properties (Abstract)
    nviews
    viewNames % [nviews]. cellstr viewnames
    viewSizes % [nviews x 2]. viewSizes(iView,:) gives [nc nr] or [width height]
  end
    
  methods (Abstract)
    
    % iView1: view index for anchor point
    % xy1: [2]. [x y] vector, cropped coords in iView1
    % iViewEpi: view index for target view (where EpiLine will be drawn)
    %
    % xEPL,yEPL: epipolar line, cropped coords, iViewEpi. Note, x and y are
    % x- and y-coords, NOT row/col coords.
    [xEPL,yEPL] = computeEpiPolarLine(obj,iView1,xy1,iViewEpi)
    
  end
  
  methods % Conceptually Abstract, stubs assert
    
    function [xRCT,yRCT] = reconstruct(obj,iView1,xy1,iView2,xy2,iViewRct)
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
    
      assert(false,'Unimplemented.');
    end

    function [u_p,v_p,w_p] = reconstruct2d(obj,x,y,iView)
      % Project 2D cropped coords in iView to 3D/world coords
      %
      % x,y: [npt] vectors, cropped coords (x and y coords, NOT row/col coords)
      %
      % u_p, v_p, w_p: [nptx2] parameters defining 3D/world lines 
      %   u_p = line parameters of real world u-coord in form 
      %              u(t) = u_p(1) + u_p(2)*t

      assert(false,'Unimplemented.');      
    end
    
    function [x,y] = project3d(obj,u,v,w,iView)
      % Project 3D/world coords to cropped coords in iView
      %
      % u,v,w: [any size] 3D/world coords
      %
      % x,y: [same size as u,v,w] cropped coords (x and y coords, NOT row/col coords)
      
      assert(false,'Unimplemented.');
    end

  end
  
  methods % Conceptually abstract, for use with CPR/RegressorCascade, shapeGt
    
    function [r,c] = projectCPR(obj,X,iView)
      % Project 3D point onto a 2D view.
      %
      % X: [3xN] 3d points in coords of iView cam.
      % iView: view index
      %
      % r: [N]. row-coordinates, cropped coords in iView. 
      % c: [N]. col-coords, cropped coords in iView.
      assert(false,'Unimplemented.');
    end
    
    function X2 = viewXformCPR(obj,X1,iView1,iView2)
      % Change extrinsic/camera 3D coord systems.
      %
      % X1: [3xN] 3d points in coords of iView1 cam.
      % iView1, iView2: view indices
      %
      % X2: [3xN] 3d points in coords of iView2 cam.
      assert(false,'Unimplemented.');
    end
    
  end
  
  methods (Static)
    
    function [obj,tfSetViewSizes] = loadCreateCalRigObjFromFile(fname)
      % Create/load a concerete CalRig object from file
      %
      % obj: Scalar CalRig object; concrete type depends on file contents
      % tfSetViewSizes: scalar logical. If true, obj.viewSizes need setting
      
      if exist(fname,'file')==0
        error('Labeler:file','File ''%s'' not found.',fname);
      end
      s = load(fname,'-mat'); % Could use whos('-file') with superclasses()
      vars = fieldnames(s);
      if numel(vars)==0
        error('CalRig:load','No variables found in file: %s.',fname);
      end
      
      if isa(s.(vars{1}),'OrthoCamCalPair')
        obj = s.(vars{1});
        tfSetViewSizes = true;
      elseif isa(s.(vars{1}),'CalRig') % Could check all vars
        obj = s.(vars{1});
        tfSetViewSizes = false;
      elseif all(ismember({'DLT_1' 'DLT_2'},vars))
        % SH
        obj = CalRigSH;
        obj.setKineData(fname);
        tfSetViewSizes = true;
      elseif all(ismember({'om' 'T' 'R' 'active_images_left' 'recompute_intrinsic_right'},vars))
        % Bouget Calib_Results_stereo.mat file
        % NOTE: could check calibResultsStereo.nx and .ny vs viewSizes
        obj = CalRig2CamCaltech(fname);
        tfSetViewSizes = true;        
      else
        error('CalRig:load',...
          'Calibration file ''%s'' has unrecognized contents.',fname);
      end
    end
  end
  
  methods % Utilities
    
    function y = cropLines(obj,y,viewIdx)
      % "Crop" lines projected on image -- replace points that lie outside
      % of image with NaN.
      %
      % y: [Nx2] (row,col) "cropped coords" (ie pixel coords on projected image)
      % viewIdx: index into viewNames/viewSizes
      %
      % y: [Nx2], with OOB points replaced with nan in both coords
      
      assert(size(y,2)==2);
      
      vSize = obj.viewSizes(viewIdx,:);
      nc = vSize(1);
      nr = vSize(2);
      rows = y(:,1);
      cols = y(:,2);
      tfOOB = rows<1 | rows>nr | cols<1 | cols>nc;
      y(tfOOB,:) = nan;
    end
    
    function y = getLineWithinAxes(obj,y,viewIdx)

      assert(size(y,2)==2);
      
      vSize = obj.viewSizes(viewIdx,:);
      nc = vSize(1);
      nr = vSize(2);
      
      r = y(:,1);
      c = y(:,2);
      
      [maxr,maxri] = max(r);
      [minr,minri] = min(r);
      [maxc,maxci] = max(c);
      [minc,minci] = min(c);
      dr = maxr-minr;
      dc = maxc-minc;
      if dr > dc,
        m = (c(maxri)-c(minri)) / dr;
        % equation of the line:
        % (y-minr) = m*(x-c(minri))
        % solve for x at y = 1 and y = nr
        rout = [1;nr];
        cout = (rout-minr)/m+c(minri);
      else
        m = (r(maxci)-r(minci)) / dc;
        % equation of the line:
        % (y-r(minci)) = m*(x-minc)
        % solve for y at x = 1 and x = nc
        cout = [1;nc];
        rout = m*(cout-minc)+r(minci);        
      end
      y = [rout,cout];
    end

  end
  
end