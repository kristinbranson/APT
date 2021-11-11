classdef CalRig < handle
  
  properties (Abstract)
    nviews
    viewNames % [nviews]. cellstr viewnames
    %viewRois % [nviews x 4]. viewRois(iView,:) gives [xlo xhi ylo yhi]
  end

  properties
    sourceFile = '';
  end
  
  methods (Abstract)
    
    % iView1: view index for anchor point
    % xy1: [2]. [x y] vector, cropped coords in iView1
    % iViewEpi: view index for target view (where EpiLine will be drawn)
    % roiEpi: [xlo xhi ylo yhi] roi (in iViewEpi) where epipolar lines 
    %   should be computed
    %
    % xEPL,yEPL: epipolar line, cropped coords, iViewEpi. Note, x and y are
    % x- and y-coords, NOT row/col coords.
    [xEPL,yEPL] = computeEpiPolarLine(obj,iView1,xy1,iViewEpi,roiEpi)
    
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

    function [X,xyrp,rpe] = triangulate(obj,xy)
      % Triangulate 2d points into 3d position
      %
      % xy: [2xnxnviews] 2d image points
      % 
      % X: [3xn] reconstructed 3d points. coord sys may depend on concrete
      %   subclass. (typically, coord sys of camera 1.)
      % xyrp: [2xnxnviews] reprojected 2d image points
      % rpe: [nxnviews] L2 reproj err
      
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
    
    function obj = loadCreateCalRigObjFromFile(fname)
      % Create/load a concerete CalRig object from file
      %
      % obj: Scalar CalRig object; concrete type depends on file contents
      % tfSetViewRois: scalar logical. If true, obj.viewRois need setting
      
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
%         tfSetViewRois = true;
      elseif isa(s.(vars{1}),'CalRig') % Could check all vars
        obj = s.(vars{1});
%         tfSetViewRois = false;
      elseif isa(s.(vars{1}),'vision.internal.calibration.tool.Session')
        obj = CalRigMLStro(s.(vars{1})); % will auto-calibrate and offer save
      elseif all(ismember({'DLT_1' 'DLT_2'},vars))
        % SH
        obj = CalRigSH;
        obj.setKineData(fname);
%         tfSetViewRois = true;
      elseif all(ismember({'om' 'T' 'R' 'active_images_left' 'recompute_intrinsic_right'},vars))
        % Bouget Calib_Results_stereo.mat file
        % NOTE: could check calibResultsStereo.nx and .ny vs viewSizes
        obj = CalRig2CamCaltech(fname);
%         tfSetViewRois = true;
      elseif ismember('type', vars) && strcmp(s.type, 'multi_caltech')
        % obj.caltech = {};
        % obj.nviews = s.nviews;
        % for i = 1:s.nviews
        %   obj.caltech{i} = CalRig2CamCaltech(s.calibrations{i});
        % end
        obj = s;
      else
        error('CalRig:load',...
          'Calibration file ''%s'' has unrecognized contents.',fname);
      end
      obj.sourceFile = fname;
      
    end
  end
  
  methods (Static) % Utilities
    
    function y = cropLines(y,roi)
      % "Crop" lines projected on image -- replace points that lie outside
      % of image with NaN.
      %
      % y: [Nx2] (row,col) "cropped coords" (ie pixel coords on projected image)
      % roi: [1x4] [xlo xhi ylo yhi] where y should be cropped
      %
      % y: [Nx2], with OOB points replaced with nan in both coords
      
      assert(size(y,2)==2);
      
%       roi = obj.viewRois(viewIdx,:);
      rows = y(:,1);
      cols = y(:,2);
      tfOOB = rows<roi(3) | rows>roi(4) | cols<roi(1) | cols>roi(2);
      y(tfOOB,:) = nan;
    end
    
    function y = getLineWithinAxes(y,roi)
      % Like cropLines

      assert(size(y,2)==2);
      
%       roi = obj.viewRois(viewIdx,:);
      clo = roi(1);
      chi = roi(2);
      rlo = roi(3);
      rhi = roi(4);
      r = y(:,1);
      c = y(:,2);
      
      badidx = r>rhi | c>chi | r<rlo | c<clo;
      r(badidx) = NaN;
      c(badidx) = NaN;
      
      [maxr,maxri] = max(r);
      [minr,minri] = min(r);
      [maxc,maxci] = max(c);
      [minc,minci] = min(c);
      dr = maxr-minr;
      dc = maxc-minc;
      if dr > dc % abs(slope)>1
        mrecip = (c(maxri)-c(minri)) / dr; % 1/slope
        % equation of the line:
        % (y-minr) = slope*(x-c(minri))
        % solve for x at y = rlo and y = rhi
        rout = [minr;maxr];
        cout = (rout-minr)*mrecip+c(minri);
      else % abs(slope)<=1
        m = (r(maxci)-r(minci)) / dc; % slope
        % equation of the line:
        % (y-r(minci)) = m*(x-minc)
        % solve for y at x = clo and x = chi
        cout = [minc;maxc];
        rout = m*(cout-minc)+r(minci);        
      end
      y = [rout,cout];
    end

  end
  
end
