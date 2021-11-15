classdef CalRigNPairwiseCalibrated < CalRig & matlab.mixin.Copyable
  % N-camera rig, with pairwise Caltech calibs  
      
  %CalRig
  properties
    nviews;
    viewNames;
  end
  properties (Dependent)
    ncams % same as nviews
  end  
  methods
    function v = get.ncams(obj)
      v = obj.nviews;
    end
  end
  
  properties
    % [ncams,ncams] cell array of pairwise calrig objs
    % This is a (strict) upper-triangular mat with an empty diagonal. 
    crigStros     
  end
      
  methods
    
    function obj = CalRigNPairwiseCalibrated(varargin)
      if nargin==1 
        if isstruct(varargin{1})
          s = varargin{1};
        end
      end
        
      ncam = s.nviews;
      obj.nviews = ncam;
      obj.crigStros = cell(ncam);
      crigs = s.calibrations;
      c = 1;
      % ordering of stereo crigs assumed
      for icam=1:ncam
      for jcam=icam+1:ncam
        obj.crigStros{icam,jcam} = crigs{c};
        c = c+1;
      end
      end
      
      assert(c==numel(crigs)+1);
      
      obj.viewNames = arrayfun(@(x)sprintf('view%d',x),(1:ncam)','uni',0);
    end     
    
  end
  
  methods

    % Helper. Get stereo crig and map "absolute" cam indices to left/right
    function [crigstro,ivwLstro,ivwRstro] = getStroCalRig(obj,iView1,iView2)
      % iView1/2: view indices
      %
      % crigstro: calrig object for iview1,2 pair
      % ivwLstro: 1 if iView1 maps to "left", 2 if "right"
      % ivwRstro: "    iView2  "
      
      if iView1<iView2
        crigstro = obj.crigStros{iView1,iView2};
        ivwLstro = 1;
        ivwRstro = 2;
      elseif iView2<iView1
        crigstro = obj.crigStros{iView2,iView1};
        ivwLstro = 2;
        ivwRstro = 1;
      else
        assert(false);
      end      
    end
        
    %CalRig
    function [xEPL,yEPL] = computeEpiPolarLine(obj,iView1,xy1,iViewEpi,imroi)
      [crigstro,ivwLstro,ivwRstro] = obj.getStroCalRig(iView1,iViewEpi);
      [xEPL,yEPL] = crigstro.computeEpiPolarLine(ivwLstro,xy1,ivwRstro,imroi);      
    end
    
    %CalRig
    function [xRCT,yRCT] = reconstruct(obj,iView1,xy1,iView2,xy2,iViewRct)
      
      assert(numel(xy1)==2);
      assert(numel(xy2)==2);
      
      [crigTri,ivwLTri,ivwRTri] = obj.getStroCalRig(iView1,iView2);
      xy = cat(3,xy1(:),xy2(:));
      idxTri = [ivwLTri ivwRTri]; % [1 2] if iView1/2 are l/r; else [2 1] 
      xy = xy(:,:,idxTri);

      iviewsTriangulate = [iView1 iView2];
      iviewLTriangulate = iviewsTriangulate(idxTri(1));
      iviewRTriangulate = iviewsTriangulate(idxTri(2));

      X = crigTri.triangulate(xy); % X in coord sys of crigstro 'L', or iviewLTriangulate
      % Triangulate 2d points into 3d position
      % xy: [2xnxnviews] 2d image points
      % X: [3xn] reconstructed 3d points. coord sys may depend on concrete
      %   subclass. (typically, coord sys of camera 1.)  
      ylTri = crigTri.x2y(crigTri.project(X,'L'),'L');
      XrTri = crigTri.camxform(X,'LR');
      yrTri = crigTri.x2y(crigTri.project(XrTri,'R'),'R');
      fprintf(1,'RP pt. View %d: %s. View %d: %s.\n',iviewLTriangulate,...
        mat2str(round(ylTri(end:-1:1))),iviewRTriangulate,mat2str(round(yrTri(end:-1:1))));
      
      [crigRC,ivwLRC] = obj.getStroCalRig(iviewLTriangulate,iViewRct);
      CAMXFORMS = {'LR' 'RL'};
      camxform = CAMXFORMS{ivwLRC}; % if ivw1RC==1, translate l->r. else translate r->l.
      Xrc = crigRC.camxform(X,camxform); % transform from iviewLTriangulate->iViewRct
      camRCT = camxform(2);
      y = crigRC.x2y(crigRC.project(Xrc,camRCT),camRCT);
      
      [crigRC2,ivwLRC2] = obj.getStroCalRig(iviewRTriangulate,iViewRct);
      camxform = CAMXFORMS{ivwLRC2};
      Xrc2 = crigRC2.camxform(XrTri,camxform);
      camRCT = camxform(2);
      y2 = crigRC2.x2y(crigRC2.project(Xrc2,camRCT),camRCT);      
      
      xRCT = [y([2 2]) y2(2)];
      yRCT = [y([1 1]) y2(1)];
    end
        
  end
      
  
end