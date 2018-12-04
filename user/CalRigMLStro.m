classdef CalRigMLStro < CalRigZhang2CamBase
    
  properties
    calSess % scalar Session from ML Stro calibration
    
    eplineComputeMode = 'mostaccurate'; % either 'mostaccurate' or 'fastest'
  end
  properties (Dependent)
    stroParams
  end
  
  % Extrinsics
  properties (Dependent,Hidden)
    RLR
    RRL
    TLR
    TRL
  end
  %Intrinsics
  properties
    int % dups state in calSess, but non-Dependent for perf
  end

  methods
    function v = get.stroParams(obj)
      v = obj.calSess.CameraParameters;
    end
    function R = get.RLR(obj)
      R = obj.stroParams.RotationOfCamera2'; 
      % ML defines R, T by 
      %[x y z] = [X Y Z]*R + t
      
      %R = obj.RRL';
    end
    function R = get.RRL(obj)
      R = obj.RLR';
    end
    function T = get.TLR(obj)
      T = obj.stroParams.TranslationOfCamera2(:);
      %T = -obj.RRL'*obj.TRL;
    end
    function T = get.TRL(obj)
      T = -obj.RLR'*obj.TLR;
    end
    %function s = get.int(obj)
    function s = getInt(obj)
      sp = obj.stroParams;
      
      s1 = struct();
      [s1.fc,s1.cc,s1.kc,s1.alpha_c] = ...
        CalRigMLStro.camParams2Intrinsics(sp.CameraParameters1);
      s2 = struct();
      [s2.fc,s2.cc,s2.kc,s2.alpha_c] = ...
        CalRigMLStro.camParams2Intrinsics(sp.CameraParameters2);
      
      s = struct(obj.viewNames{1},s1,obj.viewNames{2},s2);
    end
  end
  
  methods
    
    function obj = CalRigMLStro(calibSession)
      assert(isa(calibSession,'vision.internal.calibration.tool.Session'));
      obj.calSess = calibSession;
      obj.int = obj.getInt();
      
      obj.autoCalibrateProj2NormFuncTol();
    end
    
    function autoCalibrateProj2NormFuncTol(obj,varargin)
      % Automatically set .proj2NormFuncTol by "round-tripping" calibration
      % points through projected2normalized/normalized2projected.
      %
      % normalized2projected takes a point in normalized coordinates for a
      % camera and projects onto the final image by recentering about the
      % principal point, applying zoom/skew, and applying distortions.
      % The distortions are highly nonlinear closed-form expressions.
      %
      % projected2normalized inverts normalized2projected numerically,
      % using eg lsqnonlin. This inversion/optimization will have a functol
      % or other threshold controlling how precise this inversion must be.
      % This method/inversion is necesary for computing EP lines (as of 18b
      % there does not appear to be a MATLAB CV toolbox library fcn 
      % available; MATLAB's triangulate() does its own nonlinear
      % inversion.)
      %
      % We auto-calibrate this threshold by requiring that the round-trip
      % error result in less than l2pxerr (optionally supplied).
      
      calPtL2RTerr = myparse(varargin,...
        'calPtL2RTerr',1/8 ... % in pixels
      );
      
      cs = obj.calSess;
      bs = cs.BoardSet;
      calpts = bs.BoardPoints; % ipt, x/y, ipat, ivw
      [npts,d,npat,nvw] = size(calpts);
      calpts = permute(calpts,[1 3 2 4]);
      calpts = reshape(calpts,[npts*npat d nvw]);

      fprintf(1,'Auto-calibrating .projected2normalized funcTol with %d calibration points.\n',...
        npts*npat);
      fprintf(1,'Required round-trip L2 accuracy: %.3f px.\n',calPtL2RTerr);

      functol = 1e-6;
      
      % FUTURE: factor core into CalRigZhang2CamBase
      calptsRT = nan(size(calpts));
      while 1
        fprintf(1,' ... trying funcTol=%.3g\n',functol);
        
        for icam=1:2
          calptsNorm = arrayfun(@(i)obj.projected2normalized(calpts(i,:,icam)',icam,'functol',functol),...
            1:npts*npat,'uni',0);
          calptsNorm = cat(2,calptsNorm{:});
          calptsRT(:,:,icam) = obj.normalized2projected(calptsNorm,icam)';
        end

        d = calpts-calptsRT;
        err = squeeze(sqrt(sum(d.^2,2)));
        szassert(err,[npts*npat nvw]);
        maxerr = max(err);
        
        fprintf(' ... max roundtrip err, view1: %.3g. view2: %.3g.\n',...
          maxerr(1),maxerr(2));
        if all(maxerr<calPtL2RTerr)
          fprintf(' ... SUCCESS! Setting .proj2NormFuncTol to %.3g.\n',functol);
          obj.proj2NormFuncTol = functol;
          break;
        else
          functol = functol/2;
        end
      end
    end
    
    function autoCalibrateEplineZrange(obj,varargin)
      % Automatically set .eplineZrange by computing EPlines from view1->2 
      % and vice versa at i) the four corners and ii) center of views:
      %
      % - The start/end of the zrange must fully accomodate all EPlines 
      % 
      % - dz is chosen so that the average spacing discretization of 
      % EPlines is ~1px (see optional param)
      %
      % Note, the bulk of computational expense for EP-line computation is
      % the nonlinear inversion in projected2normalized(). Using a zrange 
      % that is too big or too fine shouldn't hurt much. This 
      % auto-calibration may still be handy however
      %
      % Currently assumes two cameras have the SAME IMAGE SIZE.
      
      [eplineDxy,zrangeWidthTitrationFac] = myparse(varargin,...
        'eplineDxy',1, ... % in pixels
        'zrangeWidthTitrationFac',1.5 ... % increase range of zwidth by this much every step
      );
    
      cs = obj.calSess;
      bs = cs.BoardSet;
      calpts = bs.BoardPoints; % ipt, x/y, ipat, ivw
      [npts,d,npat,nvw] = size(calpts);
      calpts = permute(calpts,[1 3 2 4]);
      calpts = reshape(calpts,[npts*npat d nvw]);
      roi = [1 bs.ImageSize(2) 1 bs.ImageSize(1)];

      fprintf(1,'Auto-calibrating .eplineZrange. Imagesize is: %s.\n',...
        mat2str(bs.ImageSize));

      fprintf(1,'Starting with %d calpts.\n',npts*npat);
      
      % Start with calpts.
      % Initial z-range: fits
      % Using arrayfun here due to O(n^2) behavior, see stereoTriangulateML
      Xcalpts1 = arrayfun(@(i)...
        obj.stereoTriangulateML(calpts(i,:,1)',calpts(i,:,2)'),1:npts*npat,'uni',0);
      Xcalpts1 = cat(2,Xcalpts1{:});
      Xcalpts2 = obj.camxform(Xcalpts1,[1 2]);      
      zrange0 = [ min(Xcalpts1(3,:)) max(Xcalpts1(3,:));...
                  min(Xcalpts2(3,:)) max(Xcalpts2(3,:)) ]; 
      % zrange: ivw, zmin/zmax
      
      % Auto-calibrate dz so that the spacing of EPline pts in each view is
      % ~eplineDxy.
      dz = 1;
      zrange1 = zrange0(1,1):dz:zrange0(1,2);
      zrange2 = zrange0(2,1):dz:zrange0(2,2);
      fprintf('Choosing dz. Starting with dz=%.3f.\n',dz);
      epdmu = nan(npts*npat,2);
      for i=1:npts*npat
        [ep12x,ep12y] = ...
          obj.computeEpiPolarLine(1,calpts(i,:,1)',2,roi,'z1range',zrange1);
        [ep21x,ep21y] = ...
          obj.computeEpiPolarLine(2,calpts(i,:,2)',1,roi,'z1range',zrange2);
        
        ep12dxy = diff([ep12x ep12y],1);
        ep21dxy = diff([ep21x ep21y],1);
        ep12d = sqrt(sum(ep12dxy.^2,2));
        ep21d = sqrt(sum(ep21dxy.^2,2));
        epdmu(i,1) = nanmean(ep12d);
        epdmu(i,2) = nanmean(ep21d); 
      end
      epdmu = mean(epdmu);
      fprintf('Mean spacing in EP lines in view1/2: %s.\n',mat2str(epdmu));
      epdxy = mean(epdmu);
      dz = dz * eplineDxy/epdxy;
      fprintf('Selected dz=%.3f.\n',dz);

      % Autocalibrate zrange by expanding the zrange until the number of
      % in-bounds points stops changing.
      zRangeWidthCtr = [diff(zrange0,1,2) mean(zrange0,2)];
      % zRangeWidthCtr(ivw,1) is width
      % zRangeWidthCtr(ivw,2) is center
      
      % four corners and center
      testpts = [roi([1 3]); roi([1 4]); roi([2 4]); roi([3 4]); ...
        mean(roi([1 2])) mean(roi([3 4]))];
      ntestpts = size(testpts,1);
      fprintf(1,'Selecting zrange with %d test points.\n',ntestpts);
      
      nInBounds = zeros(ntestpts,2);
      while 1      
        zrangelims1 = zRangeWidthCtr(1,2) + 0.5*zRangeWidthCtr(1,1)*[-1 1];
        zrangelims2 = zRangeWidthCtr(2,2) + 0.5*zRangeWidthCtr(2,1)*[-1 1];
        zrange1 = zrangelims1(1):dz:zrangelims1(2);
        zrange2 = zrangelims2(1):dz:zrangelims2(2);
        
        fprintf('View1 zrange (n=%d): %s. View2 zrange (n=%d): %s.\n',...
          numel(zrange1),mat2str(zrange1([1 end])),...
          numel(zrange2),mat2str(zrange2([1 end])));

        nInBoundsNew = nan(ntestpts,2);
        for i=1:ntestpts
          [~,~,~,tfOOB1] = ...
            obj.computeEpiPolarLine(1,testpts(i,:)',2,roi,'z1range',zrange1);
          [~,~,~,tfOOB2] = ...
            obj.computeEpiPolarLine(2,testpts(i,:)',1,roi,'z1range',zrange2);
          nInBoundsNew(i,:) = [nnz(~tfOOB1) nnz(~tfOOB2)];
        end
        
        disp(nInBoundsNew);
        
        if isequal(nInBounds,nInBoundsNew)
          fprintf('SUCCESS! Setting .eplineZrange.\n');
          break;
        end
        
        for ivw=1:2
          if ~isequal(nInBounds(:,ivw),nInBoundsNew(:,ivw))
            zRangeWidthCtr(ivw,1) = zRangeWidthCtr(ivw,1)*zrangeWidthTitrationFac;
          end
        end
        
        nInBounds = nInBoundsNew;
        
      end
      
      obj.eplineZrange = {zrange1 zrange2};
    end
    
    function [rperr1ml,rperr2ml] = checkRPerr(obj,varargin)
      [showplots,wbObj,randsampcorners,randsampcornersN] = myparse(varargin,...
        'showplots',true,...
        'wbObj',[],...
        'randsampcorners',false,... % if true, sample only randsampN corners
        'randsampcornersN',20 ...
        );
      tfWB = ~isempty(wbObj);
      
      % compute RP err using ML lib fcns
      
      sp = obj.stroParams;
      cp1 = sp.CameraParameters1;
      cp2 = sp.CameraParameters2;
      
      boardpts = obj.calSess.BoardSet.BoardPoints;
      [ncorners,d,ncalpairs,ncam] = size(boardpts);
      assert(d==2);
      assert(ncam==2);
      
      bpcam1 = boardpts(:,:,:,1);
      bpcam1 = permute(bpcam1,[1 3 2]); % bpcam1 is [ncorners x ncalpairs x 2]
      bpcam2 = boardpts(:,:,:,2);
      bpcam2 = permute(bpcam2,[1 3 2]);
      
      if randsampcorners
        ptmp = randperm(ncorners);
        icorners = ptmp(1:randsampcornersN);
        fprintf(1,'Selected %d/%d checkerboard corners to consider.\n',...
          randsampcornersN,ncorners);
        bpcam1 = bpcam1(icorners,:,:);
        bpcam2 = bpcam2(icorners,:,:);
        ncorners = randsampcornersN;
      end
      bpcam1 = reshape(bpcam1,ncorners*ncalpairs,2);
      bpcam2 = reshape(bpcam2,ncorners*ncalpairs,2);
      bpcam1ud = undistortPoints(bpcam1,cp1);
      bpcam2ud = undistortPoints(bpcam2,cp2);
      
      wp = triangulate(bpcam1ud,bpcam2ud,sp);
      
      bpcam1rp = worldToImage(cp1,eye(3),[0;0;0],wp,'applyDistortion',true);
      bpcam2rp = worldToImage(cp2,...
        sp.RotationOfCamera2,sp.TranslationOfCamera2,wp,'applyDistortion',true);
            
      % using CalRigZhang2CamBase
      ntot = size(bpcam1,1);
      X1 = nan(3,ntot);
      X2 = nan(3,ntot);
      d = nan(ntot,1);
      if tfWB
        wbObj.startPeriod('stro tri','shownumden',true,'denominator',ntot);
      end
      for i=1:ntot
        if tfWB
          wbObj.updateFracWithNumDen(i);
        end
        [X1(:,i),X2(:,i),d(i)] = ...
          obj.stereoTriangulate(bpcam1(i,:)',bpcam2(i,:)','cam1','cam2');
      end
      xn1 = X1(1:2,:)./X1(3,:);
      xn2 = X2(1:2,:)./X2(3,:);
      xp1 = obj.normalized2projected(xn1,'cam1');
      xp2 = obj.normalized2projected(xn2,'cam2');
      
      rperr1ml = sqrt(sum((bpcam1-bpcam1rp).^2,2));
      rperr1ml = reshape(rperr1ml,[ncorners ncalpairs]);
      rperr2ml = sqrt(sum((bpcam2-bpcam2rp).^2,2));
      rperr2ml = reshape(rperr2ml,[ncorners ncalpairs]);
      
      rperr1ours = sqrt(sum((bpcam1-xp1').^2,2));
      rperr1ours = reshape(rperr1ours,[ncorners ncalpairs]);
      rperr2ours = sqrt(sum((bpcam2-xp2').^2,2));
      rperr2ours = reshape(rperr2ours,[ncorners ncalpairs]);
      
      if showplots
        cs = obj.calSess;

        figure('Name','RP err per showReprojectionErrors()');
        ax = axes;
        showReprojectionErrors(sp,'parent',ax);
        fprintf('Calibration Session has meanRPerr = %.3f\n',...
          cs.CameraParameters.MeanReprojectionError);
        
        figure('Name','RP err per ML library functions');
        ax2 = axes;
        rpe1perim = mean(rperr1ml,1);
        rpe2perim = mean(rperr2ml,1);
        rpeBar = [rpe1perim; rpe2perim];
        hbar = bar(rpeBar','grouped');
        legend(hbar,{'cam1' 'cam2'});
        title('Mean RP err per image, using lib fcns','fontweight','bold');
        xlabel('Cal image pair');
        ylabel('Mean RP err (px)');
        fprintf('Using ML library functions gives meanRPerr = %.3f\n',...
          mean(rpeBar(:)));
        
        figure('Name','RP err per our functions');
        ax3 = axes;
        rpe1perim = mean(rperr1ours,1);
        rpe2perim = mean(rperr2ours,1);
        rpeBar = [rpe1perim; rpe2perim];
        hbar = bar(rpeBar','grouped');
        legend(hbar,{'cam1' 'cam2'});
        title('Mean RP err per image, using our fcns','fontweight','bold');
        xlabel('Cal image pair');
        ylabel('Mean RP err (px)');
        fprintf('Using our functions gives meanRPerr = %.3f\n',...
          mean(rpeBar(:)));
        
        linkaxes([ax ax2 ax3]);
      end
    end
  end
  
  methods (Static)
    function rperrPlot(rperrcam1,rperrcam2)
      % rperrcam1: [N x nphyspts] rp err
      
      szassert(rperrcam1,size(rperrcam2));
      [N,nphyspts] = size(rperrcam1);
      
      figure('Name','RP err');
      ax1 = axes;

      X = [rperrcam1 rperrcam2];
      Gcam = [repmat({'cam1'},nphyspts,1); repmat({'cam2'},nphyspts,1)];
      Gphyspt = [1:nphyspts 1:nphyspts]';
      boxplot(X,{Gphyspt Gcam},...
        ... %'colorgroup',Gcam,...
        'factorseparator',1,...
        'factorgap',[3 1]...
        );

      ylabel('RP err (px)','fontweight','bold','fontsize',14);
      title('RP err','fontweight','bold','fontsize',14);
        
      
%       
%       rpe1perim = mean(rperr1ml,1);
%       rpe2perim = mean(rperr2ml,1);
%       rpeBar = [rpe1perim; rpe2perim];
%       hbar = bar(rpeBar','grouped');
%       legend(hbar,{'cam1' 'cam2'});
    end
  end
      
  methods % CalRig
    
    function [xEPL,yEPL,Xc1,Xc1OOB] = ...
        computeEpiPolarLine(obj,iView1,xy1,iViewEpi,roiEpi,varargin)
      
      switch obj.eplineComputeMode
        case 'mostaccurate'
          [xEPL,yEPL,Xc1,Xc1OOB] = obj.computeEpiPolarLineBase(iView1,xy1,iViewEpi,roiEpi,varargin{:});
        case 'fastest'
          % This codepath will err if 3rd/4th args are requested
          [xEPL,yEPL] = obj.computeEpiPolarLineEPline(iView1,xy1,iViewEpi,roiEpi,varargin{:});
        otherwise
          error('''eplineComputeMode'' property must be either ''mostaccurate'' or ''fastest''.');          
      end
    end
      
    function [xEPL,yEPL,Xc1,Xc1OOB] = ...
        computeEpiPolarLineBase(obj,iView1,xy1,iViewEpi,roiEpi,varargin)
      %
      % Xc1: [3 x nz] World coords (coord sys of cam/iView1) of EPline
      % Xc1OOB: [nz] indicator vec for Xc1 being out-of-view in iViewEpi
      
      [projectionmeth,z1Range] = myparse(varargin,...
        'projectionmeth','worldToImage',... % either 'worldToImage' or 'normalized2projected'. AL20181204: this should make negligible diff, see diagnostics()
        'z1Range',obj.eplineZrange{iView1} ...
        );
      
      % See CalRig
      %
      % z1Range: vector of world-z-coords to project onto iViewEpi
            
      xp = xy1(:);
      szassert(xp,[2 1]);            
      xn1 = obj.projected2normalized(xp,iView1);
      
      % create 3D segment by projecting normalized coords into 3D space
      % (coord sys of cam1)
      %Zc1 = 1e2:.25:2e2; % mm
      %Zc1 = 1:.25:5e2;
      Xc1 = [xn1(1)*z1Range; xn1(2)*z1Range; z1Range];

      switch projectionmeth        
        case 'worldToImage'
          sp = obj.stroParams;
          cpEpiFld = ['CameraParameters' num2str(iViewEpi)];
          cpEpi = sp.(cpEpiFld); % camera parameters for viewEpi
          if iView1==1 && iViewEpi==2
            wp = Xc1'; % world coord sys is cam1 coord sys
            R = sp.RotationOfCamera2;
            T = sp.TranslationOfCamera2;
            imPoints = worldToImage(cpEpi,R,T,wp,'applyDistortion',true);
            imPointsUD = worldToImage(cpEpi,R,T,wp,'applyDistortion',false);            
          elseif iView1==2 && iViewEpi==1
            wp = obj.camxform(Xc1,[2 1]);
            wp = wp';
            R = eye(3);
            T = [0 0 0];
            imPoints = worldToImage(cpEpi,R,T,wp,'applyDistortion',true);
            imPointsUD = worldToImage(cpEpi,R,T,wp,'applyDistortion',false);
          else
            assert(false);
          end
          Xc1OOB = imPointsUD(:,1)<roiEpi(1) | imPointsUD(:,1)>roiEpi(2) |...
                   imPointsUD(:,2)<roiEpi(3) | imPointsUD(:,2)>roiEpi(4);
          imPoints(Xc1OOB,:) = nan;  
          xEPL = imPoints(:,1);
          yEPL = imPoints(:,2);
        case 'normalized2projected'
          XcEpi = obj.camxform(Xc1,[iView1 iViewEpi]); % 3D seg, in frame of cam2
          xnEpi = [XcEpi(1,:)./XcEpi(3,:); XcEpi(2,:)./XcEpi(3,:)]; % normalize
          xpEpi = obj.normalized2projected(xnEpi,iViewEpi); % project
          
          Xc1OOB = xpEpi(1,:)<roiEpi(1) | xpEpi(1,:)>roiEpi(2) |...
                   xpEpi(2,:)<roiEpi(3) | xpEpi(3,:)>roiEpi(4);
          xpEpi(:,Xc1OOB) = nan;
          xEPL = xpEpi(1,:)';
          yEPL = xpEpi(2,:)';
        otherwise
          assert(false);
      end
    end
    
    function [xEPL,yEPL] = ...
        computeEpiPolarLineEPline(obj,iView1,xy1,iViewEpi,roiEpi)
      % Use worldToImage
      
      warningNoTrace('Development codepath.');
      
      xEPLnstep = 20;
      
      assert(numel(xy1)==2);
      sp = obj.stroParams;
      F = sp.FundamentalMatrix;
      
      if iView1==1 && iViewEpi==2
        lines = epipolarLine(F,xy1(:)');
      elseif iView1==2 && iViewEpi==1
        lines = epipolarLine(F',xy1(:)');
      else
        assert(false);
      end
      
      assert(numel(lines)==3);
      A = lines(1);
      B = lines(2);
      C = lines(3);
      xEPL = linspace(roiEpi(1),roiEpi(2),xEPLnstep)';
      yEPL = (-C-A*xEPL)/B;
      
      tmp = obj.cropLines([yEPL xEPL],roiEpi);
      xEPL = tmp(:,2);
      yEPL = tmp(:,1);
    end
    
    function [X1,xp1rp,xp2rp,rperr1,rperr2] = stereoTriangulateML(obj,xp1,xp2)
      % Stereo triangulation using matlab/vision lib fcns
      % 
      % xp1: [2xn]. xy image coords, camera1
      % xp2: etc
      %
      % X1: [3xn]. 3D coords in frame of camera1
      % xp1rp: [2xn]. xy image coords, reprojected, cam1
      % xp2rp: etc
      % rperr1: [n]. L2 err orig vs reprojected, cam1.
      % rperr2: [n] etc.
      %
      % WARNING: lib fcns like undistortPoints seem to scale like O(n^2),
      % so break up big inputs.
      
      szassert(xp2,size(xp1));
      [d,n] = size(xp1);
      assert(d==2);
      
      sp = obj.stroParams;
      cp1 = sp.CameraParameters1;
      cp2 = sp.CameraParameters2;      
      
      xp1ud = undistortPoints(xp1',cp1);
      xp2ud = undistortPoints(xp2',cp2);      
      X1 = triangulate(xp1ud,xp2ud,sp);
      
      xp1rp = worldToImage(cp1,eye(3),[0;0;0],X1,'applyDistortion',true);
      xp2rp = worldToImage(cp2,...
        sp.RotationOfCamera2,sp.TranslationOfCamera2,X1,'applyDistortion',true);
      
      % shape conventions
      X1 = X1';
      xp1rp = xp1rp';
      xp2rp = xp2rp';
      rperr1 = sqrt(sum((xp1rp-xp1).^2,1));
      rperr2 = sqrt(sum((xp2rp-xp2).^2,1));      
    end
    
    
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
       
  end
  
  methods (Static)
    function [fc,cc,kc,alpha_c] = camParams2Intrinsics(camParams)
      % convert ML-style intrinsics to Bouget-style
      
      fc = camParams.FocalLength(:);
      assert(numel(fc)==2);
      
      cc = camParams.PrincipalPoint(:);
      assert(numel(cc)==2);
            
      kc = nan(5,1);
      raddistort = camParams.RadialDistortion;
      if numel(raddistort)==2
        raddistort(end+1) = 0;
      end
      kc([1 2 5]) = raddistort;
      
      tandistort = camParams.TangentialDistortion;
      kc([3 4]) = tandistort;
      
      alpha_c = camParams.Skew;
      assert(isscalar(alpha_c));
    end
  end

end