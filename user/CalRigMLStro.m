classdef CalRigMLStro < CalRigZhang2CamBase
  
  % Warning note: pointsToWorld() doesn't make a lot of sense.
  
  properties
    calSess % scalar Session from ML Stro calibration
    
    eplineComputeMode = 'mostaccurate'; % either 'mostaccurate' or 'fastest'
    eplineComputeBaseZrange = 0:.1:100;
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
  
  methods % CalRig
    
    function [xEPL,yEPL] = ...
        computeEpiPolarLine(obj,iView1,xy1,iViewEpi,roiEpi)
      
      switch obj.eplineComputeMode
        case 'mostaccurate'
          [xEPL,yEPL] = obj.computeEpiPolarLineBase(iView1,xy1,iViewEpi,roiEpi);
        case 'fastest'
          [xEPL,yEPL] = obj.computeEpiPolarLineEPline(iView1,xy1,iViewEpi,roiEpi);
        otherwise
          error('''eplineComputeMode'' property must be either ''mostaccurate'' or ''fastest''.');          
      end
    end
      
    function [xEPL,yEPL] = ...
        computeEpiPolarLineBase(obj,iView1,xy1,iViewEpi,roiEpi,varargin)
      
      [projectionmeth,z1Range] = myparse(varargin,...
        'projectionmeth','worldToImage',... % either 'worldToImage' or 'normalized2projected'
        'z1Range',obj.eplineComputeBaseZrange ...
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
            imPoints = worldToImage(cpEpi,R,T,wp);
            xpEpi = imPoints';
          elseif iView1==2 && iViewEpi==1
            wp = obj.camxform(Xc1,[2 1]);
            wp = wp';
            R = eye(3);
            T = [0 0 0];
            imPoints = worldToImage(cpEpi,R,T,wp);
            xpEpi = imPoints';
          else
            assert(false);
          end
        case 'normalized2projected'
          XcEpi = obj.camxform(Xc1,[iView1 iViewEpi]); % 3D seg, in frame of cam2
          xnEpi = [XcEpi(1,:)./XcEpi(3,:); XcEpi(2,:)./XcEpi(3,:)]; % normalize
          xpEpi = obj.normalized2projected(xnEpi,iViewEpi); % project
        otherwise
          assert(false);
      end

      yEpi = xpEpi';
      yEpi = yEpi(:,[2,1]); % Clearly, we have gone astray
      yEpi = obj.cropLines(yEpi,roiEpi);
      r2 = yEpi(:,1); % ... etc
      c2 = yEpi(:,2);
      xEPL = c2;
      yEPL = r2;
    end
    
    function [xEPL,yEPL] = ...
        computeEpiPolarLineEPline(obj,iView1,xy1,iViewEpi,roiEpi)
      % Use worldToImage, pointsToWorld
      
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