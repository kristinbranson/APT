classdef Shape 

  methods (Static)
  
    function p = xy2vec(xy)
      % xy: [nptsx2]
      % p: [1x2*npts]
      assert(size(xy,2)==2);
      %p = [xy(:,1);xy(:,2)]';
      p = xy(:).';
    end
    
    function xy = vec2xy(p)
      % p: [D] shape vec
      % 
      % xy: [nptsx2] x/y coords
      assert(isvector(p));
      n = numel(p);
      p = p(:);
      xy = [p(1:n/2) p(n/2+1:end)];
    end
    
    function xys = vecs2xys(ps)
      % ps: [D x nshapes] shape vec
      % 
      % xys: [npts x 2 x nshapes] x/y coords
      [n,nshapes] = size(ps);
      n2 = n/2;
      xys = cat(2,reshape(ps(1:n2,:),[n2,1,nshapes]),...
                  reshape(ps(n2+1:end,:),[n2,1,nshapes]));
    end
    
    function [nOOB,tfOOB] = normShapeOOB(p)
      % Determine if normalized shape is out-of-bounds.
      %
      % p: [NxD] normalized shapes
      % 
      % nOOB: scalar double, number of out-of-bounds els of p
      % tfOOB: logical, same size as p. If true, p(i) is out-of-bounds.
      %   NaN elements are NOT considered OOB.
      
      inBounds = (-1<=p & p<=1);
      tfOOB = ~inBounds & ~isnan(p);
      nOOB = nnz(tfOOB);
    end
    
    function xyhat = findOrientation2d(xy,iHead,iTail)
      % Compute orientation of shape.
      %
      % Only points iHead..iTail are considered.
      %
      % xy: [nptx2] landmark coordinates
      % iHead: scalar integer, 1..npt. Index for 'head' landmark.
      % iTail: etc. Index for 'tail' landmark.
      %
      % xyhat: [1x2] unit vector in "forwards"/head direction.
      %
      % This method computes xyhat by finding the long axis of the
      % covariance ellipse and picking a sign using iHead/iTail.
      
      [~,d] = size(xy);
      assert(d==2);

      assert(iHead<iTail);
      xy = xy(iHead:iTail,:); % use only "body" points iHead..iTail
      
      c = cov(xy);
      [v,d] = eig(c);
      d = diag(d);
      [~,imax] = max(d);
      vlong1 = v(:,imax);
      vlong2 = -vlong1;
      
      % pick sign
      xyH = xy(1,:);
      xyT = xy(end,:);
      xyHT = xyH-xyT;
      
      if dot(xyHT,vlong1) > dot(xyHT,vlong2)
        xyhat = vlong1;
      else
        xyhat = vlong2;
      end
    end
    
    function xyhats = findOrientations2d(xys,iHead,iTail)
      % Compute orientation of many shapes.
      % Same as findOrientation2d, but works on many shapes simultaneously
      % so that we can vectorize operations.
      %
      % Only points iHead..iTail are considered.
      %
      % xy: [npt x 2 x nshapes] landmark coordinates
      % iHead: scalar integer, 1..npt. Index for 'head' landmark.
      % iTail: etc. Index for 'tail' landmark.
      %
      % xyhat: [1 x 2 x nshapes] unit vector in "forwards"/head direction.
      %
      % This method computes xyhat by finding the long axis of the
      % covariance ellipse and picking a sign using iHead/iTail.
      
      [~,d,nshapes] = size(xys);
      assert(d==2);

      assert(iHead<iTail);
      xys = xys(iHead:iTail,:,:); % use only "body" points iHead..iTail
      npt = size(xys,1);
      
      % vectorized version of this
      % c = cov(xy);
      mu = sum(xys,1) / npt;
      % old matlab requires explicit bsxfun
      if verLessThan('matlab','9.2.0'),
        diffs = bsxfun(@minus,xys,mu);
      else
        diffs = xys - mu;
      end
      cs = zeros([2,2,nshapes]);
      cs(1,1,:) = sum(diffs(:,1,:).^2,1);
      cs(1,2,:) = sum(diffs(:,1,:).*diffs(:,2,:),1);
      cs(2,1,:) = cs(1,2,:);
      cs(2,2,:) = sum(diffs(:,2,:).^2,1);
      cs = cs / (npt-1);      

      % vectorized version of this
      % [v,d] = eig(c);
      % d = diag(d);
      % [~,imax] = max(d);
      % vlong1 = v(:,imax);
      % vlong2 = -vlong1;
      [~,vs] = eigs_2x2(cs);
      imax = 1;
      xyhats = reshape(vs(:,imax,:),[d,nshapes]);

      % pick sign
      %xyHs = xys(1,:,:);
      %xyTs = xys(end,:,:);
      xyHTs = reshape(xys(1,:,:)-xys(end,:,:),[d,nshapes]);
      d1s = sum(xyHTs.*xyhats,1);
      xyhats(:,d1s<0) = -xyhats(:,d1s<0);

%       if dot(xyHT,vlong1) > dot(xyHT,vlong2)
%         xyhat = vlong1;
%       else
%         xyhat = vlong2;
%       end
    end
      
    function p1 = rotate(p0,theta,ctr)
      % Rotate shapes 
      % 
      % p: [NxD], shapes
      % theta: [N], rotation angles
      % ctr: [Nx2] or [1x2], centers of rotation
      %
      % p1: [NxD], rotated shapes
      
      d = 2;
      assert(ismatrix(p0));
      [N,D] = size(p0);
      nfids = D/d;
      assert(isvector(theta) && numel(theta)==N);
      szctr = size(ctr);
      assert(isequal(szctr,[N,d]) || isequal(szctr,[1 d]));
      ctr = reshape(ctr,[],1,d); % [Nx1x2] or [1x1x2]
            
      ct = cos(theta); % [N]
      st = sin(theta); % [N]      
      p0 = reshape(p0,[N,nfids,d]); % [Nxnfidsxd]
      %mus = mean(p0,2); % [Nx1x2] centroids
      p0 = bsxfun(@minus,p0,ctr);
      x = bsxfun(@times,ct,p0(:,:,1)) - bsxfun(@times,st,p0(:,:,2)); % [Nxnfids]
      y = bsxfun(@times,st,p0(:,:,1)) + bsxfun(@times,ct,p0(:,:,2)); % [Nxnfids]
      
      p0 = cat(3,x,y); % [Nxnfidsx2]
      p0 = bsxfun(@plus,p0,ctr);
      p0 = reshape(p0,[N D]);
      
      p1 = p0;
    end
    
    function [p1,mu] = rotateCentroid(p0,theta,varargin)
      % Rotate shapes around centroid
      % 
      % p0: [NxD], shapes
      % theta: [N], rotation angles
      %
      % p1: [NxD], rotated shapes
      % mu: [Nx2], centroids

      d = 2;
      assert(ismatrix(p0));
      [N,D] = size(p0);
      nfids = D/d;
      assert(isvector(theta) && numel(theta)==N);
      
      iptsCentroid = myparse(varargin,...
        'iptsCentroid',1:nfids); % optional, only consider iptsCentroid when computing centroid
      
      p0 = reshape(p0,[N,nfids,d]); % [Nxnfidsxd] 
      mu = mean(p0(:,iptsCentroid,:),2); % [Nx1xd]
      mu = reshape(mu,[N d]); % [Nx2] centroids
      p1 = Shape.rotate(reshape(p0,[N D]),theta,mu);
    end
    
    function xy1 = rotateXY(xy0,theta)
      % Rotate some points about origin
      %
      % xy0: [nptx2] xy coords of points
      % theta: rotation angle
      %
      % xy1: [nptx2]
      
      assert(size(xy0,2)==2);      
      ct = cos(theta);
      st = sin(theta);      
      xy1(:,1) = ct*xy0(:,1) - st*xy0(:,2);
      xy1(:,2) = st*xy0(:,1) + ct*xy0(:,2);
    end
    
    function xy1 = rotateXYCenter(xy0,theta,xyc)
      % Rotate points about particular center
      %
      % xy0: [nptx2] xy coords of points
      % theta: rotation angle
      % xyc: [nptx2] OR [1x2] xy coords of center
      %
      % xy1: [nptx2]
      
      assert(size(xy0,2)==2);
      szxyc = size(xyc);
      assert(isequal(szxyc,size(xy0)) || isequal(szxyc,[1 2]));

      xy0 = bsxfun(@minus,xy0,xyc);
      xy1 = Shape.rotateXY(xy0,theta);
      xy1 = bsxfun(@plus,xy1,xyc);
    end
    
    function [pNA,th] = alignOrientationsOrigin(pN,iHead,iTail)
      % Align orientations of shapes by finding iHead:iTail canonicalRot 
      % angle and applying/rotating about origin
      %
      % pN: [NxD] shapes, SHOULD BE in eg NORMALIZED COORDS because we are
      %   rotating about origin (unless it is ok that output has random
      %   offsets)
      % iHead/iTail: head/tail landmarks per canonicalRot()
      %
      % pNA: [NxD] shapes, canonically rotated about oritin
      % th: [Nx1] thetas which, when applied to pN (about origin) result in
      %   pNA
      
      th = Shape.canonicalRot(pN,iHead,iTail);
      %pNA = Shape.rotateCentroid(pN,th,'iptsCentroid',iHead:iTail);
      pNA = Shape.rotate(pN,th,[0 0]);
    end
    
    function th = canonicalRot(p,iHead,iTail)
      % Find rotations that transform p to canonical coords.
      %
      % Only points iHead..iTail are considered
      % 
      % p: [NxD] shapes
      % 
      % th: [Nx1] thetas which, when applied to p, result in p's all being
      % oriented towards (x,y)=(1,0).

      % KB 20180419: vectorized version of the loop below
      xyPs = Shape.vecs2xys(p');
      vhats = Shape.findOrientations2d(xyPs,iHead,iTail);
      th = -atan2(vhats(2,:),vhats(1,:))';

%       N = size(p,1);
%       th = nan(N,1);
%       for i = 1:N
%         xyP = Shape.vec2xy(p(i,:));
%         vhat = Shape.findOrientation2d(xyP,iHead,iTail);
%         vhatTheta = atan2(vhat(2),vhat(1));
%         th(i) = -vhatTheta; % rotate by this to bring p(i,:) into canonical orientation
%       end
      
    end    
    
    function pRIDel = rotInvariantDiff(p,pTgt,iHead,iTail)
      % Rotationally-invariant difference operation.
      %
      % p: [NxD] shape (eg current/predicted), normalized coords
      % pTgt: [NxD] target (eg GT) shape, normalized coords
      % iHead/iTail: 1/3 for FlyBubble
      %
      % pRIDel: [NxD] This is pTgt-p, but taken in the coordinate system 
      % where p is canonically oriented. 
      
      assert(isequal(size(p),size(pTgt)));

      theta = Shape.canonicalRot(p,iHead,iTail); % [Nx1]
      % AL20171006 note precise rotOrigin should not impact 
      % difference/result pRIDel
      rotOrigin = [0 0]; % all shapes are in normalized coords which should be in [-1,1]
      pCanon = Shape.rotate(p,theta,rotOrigin); % make sure to rotate pCanon, pTgtCanon about same origin
      pTgtCanon = Shape.rotate(pTgt,theta,rotOrigin);
      pRIDel = pTgtCanon - pCanon;
      % AL20171006 prob better computed as 
      % pRIDel = Shape.rotate(pTgtCanon-pCanon,theta,[0 0])
    end
    
    function p1 = applyRIDiff(p0,pRIDel,iHead,iTail)
      % Apply (add) rotationally-invariant difference to shape.
      %
      % p0: [NxD] shape, normalized coords
      % pRIDel: [NxD] rot-invar diff, see eg rotInvariantDiff(). normalized
      %   coords.
      % 
      % p1: [NxD] shape, normalized coords. result of applying pRIDel to 
      %   p0. 
      
      assert(isequal(size(p0),size(pRIDel)));
      theta = Shape.canonicalRot(p0,iHead,iTail); 
      
      % pRIDel is a "difference shape", taken in coord system where p0 is
      % canonically rotated. theta rotates p0 to canonical orientation, so
      % -theta rotates from canonical orientation to p0 orientation.
      ROTORIGIN = [0 0]; % difference vectors should/must be rotated about origin
      pRIDel = Shape.rotate(pRIDel,-theta,ROTORIGIN);
      p1 = p0+pRIDel;
    end
    
    function [d,dav] = distP(p0,p1)
      % p0: [NxD]
      % p1: [NxDxM]
      %
      % d: [NxnptxM] 2-norm distances for all trials/pts/itersOrReps
      % dav: [NxM] distances averaged over pts
      % Assumes d=2
      
      d = 2;
      warning('Shape:distP','d assumed to be 2.');
      
      [N,D,RT] = size(p1);
      npt = D/d;
      assert(isequal([N,D],size(p0)));
      
      xy0 = reshape(p0,[N npt d]);
      xy1 = reshape(p1,[N npt d RT]);
      dxy = bsxfun(@minus,xy0,xy1);
      d = sqrt(sum(dxy.^2,3)); % [Nxnptx1xRT]
      d = squeeze(d); % [NxnptxRT]
      
      dav = squeeze(nanmean(d,2)); % [NxRT]      
    end
    
    % ROI utils: see also CropInfo.m
    
    function [roi,tfOOBview,xyRoi] = xyAndTrx2ROI(xy,xyTrx,nphysPts,radius)
      % Generate ROI/bounding boxes from shape and trx
      %
      % Currently we do this with a fixed radius and warn if a shape falls 
      % outside the ROI.
      % 
      % xy: [npt x 2 x n] xy coords. npt=nphysPts*nview, raster order is 
      %   ipt,iview. Can be nans
      % xyTrx: [nview x 2 x n] xy coords of trx center for this shape
      % nphysPts: scalar
      % radius: roi square radius, in px, must be integer
      %
      % roi: [n x 4*nview]. [xlo xhi ylo yhi xlo_v2 xhi_v2 ylo_v2 ... ]
      %   Square roi based on xyTrx.
      %   NOTE: roi may be outside range of image, eg xlo could be negative
      %   or xhi could exceed number of cols.
      % tfOOBview: [n x nview] logical. If true, shape is out-of-bounds of
      %   trx ROI box in that view. A shape with nan coords is not 
      %   considered OOB.
      % xyRoi: [npt x 2 x n] xy coords relative to ROIs; x==1 => first col 
      %   of ROI etc.
      
      [npt,d,n] = size(xy);
      assert(d==2);
      nview = npt/nphysPts;
      szassert(xyTrx,[nview 2 n]);
      validateattributes(radius,{'numeric'},{'positive' 'integer'});
      
      roi = nan(n,4*nview);
      for iview=1:nview
        x0 = round(xyTrx(iview,1,:)); % [1 x 1 x n]
        y0 = round(xyTrx(iview,2,:)); % etc
        xlo = x0-radius + 1; % MK 10022022. Adding 1 so that the patch is even sized
        xhi = x0+radius;
        ylo = y0-radius + 1;
        yhi = y0+radius;
        
        roicols = (1:4) + (iview-1)*4;
        roi(:,roicols) = [xlo(:) xhi(:) ylo(:) yhi(:)];
      end
      
      [xyRoi,tfOOBview] = Shape.xy2xyROI(xy,roi,nphysPts);
    end
    
    function [xyRoi,tfOOBview] = xy2xyROI(xy,roi,nphyspts)
      % Compute xyROI from xy (xy relative to roi); check for OOB.
      % 
      % xy: [nptx2xn] xy coords. npt=nphysPts*nview, raster order is
      %   ipt,iview. Can be nans
      % roi: [nx4*nview]. [xlo xhi ylo yhi xlo_v2 xhi_v2 ylo_v2 ... ]
      % nphysPts: scalar
      %
      % xyRoi: [nptx2xn] xy coords relative to ROIs; x==1 => first col of
      %   ROI in that view etc.
      % tfOOBview: [nxnview] logical. If true, shape is out-of-bounds of
      %   ROI in that view. A shape with nan coords is not considered OOB.
      
      [npt,d,n] = size(xy);
      assert(d==2);
      nview = npt/nphyspts;
      szassert(roi,[n 4*nview]);
      tfOOBview = false(n,nview);
      xyRoi = nan(npt,2,n);
      
      for iview=1:nview
        ipts = (1:nphyspts)+nphyspts*(iview-1);
        roivw = roi(:,(1:4)+4*(iview-1)); % [nx4]
        xlo = repmat(reshape(roivw(:,1),1,1,n),nphyspts,1,1);
        xhi = repmat(reshape(roivw(:,2),1,1,n),nphyspts,1,1);
        ylo = repmat(reshape(roivw(:,3),1,1,n),nphyspts,1,1);
        yhi = repmat(reshape(roivw(:,4),1,1,n),nphyspts,1,1);
        
        xs = xy(ipts,1,:); % [nphyspts x 1 x n]
        ys = xy(ipts,2,:); % etc
        tfOOBx = xs<xlo | xs>xhi;
        tfOOBy = ys<ylo | ys>yhi;
        tfOOBx = squeeze(tfOOBx)'; % [nxnphyspts]
        tfOOBy = squeeze(tfOOBy)'; % etc
        
        tfOOBview(:,iview) = any(tfOOBx,2) | any(tfOOBy,2);
        xyRoi(ipts,1,:) = xs-xlo+1;
        xyRoi(ipts,2,:) = ys-ylo+1;
      end
    end
    
    function [pRoi,tfOOBview] = p2pROI(p,roi,nphyspts)
      % Like xy2xyROI.
      %
      % p: [nxD]
      %
      % pRoi: [nxD]
      % tfOOBview: [nxnview] 
            
      xy = Shape.vecs2xys(p'); % xy: [nptx2xn] 
      [xyRoi,tfOOBview] = Shape.xy2xyROI(xy,roi,nphyspts); % xyRoi: [nptx2xn]
      [npt,d,n] = size(xyRoi);
      pRoi = reshape(xyRoi,[npt*d n])';
    end
    
    function xy = xyRoi2xy(xyRoi,roi)
      % Convert relative/roi xy to absolute xy
      %
      % xyRoi: [nptx2xN] xy coords relative to roi. npt=nphyspt*nview, with
      %   that raster order
      % roi: [Nx4*nview]. [xlo xhi ylo yhi xlo_v2 xhi_v2 ylo_v2 ... ]
      %
      % xy: [nptx2xN] xy coords, absolute.
      
      [npt,d,N] = size(xyRoi);
      assert(d==2);
      assert(size(roi,1)==N);
      nView = size(roi,2)/4;
      nPhysPt = npt/nView;
      assert(round(nView)==nView && round(nPhysPt)==nPhysPt);
      
      xy = nan(npt,2,N);
      for iView=1:nView
        xloylo = roi(:,[1 3]+4*(iView-1)); % [Nx2]
        xloyloArr = reshape(xloylo',[1 2 N]);
        ipts = (1:nPhysPt)+nPhysPt*(iView-1);
        if verLessThan('matlab','R2016b')
          xy(ipts,:,:) = xyRoi(ipts,:,:) + repmat(xloyloArr,[nPhysPt 1 1]) - 1;
        else
          xy(ipts,:,:) = xyRoi(ipts,:,:) + xloyloArr - 1; % nPhysPtx2xN, scalar expansions
        end
      end
    end
    
    function radii = suggestROIradius(xy,nphysPts)
      % Suggest an ROI radius for xyAndTrx2ROI
      %
      % xy: [nptx2xN] xy coords (row raster order: physpt,view)
      % nphysPts: scalar
      %
      % radii: [1xnview] roi square radius, in px, for each view

      [npt,d,N] = size(xy);
      assert(d==2);
      nview = npt/nphysPts;
      radii = nan(1,nview);
      if npt==0
        warningNoTrace('Shape:empty','Empty xy supplied.');
        return;
      end
      
      xy = permute(xy,[1 3 2]); % [npt x N x 2]
      xy = reshape(xy,nphysPts,nview,N,2);
      xy = permute(xy,[1 3 2 4]); % [nphysPts x N x nview x 2]
      
      xyCentroid = median(xy,1); % [1 x N x nview x 2] median for each shape/view/coord
      xy = xy-xyCentroid; % de-centroided
      xy = reshape(xy,[nphysPts*N,nview,2]);
      xymaxdev = max(abs(xy),[],1); % [1 x nview x 2], max abs deviation from centroid
      
      radii = reshape(xymaxdev,[nview 2]);
      radii = max(radii,[],2);
      szassert(radii,[nview 1]);
      radii = radii';
    end
    
  end
  
  %% Visualization
  methods (Static) 
    
    function vizSingle(I,p,idx,mdl,varargin)
      % Visualize a single Image+Shape from a trial set 
      %
      % I: [N] cell vec of images
      % p: [NxD] shape
      % mdl: model
      % idx: trial to visualize (index into I, rows of p)
      % optional pvs:
      % fig - handle to figure to use
      % labelpts - see viz()
      
      Shape.viz(I,p,mdl,'idxs',idx,'nr',1,'nc',1,varargin{:});      
    end
       
    function hax = viz(I,p,mdl,varargin)
      % Visualize many Images+Shapes from a Trial set
      % 
      % I: [N] cell vec of images
      % p: [NxDxR] shapes
      %
      % optional pvs
      % fig - handle to figure to use
      % nr, nc - subplot size
      % idxs - indices of images to plot; must have nr*nc els. if 
      %   unspecified, these are randomly selected.
      % labelpts - if true, number landmarks. default false
      % md - optional, table of MD for I
      
      opts.fig = [];
      opts.hax = []; % if supplied, overrides opts.fig
      opts.nr = 4;
      opts.nc = 5;
      opts.idxs = [];      
      opts.labelpts = false;
      opts.md = [];
      opts.p2 = [];
      opts.p2plotargs = {};
      opts.cmap = 'jet';
      opts = getPrmDfltStruct(varargin,opts);
      if isempty(opts.hax)
        if isempty(opts.fig)
          opts.fig = figure('windowstyle','docked');
        else
          figure(opts.fig);
          clf;
        end
        hax = createsubplots(opts.nr,opts.nc,.01);
      else
        hax = opts.hax;
        assert(numel(hax)==opts.nr*opts.nc);        
      end

      N = numel(I);
      assert(isequal(size(p),[N mdl.D]));
      
      tfp2 = ~isempty(opts.p2);
      if tfp2
        assert(isequal(size(opts.p2),[N mdl.D]));
      end

      tfMD = ~isempty(opts.md);
      if tfMD
        assert(size(opts.md,1)==N);
      end
      
      naxes = opts.nr*opts.nc;
      if isempty(opts.idxs)
        nplot = naxes;
        iPlot = randsample(N,nplot);
      else
        nplot = numel(opts.idxs);
        assert(nplot<=naxes,...
          'Number of ''idxs'' specified must be <= nr*nc=%d.',naxes);
        iPlot = opts.idxs;
      end
        
      if isstr(opts.cmap)
        colors = feval(opts.cmap,mdl.nfids);
      else
        colors = opts.cmap;
        colorsz = [mdl.nfids 3];
        assert(isequal(size(colors),colorsz),...
          sprintf('Color spec must be string colormap or array of size %s',mat2str(colorsz)));
      end
      for iPlt = 1:nplot
        iIm = iPlot(iPlt);
        im = I{iIm};
        imagesc(im,'Parent',hax(iPlt));
        axis(hax(iPlt),'image','off');
        hold(hax(iPlt),'on');
        colormap gray;
        for j = 1:mdl.nfids
          plot(hax(iPlt),p(iIm,j),p(iIm,j+mdl.nfids),...
            'wo','MarkerFaceColor',colors(j,:));
          if opts.labelpts
            htmp = text(p(iIm,j)+2.5,p(iIm,j+mdl.nfids)+2.5,num2str(j),'Parent',hax(iPlt));
            htmp.Color = [1 1 1];
          end
          if tfp2
            plot(hax(iPlt),opts.p2(iIm,j),opts.p2(iIm,j+mdl.nfids),...
              'MarkerFaceColor',colors(j,:),opts.p2plotargs{:});
          end
        end
        if tfMD
          movID = opts.md.mov(iIm);
          str = sprintf('%d mov%d f%d',iIm,movID,opts.md.frm(iIm));
        else
          str = num2str(iIm);
        end
        text(1,1,str,'parent',hax(iPlt),'color',[0 0 0],...
          'verticalalignment','top','interpreter','none');
      end
    end

    function montage(I,p,varargin)
      % Visualize many Images+Shapes from a Trial set
      % 
      % I: [N] cell vec of images, all same size
      % p: [NxDxntgtmax] shapes
      %
      % optional pvs
      % fig - handle to figure to use
      % nr, nc - subplot size
      % idxs - indices of images to plot; must have nr*nc els. if 
      %   unspecified, these are randomly selected.
      % framelbls - cellstr of labels (eg upper-right) for each plot
      % labelpts - if true, number landmarks. default false
      % md - (UNUSED atm) table of MD for I
      % p2 - [NxD] second set of shapes
      
      opts.fig = [];
      opts.nr = 4;
      opts.nc = 5;
      opts.idxs = [];
      opts.framelbls = [];
      opts.framelblscolor = [1 1 1];
      opts.framelblsbgcolor = [0 0 0];
      opts.framelblsIsFullSize = false; % if true, numel(opts.framelbls)==size(I,1)
      opts.labelpts = false;
      opts.labelptsdx = 10;
      opts.labelptsargs = {'fontweight' 'bold'};
      opts.md = [];
      opts.pplotargs = {}; % optional PV args for regular markers
      opts.p2 = [];
      opts.p2marker = '+';
      opts.labelpts2 = false;
      opts.colors = 'jet'; % either colormap name, or [nptsx3]
      opts.titlestr = 'Montage';
      opts.imsHeterogeneousSz = false; % if true, pad elements of I to make them the same size. all p's must be nan
      opts.imsHeterogeneousPadColor = 0; % grayscale pad color 
      opts.rois = []; % (opt), [Nx4] roi rectangles will be plotted. Each row is [xlo xhi ylo yhi]. OR, [N] cell array of rectangles; each el is [nroix4].
      opts.roisRectangleArgs = {'EdgeColor' [0 1 0] 'LineWidth',2}; % P-Vs used for plotting roi rects
      opts.masks = []; % (opt) [N] cell vec of masks, all the same size. Masks are doubles in [0,1] and scale I directly. Masks MUST have 3 chans!
      opts.maskalpha = 0.1;
      opts.ppFcn = []; % Optional preprocessing fcn for images
      opts = getPrmDfltStruct(varargin,opts);
      if isempty(opts.fig)
        opts.fig = figure('windowstyle','docked');
      else
        figure(opts.fig);
        clf;
      end
      tfMD = ~isempty(opts.md);
      tfROI = ~isempty(opts.rois);
      tfpp = ~isempty(opts.ppFcn);
      tfmask = ~isempty(opts.masks);
      
      N = numel(I);
      if opts.imsHeterogeneousSz
        szs = cellfun(@size,I,'uni',0);
        szs = cat(1,szs{:});
        assert(size(szs,2)==2,'Expect grayscale ims.');
        maxnrnc = max(szs,[],1);
        fprintf(1,'Heterogeneous image sizes. Max nr, nc: %d by %d.\n',...
          maxnrnc(1),maxnrnc(2));
        im0 = opts.imsHeterogeneousPadColor*ones(maxnrnc);
        for i=1:N
          im = im0;
          im(1:size(I{i},1),1:size(I{i},2)) = I{i};
          I{i} = im;
        end

        assert(all(isnan(p(:))),...
          'Input arg ''p'' must be all NaNs when imsHeterogeneousSz is on.');
      end      
      assert(size(p,1)==N);
      npts = size(p,2)/2;
      ntgtmax = size(p,3);
      if tfMD
        assert(size(opts.md,1)==N);
      end
      
      if ~ischar(opts.colors)
        szassert(opts.colors,[npts 3]);
      end
      
      naxes = opts.nr*opts.nc;
      if isempty(opts.idxs)
        nplot = naxes;
        iPlot = randsample(N,nplot);
      else
        nplot = numel(opts.idxs);
        assert(nplot<=naxes,...
          'Number of ''idxs'' specified must be <= nr*nc=%d.',naxes);
        iPlot = opts.idxs;
      end
      
      tfFrameLbls = ~isempty(opts.framelbls);
      if tfFrameLbls
        if opts.framelblsIsFullSize
          opts.framelbls = opts.framelbls(iPlot);
        end
        assert(iscellstr(opts.framelbls));
        assert(numel(opts.framelbls)==nplot);
      end

      tfP2 = ~isempty(opts.p2);
      if tfP2
        szassert(opts.p2,size(p));        
      end
      
      if tfROI
        tfCellRoi = iscell(opts.rois);
        if tfCellRoi
          assert(numel(opts.rois)==N);
        else
          szassert(opts.rois,[N 4]);
        end
      end
      
      [imnr,imnc,nch] = size(I{1});
      BIGIMNCHAN = 3; % use color to enable ROIs etc
      bigIm = nan(imnr*opts.nr,imnc*opts.nc,BIGIMNCHAN); 
      bigP = nan(npts,2,ntgtmax,nplot);
      bigP2 = nan(npts,2,ntgtmax,nplot);
      bigROI = cell(nplot);
      for iRow=1:opts.nr
        for iCol=1:opts.nc
          iPlt = iCol+opts.nc*(iRow-1);
          if iPlt>nplot
            % only breaks inner loop, so we will come back here but that's
            % ok
            break;
          end
          iIm = iPlot(iPlt);
          Ithis = I{iIm};
          szthis = size(Ithis);
          if tfpp
            Ithis = opts.ppFcn(Ithis);
          end
          if size(Ithis,3)<BIGIMNCHAN
            Ithis = repmat(Ithis,1,1,BIGIMNCHAN);
          end
          if tfmask
            maskthis = opts.masks{iIm};
            Ithis = Ithis*(1-opts.maskalpha) + maskthis*opts.maskalpha;
          end
          bigIm( (1:imnr)+imnr*(iRow-1), (1:imnc)+imnc*(iCol-1), :) = Ithis;
          
          for itgt=1:ntgtmax
            % surely suboptimal indexing
            xytmp = reshape(p(iIm,:,itgt),npts,2);
            outofbounds = xytmp(:,1) < 1 | xytmp(:,2) < 1 | xytmp(:,1) > szthis(2) | xytmp(:,2) > szthis(1);
            xytmp(outofbounds,:) = nan;
            xytmp(:,1) = xytmp(:,1)+imnc*(iCol-1);
            xytmp(:,2) = xytmp(:,2)+imnr*(iRow-1);
            bigP(:,:,itgt,iPlt) = xytmp;
          
            if tfP2
              xytmp = reshape(opts.p2(iIm,:,itgt),npts,2);
              xytmp(:,1) = xytmp(:,1)+imnc*(iCol-1);
              xytmp(:,2) = xytmp(:,2)+imnr*(iRow-1);
              bigP2(:,:,itgt,iPlt) = xytmp;            
            end
          end
          
          if tfROI
            if tfCellRoi
              roi = opts.rois{iIm}; % [nroi x 4]
            else
              roi = opts.rois(iIm,:); % [nroi==1 x 4]
            end
            roi(:,1:2) = roi(:,1:2) + imnc*(iCol-1);
            roi(:,3:4) = roi(:,3:4) + imnr*(iRow-1);
            bigROI{iPlt} = CropInfo.roi2RectPos(roi);
          end
        end
      end
      %bigROIRectPosn = CropInfo.roi2RectPos(bigROI);
      
      hIm = imagesc(bigIm);
      hIm.PickableParts = 'none';
      axis image off
      hold on
      colormap gray
      if ischar(opts.colors)
        colors = feval(opts.colors,npts);
      else
        colors = opts.colors;
      end
      hP1 = gobjects(npts,1);
      hP2 = gobjects(npts,1);
      for ipt=1:npts
        bigx = bigP(ipt,1,:,:);
        bigy = bigP(ipt,2,:,:);
        bigx = bigx(:);
        bigy = bigy(:);
        hP1(ipt) = plot(bigx,bigy,'o','Color',colors(ipt,:),...
          opts.pplotargs{:});
        if opts.labelpts
          text(bigx+opts.labelptsdx,bigy+opts.labelptsdx,num2str(ipt),...
            'color',colors(ipt,:),opts.labelptsargs{:});
        end
        if tfP2
          bigx = squeeze(bigP2(ipt,1,:,:));
          bigy = squeeze(bigP2(ipt,2,:,:));
          bigx = bigx(:);
          bigy = bigy(:);
          hP2(ipt) = plot(bigx,bigy,...          
            opts.p2marker,'MarkerFaceColor',colors(ipt,:),...
            'MarkerEdgeColor',colors(ipt,:),'linewidth',2);
          if opts.labelpts && opts.labelpts2
            text(bigx+opts.labelptsdx,bigy+opts.labelptsdx,num2str(ipt),...
            'color',colors(ipt,:),opts.labelptsargs{:});
          end
        end
      end
      hPall = hP1;
      if tfP2
        hPall = [hPall;hP2];
      end
      ax = gca;
      hCM = uicontextmenu(opts.fig);
      ax.UIContextMenu = hCM;
      uimenu(hCM,'Label','Increase marker size','Callback',Shape.makeCbkMarkerSize(hPall,'increase'));
      uimenu(hCM,'Label','Decrease marker size','Callback',Shape.makeCbkMarkerSize(hPall,'decrease'));
      set(ax,'Visible','on','XTickLabel',[],'YTickLabel',[]);
      
      for iRow=1:opts.nr
        for iCol=1:opts.nc
          iPlt = iCol+opts.nc*(iRow-1);
          if iPlt>nplot
            break;
          end
          %iIm = iPlot(iPlt);
          if tfFrameLbls
            h = text( imnc/15+imnc*(iCol-1),imnr/15+imnr*(iRow-1), opts.framelbls{iPlt} );
            set(h,...
              'Color',opts.framelblscolor,...
              'BackgroundColor',opts.framelblsbgcolor);
          end
          if tfROI
            posns = bigROI{iPlt};
            for iposn = 1:size(posns,1)
              rectangle('Position',posns(iposn,:),opts.roisRectangleArgs{:});
            end
          end
        end
      end
      set(opts.fig,'Color',[0 0 0]);
      title(opts.titlestr,'fontweight','bold','Color','w');
        
%         if tfMD
%           movID = opts.md.movID{hIm};
%           [~,movS] = myfileparts(movID);
%           str = sprintf('%d %s f%d',iIm,movS,opts.md.frm(iIm));
%         else
%           str = num2str(iIm);
%         end
%         text(1,1,str,'parent',hax(iPlt),'color',[1 1 .2],...
%           'verticalalignment','top','interpreter','none');
    end
    
    function h = montageTabbed(I,p,plotnr,plotnc,varargin)
      % If you pass framelbls, make sure to set framelblsIsFullSize==true
      
      n = size(I,1);
      nplot = plotnr*plotnc;
      startIdxs = 1:nplot:n;
      h = [];
      for i=1:numel(startIdxs)
        plotIdxs = startIdxs(i):min(startIdxs(i)+nplot-1,n);
        h(end+1,1) = figure('windowstyle','docked'); %#ok<AGROW>
        Shape.montage(I,p,'fig',h(end),...
          'nr',plotnr,'nc',plotnc,'idxs',plotIdxs,...
          varargin{:} ...
          );
      end
    end
    
    function cbk = makeCbkMarkerSize(hPlot,action)
      
      INCDECFAC = 1.2;
      cbk = @nst;
      function nst(src,evt)
        if isempty(hPlot)
          return;
        end
        sz = hPlot(1).MarkerSize;            
        switch action
          case 'increase'
            sz = round(sz*INCDECFAC);
          case 'decrease'
            sz = round(sz/INCDECFAC);            
          otherwise
            assert(false);
        end
        [hPlot.MarkerSize] = deal(sz);
      end
    end

    function vizRepsOverTimeTracks(I,pT,iTrl,mdl,varargin)
      % Visualize Replicates over time for a single Trial from a Trial set
      %
      % I: [N] cell vec of images
      % pT: [NxRTxDx(T+1)] shapes
      %      % iTrl: index into I of trial to follow

      % optional pvs    
      % fig - handle to figure to use
      % nr, nc - subplot size
      % t0 - starting iteration to show (defaults to 1)
      % t1 - ending iteration to show (defaults to T+1)
      
      N = numel(I);
      assert(size(pT,1)==N);
      RT = size(pT,2);
      assert(size(pT,3)==mdl.D);
      Tp1 = size(pT,4);
      
      opts.fig = [];
      opts.nr = 4;
      opts.nc = 5;
      opts.t0 = 1;
      opts.t1 = Tp1;
      opts = getPrmDfltStruct(varargin,opts);      
      if isempty(opts.fig)
        opts.fig = figure('windowstyle','docked');
      else
        figure(opts.fig);
        clf;
      end
      nplot = opts.nr*opts.nc;
      hax = createsubplots(opts.nr,opts.nc,.01);

      % plot the image for iTrl; initialize hlines
      im = I{iTrl};
      colors = jet(mdl.nfids);
      iReps = randsample(RT,nplot); % pick nplot replicates to follow
      for iPlt = 1:nplot
        ax = hax(iPlt);
        
        imagesc(im,'Parent',ax,[0,255]);
        axis(ax,'image','off');
        hold(ax,'on');
        colormap gray;
        iRT = iReps(iPlt);
        text(1,1,num2str(iRT),'parent',ax);

        for iPt = 1:mdl.nfids
          plot(ax,...
            squeeze(pT(iTrl,iRT,iPt,opts.t0:opts.t1-1)),...
            squeeze(pT(iTrl,iRT,iPt+mdl.nfids,opts.t0:opts.t1-1)),...
            '--','Color',colors(iPt,:)*.7,'MarkerSize',12,'LineWidth',2);
          plot(ax,...
            squeeze(pT(iTrl,iRT,iPt,opts.t1-1:opts.t1)),...
            squeeze(pT(iTrl,iRT,iPt+mdl.nfids,opts.t1-1:opts.t1)),...
            'x-','Color',colors(iPt,:)*.7,'MarkerSize',10,'LineWidth',3);
          plot(ax,...
            pT(iTrl,iRT,iPt,opts.t1),...
            pT(iTrl,iRT,iPt+mdl.nfids,opts.t1),...
            'wo','MarkerFaceColor',colors(iPt,:),'MarkerSize',8,'LineWidth',2);
        end
      end
      
    end
    
    % See MakeTrackingResultsHistogramVideo
    
    function hFig = vizReps(I,pT,iTrl,t,mdl,varargin)
      % I: [N] cell vec of images
      % pT: [NxRTxDx(T+1)] shapes
      % iTrl: index into I of trial to follow
      % t: iteration index (into 1..(T+1)) to visualize
      % 
      % optional PVs
      %  fig - handle to figure to use

      N = numel(I);
      assert(size(pT,1)==N);
      RT = size(pT,2);
      assert(size(pT,3)==mdl.D);
      Tp1 = size(pT,4);
      npts = mdl.nfids;
      
      opts.fig = [];
      opts = getPrmDfltStruct(varargin,opts);      
      if isempty(opts.fig)
        hFig = figure('windowstyle','docked');
      else
        figure(opts.fig);
        hFig = opts.fig;
        clf;
      end
      ax = axes;
      
      % plot the image for iTrl; initialize hlines
      im = I{iTrl};
      colors = jet(npts);
      hold(ax,'off');
      imagesc(im,'Parent',ax,[0,255]);
      axis(ax,'image','off');
      hold(ax,'on');  
      colormap gray
      lims = axis;
        
      for r = 1:RT
        for iPt = 1:npts
          x = pT(iTrl,r,iPt,t);
          y = pT(iTrl,r,iPt+npts,t);
          if x < lims(1) || x > lims(2) || ...
              y < lims(3) || y > lims(4)
            continue;
          end
          plot(x,y,'o','Color',colors(iPt,:),...
            'MarkerFaceColor',colors(iPt,:),'MarkerSize',2,'LineWidth',1);
        end
      end
      
      text(lims(1),lims(3),sprintf('  iTrl%d Iter%d',iTrl,t),...
        'FontSize',24,'HorizontalAlignment','left','VerticalAlignment','top','Color',[1 1 1]);
    end
    
    function vizDiff(I,p0,p1,mdl,varargin)
      % I: [N] cell vec of images
      % p0,p1: [NxD] shapes
      % mdl: model
      %
      % optional pvs
      % fig - handle to figure to use
      % nr, nc - subplot size
      % idxs - indices of images to plot; must have nr*nc els. if 
      %   unspecified, these are randomly selected.
      % labelpts - if true, number landmarks. default false
      % md - if specified, table of metadata for I
      
      % Very Similar to Shape.viz()
      
      opts.fig = [];
      opts.nr = 4;
      opts.nc = 5;
      opts.idxs = [];      
      opts.labelpts = false;
      opts.md = [];
      opts = getPrmDfltStruct(varargin,opts);
      if isempty(opts.fig)
        opts.fig = figure('windowstyle','docked');
      else
        figure(opts.fig);
        clf;
      end
      tfMD = ~isempty(opts.md);
      hax = createsubplots(opts.nr,opts.nc,.01);

      N = numel(I);
      assert(isequal(size(p0),size(p1),[N mdl.D]));
      if tfMD
        assert(size(opts.md,1)==N);
      end

      naxes = opts.nr*opts.nc;
      if isempty(opts.idxs)
        nplot = naxes;
        iPlot = randsample(N,nplot);
      else
        nplot = numel(opts.idxs);
        assert(nplot<=naxes,...
          'Number of ''idxs'' specified must be <= nr*nc=%d.',naxes);
        iPlot = opts.idxs;
      end
      
      colors = jet(mdl.nfids);
      for iPlt = 1:nplot
        iIm = iPlot(iPlt);
        im = I{iIm};
        imagesc(im,'Parent',hax(iPlt),[0,255]);
        axis(hax(iPlt),'image','off');
        hold(hax(iPlt),'on');
        colormap gray;
        for j = 1:mdl.nfids
          plot(hax(iPlt),p0(iIm,j),p0(iIm,j+mdl.nfids),...
            'wo','MarkerFaceColor',colors(j,:));
          plot(hax(iPlt),p1(iIm,j),p1(iIm,j+mdl.nfids),...
            'ws','MarkerFaceColor',colors(j,:));          
        end
        
        if tfMD
          str = sprintf('%d iLbl%d f%d',iIm,opts.md.iLbl(iIm),...
            opts.md.frm(iIm));
        else
          str = num2str(iIm);
        end
        htmp = text(1,size(im,2),str,'Parent',hax(iPlt));
        htmp.Color = [1 1 1];
        htmp.VerticalAlignment = 'bottom';
      end
    end
    
  end
  
end
