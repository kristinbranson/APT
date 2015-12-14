classdef Shape 

  methods (Static)
  
    function p = xy2vec(xy)
      % xy: [nptsx2]
      % p: [1x2*npts]
      assert(size(xy,2)==2);
      p = [xy(:,1);xy(:,2)]';
    end
    
    function xy = vec2xy(p)
      assert(isvector(p));
      n = numel(p);      
      p = p(:);
      xy = [p(1:n/2) p(n/2+1:end)];
    end
    
    function p0 = randrot(p0,d)
      % Randomly rotate shapes
      % 
      % p0 (in): [Lx2d] shapes
      % d: model.d. Must be 2.
      %
      % p0 (out): [Lx2d] randomly rotated shapes (each row of p0 randomly
      %   rotated)
      
      assert(d==2);
      L = size(p0,1); 
      thetas = 2*pi*rand(L,1);
      p0 = Shape.rotate(p0,thetas);
    end
    
    function p1 = randsamp(p0,i,L)
      % Randomly sample shapes
      %
      % p0 (in): [NxD] all shapes
      % i: index of 'current' shape (1..L)
      % L: number of shapes to return
      %
      % p1: [LxD] shapes randomly sampled from p0, *omitting* the nth
      %   shape, ie p0(n,:).
            
      [N,D] = size(p0);
      assert(any(i==1:N));      
      
      iOther = [1:i-1,i+1:N];
      if L <= N-1
        i1 = randSample(iOther,L,true); %[n randSample(1:N,L-1)];
        p1 = p0(i1,:);
      else % L>N-1
        % Not enough other shapes. Select pairs of shapes and average them
        % AL: min() seems unnecessary if use (N-1)*rand(...)
        nExtra = L-(N-1);
        iAv = iOther(ceil((N-1)*rand([nExtra,2]))); % [nExtrax2] random elements of iOther
        pAv = (p0(iAv(:,1),:) + p0(iAv(:,2),:))/2; % [nExtraxD] randomly averaged shapes
        p1 = cat(1,p0(iOther,:),pAv);
      end
      
      % p1: Set of L shapes, explicitly doesn't include p0(n,:)
      assert(isequal(size(p1),[L D]));      
    end
    
    function bbJ = jitterBbox(bb,L,d,uncertfac)
      % Randomly jitter bounding box
      % bb: [1x2d] bounding box [x1 x2 .. xd w1 w2 .. wd]
      % L: number of replicates
      % d: dimension
      % uncertfac: scalar double
      
      assert(isequal(size(bb),[1 2*d]));
      szs = bb(d+1:end);
      maxDisp = szs/uncertfac;
      uncert = bsxfun(@times,(2*rand(L,d)-1),maxDisp);
      
      bbJ = repmat(bb,[L,1]);
      bbJ(:,1:d) = bbJ(:,1:d) + uncert;
    end
    
    function [nOOB,tfOOB] = normShapeOOB(p)
      % Determine if normalized shape is out-of-bounds.
      %
      % p: [NxD] normalized shapes
      % 
      % nOOB: scalar double, number of out-of-bounds els of p
      % tfOOB: logical, same size as p. If true, p(i) is out-of-bounds.
      tfOOB = ~(-1<=p & p<=1);
      nOOB = nnz(tfOOB);
    end
    
    function xyhat = findOrientation2d(xy,iHead,iTail)
      % Compute orientation of shape.
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
      
      c = cov(xy);
      [v,d] = eig(c);
      d = diag(d);
      [~,imax] = max(d);
      vlong1 = v(:,imax);
      vlong2 = -vlong1;
      
      % pick sign
      xyH = xy(iHead,:);
      xyT = xy(iTail,:);
      xyHT = xyH-xyT;
      
      if dot(xyHT,vlong1) > dot(xyHT,vlong2)
        xyhat = vlong1;
      else
        xyhat = vlong2;
      end
    end
    
    function p1 = rotate(p0,theta)
      % Rotate shapes
      % 
      % p: [NxD], shapes
      % theta: [N], rotation angles
      % mdl: model
      %
      % p1: [NxD], rotated shapes
      
      d = 2;
      [N,D] = size(p0);
      nfids = D/d;
      assert(isvector(theta) && numel(theta)==N);
      
      ct = cos(theta); % [N]
      st = sin(theta); % [N]      
      p0 = reshape(p0,[N,nfids,d]); % [Nxnfidsxd]
      mus = mean(p0,2); % [Nx1x2] centroids
      p0 = bsxfun(@minus,p0,mus);
      x = bsxfun(@times,ct,p0(:,:,1)) - bsxfun(@times,st,p0(:,:,2)); % [Nxnfids]
      y = bsxfun(@times,st,p0(:,:,1)) + bsxfun(@times,ct,p0(:,:,2)); % [Nxnfids]
      
      p0 = cat(3,x,y); % [Nxnfidsx2]
      p0 = bsxfun(@plus,p0,mus);
      p0 = reshape(p0,[N D]);
      
      p1 = p0;
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
      % xyc: [1x2] xy coords of center
      %
      % xy1: [nptx2]
      
      assert(size(xy0,2)==2);
      assert(isequal(size(xyc),[1 2]));
      
      xy0 = bsxfun(@minus,xy0,xyc);
      xy1 = Shape.rotateXY(xy0,theta);
      xy1 = bsxfun(@plus,xy1,xyc);
    end
    
    function th = canonicalRot(p,iHead,iTail)
      % Find rotations that transform p to canonical coords.
      % 
      % p: [NxD] shapes
      % 
      % th: [Nx1] thetas which, when applied to p, result in p's all being
      % oriented towards (x,y)=(1,0).
      
      N = size(p,1);
      th = nan(N,1);
      for i = 1:N
        xyP = Shape.vec2xy(p(i,:));
        vhat = Shape.findOrientation2d(xyP,iHead,iTail);
        vhatTheta = atan2(vhat(2),vhat(1));
        th(i) = -vhatTheta; % rotate by this to bring p(i,:) into canonical orientation
      end
    end    
    
    function pRIDel = rotInvariantDiff(p,pTgt,iHead,iTail)
      % Rotationally-invariant difference operation.
      %
      % p: [NxD] shape (eg current/predicted), normalized coords
      % pTgt: [NxD] target (eg GT) shape, normalized coords
      % iHead/iTail: 1/3 for FlyBubble
      %
      % pDiff: [NxD] This is pTgt-p, but taken in the coordinate system 
      % where p is canonically oriented. 
      
      assert(isequal(size(p),size(pTgt)));
      pRIDel = nan(size(p));
      theta = Shape.canonicalRot(p,iHead,iTail);
      N = size(p,1);      
      for i = 1:N
        pCanon = Shape.rotate(p(i,:),theta(i));
        pTgtCanon = Shape.rotate(pTgt(i,:),theta(i));
        pRIDel(i,:) = pTgtCanon - pCanon;
      end
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
      %
      % p1 is computed by:
      %   - rotating p0 into canonical orientation
      %   - adding pRIDel
      %   - rotating back
      
      assert(isequal(size(p0),size(pRIDel)));
      p1 = nan(size(p0));
      theta = Shape.canonicalRot(p0,iHead,iTail);
      N = size(p0,1);
      for i = 1:N      
        pTmp = Shape.rotate(p0(i,:),theta(i));
        pTmp = pTmp + pRIDel(i,:);
        p1(i,:) = Shape.rotate(pTmp,-theta(i));
      end
    end
    
  end
  
  %% Visualization
  methods (Static) 
    
    function viz(I,p,mdl,varargin)
      % I: [N] cell vec of images
      % p: [NxDxR] shapes
      %
      % optional pvs
      % fig - handle to figure to use
      % nr, nc - subplot size
      % idxs - indices of images to plot; must have nr*nc els. if 
      %   unspecified, these are randomly selected.
      % labelpts - if true, number landmarks. default false
      
      opts.fig = [];
      opts.nr = 4;
      opts.nc = 5;
      opts.idxs = [];      
      opts.labelpts = false;
      opts = getPrmDfltStruct(varargin,opts);
      if isempty(opts.fig)
        opts.fig = figure('windowstyle','docked');
      else
        figure(opts.fig);
        clf;
      end
      nplot = opts.nr*opts.nc;
      hax = createsubplots(opts.nr,opts.nc,.01);

      N = numel(I);
      assert(size(p,1)==N);
      assert(size(p,2)==mdl.D);
      R = size(p,3);

      if isempty(opts.idxs)
        iPlot = randsample(N,nplot);
      else
        assert(numel(opts.idxs)==nplot,'Number of ''idxs'' specified must equal nr*nc=%d.',nplot);
        iPlot = opts.idxs;
      end
        
      colors = jet(mdl.nfids);
      for iR = 1:R  
        for iPlt = 1:nplot
          iIm = iPlot(iPlt);
          im = I{iIm};
          imagesc(im,'Parent',hax(iPlt),[0,255]);
          axis(hax(iPlt),'image','off');
          hold(hax(iPlt),'on');
          colormap gray;
          for j = 1:mdl.nfids
            plot(hax(iPlt),p(iIm,j),p(iIm,j+mdl.nfids),...
              'wo','MarkerFaceColor',colors(j,:));
            if opts.labelpts
              htmp = text(p(iIm,j)+5,p(iIm,j+mdl.nfids)+5,num2str(j)); 
              htmp.Color = [1 1 1];             
            end
          end
          text(1,1,num2str(iIm),'parent',hax(iPlt));
        end
        
%        input(sprintf('Replicate %d/%d
      end
    end
    
    function vizBig(I,pT,iTrl,mdl,varargin)
      % I: [N] cell vec of images
      % pT: [NxRTxDx(T+1)] shapes
      %      % iTrl: index into I of trial to follow

      % optional pvs
      % fig - handle to figure to use
      % nr, nc - subplot size
      
      opts.fig = [];
      opts.nr = 4;
      opts.nc = 5;
      opts = getPrmDfltStruct(varargin,opts);      
      if isempty(opts.fig)
        opts.fig = figure('windowstyle','docked');
      else
        figure(opts.fig);
        clf;
      end
      nplot = opts.nr*opts.nc;
      hax = createsubplots(opts.nr,opts.nc,.01);

      N = numel(I);
      assert(size(pT,1)==N);
      RT = size(pT,2);
      assert(size(pT,3)==mdl.D);
      Tp1 = size(pT,4);

      % plot the image for iTrl; initialize hlines
      im = I{iTrl};
      hlines = cell(size(hax));
      colors = jet(mdl.nfids);
      iPlot = randsample(RT,nplot); % pick nplot replicates to follow
      for iPlt = 1:nplot
        imagesc(im,'Parent',hax(iPlt),[0,255]);
        axis(hax(iPlt),'image','off');
        hold(hax(iPlt),'on');
        colormap gray;
        iRT = iPlot(iPlt);
        text(1,1,num2str(iRT),'parent',hax(iPlt));
        
        for iPt = 1:mdl.nfids
          hlines{iPlt}(iPt) = plot(hax(iPlt),nan,nan,'wo',...
            'MarkerFaceColor',colors(iPt,:));
        end
      end
      
      % pick nplot replicates out of RT to follow
      for t = 1:Tp1
        for iPlt = 1:nplot
          iRT = iPlot(iPlt);
          for iPt = 1:mdl.nfids
            set(hlines{iPlt}(iPt),...
              'XData',pT(iTrl,iRT,iPt,t),'YData',pT(iTrl,iRT,iPt+mdl.nfids,t));
          end
        end
        
        input(sprintf('t= %d/%d',t,Tp1));
      end
    end
    
  end
  
end
