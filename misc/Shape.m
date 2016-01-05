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
      % Randomly rotate shapes about centroids
      % 
      % p0 (in): [Lx2d] shapes
      % d: model.d. Must be 2.
      %
      % p0 (out): [Lx2d] randomly rotated shapes (each row of p0 randomly
      %   rotated)
      
      assert(d==2);
      L = size(p0,1); 
      thetas = 2*pi*rand(L,1);
      p0 = Shape.rotateCentroid(p0,thetas);
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
    
    function p1 = rotate(p0,theta,ctr)
      % Rotate shapes 
      % 
      % p: [NxD], shapes
      % theta: [N], rotation angles
      % ctr: [Nx2] or [1x2], centers of rotation
      %
      % p1: [NxD], rotated shapes
      
      d = 2;
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
    
    function [p1,mu] = rotateCentroid(p0,theta)
      % Rotate shapes around centroid
      % 
      % p0: [NxD], shapes
      % theta: [N], rotation angles
      %
      % p1: [NxD], rotated shapes
      % mu: [Nx2], centroids
      
      d = 2;
      [N,D] = size(p0);
      nfids = D/d;
      assert(isvector(theta) && numel(theta)==N);
      
      p0 = reshape(p0,[N,nfids,d]); % [Nxnfidsxd] 
      mu = squeeze(mean(p0,2)); % [Nx2] centroids
      p1 = Shape.rotate(p0,theta,mu);
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

      theta = Shape.canonicalRot(p,iHead,iTail); % [Nx1]
      rotOrigin = [0 0]; % all shapes are in normalized coords which should be in [-1,1]
      pCanon = Shape.rotate(p,theta,rotOrigin); % make sure to rotate pCanon, pTgtCanon about same origin
      pTgtCanon = Shape.rotate(pTgt,theta,rotOrigin);
      pRIDel = pTgtCanon - pCanon;
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
      ROTORIGIN = [0 0]; % difference vectors should be rotated about origin
      pRIDel = Shape.rotate(pRIDel,-theta,ROTORIGIN);
      p1 = p0+pRIDel;
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
       
    function viz(I,p,mdl,varargin)
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
      hax = createsubplots(opts.nr,opts.nc,.01);

      N = numel(I);
      assert(isequal(size(p),[N mdl.D]));

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
          plot(hax(iPlt),p(iIm,j),p(iIm,j+mdl.nfids),...
            'wo','MarkerFaceColor',colors(j,:));
          if opts.labelpts
            htmp = text(p(iIm,j)+2.5,p(iIm,j+mdl.nfids)+2.5,num2str(j),'Parent',hax(iPlt));
            htmp.Color = [1 1 1];
          end
        end
        text(1,1,num2str(iIm),'parent',hax(iPlt),'color',[1 1 .2]);
      end
    end
    
    function vizRepsOverTime(I,pT,iTrl,mdl,varargin)
      % Visualize Replicates over time for a single Trial from a Trial set
      % 
      % I: [N] cell vec of images
      % pT: [NxRTxDx(T+1)] shapes
      % iTrl: index into I of trial to follow
      % mdl: model
      %
      % 
      % optional pvs
      % fig - handle to figure to use
      % nr, nc - subplot size
      % pGT: [NxD], GT labels; shown if supplied
      % regs: Tx1 struct array of regressors (fields: regInfo, ftrPos). If
      %   supplied, mini-iterations will be shown with selected features
      
      opts.fig = [];
      opts.nr = 4;
      opts.nc = 5;
      opts.pGT = [];
      opts.regs = [];
      opts = getPrmDfltStruct(varargin,opts);      
      if isempty(opts.fig)
        opts.fig = figure('windowstyle','docked');
      else
        figure(opts.fig);
        clf;
      end
      tfGT = ~isempty(opts.pGT);
      tfRegs = ~isempty(opts.regs);
      nplot = opts.nr*opts.nc;
      hax = createsubplots(opts.nr,opts.nc,.01);

      N = numel(I);
      assert(size(pT,1)==N);
      RT = size(pT,2);
      assert(size(pT,3)==mdl.D);
      Tp1 = size(pT,4);
      if tfGT
        assert(size(opts.pGT,1)==N);
        assert(size(opts.pGT,2)==mdl.D);
      end
      if tfRegs
        assert(isstruct(opts.regs) && numel(opts.regs)==Tp1-1);
      end

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
          hlines{iPlt}(iPt) = plot(hax(iPlt),nan,nan,'ws',...
            'MarkerFaceColor',colors(iPt,:));
          if tfGT
            plot(hax(iPlt),opts.pGT(iTrl,iPt),opts.pGT(iTrl,iPt+mdl.nfids),'wo',...
              'MarkerFaceColor',colors(iPt,:));
          end
        end        
      end
      
      % pick nplot replicates out of RT to follow
      if ~tfRegs
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
      else % regs
        nMini = arrayfun(@(x)numel(x.regInfo),opts.regs);
        assert(all(nMini==nMini(1)));
        for t = 2:Tp1
          for iMini = 1:nMini
            if exist('hMiniFtrs','var')>0
              deleteValidHandles(hMiniFtrs);
            end 
            hMiniFtrs = [];
            for iPlt = 1:nplot
              iRT = iPlot(iPlt);
              if iMini==1
                for iPt = 1:mdl.nfids 
                  set(hlines{iPlt}(iPt),...
                    'XData',pT(iTrl,iRT,iPt,t),'YData',pT(iTrl,iRT,iPt+mdl.nfids,t));
                end
              end
              
              reg = opts.regs(t-1); % when t==2, we are plotting result of first iteraton, which used first regressor
              
              p = reshape(pT(iTrl,iRT,:,t),1,mdl.D); % absolute shape for trl/rep/it
              [xF,yF,infoF] = Features.compute2LM(reg.ftrPos.xs,p(1:mdl.nfids),p(mdl.nfids+1:end));
              assert(isrow(xF));
              assert(isrow(yF));
              
              fids = reg.regInfo{iMini}.fids;
              nfids = size(fids,2);              
              colors = jet(nfids);
              for iFid = 1:nfids
                fid1 = fids(1,iFid);
                fid2 = fids(2,iFid);
                hMiniFtrs(end+1) = plot(hax(iPlt),xF([fid1 fid2]),yF([fid1 fid2]),'-',...
                  'Color',colors(iFid,:));
%                 hMiniFtrs(end+1) = plot(hax(iPlt),xF(fid2),yF(fid2),'v',...
%                   'markerfacecolor',colors(iFid,:),'MarkerSize',4);
%                 hMiniFtrs(end+1) = plot(hax(iPlt),xF(fid2),yF(fid2),'v',...
%                   'markerfacecolor',colors(iFid,:),'MarkerSize',4);
              end              
            end
            fprintf(1,'fids:\n');
            disp(fids);
            input(sprintf('it %d.%03d\n',t,iMini));
          end
        end
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
    
    function vizRepsOverTimeDensity(I,pT,iTrl,mdl,varargin)
      % Visualize Replicate-density over time for a single Trial from a Trial set
      % 
      % I: [N] cell vec of images
      % pT: [NxRTxDx(T+1)] shapes
      %      % iTrl: index into I of trial to follow

      % optional pvs    
      % fig - handle to figure to use
      % t0 - starting iteration to show (defaults to 1)
      % t1 - ending iteration to show (defaults to T+1)
      % smoothsig - sigma for gaussian smoothing (defaults to 2)
      
      N = numel(I);
      assert(size(pT,1)==N);
      RT = size(pT,2);
      assert(size(pT,3)==mdl.D);
      Tp1 = size(pT,4);
      npts = mdl.nfids;
      
      opts.fig = [];
      opts.t0 = 1;
      opts.t1 = Tp1;
      opts.smoothsig = 2;
      opts = getPrmDfltStruct(varargin,opts);      
      if isempty(opts.fig)
        opts.fig = figure('windowstyle','docked');
      else
        figure(opts.fig);
        clf;
      end

      % plot the image for iTrl; initialize hlines
      ax = axes;
      im = I{iTrl};
      colors = jet(npts);
      t = opts.t0;
      while isnumeric(t)
        hold(ax,'off');
        imagesc(im,'Parent',ax,[0,255]);
        axis(ax,'image','off');
        hold(ax,'on');  
        lims = axis;
        colormap gray;
        
        binedges{1} = floor(lims(1)):ceil(lims(2));
        binedges{2} = floor(lims(3)):ceil(lims(4));
        bincenters{1} = (binedges{1}(1:end-1)+binedges{1}(2:end))/2;
        bincenters{2} = (binedges{2}(1:end-1)+binedges{2}(2:end))/2;
        counts = cell(1,npts);
        fil = fspecial('gaussian',6*opts.smoothsig+1,opts.smoothsig);
        maxv = .15;
        for iPt = 1:npts
          xy = [squeeze(pT(iTrl,:,iPt,t))' squeeze(pT(iTrl,:,iPt+npts,t))'];
          cnts = hist3(xy,'edges',binedges);
          
          sumcnts = sum(cnts(:));
          if sumcnts<RT
            warningNoTrace('Shape:viz','%d/%d points omitted from histogram.',RT-sumcnts,RT);
          end
          %assert(sum(cnts(:))==RT);
          cnts = cnts(1:end-1,1:end-1)/RT;
          counts{iPt} = imfilter(cnts,fil,'corr','same',0); % smoothed
          him2 = image(...
            [bincenters{1}(1),bincenters{1}(end)],...
            [bincenters{2}(1),bincenters{2}(end)],...
            repmat(reshape(colors(iPt,:),[1,1,3]),size(counts{iPt}')),...
            'AlphaData',min(1,3*sqrt(counts{iPt}')/sqrt(maxv)),'AlphaDataMapping','none');
        end
        
        for r = 1:RT
          for iPt = 1:npts
            %plot(squeeze(ptcurr(r,j,1,1:i)),squeeze(ptcurr(r,j,2,1:i)),'-','Color',colors(j,:)*.7,'LineWidth',1);
            x = pT(iTrl,r,iPt,t);
            y = pT(iTrl,r,iPt+npts,t);
            if x < lims(1) || x > lims(2) || ...
               y < lims(3) || y > lims(4)
              continue;
            end
            plot(x,y,'o','Color',colors(iPt,:),'MarkerFaceColor',colors(iPt,:),'MarkerSize',6,'LineWidth',1);
          end
        end
        
        text(lims(1),lims(3),sprintf('  iTrl%d Iter%d',iTrl,t),'FontSize',24,'HorizontalAlignment','left','VerticalAlignment','top','Color',[1 1 1]);
        tinput = input('Enter t (default to next iteration, char to end)');
        if ~isempty(tinput) 
          if isnumeric(tinput)
            t = tinput;
          else
            break;
          end
        else
          t = t+1;
        end
      end
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
      
      % Very Similar to Shape.viz()
      
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
      hax = createsubplots(opts.nr,opts.nc,.01);

      N = numel(I);
      assert(isequal(size(p0),size(p1),[N mdl.D]));

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
      end
    end
      
  end
  
end
