classdef Shape 

  methods (Static)
  
    function p0 = randrot(p0,d)
      % Randomly rotate shapes
      % 
      % p0 (in): [Lx2d] shapes
      % d: model.d. Must be 2.
      %
      % p0 (out): [Lx2d] randomly rotated shapes (each row of p0 randomly
      %   rotated)
      
      assert(d==2);
      [L,D] = size(p0);      
      nfids = D/d;
      
      thetas = 2*pi*rand(L,1);
      ct = cos(thetas); % [Lx1]
      st = sin(thetas); % [Lx1]
      p0 = reshape(p0,[L,nfids,d]);
      mus = mean(p0,2); % [Lx1x2] centroids
      p0 = bsxfun(@minus,p0,mus);
      x = bsxfun(@times,ct,p0(:,:,1)) - bsxfun(@times,st,p0(:,:,2)); % [Lxnfids]
      y = bsxfun(@times,st,p0(:,:,1)) + bsxfun(@times,ct,p0(:,:,2)); % [Lxnfids]
      p0 = cat(3,x,y); % [Lxnfidsx2]
      p0 = bsxfun(@plus,p0,mus);
      p0 = reshape(p0,[L,nfids*d]);
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
    
    function viz(I,p,mdl,varargin)
      % I: [N] cell vec of images
      % p: [NxDxR] shapes
      %
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
      assert(size(p,1)==N);
      assert(size(p,2)==mdl.D);
      R = size(p,3);

      iPlot = randsample(N,nplot);
      colors = jet(obj.nfids);
      for iR = 1:R      
        for iPlt = 1:nplot
          iIm = iPlot(iPlt);
          im = obj.I{iIm};
          imagesc(im,'Parent',hax(iPlt),[0,255]);
          axis(hax(iPlt),'image','off');
          hold(hax(iPlt),'on');
          colormap gray;
          for j = 1:obj.nfids
            plot(hax(iPlt),...
              obj.pGT(iIm,j),obj.pGT(iIm,j+obj.nfids),...
              'wo','MarkerFaceColor',colors(j,:));
          end
          text(1,1,num2str(iIm),'parent',hax(iPlt));
        end
        
%        input(sprintf('Replicate %d/%d
      end
    end
    
    function vizBig(I,pT,iTrl,mdl,varargin)
      % I: [N] cell vec of images
      % pT: [NxRTxDx(T+1)] shapes
      % iTrl: index into I of trial to follow
      %
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
