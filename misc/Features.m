classdef Features
  
  properties (Constant)
    % take sgs = SGS{i1,i2}; it will have a hist in range [0,max]; this
    % array gives the maxes for sig1,sig2 = [0 2 4 8]
    JAN_SGS_MAX_SIG0248 = [
      103 44 30 17;37 25 19 12;16 15 13 9;9 9 9 7];
    
    % take sls = SLS{i1,i2}; it will have a mean of ~0 and a span; this
    % array gives the spans for sig1,sig2 = [0 2 4 8]
    JAN_SLS_SPAN_SIG0248 = [
      294 34 9 2; 135 73 28 7; 145 113 62 26; 117 111 102 80];
    
    % span of prctile(SLS{i1,i2},[2 98]) for sig1,sig2 = [0 2 4 8]
    JAN_SLS_SPAN98_SIG0248 = [
      26 5 2.3 1.1; 20 13 8.1 4.2; 37 32 25 15; 69 66 60 43];    

    JAN_SLS_SPAN99_SIG0248 = [
      38 7 3 1.3;28 18 11 5.1;48 42 32 19;85 82 74 54];
  end
  %% preprocessing/channels
  methods (Static)
    
    function [S,SGS,SLS] = pp(Is,sig1,sig2,varargin)
      % Preprocess images for feature comp
      %
      % Is: [N] cell array of images
      % sig1: [n1] vector of SDs for first blur. Include 0 to get no-blur.
      % sig2: [n2] vector of SDs for second blur. Include 0 to get no-blur.
      %
      % optional PVs:
      % - gaussFiltRadius. Radius of gaussian filter in units of SD,
      %   defaults to 2.5. Applied to blurring by both sig1 and sig2.
      % - laplaceKernel. Defaults to fspecial('laplacian',0).
      %
      % - sgsRescale. Scalar logical. If true (default), multiply SGS by
      % sgsRescaleFacs and convert to uint8.
      % - sgsRescaleFacs. [n1xn2] Defaults to values for Jan data, sig=[0 2 4 8].
      %
      % - slsRescale. Scalar logical. If true (default), etc.
      % - slsRescaleFacs. [n1xn2] Defaults to values for Jan data, sig=[0 2 4 8].
      %
      % S: [Nxn1] cell array of smoothed images (using sig1)
      % SGS: [Nxn1xn2] blur2(gradmag(blur1(S)))
      % SLS: [Nxn1xn2] blur2(laplacian(blur1(S)))
      
      opts = struct();
      opts.gaussFiltRadius = 2.5;
      opts.laplaceKernel = fspecial('laplacian',0);
      opts.sgsRescale = true;
      opts.sgsRescaleFacs = 200./Features.JAN_SGS_MAX_SIG0248; % 200 instead of 256 for safety buffer
      opts.slsRescale = true;
      opts.slsRescaleFacs = 200./Features.JAN_SLS_SPAN99_SIG0248; % 200 instead of 256 for safety buffer
      opts = getPrmDfltStruct(varargin,opts);      
      
      assert(iscell(Is) && isvector(Is));
      N = numel(Is);
      validateattributes(sig1,{'numeric'},{'vector' 'real' 'nonnegative'});
      validateattributes(sig2,{'numeric'},{'vector' 'real' 'nonnegative'});
      n1 = numel(sig1);
      n2 = numel(sig2);
      validateattributes(opts.gaussFiltRadius,{'numeric'},{'scalar' 'real' 'positive'});
      if opts.sgsRescale
        assert(isequal(size(opts.sgsRescaleFacs),[n1 n2]));
      end
      if opts.slsRescale
        assert(isequal(size(opts.slsRescaleFacs),[n1 n2]));
      end

      [tmp1,tmp2] = size(opts.laplaceKernel);
      assert(tmp1==tmp2);
      lkRad = (tmp1-1)/2;
      assert(round(lkRad)==lkRad);

%       G = cell(N,1);
%       L = cell(N,1);
      S = cell(N,n1);
%       GS = cell(N,n1);
%       LS = cell(N,n1);
      SGS = cell(N,n1,n2);
      SLS = cell(N,n1,n2);
      
      for iTrl = 1:N
        fprintf(1,'Working on iTrl=%d/%d\n',iTrl,N);
        im = Is{iTrl};
%         G{iTrl} = gradientMag(im);
%         L{iTrl} = filter2(opts.laplaceKernel,im,'same');
        
        for i1 = 1:n1
          s1 = sig1(i1);
          sIm = gaussSmooth(im,s1,'same',opts.gaussFiltRadius);
          S{iTrl,i1} = sIm;
          GsIm = gradientMag(single(sIm));
          LsIm = zeros(size(sIm));
          LsIm(1+lkRad:end-lkRad,1+lkRad:end-lkRad) = ...
            filter2(opts.laplaceKernel,sIm,'valid');
          if true
            if s1==0
              % gaussian smoothing not performed; maybe should normalize by
              % something but not sure what value
              % warning('Features:laplace','Laplacian: not sure what to normalize by for sd=0 (no smoothing');

              % Normalizing by 1.0 (or not normalizing) seems rightish
            else
              % scale-normalized laplacian
              LsIm = s1.^2*LsIm;
            end
          end
          
          for i2 = 1:n2
            s2 = sig2(i2);
            SGS{iTrl,i1,i2} = gaussSmooth(GsIm,s2,'same',opts.gaussFiltRadius);
            SLS{iTrl,i1,i2} = gaussSmooth(LsIm,s2,'same',opts.gaussFiltRadius);
            
            if opts.sgsRescale
              sgs = SGS{iTrl,i1,i2}*opts.sgsRescaleFacs(i1,i2);
              tfOOB = sgs<0 | sgs>255;
              if any(tfOOB(:))
                warningNoTrace('Features:oob','SGS out of bounds: %d pixels.',nnz(tfOOB));
              end
              SGS{iTrl,i1,i2} = uint8(sgs);
            end
            if opts.slsRescale
              sls = SLS{iTrl,i1,i2}*opts.slsRescaleFacs(i1,i2)+128;
              tfOOB = sls<0 | sls>255;
              if any(tfOOB(:))
                warningNoTrace('Features:oob','Rescaling SLS. %d/%d pxs (%.2f %%) clipped.',...
                  nnz(tfOOB),numel(tfOOB),nnz(tfOOB)/numel(tfOOB)*100);
              end
              SLS{iTrl,i1,i2} = uint8(sls);
            end
          end
        end
      end
    end
    
    function [axS,axSGS,axSLS] = ppViz(S,SGS,SLS,sig1,sig2,iTrl)
      % Visualize preprocessed images
      %
      % S/SGS/SLS/sig1/sig2: see Features.pp
      % iTrl: scalar index into Is
      %
      % axS: [n1] axes for visualizing S
      % axSGS: [n2xn1]  " SGS
      % axSLS: [n2xn1] etc
      
      n1 = numel(sig1);
      n2 = numel(sig2);
      figure;
      borders = [.05 0.01;.05 0.01];
      axS = createsubplots(n1,1,borders);
      figure;
      axSGS = createsubplots(n1,n2,borders); 
      figure;
      axSLS = createsubplots(n1,n2,borders);
      
      [smax,smin] = lclImMaxMin(S);
%       [sgsmax,sgsmin] = lclImMaxMin(SGS);
%       [slsmax,slsmin] = lclImMaxMin(SLS);

      for i1 = 1:n1
        s1 = sig1(i1);
        
        imshow(S{iTrl,i1},'parent',axS(i1),'DisplayRange',[smin,smax]);
        args1 = {sprintf('\\sigma_1=%.3g',s1),...
          'interpreter','tex','fontsize',16,'fontweight','bold'};
        ylabel(axS(i1),args1{:},'verticalalignment','top');
        
        for i2 = 1:n2
          % i1 is ROW index, i2 is COL index
          iAx = i1+(i2-1)*n1;
          s2 = sig2(i2);
          args2 = {sprintf('\\sigma_2=%.3g',s2),...
            'interpreter','tex','fontsize',16,'fontweight','bold'};
          
          imshow(SGS{iTrl,i1,i2},'parent',axSGS(iAx));
          imshow(SLS{iTrl,i1,i2},'parent',axSLS(iAx));
          %imagesc(SLS{iTrl,i1,i2},'parent',axSLS(iAx));
          axSLS(iAx).XTickLabel = [];
          axSLS(iAx).YTickLabel = [];
          
          if i1==1 % ROW 1
            title(axSGS(iAx),args2{:});
            title(axSLS(iAx),args2{:});
          end
          if i2==1 % COL 1
            ylabel(axSGS(iAx),args1{:});
            ylabel(axSLS(iAx),args1{:});
          end
        end
      end
      
      lp = linkprop(axS,'CLim');
      axS(1).UserData = lp;
      lp = linkprop(axSGS,'CLim');
      axSGS(1).UserData = lp;
      lp = linkprop(axSLS,'CLim');
      axSLS(1).UserData = lp;
      linkaxes([axS(:);axSGS(:);axSLS(:)]);
    end
    
  end
  
  
  %%
  methods (Static)
  
    function visualize(type,model,pGt,pCur,Is,imgIds,xF,yF,info,varargin)
      defaultPrms.ax = [];  
      defaultPrms.txtDx = 2;
      prms = getPrmDfltStruct(varargin,defaultPrms);
      
      if isempty(prms.ax)
        figure('windowstyle','docked');
        ax = axes;
      else
        ax = prms.ax;
        cla(ax,'reset');
      end
      
      [N,F] = size(xF);
      for iN = 1:N
        cla(ax,'reset');
        imagesc(Is{imgIds(iN)},'Parent',ax);
        colormap gray;
        hold(ax,'on');        
        for iPt = 1:model.nfids
          plot(ax,pGt(iN,iPt),pGt(iN,iPt+model.nfids),'wx','LineWidth',3);
          plot(ax,pCur(iN,iPt),pCur(iN,iPt+model.nfids),'cx','LineWidth',3,'MarkerSize',15);
          text(pCur(iN,iPt)+prms.txtDx,pCur(iN,iPt+model.nfids),num2str(iPt),'Color',[0 1 1]);
        end
        axis(ax,'xy');
        h = [];
        for iF = 1:F
          delete(h(ishandle(h)));          
          switch type
            case '1lm'
              [h,str] = Features.visualize1LM(ax,xF(iN,iF),yF(iN,iF),info,iN,iF);
            case '2lm'
              [h,str] = Features.visualize2LM(ax,xF(iN,iF),yF(iN,iF),info,iN,iF);
          end
          
          input(str);
        end        
      end
    end
    
  end
  
  methods (Static) % Two-landmark features
    
    function [xs,prms] = generate2LM(model,varargin)
      % Generate 2-landmark features
      % 
      % xs: F x 5 double array. xs(i,:) specifies the ith feature
      %   col 1: landmark index 1
      %   col 2: landmark index 2
      %   col 3: radius FACTOR in pixels
      %   col 4: theta
      %   col 5: "interpolation" factor in [0,1] for location of center
      %          between landmarks 1/2
      %   col 6: channel index (identically equal to 1 unless optional
      %     param 'nchan' specified
      %
      % prms: scalar struct, params used to compute xs
            
      %%% Optional input P-V params
      
      % number of ftrs to generate
      defaultPrms.F = 20;
      % maximum distance within which to sample; dimensionless scaling
      % factor applied to ellipse size
      defaultPrms.radiusFac = 1;
      % if true, the feature "center" will be a random location chosen on 
      % the line segement connecting landmarks 1/2. If false, the center
      % will be the centroid of landmarks 1/2.
      defaultPrms.randctr = false;
      % vector of landmark indices from which to sample. If unspecified,
      % defaults (conceptually) to 1:model.nfids.
      defaultPrms.fids = [];
      % model.nfids x 1 cell vector. dfs.neighbors{i} is a vector of 
      % landmark indices indicating neighbors of landmark i. If not 
      % supplied, all landmarks are considered mutual neighbors. If
      % supplied, landmarks 1 and 2 will always be chosen to be neighbors.
      defaultPrms.neighbors = [];
      % number of channels. 
      defaultPrms.nchan = 1;
                                  
      prms = getPrmDfltStruct(varargin,defaultPrms);
      
      tfFidsSpeced = ~isempty(prms.fids);
      tfNeighborsSpeced = ~isempty(prms.neighbors);
      if tfFidsSpeced && tfNeighborsSpeced
        warning('TwoLandmarkFeatures:params',...
          '.fids and .neighbors both specified; landmark1 chosen from .fids, landmark2 from neighbors.');
      end      
      if tfFidsSpeced      
        assert(all(ismember(prms.fids,1:model.nfids)),'Invalid ''fids''.');
      else
        prms.fids = 1:model.nfids;
      end
      if tfNeighborsSpeced
        assert(iscell(prms.neighbors) && numel(prms.neighbors)==model.nfids);
        cellfun(@(x)assert(all(ismember(x,1:model.nfids))),prms.neighbors);
      end

      F = prms.F;
      xs = nan(F,6);

      nfidsFtr = numel(prms.fids); % can differ from model.nfids
      if ~tfNeighborsSpeced
        % choose two random, distinct landmarks
        xs(:,1) = randint2(F,1,[1,nfidsFtr]);
        xs(:,2) = randint2(F,1,[1,nfidsFtr-1]);
        tf = xs(:,2)>=xs(:,1);
        xs(tf,2) = xs(tf,2)+1;
        xs(:,1:2) = prms.fids(xs(:,1:2));
      else
        % choose one random landmark from prms.fids
        xs(:,1) = randint2(F,1,[1,nfidsFtr]);
        xs(:,1) = prms.fids(xs(:,1));
        % select landmark 2 using neighbors; NOTE: this could include 
        % landmarks outside prms.fids
        for i = prms.fids(:)'
          tf = xs(:,1)==i;
          neighbors = prms.neighbors{i};
          xs(tf,2) = neighbors(randint2(nnz(tf),1,[1 numel(neighbors)]));
        end
      end
      
      xs(:,3) = prms.radiusFac*rand(F,1);
      xs(:,4) = (2*pi*rand(F,1))-pi;
      
      if prms.randctr
        xs(:,5) = rand(F,1);
      else
        xs(:,5) = 0.5*ones(F,1);
      end
      
      xs(:,6) = randint2(F,1,[1,prms.nchan]);
    end
    
    function [xF,yF,chan,info] = compute2LM(xs,xLM,yLM)
      % xs: F x 6, from generate2LM()
      % xLM: N x npts. xLM(i,j) gives x-positions ('columns') for ith
      % instance, jth landmark
      % yLM: etc
      %
      % xF: N x F. xF(i,j) gives x-pos for ith instance, jth feature
      % yF: N x F.
      % chan: N x F. chan(i,j) gives channel in which to compute feature
      %   for ith instance, jth feature.
      % info: struct with miscellaneous/intermediate vars
      %
      % Note: xF and yF are in whatever units xLM and yLM are in.
      
      l1 = xs(:,1);
      l2 = xs(:,2);
      rfac = xs(:,3);
      theta = xs(:,4);
      ctrFac = xs(:,5);
      chan = xs(:,6);
      
      N = size(xLM,1);
      x1 = xLM(:,l1); % N X F. x, landmark 1 for each instance/feature
      y1 = yLM(:,l1); % etc
      x2 = xLM(:,l2);
      y2 = yLM(:,l2);
      theta = repmat(theta',N,1);      
      chan = repmat(chan',N,1);
      
      alpha = atan2(y2-y1,x2-x1); 
      r = sqrt((x2-x1).^2+(y2-y1).^2)/2; 
      a = repmat(rfac',N,1).*r; 
      b = repmat(rfac',N,1).*r/2;
      % AL: previously, rfac was being specified in units of pixels, 
      % leading to (apparently) very large a/b
      ctrX = bsxfun(@times,x1,ctrFac') + bsxfun(@times,x2,(1-ctrFac)');
      ctrY = bsxfun(@times,y1,ctrFac') + bsxfun(@times,y2,(1-ctrFac)');

      cost = cos(theta);
      sint = sin(theta);
      cosa = cos(alpha);
      sina = sin(alpha);
      xF = ctrX + a.*cost.*cosa - b.*sint.*sina;
      yF = ctrY + a.*cost.*sina + b.*sint.*cosa;
      % cs1=ctrX+(repmat(xs',size(r,1),1).*r.*cos(theta+alpha));
      % rs1=ctrY+(repmat(xs',size(r,1),1).*r.*sin(theta+alpha));      
      xF = round(xF);
      yF = round(yF);      
      
      if nargout>=4
        info = struct();
        info.l1 = l1;
        info.l2 = l2;
        info.xLM1 = x1;
        info.yLM1 = y1;
        info.xLM2 = x2;
        info.yLM2 = y2;
        info.theta = theta;
        info.alpha = alpha;
        info.r = r;
        info.araw = r;
        info.braw = r/2;
        info.a = a; % after scaling by rfac
        info.b = b; % etc
        info.xc = ctrX;
        info.yc = ctrY;
      end
    end
    
    function [h,str] = visualize2LM(ax,xf,yf,info,iN,iF)
      % Visualize feature pts from compute2LM
      % 
      % xf/yf/info: from compute2LM
      % iN: trial index
      % iF: feature index
      
      x1 = info.xLM1(iN,iF);
      x2 = info.xLM2(iN,iF);
      y1 = info.yLM1(iN,iF);
      y2 = info.yLM2(iN,iF);
      xc = info.xc(iN,iF);
      yc = info.yc(iN,iF);
      
      h = [];
      h(end+1) = plot(ax,[x1;xc;x2],[y1;yc;y2],'r-','markerfacecolor',[1 0 0]); %#ok<*AGROW>
      h(end+1) = plot(ax,[x1;xc;x2],[y1;yc;y2],'ro','markerfacecolor',[1 0 0]);
      h(end+1) = text(x1,y1,'1','parent',ax);
      set(h(end),'color',[0 0 1]);
      h(end+1) = text(x2,y2,'2','parent',ax);
      set(h(end),'color',[0 0 1]);
      h(end+1) = plot(ax,xf(iN,iF),yf(iN,iF),'go','markerfacecolor',[0 1 0]);
      %h(end+1) = ellipsedraw(info.araw(iN,iF),info.braw(iN,iF),xc,yc,info.alpha(iN,iF),'-g','parent',ax);
      %h(end+1) = plot(ax,[xc;xf],[yc;yf],'-g');
      str = sprintf('n=%d,f=%d(%d,%d). r=%.3f, theta=%.3f',iN,iF,...
        info.l1(iF),info.l2(iF),info.r(iN,iF),info.theta(iN,iF));
      title(ax,str,'interpreter','none','fontweight','bold'); 
    end
    
  end
  
  methods (Static) % One-landmark features
    
    function [xs,prms] = generate1LM(model,varargin)
      % Generate 1-landmark features
      % 
      % xs: F x 3 double array. xs(i,:) specifies the ith feature
      %   col 1: landmark index 1
      %   col 2: radius in pixels
      %   col 3: theta
      %
      % prms: scalar struct, params used to compute xs
            
      %%% Optional input P-V params
      
      % number of ftrs to generate
      defaultPrms.F = 20;
      % maximum distance within which to sample; dimensionless scaling
      % factor applied to ellipse size
      defaultPrms.radius = 1;
      
      prms = getPrmDfltStruct(varargin,defaultPrms);      
      F = prms.F;
      nfids = model.nfids;
      
      xs = nan(F,3);
      xs(:,1) = randint2(F,1,[1,nfids]);
      xs(:,2) = prms.radius*rand(F,1);
      xs(:,3) = (2*pi*rand(F,1))-pi;
    end
    
    function [xF,yF,info] = compute1LM(xs,xLM,yLM)

      l1 = xs(:,1);
      r = xs(:,2);
      theta = xs(:,3);
      
      x = xLM(:,l1); % N X F
      y = yLM(:,l1); % etc
      
      dx = r.*cos(theta);
      dy = r.*sin(theta);
      xF = round(bsxfun(@plus,x,dx'));
      yF = round(bsxfun(@plus,y,dy'));
      
      if nargout>=3
        info = struct();
        info.l1 = l1;
        info.r = r;
        info.theta = theta;
        info.xLM = x;
        info.yLM = y;
      end
    end
    
    function [h,str] = visualize1LM(ax,xf,yf,info,iN,iF)
      x1 = info.xLM(iN,iF);
      y1 = info.yLM(iN,iF);
      
      h = [];
      h(end+1) = plot(ax,x1,y1,'ro','markerfacecolor',[1 0 0]);
      h(end+1) = plot(ax,xf,yf,'go','markerfacecolor',[0 1 0]);
      h(end+1) = ellipsedraw(info.r(iF),info.r(iF),x1,y1,0,'-g','parent',ax);
      h(end+1) = plot(ax,[x1;xf],[y1;yf],'-g');
      str = sprintf('n=%d,f=%d(%d). r=%.3f, theta=%.3f',iN,iF,info.l1(iF),...
        info.r(iF),info.theta(iF));
      title(ax,str,'interpreter','none','fontweight','bold');
    end
    
  end
    
end

function [mx,mn] = lclImMaxMin(Is)
% Is: cell array of images of same size

Is = cat(1,Is{:});
Is = Is(:);
mx = max(Is);
mn = min(Is);
end
