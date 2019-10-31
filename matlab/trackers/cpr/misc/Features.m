classdef Features
  
  properties (Constant)      
    % take sgs = SGS{i1,i2}; it will have a hist in range [0,max]; this
    % array gives the maxes for sig1,sig2 = [0 2 4 8]
    JAN_SGS_MAX_SIG0248 = [
      103 44 30 17;37 25 19 12;16 15 13 9;9 9 9 7];
    
    JAN_SGS_99P9_SIG0248_160211 = [
      53.5000   34.5000   25.5000   18.5000
      29.5000   24.5000   18.5000   12.5000
      18.5000   17.5000   14.5000    9.5000
      10.5000   10.5000    9.5000    7.5000];
            
    % take sls = SLS{i1,i2}; it will have a mean of ~0 and a span; this
    % array gives the spans for sig1,sig2 = [0 2 4 8]
    JAN_SLS_SPAN_SIG0248 = [
      294 34 9 2; 135 73 28 7; 145 113 62 26; 117 111 102 80];
    
    % span of prctile(SLS{i1,i2},[2 98]) for sig1,sig2 = [0 2 4 8]
    JAN_SLS_SPAN98_SIG0248 = [
      26 5 2.3 1.1; 20 13 8.1 4.2; 37 32 25 15; 69 66 60 43];    

    JAN_SLS_SPAN99_SIG0248 = [
      38 7 3 1.3;28 18 11 5.1;48 42 32 19;85 82 74 54];
    
    JAN_SLS_SPAN99_SIG0248_160211 = [
      51     9     5     2
      35    25    15     7
      63    56    41    22
      97    93    83    60];
    
    TYPE2XSCOLLBLS = containers.Map(...
      {'single landmark' '2lm' '2lmdiff' 'two landmark elliptical'},...
      { {'lm' 'radius' 'theta' 'view'},...
        {'lm1' 'lm2' 'rfac' 'theta' 'ctrfac' 'chan' 'view'},...
        [],...
        []});
      % TODO: 2lmdiff. TODO: hardcoded here -- see generate/compute/visualize1LM, 2LM etc
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
      % - sRescale. Scalar logical. Defaults FALSE. If true, rescale S by
      % sRescaleFacs and convert to uint8.
      % - sRescaleFacs. [n1]. Must be supplied if sRescale is true.
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
      opts.sRescale = false;
      opts.sRescaleFacs = nan;
      opts.sgsRescale = true;
      opts.sgsRescaleFacs = 200./Features.JAN_SGS_99P9_SIG0248_160211; % 200 instead of 256 for safety buffer
      opts.slsRescale = true;
      opts.slsRescaleFacs = 200./Features.JAN_SLS_SPAN99_SIG0248_160211; % 200 instead of 256 for safety buffer
      opts.sgsRescaleClipPctThresh = 1; % warn if pct SLS/SGS pixels clipped greater than this val
      opts.hWaitBar = [];
      opts = getPrmDfltStruct(varargin,opts);      
      
      assert(iscell(Is) && isvector(Is));
      N = numel(Is);
      validateattributes(sig1,{'numeric'},{'vector' 'real' 'nonnegative'});
      validateattributes(sig2,{'numeric'},{'vector' 'real' 'nonnegative'});
      n1 = numel(sig1);
      n2 = numel(sig2);
      validateattributes(opts.gaussFiltRadius,{'numeric'},{'scalar' 'real' 'positive'});
      if opts.sRescale
        assert(numel(opts.sRescaleFacs)==n1);
      end
      if opts.sgsRescale
        assert(isequal(size(opts.sgsRescaleFacs),[n1 n2]));
      end
      if opts.slsRescale
        assert(isequal(size(opts.slsRescaleFacs),[n1 n2]));
      end
      tfWB = ~isempty(opts.hWaitBar);

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
      
      if tfWB
        hWB = opts.hWaitBar;
        hWB.Name = 'Preprocessing';        
        waitbar(0,hWB,'Preprocessing images');
      end
      for iTrl = 1:N
        if tfWB
          waitbar(iTrl/N,hWB);
        else
          if mod(iTrl,10)==0
            fprintf(1,'Working on iTrl=%d/%d\n',iTrl,N);
          end
        end
        im = Is{iTrl};
%         G{iTrl} = gradientMag(im);
%         L{iTrl} = filter2(opts.laplaceKernel,im,'same');
        
        for i1 = 1:n1
          s1 = sig1(i1);
          sIm = gaussSmooth(im,s1,'same',opts.gaussFiltRadius);
          S{iTrl,i1} = sIm;
          if opts.sRescale
            s = S{iTrl,i1}*opts.sRescaleFacs(i1);
            tfOOB = s<0 | s>255;
            pctOOB = nnz(tfOOB)/numel(tfOOB)*100;
            if pctOOB>opts.sgsRescaleClipPctThresh
              warningNoTrace('Features:oob','Rescaling S. %d/%d pxs (%.2f %%) clipped.',...
                nnz(tfOOB),numel(tfOOB),pctOOB);
            end
            S{iTrl,i1} = uint8(s);
          else
            S{iTrl,i1} = uint8(S{iTrl,i1});
          end
          
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
              pctOOB = nnz(tfOOB)/numel(tfOOB)*100;
              if pctOOB>opts.sgsRescaleClipPctThresh
                warningNoTrace('Features:oob','Rescaling SGS. %d/%d pxs (%.2f %%) clipped.',...
                  nnz(tfOOB),numel(tfOOB),pctOOB);
              end
              SGS{iTrl,i1,i2} = uint8(sgs);
            else
              SGS{iTrl,i1,i2} = uint8(SGS{iTrl,i1,i2});
            end
            if opts.slsRescale
              sls = SLS{iTrl,i1,i2}*opts.slsRescaleFacs(i1,i2)+128;
              tfOOB = sls<0 | sls>255;
              pctOOB = nnz(tfOOB)/numel(tfOOB)*100;
              if pctOOB>opts.sgsRescaleClipPctThresh
                warningNoTrace('Features:oob','Rescaling SLS. %d/%d pxs (%.2f %%) clipped.',...
                  nnz(tfOOB),numel(tfOOB),pctOOB);
              end
              SLS{iTrl,i1,i2} = uint8(sls);
            else
              SLS{iTrl,i1,i2} = uint8(SLS{iTrl,i1,i2});
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
          %imshow(SLS{iTrl,i1,i2},'parent',axSLS(iAx));
          imagesc(SLS{iTrl,i1,i2},'parent',axSLS(iAx));
          tmp = SLS{iTrl,i1,i2}(:);
          tmpclass = class(tmp);
          ptilesclass = feval(tmpclass,[5 95]);
          p5_95 = prctile(SLS{iTrl,i1,i2}(:),ptilesclass);
          caxis(axSLS(iAx),p5_95);

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
    
    function [S99p9,SGSmax,SGS99p9,SLSspn,SLSspn99,SLSspn98,SLSmu,SLSmdn] = ...
        ppCalib(S,SGS,SLS,sig1,sig2)
      % Compute scale factors for SGS and SLS.
      %
      % SGS matrices take values in [0,maxgradient]. For channel images we
      % need these to fit in uint8s [0,256). So we need to compute the
      % max/maxish of SGS.
      %
      % SLS matrices take values in [minLaplace,maxLaplace], where
      % minLaplace is negative and |minLaplace|~|maxLaplace|. Again, to
      % utilize SLS data as 'channel' images, we need something in range
      % [0,256). So we compute the spans of SLS.
      %
      % S: [nTrl x numel(sig1)];
      % SGS/SLS: [nTrlxnumel(sig1)xnumel(sig2)]
      % 
      % SGSmax: [numel(sig1)xnumel(sig2)]. maximum of each SGS taken over
      % all trials.
      % SGS99p9: 99.9 percentile of each SGS, etc.
      % SLSspn: [numel(sig1)xnumel(sig2)] cell. Each element is a two-vec
      % [lo hi] giving the span.
      
      n1 = numel(sig1);
      n2 = numel(sig2);
      
      % aggregate data
      sbig = cell(n1,1);
      sgsbig = cell(n1,n2);
      slsbig = cell(n1,n2);
      for i1=1:n1
        sall = S(:,i1);
        sall = cellfun(@(x)x(:),sall,'uni',0);
        sbig{i1} = cat(1,sall{:});
        for i2=1:n2
          sgsall = SGS(:,i1,i2);
          sgsall = cellfun(@(x)x(:),sgsall,'uni',0);
          sgsbig{i1,i2} = cat(1,sgsall{:});
          
          slsall = SLS(:,i1,i2);
          slsall = cellfun(@(x)x(:),slsall,'uni',0);
          slsbig{i1,i2} = cat(1,slsall{:});          
        end
      end
      
      % compute stats
      S99p9 = nan(n1,1);
      SGSmax = nan(n1,n2);
      SGS99p9 = nan(n1,n2);
      SLSspn = cell(n1,n2);
      SLSspn99 = cell(n1,n2);
      SLSspn98 = cell(n1,n2);
      SLSmu = nan(n1,n2);
      SLSmdn = nan(n1,n2);
      for i1=1:n1
        s = sbig{i1};
        assert(iscolumn(s) && all(s>=0));
        S99p9(i1) = prctile(s,99.9);        
        for i2=1:n2
          sgs = sgsbig{i1,i2};
          assert(iscolumn(sgs) && all(sgs>=0));
          SGSmax(i1,i2) = max(sgs);
          SGS99p9(i1,i2) = prctile(sgs,99.9);
          
          sls = slsbig{i1,i2};
          assert(iscolumn(sls));  
          span = [min(sls) max(sls)];          
          span99 = prctile(sls,[1 99]);
          span98 = prctile(sls,[2 98]);
          assert(span(1)<0 && span(2)>0);
          assert(span99(1)<0 && span99(2)>0);
          assert(span98(1)<0 && span98(2)>0);
%           skewfcn = @(x)sum(x)/mean(abs(x));
%           spanskew = skewfcn(span);
%           span99skew = skewfcn(span99);
%           span98skew = skewfcn(span98);
%           UNEXPECTED_SKEW_THRESH = .25;
%           if abs(spanskew)>UNEXPECTED_SKEW_THRESH || ...
%              abs(span99skew)>UNEXPECTED_SKEW_THRESH || ...
%              abs(span98skew)>UNEXPECTED_SKEW_THRESH 
%             warning('Features:ppCalib','Unexpected skew, SLS(:,%d,%d). spanskew span99skew span98skew: %.2f %.2f %.2f',....
%               i1,i2,spanskew,span99skew,span98skew);
%           end
          
          SLSspn{i1,i2} = span;
          SLSspn99{i1,i2} = span99;
          SLSspn98{i1,i2} = span98;          
          SLSmu(i1,i2) = mean(sls); 
          SLSmdn(i1,i2) = median(sls);
        end
      end
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
            case 'single landmark'
              [h,str] = Features.visualize1LM(ax,xF(iN,iF),yF(iN,iF),info,iN,iF);
            case '2lm'
              [h,str] = Features.visualize2LM(ax,xF(iN,iF),yF(iN,iF),info,iN,iF);
          end
          
          input(str);
        end        
      end
    end
    
  end
  
  %#3DOK
  methods (Static) % Two-landmark features
    
    function [xs,prms,xslbls] = generate2LM(model,varargin)
      % Generate 2-landmark features
      % 
      % xs: F x 7 double array. xs(i,:) specifies the ith feature
      %   col 1: landmark index 1
      %   col 2: landmark index 2
      %   col 3: radius FACTOR. In [0,prms.radiusFac].
      %   col 4: theta in (-pi,pi). NOTE: this is not the actual angle on
      %   the ellipse; it is an angle-like parameterization of pts on the
      %   surface of an ellipse!
      %   col 5: "interpolation" factor in [0,1] for location of center
      %          between landmarks 1/2
      %   col 6: channel index, shared by landmarks 1 and 2. Identically 
      %     equal to 1 unless optional param 'nchan' specified.
      %   col 7: view index, shared by landmarks 1 and 2. Identically equal
      %     to 1 unless model.nviews>1.
      %
      % prms: scalar struct, params used to compute xs
      %
      % xslbls: [1x7] labels for cols of xs
      
      %%% Optional input P-V params
      %
      % Note. These options are overly complicated. For a random pt within
      % the ellipse (a/b=2) fitting snugly between landmarks 1 and 2, use
      % defaults of radiusFac==1, randctr=false.
      % 
      % number of ftrs to generate
      defaultPrms.F = 20;
      % maximum distance within which to sample; dimensionless scaling
      % factor applied to ellipse size. See note above.
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
      xs = nan(F,7);
      d = model.d;
      nviews = model.nviews;
      assert(d==2 && nviews==1 || d==3 && nviews>1);

      nfidsFtr = numel(prms.fids); % can differ from model.nfids
      assert(nfidsFtr>1,'Must sample from 2 or more features.');
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
      xs(:,7) = randint2(F,1,[1,nviews]);
      
      xslbls = {'lm1' 'lm2' 'rfac' 'theta' 'ctrfac' 'chan' 'view'};
    end
    
    function [tbl,prms] = generate2LMelliptical(model,varargin)
      % Generate new-style elliptical 2-landmark features
      %
      % At feature computation/instantiation time, there will be a shape
      % represented by a set of landmarks. A feature (row of table) is
      % defined as follows. Two landmarks are specified: .lm1 and .lm2. A
      % line segment of length 2d connects these two lms. Consider an
      % ellipse centered at the midpoint of this line segment, and oriented
      % with major axis along the line segment. This ellipse has major axis
      % a=d*prm.radiusFac and minor axis b=a/prm.abratio. (If radiusFac=1,
      % then the ellipse is "squeezed" tangentially between the two
      % landmarks.)
      %
      % A random point (uniformly drawn) is specified within this ellipse 
      % via two parameters, .reff (r_effective, dimensionless) and .theta
      % (radians). These two parameters serve to specify the feature point
      % location in a scale/orientation-independent manner relative to the
      % two landmarks (and more generally, the shape).
      %
      % tbl: [FxnTblFld] feature defn table. Note, in future this table can
      %         be expanded by adding further fields.
      %     .lm1 -- landmark 1 (point index) 
      %     .lm2 -- landmark 2 etc
      %     .reff -- see mathematical notes below
      %     .xeff -- etc
      %     .yeff -- etc
      %     .chan -- channel index, shared by landmarks 1 and 2. Identically 
      %              equal to 1 unless optional param 'nchan' specified.
      %     .view -- view index, shared by landmarks 1 and 2. Identically equal
      %              to 1 unless model.nviews>1.
      %
      % prms: [1] struct, parameters used to generate tbl
      
      %%% Optional input P-V params
      %
      % 
      % number of ftrs to generate
      defaultPrms.F = 20;
      % dimensionless scaling factor applied to ellipse size, see above.
      defaultPrms.radiusFac = 1.5;
      % ratio a/b of major/minor axis lengths for ellipse.
      defaultPrms.abratio = 2;
      % number of channels. 
      defaultPrms.nchan = 1;
                                  
      prms = getPrmDfltStruct(varargin,defaultPrms);
      
      prms.fids = 1:model.nfids; % for moment draw from all landmarks
      F = prms.F;      
      d = model.d;
      nviews = model.nviews;
      assert(d==2 && nviews==1 || d==3 && nviews>1);

      nfidsFtr = numel(prms.fids); % in concept can differ from model.nfids
      assert(nfidsFtr>1,'Must sample from 2 or more features.');
      
      % choose two random, distinct landmarks
      xs(:,1) = randint2(F,1,[1,nfidsFtr]);
      xs(:,2) = randint2(F,1,[1,nfidsFtr-1]);
      tf = xs(:,2)>=xs(:,1);
      xs(tf,2) = xs(tf,2)+1;
      xs(:,1:2) = prms.fids(xs(:,1:2));
      lm1 = xs(:,1);
      lm2 = xs(:,2);

      % We use this mathematical fact. Given the major/minor axes a and b
      % for an ellipse centered at the origin with major axis oriented 
      % in the x-dir, the following computation of (x,y) draws a random 
      % point uniformly within that ellipse.
      %
      % r = rand;
      % theta = rand*2*pi-pi;
      % x = a*sqrt(r)*cos(theta);
      % y = b*sqrt(r)*sin(theta);
      %
      % Once an (r,theta) pair is generated, (x,y) are linear in the axes
      % lengths (a,b). A given (r,theta) pair specifies a scale-invariant
      % point relative to the ellipse.
      %
      % In our case, at feature computation/instantiation time we have 
      %   a_final = d*prm.radiusFac and
      %   b_final = d*prm.radiusFac/prm.abRatio => 
      % 
      %   x = d*prm.radiusFac*sqrt(r)*cos(theta)
      %   y = d*prm.radiusFac/prm.abRatio*sqrt(r)*sin(theta)
      %
      % We define 
      %   reff = prm.radiusFac*sqrt(r) 
      %   xeff = reff*cos(theta) 
      %   yeff = reff/abratio*sin(theta)
      % So that
      %   x = d*reff*cos(theta) = d*xeff
      %   y = d*reff/abRatio*sin(theta) = d*yeff
      %
      % Note, we save reff, xeff, yeff to the table which enables backing
      % out theta and abratio.

      r = rand(F,1);
      reff = prms.radiusFac*sqrt(r);
      theta = (2*pi*rand(F,1))-pi;
      abratio = prms.abratio;
      xeff = reff.*cos(theta);
      yeff = reff.*sin(theta)/abratio;
      chan = randint2(F,1,[1,prms.nchan]);
      view = randint2(F,1,[1,nviews]);
      
      tbl = table(lm1,lm2,reff,theta,xeff,yeff,chan,view);
    end
    
    function [tbl,prms] = ...
        generate2LMellipticalForSetParamViz(model,radiusFac,abratio)
      % Like generate2LMelliptical with maximum radius for set-parameter 
      % visualization
      [tbl,prms] = Features.generate2LMelliptical(model,'F',1,...
        'radiusFac',radiusFac,'abratio',abratio,'nchan',1);
      tbl.reff = radiusFac; 
      tbl.xeff = tbl.reff.*cos(tbl.theta);
      tbl.yeff = tbl.reff.*sin(tbl.theta)/abratio;
      tbl.view = 1;
    end
    
    function [xF,yF,chan,iview,info] = compute2LM(xs,xLM,yLM)
      % xs: [Fx7], from generate2LM(). Legacy [Fx6] okay, assume nView==1.
      % xLM: [NxnPtsxnView]. xLM(i,:,:) gives x-coords (column pixel
      %  indices) of ith shape, across all pts/views
      % yLM: [NxnPtxnView]. y-coords
      %
      % xF: N x F. xF(i,j) gives x-coords (in pixels) for ith instance, jth 
      %   feature. Values are rounded. Could lie "outside" image. xF(i,j)
      %   should be read in channel chan(i,j), view iview(j).
      % yF: N x F.
      % chan: N x F. Channels in which to read features.
      % iview: [Fx1]. View indices in which to read features. Labels
      % columns of xF, yF.
      % info: struct with miscellaneous/intermediate vars
      %
      % Note: xF and yF are in whatever units xLM and yLM are in (pixels)
      
      if size(xs,2)==6
        xs(:,7) = 1;
      end
      
      l1 = xs(:,1);
      l2 = xs(:,2);
      rfac = xs(:,3);
      theta = xs(:,4); % NOTE: this is not the actual angle, this is an angle-like parameterization
      ctrFac = xs(:,5);
      chan = xs(:,6);
      iview = xs(:,7);
      
      [N,nPt,nView] = size(xLM);
      nFlat = nPt*nView;
      xLMFlat = reshape(xLM,N,nFlat); 
      yLMFlat = reshape(yLM,N,nFlat);
      iFlat1 = sub2ind([nPt nView],l1,iview);
      iFlat2 = sub2ind([nPt nView],l2,iview);
      x1 = xLMFlat(:,iFlat1); % N X F. x, landmark 1 x-coord for each instance/feature
      y1 = yLMFlat(:,iFlat1); % etc
      x2 = xLMFlat(:,iFlat2);
      y2 = yLMFlat(:,iFlat2);
         
      theta = repmat(theta',N,1);
      chan = repmat(chan',N,1);
      %iview = repmat(iview',N,1);
      
      alpha = atan2(y2-y1,x2-x1); % [NxF]
      r = sqrt((x2-x1).^2+(y2-y1).^2)/2; % [NxF], pixels
      a = repmat(rfac',N,1).*r; % [NxF], pixels
      b = repmat(rfac',N,1).*r/2; % [NxF], pixels
      % AL: previously, rfac was being specified in units of pixels, 
      % leading to (apparently) very large a/b
      ctrX = bsxfun(@times,x1,ctrFac') + bsxfun(@times,x2,(1-ctrFac)');
      ctrY = bsxfun(@times,y1,ctrFac') + bsxfun(@times,y2,(1-ctrFac)');

      cost = cos(theta);
      sint = sin(theta);
      cosa = cos(alpha);
      sina = sin(alpha);
      % 1. First we have parametrized pts on an ellipse with major/minor 
      % axis a/b via (x,y)=(a*cos(theta),b*sin(theta)). Note, here theta 
      % IS NOT THE ACTUAL ANGLE of (x,y) relative to the origin.
      % 2. We have rotated the entire ellipse counterclockwise by alpha      
      xF = ctrX + a.*cost.*cosa - b.*sint.*sina;
      yF = ctrY + a.*cost.*sina + b.*sint.*cosa;
      % cs1=ctrX+(repmat(xs',size(r,1),1).*r.*cos(theta+alpha));
      % rs1=ctrY+(repmat(xs',size(r,1),1).*r.*sin(theta+alpha));      
      xF = round(xF);
      yF = round(yF);
      
      if nargout>=5
        info = struct();
        info.l1 = l1;
        info.l2 = l2;
        info.xLM1 = x1;
        info.yLM1 = y1;
        info.xLM2 = x2;
        info.yLM2 = y2;
        info.rfac = rfac;
        info.ctrFac = ctrFac;
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
    
    function [xF,yF,chan,iview,info] = compute2LMelliptical(tbl,xLM,yLM)
      % Compute/"instantiate" feature point positions for given landmark 
      % positions
      % 
      % tbl: [FxnTblFlds], from generate2LMelliptical(). 
      % xLM: [NxnPtsxnView]. xLM(i,:,:) gives x-coords (column pixel
      %  indices) of ith shape, across all pts/views
      % yLM: [NxnPtxnView]. y-coords
      %
      % xF: [NxF]. xF(i,j) gives x-coords (in pixels) for ith instance, jth 
      %   feature. Values are rounded. Could lie "outside" image. xF(i,j)
      %   should be read in channel chan(i,j), view iview(j).
      % yF: [NxF]. etc
      % chan: [NxF]. Channels in which to read features.
      % iview: [Fx1]. View indices in which to read features. Labels
      %   columns of xF, yF.
      % info: struct with miscellaneous/intermediate vars
      %
      % Note: xF and yF are in whatever units xLM and yLM are in (pixels)
            
      %F = height(tbl);
      l1 = tbl.lm1;
      l2 = tbl.lm2;
      %reff = tbl.reff;
      xeff = tbl.xeff;
      yeff = tbl.yeff;
      chan = tbl.chan;
      iview = tbl.view;
      
      [N,nPt,nView] = size(xLM);
      nFlat = nPt*nView;
      xLMFlat = reshape(xLM,N,nFlat); 
      yLMFlat = reshape(yLM,N,nFlat);
      iFlat1 = sub2ind([nPt nView],l1,iview); % [Fx1] "flat" indices into nPt*nView for feature lm 1's
      iFlat2 = sub2ind([nPt nView],l2,iview); % [Fx1] "flat" indices into nPt*nView for feature lm 2's
      x1 = xLMFlat(:,iFlat1); % [NxF]. x, landmark 1 x-coords for each instance/feature
      y1 = yLMFlat(:,iFlat1); % etc
      x2 = xLMFlat(:,iFlat2);
      y2 = yLMFlat(:,iFlat2);

      % See mathematical notes in generate2LMelliptical
      %   x = d*reff*cos(theta) = d*xeff
      %   y = d*reff/abRatio*sin(theta) = d*yeff
      dx2 = (x2-x1)/2; % [NxF]
      dy2 = (y2-y1)/2; % [NxF]
      d = nan;
      %d = sqrt(dx.^2+dy.^2)/2; % [NxF]. 2d=dist between lms
      %xEll = bsxfun(@times,d,xeff'); % [NxF]
      %yEll = bsxfun(@times,d,yeff'); % [NxF]
      % xEll and yEll are x- and y-coords of feature point, relative to
      % ellipse centered at origin and oriented along x-axis.
            
      xc = (x1+x2)/2; % [NxF]
      yc = (y1+y2)/2; % [NxF]
      %alpha = atan2(y2-y1,x2-x1); % [NxF] % OPTIM skip the trig just use dx, dy, d
      %cosa = cos(alpha); % [NxF]
      %sina = sin(alpha); % [NxF]
      alpha = nan; % no longer used/computed
      %cosa = dx./(2*d);
      %sina = dy./(2*d);
      xF = xc + xeff'.*dx2 - yeff'.*dy2;
      yF = yc + xeff'.*dy2 + yeff'.*dx2;
      %xF = xc + xEll.*cosa - yEll.*sina;
      %yF = yc + xEll.*sina + yEll.*cosa;
      xF = round(xF);
      yF = round(yF);
                
      % TODO: can prob optimize away repmat
      chan = repmat(chan',N,1);
      
      if nargout>=5
        info = struct();
        info.tbl = tbl;
        info.xLM1 = x1;
        info.yLM1 = y1;
        info.xLM2 = x2;
        info.yLM2 = y2;
        info.d = d;
        info.alpha = alpha;
        info.xc = xc;
        info.yc = yc;
        info.abratio = tbl.reff./tbl.yeff.*sin(tbl.theta);
      end
    end
    
    function [hPlot,str] = visualize2LM(ax,xF,yF,iView,info,iN,iF,clr,varargin)
      % Visualize feature pts from compute2LM
      % 
      % xf/yf/info: from compute2LM
      % iN: trial index
      % iF: feature index
      
      hPlot = myparse(varargin,...
        'hPlot',[]);
      
      assert(isequal(...
        size(xF),size(yF),...
        size(info.xLM1),size(info.yLM1),...
        size(info.xLM2),size(info.yLM2)));
      
      xf = xF(iN,iF);
      yf = yF(iN,iF);
      x1 = info.xLM1(iN,iF);
      x2 = info.xLM2(iN,iF);
      y1 = info.yLM1(iN,iF);
      y2 = info.yLM2(iN,iF);
      xc = info.xc(iN,iF);
      yc = info.yc(iN,iF);
      ivw = iView(iF);
      
      axplot = ax(ivw);
      
      if isequal(hPlot,[])
        hPlot = gobjects(5,1);
        hPlot(1) = plot(axplot,[x1;xc;x2],[y1;yc;y2],'-','Color',clr,'markerfacecolor',clr); %#ok<*AGROW>
        hPlot(2) = plot(axplot,xc,yc,'o','Color',clr,'markerfacecolor',clr);
        hPlot(3) = plot(axplot,nan,nan,'o','Color',[1 1 1],'markerfacecolor',[1 1 1]);        
        hPlot(4) = plot(axplot,xf,yf,'s','Color','w','MarkerSize',8,'markerfacecolor',[1 1 1]);
        hPlot(5) = ellipsedraw(info.a(iN,iF),info.b(iN,iF),xc,yc,...
          info.alpha(iN,iF),'-','plotArgs',{'parent',axplot,'Color',clr});
        hPlot(6) = plot(axplot,[xc;xf],[yc;yf],'-','Color',clr);
        [hPlot.LineWidth] = deal(1);
      else
        assert(numel(hPlot)==6);
        set(hPlot(1),'XData',[x1;xc;x2],'YData',[y1;yc;y2]);
        set(hPlot(2),'XData',[x1;xc;x2],'YData',[y1;yc;y2]);
        %set(hPlot(3),'XData',x1,'YData',y1);
        set(hPlot(4),'XData',xf,'YData',yf);
        ellipsedraw(info.a(iN,iF),info.b(iN,iF),xc,yc,info.alpha(iN,iF),...
          '-','hEllipse',hPlot(5),'plotArgs',{'parent',axplot});
        set(hPlot(6),'XData',[xc;xf],'YData',[yc;yf]);
      end
      str = sprintf('n=%d,f=%d(%d,%d). randctr=%.3f,rfac=%.3f,r=%.3f,theta=%.3f',iN,iF,...
        info.l1(iF),info.l2(iF),info.ctrFac(iF),info.rfac(iF),info.r(iN,iF),info.theta(iN,iF)/pi*180);
      title(axplot,str,'interpreter','none','fontweight','bold'); 
    end
    
    function [hPlot,str] = visualize2LMelliptical(ax,xF,yF,iView,info,iN,iF,clr,varargin)
      % Visualize feature pts from compute2LM
      % 
      % xf/yf/info: from compute2LM
      % iN: trial index
      % iF: feature index
      
      [hPlot,plotEllAtReff,doTitleStr,ellipseOnly] = myparse(varargin,...
        'hPlot',[],...
        'plotEllAtReff',false,... % If false (default), plot ellipse with (a,b)=(info.d,info.d/abratio). If true, plot ellipse at ultimate ftr loc.
        'doTitleStr',true,...
        'ellipseOnly',false...
        );
      
      assert(isequal(...
        size(xF),size(yF),...
        size(info.xLM1),size(info.yLM1),...
        size(info.xLM2),size(info.yLM2)));
      
      xf = xF(iN,iF);
      yf = yF(iN,iF);
      x1 = info.xLM1(iN,iF);
      x2 = info.xLM2(iN,iF);
      y1 = info.yLM1(iN,iF);
      y2 = info.yLM2(iN,iF);
      xc = info.xc(iN,iF);
      yc = info.yc(iN,iF);
      ivw = iView(iF);
      
      axplot = ax(ivw);
      
      if isequal(hPlot,[])
        hPlot = gobjects(6,1);
        aplot = info.d(iN,iF);          
        if plotEllAtReff
          aplot = aplot*info.tbl.reff(iF);
        end
        bplot = aplot/info.abratio(iF);
        hPlot(1) = ellipsedraw(aplot,bplot,xc,yc,info.alpha(iN,iF),'-',...
          'plotArgs',{'parent' axplot 'Color' clr});
        hPlot(2) = plot(axplot,[x1 x2],[y1 y2],'s','Color',clr,'linewidth',2);
        if ~ellipseOnly
          hPlot(3) = plot(axplot,[x1;xc;x2],[y1;yc;y2],'-','Color',clr,'markerfacecolor',clr); %#ok<*AGROW>
          hPlot(4) = plot(axplot,xc,yc,'o','Color',clr,'markerfacecolor',clr);
          hPlot(5) = plot(axplot,xf,yf,'s','Color','w','MarkerSize',8,'markerfacecolor',[1 1 1]);
          hPlot(6) = plot(axplot,[xc;xf],[yc;yf],'-','Color',clr);
          [hPlot.LineWidth] = deal(1);
        end
      else
        assert(numel(hPlot)==6);
        aplot = info.d(iN,iF);          
        if plotEllAtReff
          aplot = aplot*info.tbl.reff(iF);
        end
        bplot = aplot/info.abratio(iF);
        ellipsedraw(aplot,bplot,xc,yc,info.alpha(iN,iF),'-',...
          'hEllipse',hPlot(1),'plotArgs',{'parent' axplot});
        set(hPlot(2),'XData',[x1 x2],'YData',[y1 y2]);
        if ~ellipseOnly
          set(hPlot(3),'XData',[x1;xc;x2],'YData',[y1;yc;y2]);
          set(hPlot(4),'XData',xc,'YData',yc);
          set(hPlot(5),'XData',xf,'YData',yf);
          set(hPlot(6),'XData',[xc;xf],'YData',[yc;yf]);
        end
      end
      if doTitleStr
        str = sprintf('n=%d,f=%d(%d,%d). reff=%.3f,theta=%.3f',iN,iF,...
          info.tbl.lm1(iF),info.tbl.lm2(iF),info.tbl.reff(iF),info.tbl.theta(iF)/pi*180);
        title(axplot,str,'interpreter','none','fontweight','bold'); 
      end
    end
    
    function [xs,prms] = generate2LMDiff(model,varargin)
      % Generate 2-landmark features, diff
      % 
      % xs: F x 13 double array. xs(i,:) specifies the ith feature
      %   col 1: pt 1, landmark index 1
      %   col 2: pt 1, landmark index 2
      %   col 3: pt 1, radius FACTOR 
      %   col 4: pt 1, theta in (-pi.pi)
      %   col 5: pt 1, "interpolation" factor in [0,1] for location of center
      %          between landmarks 1/2
      %   col 6: pts 1 AND 2, channel index 
      %   col 7: pts 1 AND 2, view index
      %
      %   col 8-12: same as cols 1-5, but for pt 2.
      %   col 13: dummy col (unused)
      %
      % Note cols 6 and 7 apply to both points, ie differences are always
      % taken between features in the same channel and view.
      %
      % prms: scalar struct, params used to compute xs
            
      %%% Optional input P-V params: same as available for generate2LM.
      
      [xs1,prms1] = Features.generate2LM(model,varargin{:});
      [xs2,prms2] = Features.generate2LM(model,varargin{:});
      assert(isequal(prms1,prms2));
      assert(size(xs1,2)==7);
      assert(size(xs2,2)==7);
      
      % Use channel/view specification in xs1 for both pts; should be
      % randomly drawn from {1,2,...,nChan} and {1,2,...,nView} resp.
      F = size(xs1,1);
      xs = [xs1 xs2(:,1:5) nan(F,1)];
      prms = prms1;
    end
    
    function [xF1,yF1,xF2,yF2,chan,view,info] = compute2LMDiff(xs,xLM,yLM)
      % xs: [Fx13] from generate2LMDiff(). [Fx12] legacy accepted.
      % Rest: in analogy with compute2LM
      %
      % To compute jth feature for ith trial:
      %  ftrval = intensity-of-view(i,j)-chan(i,j)-at-(xF1(i,j),yF1(i,j)) - 
      %           intensity-of-view(i,j)-chan(i,j)-at-(xF2(i,j),yF2(i,j)) - 
      
      if size(xs,2)==12
        % legacy format. cols 1-6 for pt1 and 7-12 for pt2; cols 6 and 12        
        % match.
        assert(isequal(xs(:,6),xs(:,12)));
        F = size(xs,1);
        % convert to new format. assume all iview==1.
        xs = [xs(:,1:6) ones(F,1) xs(:,7:11) nan(F,1)];
      end
      xs1 = xs(:,1:7); % [Fx7] in format of generate2LM
      xs2 = [xs(:,8:12) xs(:,6:7)]; % [Fx7] etc
      [xF1,yF1,chan1,view1,info1] = Features.compute2LM(xs1,xLM,yLM);
      [xF2,yF2,chan2,view2,info2] = Features.compute2LM(xs2,xLM,yLM);
      assert(isequal(chan1,chan2));
      assert(isequal(view1,view2));
      chan = chan1;
      view = view1;
      info = struct('info1',info1,'info2',info2);
    end
    
    function [h,str] = visualize2LMDiff(ax,xF1,yF1,xF2,yF2,iView,info,iN,iF)
      [h1,str1] = Features.visualize2LM(ax,xF1,yF1,iView,info.info1,iN,iF,[1 0 0]);
      [h2,str2] = Features.visualize2LM(ax,xF2,yF2,iView,info.info2,iN,iF,[0 1 0]);
      h = [h1(:);h2(:)];
      str = [str1 '#' str2];
      title(ax(iView(iF)),str,'interpreter','none');
    end
    
  end
  
  %#3DOK
  methods (Static) % One-landmark features
    
    function [xs,prms] = generate1LM(model,varargin)
      % Generate 1-landmark features
      % 
      % xs: F x 4 double array. xs(i,:) specifies the ith feature
      %   col 1: pt/landmark index
      %   col 2: radius in pixels
      %   col 3: theta in (-pi,pi). This is angle relative to
      %     absolute/fixed frame of image
      %   col 4: view index
      %
      % prms: scalar struct, params used to compute xs
            
      %%% Optional input P-V params      
      % number of ftrs to generate
      defaultPrms.F = 20;
      % maximum pixel radius within which to sample
      defaultPrms.radius = 1;
      
      prms = getPrmDfltStruct(varargin,defaultPrms);      
      F = prms.F;
      d = model.d;
      nfids = model.nfids;
      nviews = model.nviews;
      assert(d==2 && nviews==1 || d==3 && nviews>1);
      
      xs = nan(F,4);
      xs(:,1) = randint2(F,1,[1,nfids]);
      xs(:,2) = prms.radius*rand(F,1);
      xs(:,3) = (2*pi*rand(F,1))-pi;
      xs(:,4) = ceil(rand(F,1)*nviews);
    end
    
    function [xs,prms] = generate1LMforSetParamViz(model,radius)
      % Generate a single 1LM feature with maximum radius
      [xs,prms] = Features.generate1LM(model,'F',1,'radius',radius);
      xs(:,2) = radius;
      xs(:,4) = 1;
    end
    
    function [xF,yF,iView,info] = compute1LM(xs,xLM,yLM)
      % xs: [Fx4] from generate1LM. Allow [Fx3] for legacy/historical with
      %     nView==1
      % xLM: [NxnPtxnView]. x-coords (column pixel indices) of landmark
      %   points
      % yLM: [NxnPtxnView]. y-coords
      %
      % xF: [NxF]. xF(iN,iF) is x-coord of iF'th feature for iN'th shape in
      %   image/view iView(iF)
      % yF: [NxF].
      % iView: [Fx1]. view indices labeling cols of xF/yF.
      % info: scalar struct.

      if size(xs,2)==3
        xs(:,4) = 1;
      end
      
      iPt = xs(:,1);
      r = xs(:,2);
      theta = xs(:,3);
      iView = xs(:,4);
      
      [N,nPt,nView] = size(xLM);
      nFlat = nPt*nView;
      xLMFlat = reshape(xLM,N,nFlat); 
      yLMFlat = reshape(yLM,N,nFlat);
      iFlat = sub2ind([nPt nView],iPt,iView);
      
      x = xLMFlat(:,iFlat); % N X F 
      y = yLMFlat(:,iFlat); % etc
      
      dx = r.*cos(theta); % [Fx1] angle relative to fixed frame of image, regardless of shape
      dy = r.*sin(theta); % [Fx1]
      xF = round(bsxfun(@plus,x,dx'));
      yF = round(bsxfun(@plus,y,dy'));
      
      %iView = repmat(iView',N,1);
      
      if nargout>=4
        info = struct();
        info.l1 = iPt;
        info.r = r;
        info.theta = theta;
        info.xLM = x;
        info.yLM = y;
      end
    end
    
    function [hPlot,str] = visualize1LM(ax,xF,yF,iView,info,iN,iF,clr,varargin)
      % xF, yF, iView, info: from compute1LM
      % iN: trial index (row index of xF/yF)
      % iF: feature index (col index of xF/yF)

      [hPlot,doTitleStr,ellipseOnly] = myparse(varargin,...
        'hPlot',[],...
        'doTitleStr',true,...
        'ellipseOnly',false);

      assert(isequal(size(xF),size(yF),size(info.xLM),size(info.yLM)));
      
      xf = xF(iN,iF);
      yf = yF(iN,iF);
      x1 = info.xLM(iN,iF);
      y1 = info.yLM(iN,iF);
      ivw = iView(iF);
      
      axplot = ax(ivw);
      
      if isequal(hPlot,[])
        hPlot = gobjects(4,1);
        hPlot(1) = ellipsedraw(info.r(iF),info.r(iF),x1,y1,0,'-',...
          'plotArgs',{'parent',axplot,'Color',clr});
        hPlot(2) = plot(axplot,x1,y1,'s','color',clr);
        if ~ellipseOnly
          hPlot(3) = plot(axplot,xf,yf,'s','color','w','markerfacecolor',[1 1 1]);
          hPlot(4) = plot(axplot,[x1;xf],[y1;yf],'-','Color',clr);
        end
      else
        assert(numel(hPlot)==4);
        ellipsedraw(info.r(iF),info.r(iF),x1,y1,0,'-','hEllipse',hPlot(1));
        set(hPlot(2),'XData',x1,'YData',y1);
        if ~ellipseOnly
          set(hPlot(3),'XData',xf,'YData',yf);
          set(hPlot(4),'XData',[x1;xf],'YData',[y1;yf]);
        end
      end
      if doTitleStr
        str = sprintf('n=%d,f=%d(%d). r=%.3f, theta=%.3f',iN,iF,info.l1(iF),...
          info.r(iF),info.theta(iF)/pi*180);
        title(axplot,str,'interpreter','none','fontweight','bold');
      end
    end
    
  end
    
end

function [mx,mn] = lclImMaxMin(Is)
% Is: cell array of images of same size

Ismax = cellfun(@(x)max(x(:)),Is);
Ismin = cellfun(@(x)min(x(:)),Is);
mx = max(Ismax(:));
mn = min(Ismin(:));
end
