classdef Features
  
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
      xs = nan(F,5);

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
    end
    
    function [xF,yF,info] = compute2LM(xs,xLM,yLM)
      % xs: F x 5, from generate2LM()
      % xLM: N x npts. poscs(i,j) gives x-positions ('columns') for ith
      % instance, jth landmark
      % yLM: etc
      %
      % xF: N X F. cs1(i,j) gives x-pos for ith instance, jth feature
      % yF: N X F.
      % info: struct with miscellaneous/intermediate vars
      
      l1 = xs(:,1);
      l2 = xs(:,2);
      rfac = xs(:,3);
      theta = xs(:,4);
      ctrFac = xs(:,5);
      
      N = size(xLM,1);
      x1 = xLM(:,l1); % N X F. x, landmark 1
      y1 = yLM(:,l1); % etc
      x2 = xLM(:,l2);
      y2 = yLM(:,l2);
      theta = repmat(theta',N,1);      
      
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
      
      if nargout>=3
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