classdef JanPostProc
  
  methods (Static)
    
    function [xyGT,xyTrk,md,xyErrRed,xyErrRep,dErrRed,dErrRep] = trackErr(res,td,iTst)
      % [md,xyErrRed,xyErrRep] = trackErr(res,td,iTst)
      %
      % Compute tracking error
      %
      % res: results structure
      % td: cprdata
      % iTst: [N] vector of iTrl indices into td corresponding to res
      %
      % xyGT: [Nxdx2]
      % xyTrk: [Nxdx2]
      % md: [Nxsomething] table of MD for tracked frames
      % xyErrRed: [Nxdx2] distance in pixels from res-reduced to GT for given frame, landmark, x/y coord
      % xyErrRep: [Nxdx2xRT] distance in pixels for res-replicate to GT "
      % dErrRed: [Nxd] L2 distance in pixels from res-reduced to GT
      % dErrRep: [NxdxRT] etc
      
      [xyRed,~,xy] = JanPostProc.getResults(res);
      Ntst = numel(iTst);
      assert(isequal(Ntst,size(xyRed,1),size(xy,1)));
      d = size(xyRed,2);
      
      pGT = td.pGT(iTst,:);
      xyGT = reshape(pGT,Ntst,d,2);
      xyTrk = xyRed;
      
      xyErrRed = xyGT-xyRed;
      xyErrRep = bsxfun(@minus,xyGT,xy);
      md = td.MD(iTst,:);
      
      dErrRed = sqrt(sum(xyErrRed.^2,3));
      dErrRep = squeeze(sqrt(sum(xyErrRep.^2,3)));
    end
    
    function [...
        xyRed,... % [Nxdx2]
        xyRed47,... % [Nx4x2]
        xy,... % [Nx7x2xRT]
        xy47,... % [Nx4x2xRT]
        djump47,... % [Nx4]. djump47(1,:) is all nan
        djump47av,... % [N]
        xyRepMad,... % [Nx7]. Mad-of-L2norm-of-rep-from-repcentroid
        xyRepMad47Av] ... % [N]
        = getResults(res)
      % Extract Jan results
      %
      % res: results structure, fields .pTstT, .pTstTRed
      
      [N,RT,D,Tp1] = size(res.pTstT);
      assert(isequal(size(res.pTstTRed),[N D Tp1]));
      d = D/2;
      
      pTstT = res.pTstT(:,:,:,end);
      pTstTRed = res.pTstTRed(:,:,end);
      xy = reshape(permute(pTstT,[1 3 2]),N,d,2,RT);
      xy47 = xy(:,4:7,:,:);
      xyRed = reshape(pTstTRed,N,d,2);
      xyRed47 = xyRed(:,4:7,:);
      
      djump47 = nan(N,4);
      for i = 2:N
        dxyRed = squeeze(xyRed47(i,:,:)-xyRed47(i-1,:,:)); % 4x2
        djump47(i,:) = sqrt(sum(dxyRed.^2,2));
      end
      djump47av = mean(djump47,2);
      
      % xy = [N 7 2 RT]
      xyCent = mean(xy,4); % [Nx7x2, replicate-centroid]
      xyDev = bsxfun(@minus,xy,xyCent); % [Nx7x2xRT], deviation of each iTrl/iRep from rep-centroid
      xyDev = squeeze(sqrt(sum(xyDev.^2,3))); % [Nx7XRT], L2norm of each pt/Rep from rep-centroid
      xyRepMad = median(xyDev,3); % [Nx7] % median-across-reps
      xyRepMad47Av = mean(xyRepMad(:,4:7),2); % [N]
    end
    
    function [dmin,iTrn] = minDistToShapeSet(p,pTrn)
      % Compute minimum distance from shapes to set of shapes, eg training set.
      %
      % p: [NxD] shapes
      % pTrn: [NTrnxD] training shapes
      %
      % dmin: [N] for each row/shape in p, compute minimum distance (L2,
      % averaged over landmarks 4:7) to pTrn
      % iTrn: [N] index into pTrn for closest training shape for each dmin
      
      n = size(p,1);
      nTrn = size(pTrn,1);
      assert(size(p,2)==size(pTrn,2));
      
      dmin = nan(n,1); % min distance to training set from each Test (Tracked) pt
      iTrn = nan(n,1); % argmin(dist-to-trning set) for each iTst
      warnst = warning('off','Shape:distP');
      for i = 1:n
        pCurr = p(i,:);
        d = Shape.distP(pTrn,repmat(pCurr,nTrn,1)); % [nTrnx7]
        d = mean(d(:,4:7),2);
        [dmin(i),iTrn(i)] = min(d);
                
        if mod(i,10)==0
          fprintf(1,'iTest=%d/%d\n',i,n);
        end
      end
      warning(warnst);
    end
    
    function vizTrkErrByLoc(I,xyGT,xyTrk,dErrRed,varargin)
      
      [iPt,markersize] = myparse(varargin,...
        'iPt',5,...
        'markersize',20);
      
      xyGTPt = squeeze(xyGT(:,iPt,:));
      xyTrkPt = squeeze(xyTrk(:,iPt,:));
      dErrRedPt = dErrRed(:,iPt);
      %n = size(xyGTPt,1);
      
      dErrRedPtPtiles = prctile(dErrRedPt,0:1:100);
      dErrRedPtPtiles(1) = 0.0;
      dErrRedPtBar = discretize(dErrRedPt,dErrRedPtPtiles);
      assert(~any(isnan(dErrRedPtBar)));
      
      %dErrMax = max(dErrRedPt);
      %fprintf('dErrMax = %.3g\n',dErrMax);
      %cmap = cool(ceil(dErrMax));
      %cmap = cool(max(dErrRedPtBar));

      % For Log xform 
      logxform_ticksRaw = [0.125 0.5 1 2 5 10 20 40 80];
      logxform_ticksRaw = logxform_ticksRaw(logxform_ticksRaw<max(dErrRedPt));
      logxform_tickLabels = arrayfun(@num2str,logxform_ticksRaw,'uni',0);
      logxform_ticks = log(logxform_ticksRaw);
      
      % For Log xform + +1
      logxform2_ticksRaw = [1 2 5 10 20 40 80];
      logxform2_ticksRaw = logxform2_ticksRaw(logxform2_ticksRaw<max(dErrRedPt));
      logxform2_tickLabels = arrayfun(@num2str,logxform2_ticksRaw,'uni',0);
      logxform2_ticks = log(logxform2_ticksRaw+1);
      
      % For Ptilexform
      ptile_ticksRaw = [0.25 0.5 1 2 20]; % in px
      ptile_ticksRaw = ptile_ticksRaw(ptile_ticksRaw<max(dErrRedPt));
      xx = sort(dErrRedPt);
      idxTicksRaw = arrayfun(@(x)argmin(abs(x-xx)),ptile_ticksRaw);
      ptile_ticks = idxTicksRaw/numel(dErrRedPt)*100; 
      ptile_ticklabels = arrayfun(@num2str,ptile_ticksRaw,'uni',0);
      [ptile_ticks,iTmp] = unique(ptile_ticks);
      ptile_ticklabels = ptile_ticklabels(iTmp);

      figure('windowstyle','docked','name','trkerrbyloc');
      args = {'fontweight','bold','interpreter','none','fontsize',12};

%       ax = subplot(2,4,1); 
%       pause(2);
%       JanPostProc.hlpScatter(ax,I,xyTrkPt(:,1),xyTrkPt(:,2),markersize,dErrRedPt);
%       title(ax,'Tracking error by location',args{:});
%       
%       ax = subplot(2,4,2);
%       pause(2);
%       hCB = JanPostProc.hlpScatter(ax,I,xyTrkPt(:,1),xyTrkPt(:,2),markersize,dErrRedPtBar);
%       title(ax,'Tracking error by location',args{:});
%       hCB.Ticks = ptile_ticks;
%       hCB.TickLabels = ptile_ticklabels;
%       
%       ax = subplot(2,4,3);
%       pause(2);
%       hCB = JanPostProc.hlpScatter(ax,I,xyTrkPt(:,1),xyTrkPt(:,2),markersize,log(dErrRedPt));
%       title(ax,'Tracking error by location',args{:});
%       hCB.Ticks = logxform_ticks;
%       hCB.TickLabels = logxform_tickLabels;
%       
%       ax = subplot(2,4,4);
%       pause(2);
%       hCB = JanPostProc.hlpScatter(ax,I,xyTrkPt(:,1),xyTrkPt(:,2),markersize,log(dErrRedPt+1));
%       title(ax,'Tracking error by location',args{:});
%       hCB.Ticks = logxform2_ticks;
%       hCB.TickLabels = logxform2_tickLabels;
% 
%       ax = subplot(2,4,5);
%       pause(2);
%       JanPostProc.hlpScatter(ax,I,xyGTPt(:,1),xyGTPt(:,2),markersize,dErrRedPt);
%       title(ax,'Tracking error by location',args{:});
% 
%       ax = subplot(2,4,6);
%       pause(2);
%       hCB = JanPostProc.hlpScatter(ax,I,xyGTPt(:,1),xyGTPt(:,2),markersize,dErrRedPtBar);        
%       title(ax,'Tracking error by location',args{:});
%       hCB.Ticks = ptile_ticks;
%       hCB.TickLabels = ptile_ticklabels;
%       
%       ax = subplot(2,4,7);
%       pause(2);
%       hCB = JanPostProc.hlpScatter(ax,I,xyGTPt(:,1),xyGTPt(:,2),markersize,log(dErrRedPt));
%       title(ax,'Tracking error by location',args{:});
%       hCB.Ticks = logxform_ticks;
%       hCB.TickLabels = logxform_tickLabels;
%       
%       ax = subplot(2,4,8);
      ax = axes;
      pause(2);
      hCB = JanPostProc.hlpScatter(ax,I,xyGTPt(:,1),xyGTPt(:,2),markersize,log(dErrRedPt+1));
      title(ax,'Tracking error by location',args{:});
      hCB.Ticks = logxform2_ticks;
      hCB.TickLabels = logxform2_tickLabels;
    end
    
    function hCB = hlpScatter(ax,I,x,y,markersz,clr)
      imagesc(I);
      colormap gray;
      hold on;
      axis image
      ax.XTick = [];
      ax.YTick = [];
      
      ax1 = axes('Parent',ax.Parent);
      axes(ax1);
      scatter(x,y,markersz,clr);
      ax1.Visible = 'off';
      colormap(ax1,'cool');      
      PROPS = {'XLim' 'YLim' 'YDir' 'Position' 'PlotBoxAspectRatio'};
      for p = PROPS,p=p{1};
        ax1.(p) = ax.(p);
      end
      
            
      hCB = colorbar(ax1,'Location','East','Box','off','Color',[0.8 0.8 0.8]);
      pos = hCB.Position;
      dx = pos(3)/2;
      pos(3) = pos(3)-dx;
      pos(1) = pos(1)+dx;
      hCB.Position = pos;
      %hCB.AxisLocation = 'out'
      
      linkaxes([ax ax1]);

      
    end
        
  end
  
end