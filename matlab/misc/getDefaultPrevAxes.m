function paModeInfo = getDefaultPrevAxes(labeler, inputPAModeInfo)  
  paModeInfo = inputPAModeInfo ;
  borderfrac = .5;
  if ~labeler.hasMovie,
    return;
  end
  if ~isPrevAxesModeInfoValid(paModeInfo),
    return;
  end
  if ~isfield(paModeInfo,'isrotated'),
    paModeInfo.isrotated = false;
  end
  viewi = 1;
  ptidx = labeler.labeledposIPt2View == viewi;      
  [~,poscurr,~] = ...
    labeler.labelPosIsLabeled(paModeInfo.frm, ...
                              paModeInfo.iTgt, ...
                              'iMov',paModeInfo.iMov, ...
                              'gtmode',paModeInfo.gtmode) ;
  poscurr = poscurr(ptidx,:);
  if labeler.hasTrx,
    poscurr = [poscurr,ones(size(poscurr,1),1)]*paModeInfo.A;
    poscurr = poscurr(:,1:2);
  end
  
  minpos = min(poscurr,[],1);
  maxpos = max(poscurr,[],1);
  centerpos = (minpos+maxpos)/2;
  % border defined by borderfrac
  r = max(1,(maxpos-minpos)/2*(1+borderfrac));
  xlim = centerpos(1)+[-1,1]*r(1);
  ylim = centerpos(2)+[-1,1]*r(2);      
  
  [axw,axh] = labeler.GetPrevAxesSizeInPixels();
  axszratio = axw/axh;
  dx = diff(xlim);
  dy = diff(ylim);
  limratio = dx / dy;
  % need to extend 
  if axszratio > limratio,
    extendratio = axszratio/limratio;
    xlim = centerpos(1)+[-1,1]*r(1)*extendratio;
  elseif axszratio < limratio,
    extendratio = limratio/axszratio;
    ylim = centerpos(2)+[-1,1]*r(2)*extendratio;
  end
  if isfield(paModeInfo,'dxlim'),
    xlim0 = xlim;
    ylim0 = ylim;
    xlim = xlim + paModeInfo.dxlim;
    ylim = ylim + paModeInfo.dylim;
    % make sure all parts are visible
    if minpos(1) < xlim(1) || minpos(2) < ylim(1) || ...
        maxpos(1) > xlim(2) || maxpos(2) < ylim(2),
      paModeInfo.dxlim = [0,0];
      paModeInfo.dylim = [0,0];
      xlim = xlim0;
      ylim = ylim0;
      fprintf('Templates zoomed axes would not show all labeled points, using default axes.\n');
    end
  else
    paModeInfo.dxlim = [0,0];
    paModeInfo.dylim = [0,0];
  end
  xlim = fixLim(xlim);
  ylim = fixLim(ylim);
  paModeInfo.xlim = xlim;
  paModeInfo.ylim = ylim;
  
  paModeInfo.axes_curr = labeler.determinePrevAxesProperties(paModeInfo);  
end
