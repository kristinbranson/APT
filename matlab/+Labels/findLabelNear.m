function [tf,f0,p0] = findLabelNear(s,frm,itgt,fdir)
  % find labeled frame for itgt 'near' frm
  %
  % fdir: optional. one of +/-1, +/-2, [] (default) to search above, 
  % below, or in either direction relative to frm
  
  if nargin<4
    fdir = [];
  end
  if isempty(itgt),
    istgtmatch = true(size(s.frm));
  else
    istgtmatch = s.tgt==itgt;
  end
  if isequal(fdir,1)
    i = find(istgtmatch & s.frm>=frm);
  elseif isequal(fdir,-1)
    i = find(istgtmatch & s.frm<=frm,1,'last');
  elseif isequal(fdir,2)
    i = find(istgtmatch & s.frm>frm);
  elseif isequal(fdir,-2)
    i = find(istgtmatch & s.frm<frm,1,'last');
  else
    i = find(istgtmatch);
  end
  fs = s.frm(i);
  tf = ~isempty(fs);
  if tf
    d = abs(fs-frm);
    [~,j] = min(d); % j is argmin of d; index into i
    f0 = fs(j);
    p0 = s.p(:,i(j));
  else
    f0 = nan;
    p0 = nan(s.npts*2,1);
  end
end  % function
