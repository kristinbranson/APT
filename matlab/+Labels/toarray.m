function [lpos,lposTS,lpostag] = toarray(s,varargin)
  % Convert to old-style full arrays
  %
  % lpos: [npt x 2 x nfrm x ntgt]
  % lposTS: [npt x nfrm x ntgt]
  % lpostag: [npt x nfrm x ntgt] logical

  [nfrm,ntgt] = myparse(varargin,...
    'nfrm',[],... % num frames in return arrays
    'ntgt',[] ... % num tgts "
    );
  if isempty(nfrm)
    nfrm = 1;
  end
  if ~isempty(s.frm),
    nfrm = max(nfrm,max(s.frm));
  end
  % KB 20201224: ntgt was not being set right
  if isempty(ntgt)
    ntgt = 1;
  end
  if ~isempty(s.tgt),
    ntgt = max(ntgt,max(s.tgt));
  end
  
  lpos = nan(s.npts,2,nfrm,ntgt);
  lposTS = -inf(s.npts,nfrm,ntgt);
  lpostag = false(s.npts,nfrm,ntgt);

  % KB 20201224 this loop was slow!
  idx = sub2ind([nfrm,ntgt],s.frm,s.tgt);
  lpos(:,:,idx) = reshape(s.p,[s.npts,2,numel(s.frm)]);
  lposTS(:,idx) = s.ts;
  lpostag(:,idx) = s.occ;
%       n = numel(s.frm);
%       for i=1:n
%         f = s.frm(i);
%         itgt = s.tgt(i);
%         lpos(:,:,f,itgt) = reshape(s.p(:,i),s.npts,2);
%         lposTS(:,f,itgt) = s.ts(:,i);
%         lpostag(:,f,itgt) = s.occ(:,i);
%       end
end  % function
