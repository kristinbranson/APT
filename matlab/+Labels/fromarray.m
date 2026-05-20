function s = fromarray(lpos,varargin)
  % s = fromarray(lpos,'lposTS',lposTS,'lpostag',lpostag)
  %
  % lpos: [npt x 2 x nfrm x ntrx]
  %
  % if lposTS not provided, ts will be 'nan' (default upon Labels.new),
  % and similarly for lpostag      

  [lposTS,lpostag,frms,tgts] = myparse(varargin,...
    'lposTS',[],...
    'lpostag',[],...
    'frms',[], ... % optional, frame labels for 3rd dim
    'tgts',[] ... % optional, tgt labels for 4th dim
    );
  
  if isstruct(lpos)
    lpos = SparseLabelArray.full(lpos);
  end
  [npts,d,nfrm,ntgt] = size(lpos);
  assert(d==2);
  
  tfTS = ~isequal(lposTS,[]);
  tfTag = ~isequal(lpostag,[]);
  if tfTS, szassert(lposTS,[npts nfrm ntgt]); end
  if tfTag, szassert(lpostag,[npts nfrm ntgt]); end
  if isempty(frms)
    frms = 1:nfrm;
  else
    assert(numel(frms)==nfrm);
  end
  if isempty(tgts)
    tgts = 1:ntgt;
  else
    assert(numel(tgts)==ntgt);
  end
  
  nnan = ~isnan(lpos);
  nnanft = reshape(any(any(nnan,1),2),nfrm,ntgt);
  [ifrms,itgts] = find(nnanft);
  n = numel(ifrms);
  s = Labels.new(npts,n);
  for i=1:n
    fi = ifrms(i);
    itgti = itgts(i);
    s.p(:,i) = reshape(lpos(:,:,fi,itgti),2*npts,1);
    if tfTS
      s.ts(:,i) = lposTS(:,fi,itgti);
    end
    if tfTag
      s.occ(:,i) = lpostag(:,fi,itgti); % true->1, false->0
    end
  end      
  s.frm(:) = frms(ifrms(:));
  s.tgt(:) = tgts(itgts(:));
end  % function
