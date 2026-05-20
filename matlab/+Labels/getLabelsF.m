function [p,occ] = getLabelsF(s,frm,ntgtsmax)
  % prob rename to "getLabelsFFull" etc
  % get labels/occ for given frame, all tgts. "All tgts" here is
  % MA-style ie "all labeled tgts which often will be zero"
  %
  % p: [2npts x ntgtslbled], or [2npts x ntgtsmax] if ntgtsmax provided
  % occ: [npts x ntgtslbled], etc

  tf = s.frm==frm;
  itgts = s.tgt(tf);
  if max(itgts)==0
    itgts = ones(size(s.frm));
  end
  
  % for MA, itgts will be compaticified ie always equal to 1:max(itgts)
  % but possibly out of order. for now don't rely on compactness in 
  % this meth.
  
  if isempty(itgts)
    ntgts = 0;
  else
    ntgts = max(itgts);
  end
  
  if nargin>=3
    assert(ntgts<=ntgtsmax,'Too many targets found.');
    ntgtsreturn = ntgtsmax;
  else
    ntgtsreturn = ntgts;
  end
  
  p = nan(2*s.npts,ntgtsreturn);
  occ = zeros(s.npts,ntgtsreturn,Labels.CLS_OCC());
  p(:,itgts) = s.p(:,tf);
  occ(:,itgts) = s.occ(:,tf);
end  % function
