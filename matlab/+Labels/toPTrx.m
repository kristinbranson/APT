function ptrx = toPTrx(s)
  tgtsUn = unique(s.tgt);
  ntgts = numel(tgtsUn);
  ptrx = TrxUtil.newptrx(ntgts,s.npts);
  
  % default x/y fcns (centroid)
  xfcn = @(p)nanmean(p(1:s.npts,:),1); %#ok<NANMEAN> 
  yfcn = @(p)nanmean(p(s.npts+1:2*s.npts,:),1); %#ok<NANMEAN> 
  
  for jtgt=1:ntgts
    iTgt = tgtsUn(jtgt);
    tf = s.tgt==iTgt;
    frms = double(s.frm(tf)); % KB 20201224 - doesn't work if uint32, off should be negative
    
    f0 = min(frms);
    f1 = max(frms);
    nf = f1-f0+1;
    off = 1-f0;
    
    p = nan(2*s.npts,nf);
    occ = false(s.npts,nf);
    p(:,frms+off) = s.p(:,tf); % f0->1, f1->nf
    occ(:,frms+off) = s.occ(:,tf);
    
    ptrx(jtgt).id = iTgt;
    ptrx(jtgt).p = p;
    ptrx(jtgt).pocc = occ;
    ptrx(jtgt).x = xfcn(p);
    ptrx(jtgt).y = yfcn(p);
    ptrx(jtgt).firstframe = f0;
    ptrx(jtgt).off = off;
    ptrx(jtgt).nframes = nf;
    ptrx(jtgt).endframe = f1;
  end
end  % function
