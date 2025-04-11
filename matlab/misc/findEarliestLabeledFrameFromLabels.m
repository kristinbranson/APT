function [tffound,iMov,frm,iTgt,xyLbl,mints] = findEarliestLabeledFrameFromLabels(labels)
  % Look only in labeledposGTaware, and look for the earliest labeled 
  % frame.
  
  tffound = false;
  mints = inf;
  for jmov = 1:numel(labels)
    s = labels{jmov};        
    [mintscurr,i] = min( min(s.ts,[],1) );
    if mintscurr < mints
      frm = s.frm(i);
      iTgt = s.tgt(i);
      p = s.p(:,i);
      iMov = jmov;
      mints = mintscurr;
      tffound = true;
    end
  end
  
  if tffound
    xyLbl = reshape(p,numel(p)/2,2);
  else
    iMov = [];
    frm = [];
    iTgt = [];
    xyLbl = [];
  end
end  % function
