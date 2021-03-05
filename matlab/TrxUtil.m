classdef TrxUtil 
  
  methods (Static)
   
    function ptrx = newptrx(ntrx,npts)
      s = struct();
      s.id = 0;
      s.p = nan(2*npts,0);
      s.pocc = nan(npts,0);
      s.x = nan(1,0);
      s.y = nan(1,0);
      s.theta = nan(1,0);
      s.firstframe = 1;
      s.off = 0;
      s.nframes = 0;
      s.endframe = 0;
      
      n = max(ntrx,1);
      for i = n:-1:1
        ptrx(i,1) = s;
        ptrx(i,1).id = i-1;
      end
      
      if ntrx==0
        ptrx = ptrx([],:);
      end
    end
    function ptrx = ptrxFromTracklet(t)
      % t: py-generated tracklet
      ntrx = numel(t.pTrk);
      if ntrx>0
        npts = size(t.pTrk{1},1);
      else
        npts = 0;
      end
      ptrx = TrxUtil.newptrx(ntrx,npts);
      
      assert(isequal(numel(t.endframes),...
                     numel(t.pTrkTS),...
                     numel(t.pTrkTag),...
                     numel(t.startframes),...
                     ntrx));
      for i=1:ntrx
        p = t.pTrk{i};
        nfrm = size(p,3);
        ptrx(i).p = reshape(p,npts*2,nfrm);
        ptrx(i).pocc = reshape(t.pTrkTag{i},npts,nfrm); 
        xymu = mean(p,1);
        ptrx(i).x = reshape(xymu(1,1,:),1,nfrm); 
        ptrx(i).y = reshape(xymu(1,2,:),1,nfrm);
        %ptrk(i).theta 
        f0 = t.startframes(i);
        f1 = t.endframes(i);
        assert(nfrm==f1-f0+1);
        ptrx(i).firstframe = f0;
        ptrx(i).off = 1-f0;
        ptrx(i).nframes = nfrm;
        ptrx(i).endframe = f1;
      end
    end
    function tblFT = tableFT(ptrx)
      % table with .frm, .ntgt
      frmmax = max([ptrx.endframe]);
      ntgt = zeros(frmmax,1);
      for i=1:numel(ptrx)
        f = ptrx(i).firstframe:ptrx(i).endframe;
        ntgt(f) = ntgt(f)+1;
      end
      
      tf = ntgt>0;
      frm = find(tf);
      ntgt = ntgt(tf);
      tblFT = table(frm,ntgt);
    end
      
    function trx = initStationary(trx,x0,y0,th0,frm0,frm1)
      % initialize all trxs to be stationary/fixed at position (x0,y0,th0)
      % for duration [frm0,frm1]. .firstframe will be set to frm0,
      % .endframe to frm1, etc.
      
      assert(isscalar(x0));
      assert(isscalar(y0));
      assert(isscalar(th0));
      nfrm = frm1-frm0+1;
      x0 = repmat(x0,1,nfrm);
      y0 = repmat(y0,1,nfrm);
      th0 = repmat(th0,1,nfrm);
      for i = 1:numel(trx)
        trx(i).x = x0;
        trx(i).y = y0;
        trx(i).theta = th0;
        trx(i).firstframe = frm0;
        trx(i).off = 1-frm0;
        trx(i).nframes = nfrm;
        trx(i).endframe = frm1;
      end
    end
    
  end
  
end