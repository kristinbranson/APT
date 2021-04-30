classdef TrxUtil 
  
  methods (Static)
   
    function ptrx = newptrx(ntrx,npts)
      s = struct();
      s.id = 0;
      s.p = nan(npts,2,0);
      s.pocc = nan(npts,0);
      s.TS = nan(npts,0);
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
    function ptrx = ptrxAddXY(ptrx)
      % ptrx.x, .y are currently computed as the centroid of .p.
      % The .x, .y fields are added for visualization purposes. By using
      % these fieldnames we can reuse visualization code originally 
      % designed for Ctrax-style trx.

      ntrx = numel(ptrx);
      for i=1:ntrx
        xymu = nanmean(ptrx(i).p,1);
        ptrx(i).x = reshape(xymu(1,1,:),1,[]); 
        ptrx(i).y = reshape(xymu(1,2,:),1,[]);
      end
    end
    function ptrx = ptrxmerge(ptrx1,ptrx2)
      % merge/concat two ptrx's.
      % just a straight concat except for .id remapping.
      % 
      % .id's for ptrx 1 remain unchanged; ptrx2 will be reassigned as nec
      
      ids1 = [ptrx1.id]; % assume mutually unique
      ids2 = [ptrx2.id]; % assume mutually unique
      idmax = max(max(ids1),max(ids2));
      for i2=1:numel(ptrx2)
        if any(ptrx2(i2).id==ids1)
          idmax = idmax+1;
          ptrx2(i2).id = idmax;
        end
      end
      ptrx = cat(1,ptrx1(:),ptrx2(:));
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
    function [lpos,occ] = getLabelsFull(ptrx0,nfrmtot)
      % get full label/occ timeseries for scalar trx
      %
      % ptrx0: scalar ptrx
      % 
      % lpos: [npts x 2 x nfrmtot]
      % occ: [npts x nfrmtot] logical
      
      npts = size(ptrx0.p,1);
      lpos = nan(npts,2,nfrmtot);
      occ = false(npts,nfrmtot);
      
      ftrx = ptrx0.firstframe:ptrx0.endframe;
      lpos(:,:,ftrx) = ptrx0.p;
      occ(:,ftrx) = ptrx0.pocc;
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