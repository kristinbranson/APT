classdef GridViterbi < handle
  
  % Coordinate notes
  % 
  % Heatmaps are indexed by coords (p,q). Heatmaps are typically "big". The
  % heatmap location may move relative to the original/raw movie so that
  % hm(1,1) in general represents a different raw movie location in each 
  % frame.
  %
  % AC-windows are square crops from a Heatmap. AC-windows are indexed by
  % coords (u,v). AC windows are small enough that grid-viterbi can be
  % performed over them. They are hopefully large enough to capture all
  % significant regions of a heatmap. An AC window is typically centered at
  % a heatmap peak etc.
  
  properties
    maxvx
    dx
    dampen
    l2dist % [sz2 x sz2 x sz1 x sz1]. l2dist(p2,q2,p1,q1). See velocityMotionModel
    l2midij2
    l2midij1
    x2g
    y2g
    x1g
    y1g
  end
  
  methods 

    function obj = GridViterbi(maxvx,dx,dampen)
      [obj.l2dist,obj.l2midij2,obj.l2midij1,...
        obj.x2g,obj.y2g,obj.x1g,obj.y1g] = ...
        GridViterbi.velocityMotionModelL2(maxvx,dx,dampen);
      
      obj.maxvx = maxvx;
      obj.dx = dx;
      obj.dampen = dampen;
    end
    
    function mctL2 = getMotionCostL2(obj,ut,vt,acCtrPQ,acrad)
      % Compute/fetch motion cost for transitioning from
      % (u1,v1,t-2)->(u2,v2,t-1)->(ut,vt,t)
      %
      % ut: row index into acwin @ time t
      % vt: col "
      % acCtrPQ: [3x2]. locations of accwindows at times t-2:t.
      %   * acCtrPQ(1,:) gives (pctr,qctr) at time t-2, where pctr,qctr are
      %   (row,col) into heatmap at time t-2. 
      %   * acCtrPQ(2,:) gives (pctr,qctr) at time t-1 etc. Note, the
      %   heatmaps as a whole might have moved rel to each other, but our
      %   motion model is within the heatmap frame (motion considered is 
      %   relative to body/COM motion)
      %   * acCtrPQ(3,:) gives (pctr,qctr) at time t
      % acrad: radius of acwindow. acwindow is a square with side 2*acrad+1
      %
      % mct: [acsz x acsz x acsz x acsz]. mct(u2,v2,u1,v1) is l2
      % distance/motion cost for transitioning from 
      %   (u1,v1,t-2)->(u2,v2,t-1)->(ut,vt,t) where 
      %   (u1,v1),(u2,v2),(ut,vt) index the acwindows at t-2,t-1,t resp

      szassert(acCtrPQ,[3 2]);
      
      pt = acCtrPQ(3,1)-acrad+ut-1;
      qt = acCtrPQ(3,2)-acrad+vt-1;      
      ptminus1lohi = acCtrPQ(2,1)+[-acrad acrad];
      qtminus1lohi = acCtrPQ(2,2)+[-acrad acrad];
      ptminus2lohi = acCtrPQ(1,1)+[-acrad acrad];
      qtminus2lohi = acCtrPQ(1,2)+[-acrad acrad];          
      
      % Put everything in a heatmap/aligned coord system where (pt,qt) is 
      % at (x,y)=(0,0).
      ptminus1lohi = ptminus1lohi-pt;
      qtminus1lohi = qtminus1lohi-qt;
      ptminus2lohi = ptminus2lohi-pt;
      qtminus2lohi = qtminus2lohi-qt;
      
      itminus1lohi = ptminus1lohi+obj.l2midij2; % i,j: indices into .l2dist
      jtminus1lohi = qtminus1lohi+obj.l2midij2;
      itminus2lohi = ptminus2lohi+obj.l2midij1;
      jtminus2lohi = qtminus2lohi+obj.l2midij1;
     
      assert(isequal( obj.y2g(itminus1lohi,1),ptminus1lohi(:) ));
      assert(isequal( obj.y1g(itminus2lohi,1),ptminus2lohi(:) ));
      assert(isequal( obj.x2g(1,jtminus1lohi),qtminus1lohi ));
      assert(isequal( obj.x1g(1,jtminus2lohi),qtminus2lohi ));
      
      l2 = obj.l2dist;
      mctL2 = l2( itminus1lohi(1):itminus1lohi(2), ...
                        jtminus1lohi(1):jtminus1lohi(2), ...
                        itminus2lohi(1):itminus2lohi(2), ...
                        jtminus2lohi(1):jtminus2lohi(2) );
    end
    
  end
  
  methods (Static)
    
    function [l2dist,l2midij2,l2midij1,x2g,y2g,x1g,y1g] = ...
        velocityMotionModelL2(maxvx,dx,dampen)
      % Compute motion model cost, velocity+damping 
      %
      % l2dist: [sz2 x sz2 x sz1 x sz1]. l2dist(i2,j2,i1,j1) is L2 distance 
      %   from predicted loc (x,y) to (0,0) at time t, assuming object is:
      %     * at (row,col)=(i1,j1)~(x,y)=(x1g(i1,j1),y1g(i1,j1)) at t-2
      %     * at (row,vol)=(i2,j2)~(x,y)=(x2g(i2,j2),y2g(i2,j2)) at t-1
      %     * at (x,y)=(0,0) at t
      %     * The foregoing (x,y) etc assume aligned windows/coordinate
      %       systems across t-2,t-1,t, ie they are in HEATMAP COORDS. 
      %
      % l2midij2: x2g(l2midij2,l2midij2) and y2g(l2midij2,l2midij2) are 
      %   both 0. l2midij2 is the center of the x2g/y2g grids.
      % l2midij1: x1g(l2midij1,l2midij1) and y1g(l2midij1,l2midij1) are both 0. 
      %
      % x2g/y2g: [sz2 x sz2] grid with x2g(l2midij2,l2midij2), y2g(etc)
      %   = (0,0). x2g/y2g ranges over -maxvx:dx:maxvx.
      %
      % x1g/y1g: [sz1 x sz1] grid etc. x1g/y1g ranges over -2*maxvx:dx:2*maxvx.
      
      % for now
      assert(maxvx==round(maxvx));
      assert(dx==1);
      
      x2gv = -maxvx:dx:maxvx;
      y2gv = x2gv;
      [x2g,y2g] = meshgrid(x2gv,y2gv);
      sz2 = numel(x2gv);
      l2midij2 = maxvx+1;
      assert(isequal(0,x2g(l2midij2,l2midij2),y2g(l2midij2,l2midij2)));
      
      x1gv = -2*maxvx:dx:2*maxvx;
      y1gv = x1gv;
      [x1g,y1g] = meshgrid(x1gv,y1gv);
      sz1 = numel(x1gv);
      l2midij1 = 2*maxvx+1;
      assert(isequal(0,x1g(l2midij1,l2midij1),y1g(l2midij1,l2midij1)));
      
      dx = x2g-reshape(x1g,[1 1 sz1 sz1]);
      dy = y2g-reshape(y1g,[1 1 sz1 sz1]);
      szassert(dx,[sz2 sz2 sz1 sz1]);
      szassert(dy,[sz2 sz2 sz1 sz1]);
      
      xpred = x2g+dampen*dx;
      ypred = y2g+dampen*dy;      
      l2dist = xpred.^2+ypred.^2;
      szassert(l2dist,[sz2 sz2 sz1 sz1]);      
    end
    
    function [l2Big,mmcBigx1,mmcBigy1,mmcBigx2,mmcBigy2] = ...
        precompMMC(maxvx,dx,dampen)
      
      %%
      % precompute motion model cost. We imagine starting at (0,0). The 1st step
      % takes us to a maximum range of [-maxvx,maxvx] (2*maxvx+1 elements). The
      % 2nd step goes to a maximum range of [-2*maxvx,2*maxvx].
      mmcBigxgv1 = -maxvx:dx:maxvx;
      mmcBigygv1 = mmcBigxgv1;
      [mmcBigx1,mmcBigy1] = meshgrid(mmcBigxgv1,mmcBigygv1);
      mmcBigxgv2 = -2*maxvx:dx:2*maxvx;
      mmcBigygv2 = mmcBigxgv2;
      [mmcBigx2,mmcBigy2] = meshgrid(mmcBigxgv2,mmcBigygv2);
      
      mmcBigx2pred = (1+dampen)*mmcBigx1;
      mmcBigy2pred = (1+dampen)*mmcBigy1;
      
      nstep1 = numel(mmcBigxgv1);
      nstep2 = numel(mmcBigxgv2);
      % mmcBig(r1,c1,r2,c2) is the motion-model cost of starting at (0,0),
      % moving to (mmcBigx1(r1,c1),mmcBigy1(r1,c1), then moving to
      % (mmcBigx2(r2,c2),mmcBigy2(r2,c2)).
      l2Big = nan(nstep1,nstep1,nstep2,nstep2);
      for i1=1:nstep1
        for j1=1:nstep1
          l2Big(i1,j1,:,:) = sqrt( (mmcBigx2pred(i1,j1)-mmcBigx2).^2 + (mmcBigy2pred(i1,j1)-mmcBigy2).^2 );
        end
      end      
    end
    
    function [pctr,qctr,pidx,qidx] = hm2acwin(hm,hmconsidr)
      
      [hmnr,hmnc] = size(hm);
      [~,idx] = max(hm(:));
      [pctr,qctr] = ind2sub([hmnr hmnc],idx);
      
      pctrmin = hmconsidr+1;
      pctrmax = hmnr-hmconsidr;
      qctrmin = hmconsidr+1;
      qctrmax = hmnc-hmconsidr;
      
      pctr = min(max(pctr,pctrmin),pctrmax);
      qctr = min(max(qctr,qctrmin),qctrmax);
      
      pidx = pctr-hmconsidr:pctr+hmconsidr;
      qidx = qctr-hmconsidr:qctr+hmconsidr;
    end
    
    function [ac,pctr,qctr,pidx,qidx,x1orig,y1orig] = ...
                                          hm2ac(hm,hmconsidr,hm11xyorig)
      % hm: [hmnrxhmnc] heatmap
      % hmconsidr: radius of hm consider-window
      % hm11xyorig: [2] (x,y) in original/raw movie coords corresponding to
      %   hm(1,1)
      %
      % ac: [hmconsidsz x hmconsidsz] appcost
      % pctr,qctr: row,col indices into hm for center of ac
      % pidx,qidx: rol,col index vectors into hm for ac
      % x1orig,y1orig: (x,y) in original movie cooordinates for ac(1,1)
      
      [pctr,qctr,pidx,qidx] = GridViterbi.hm2acwin(hm,hmconsidr);
      ac = -log(hm(pidx,qidx));
      
      x1orig = hm11xyorig(1)+qidx(1)-1;
      y1orig = hm11xyorig(2)+pidx(1)-1;
    end
    
  end
  
end