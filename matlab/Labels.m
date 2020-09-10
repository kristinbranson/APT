classdef Labels
  properties (Constant)
    CLS_OCC = 'int8';
    CLS_MD = 'uint32';
  end
  methods (Static)
    function s = new(npts,n)
      if nargin<2
        n = 0;
      end
      s = struct();
      s.npts = npts;
      s.p = nan(npts*2,n); % pos
      s.ts = nan(npts,n); % ts -- time a bout was added
      s.occ = zeros(npts,n,Labels.CLS_OCC); % "tag"
      s.md = zeros(2,n,Labels.CLS_MD); % frm-tgt. anything else?
      
      % size(s.p,2) is the number of labeled rows.
      
    end
    function s = setpFT(s,frm,itgt,xy)
      i = find(s.md(1,:)==frm & s.md(2,:)==itgt);
      if isempty(i)
        % new label
        s.p(:,end+1) = xy(:);
        s.ts(:,end+1) = now;
        s.occ(:,end+1) = 0;
        s.md(:,end+1) = [frm; itgt];
      else
        % updating existing label
        s.p(:,i) = xy(:);
        s.ts(:,i) = now;
        % s.occ(:,i) unchanged
        % s.md(:,i) unchanged
      end
    end
    function s = setpFTI(s,frm,itgt,ipt,xy)
      i = find(s.md(1,:)==frm & s.md(2,:)==itgt);
      if isempty(i)
        % new label
        s.p(:,end+1) = nan;
        s.ts(:,end+1) = now;
        s.occ(:,end+1) = 0;
        s.md(:,end+1) = [frm; itgt];
        i = size(s.p,2);
      end
      s.p([ipt ipt+s.npts],i) = xy(:);
      s.ts(ipt,i) = now;
    end
    function s = setoccFTI(s,frm,itgt,ipt)
      % ipt can be vector
      s = Labels.setoccvalFTI(s,frm,itgt,ipt,1);
    end
    function s = clroccFTI(s,frm,itgt,ipt)
      % ipt can be vector
      % note, this will create a new label if nec      
      s = Labels.setoccvalFTI(s,frm,itgt,ipt,0);
    end
    function s = setoccvalFTI(s,frm,itgt,ipt,val)
      % creates a new label if nec
      % ipt can be vector
      i = find(s.md(1,:)==frm & s.md(2,:)==itgt);
      if isempty(i)
        % new label
        s.p(:,end+1) = nan;
        s.ts(:,end+1) = now;
        s.occ(:,end+1) = 0;
        s.md(:,end+1) = [frm; itgt];
        i = size(s.p,2);
      end
      s.occ(ipt,i) = val;
      s.ts(:,i) = now;
    end    
    function [s,tfchanged] = rmFT(s,frm,itgt)
      % remove labels for given frm/itgt
      
      i = find(s.md(1,:)==frm & s.md(2,:)==itgt);
      tfchanged = ~isempty(i); % found our (frm,tgt)
      if tfchanged
        s.p(:,i) = [];
        s.ts(:,i) = [];
        s.occ(:,i) = [];
        s.md(:,i) = [];
      end
    end
    function [s,tfchanged] = clearFTI(s,frm,itgt,ipt)
      i = find(s.md(1,:)==frm & s.md(2,:)==itgt);
      tfchanged = ~isempty(i); % found our (frm,tgt)
      if tfchanged
        s.p([ipt ipt+s.npts],i) = nan;
        s.ts(ipt,i) = now;
        s.occ(ipt,i) = 0;
      end
    end
    function [s,tfchanged,ntgts] = compact(s,frm)
      % Arbitrarily renames/remaps target indices for given frm to fall
      % into 1:ntgts. No consideration is given for continuity or
      % identification across frames.
      %
      % tfchanged: true if any change/edit was made to s
      % ntgts: number of tgts for frm
      
      tf = s.md(1,:)==frm;
      ntgts = nnz(tf);
      itgts = s.md(2,tf);
      tfchanged = ~isequal(sort(itgts),1:ntgts); 
      % order currently never matters in s.md
      if tfchanged
        s.md(2,tf) = 1:ntgts;
      end
    end
    function [tf,p,occ] = isLabeledFT(s,frm,itgt)
      i = find(s.md(1,:)==frm & s.md(2,:)==itgt,1);
      tf = ~isempty(i);
      if tf
        p = s.p(:,i);
        occ = s.occ(:,i);
      else
        p = nan(2*s.npts,1);
        occ = zeros(s.npts,1,Labels.CLS_OCC);
      end
    end
    function itgts = isLabeledF(s,frm)
      % Find labeled targets (if any) for frame frm
      %
      % itgts: [ntgtlbled] vec of targets that are labeled in frm
      
      i = find(s.md(1,:)==frm);
      itgts = unique(s.md(2,i));
    end
    function [p,occ] = getlabelsT(s,itgt,nf)
      % get labels/occ for given target.
      % nf: total number of frames for target/mov
      %
      % p: [2npts x nf]
      % occ: [npts x nf] logical
      
      p = nan(2*s.npts,nf);
      occ = false(s.npts,nf);      
      tf = s.md(2,:)==itgt;
      frms = s.md(1,tf);
      p(:,frms) = s.p(:,tf);
      occ(:,frms) = s.occ(:,tf);      
    end
    function [p,occ] = getLabelsF(s,frm,ntgtsmax)
      % get labels/occ for given frame, all tgts. "All tgts" here is
      % MA-style ie "all labeled tgts which often will be zero"
      %
      % p: [2npts x ntgtslbled], or [2npts x ntgtsmax] if ntgtsmax provided
      % occ: [npts x ntgtslbled], etc

      tf = s.md(1,:)==frm;
      itgts = s.md(2,tf);
      
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
      occ = zeros(s.npts,ntgtsreturn,Labels.CLS_OCC);
      p(:,itgts) = s.p(:,tf);
      occ(:,itgts) = s.occ(:,tf);
    end
    function tf = labeledFrames(s,nfrm)
      tf = false(nfrm,1);
      tf(s.md(1,:)) = true;
    end
    function [tf,f0,p0] = findLabelNear(s,frm,itgt)
      % find labeled frame for itgt 'near' frm
      i = find(s.md(2,:)==itgt);
      fs = s.md(1,i);
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
    end
    function t = totable(s,imov)
      frm = s.md(1,:)';
      iTgt = s.md(2,:)';
      p = s.p';
      pTS = s.ts';
      tfocc = s.occ';
      if exist('imov','var')==0
        t = table(frm,iTgt,p,pTS,tfocc);
      else
        mov = repmat(imov,numel(frm),1);
        t = table(mov,frm,iTgt,p,pTS,tfocc);
      end
    end
    function s = fromtable(t)
      if any(strcmp(t.Properties.VariableNames,'mov'))
        assert(all(t.mov==t.mov(1)));
        warningNoTrace('.mov column will be ignored.');        
      end
      
      n = height(t);
      npts = size(t.p,2)/2;
      s = Labels.new(npts);
      s.p = t.p.';
      s.md(1,1:n) = t.frm.';
      s.md(2,1:n) = t.iTgt.';
      s.ts = t.pTS.';
      s.occ(:,1:n) = t.tfocc.';
    end
    function s = fromarray(lpos,varargin)
      % s = fromarray(lpos,'lposTS',lposTS,'lpostag',lpostag)
      %
      % lpos: [npt x 2 x nfrm x ntrx]
      %
      % if lposTS not provided, ts will be 'nan' (default upon Labels.new),
      % and similarly for lpostag      

      [lposTS,lpostag] = myparse(varargin,...
        'lposTS',[],...
        'lpostag',[]...
        );
      
      [npts,d,nfrm,ntgt] = size(lpos);
      assert(d==2);
      
      tfTS = ~isequal(lposTS,[]);
      tfTag = ~isequal(lpostag,[]);
      if tfTS, szassert(lposTS,[npts nfrm ntgt]); end
      if tfTag, szassert(lpostag,[npts nfrm ntgt]); end
      
      nnan = ~isnan(lpos);
      nnanft = reshape(any(any(nnan,1),2),nfrm,ntgt);
      [frms,itgt] = find(nnanft);
      n = numel(frms);
      s = Labels.new(npts,n);
      for i=1:n
        fi = frms(i);
        itgti = itgt(i);
        s.p(:,i) = reshape(lpos(:,:,fi,itgti),2*npts,1);
        if tfTS
          s.ts(:,i) = lposTS(:,fi,itgti);
        end
        if tfTag
          s.occ(:,i) = lpostag(:,fi,itgti); % true->1, false->0
        end
      end      
      s.md(1,:) = frms;
      s.md(2,:) = itgt;
    end
  end
  
  methods (Static) % MA
    function ntgt = getNumTgts(s,frm)
      tf = s.md(1,:)==frm;
      ntgt = nnz(tf);  
    end
    function [s,itgt] = addTgtWithP(s,frm,xy)
      % itgt: new target index (also new number of tgts in frm)
      ntgt = Labels.getNumTgts(s,frm);
      itgt = ntgt+1;
      s.p(:,end+1) = xy(:);
      s.ts(:,end+1) = now;
      s.occ(:,end+1) = 0;
      s.md(:,end+1) = [frm; itgt];
    end
  end
end