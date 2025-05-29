classdef Labels  
  % A class that exists only as a holder of a bunch of static methods
  % for dealing with labels.  -- ALT, 2024-11-08

  % Labels datastruct revisit 2020
  
  % # Narrow vs Wide. Standardize on Narrow.
  % 
  % A *row* has
  %  - (x,y) for each pt (under consid). or 'shape'
  %  - ts " (questionable utility but its there)
  %  - occ "
  % 
  % Currently we almost always consider rows as atomic in that a shape is 
  % either labeled entirely (all pts) or not. HT mode does not fit this but
  % HT mode is an edge case.
  % 
  % The Labels format is the 'narrow' fmt.
  % 
  % The TrkFile fmt is a 'wide' fmt assuming a cartesian F x T set of 
  % (frm,tgt) pairs. In the single-target case, this is the same as the
  % narrow fmt. TrkFiles were recently updated to enable a narrow fmt 'under
  % the hood' ie produced by DL; but this is converted to a wide fmt upon
  % TrkFile.load.
  %
  % The narrow fmt is more general and flexible so long term we should 
  % probably standardize to that where possible.
  % 
  % # Lbls vs Preds.
  % 
  % Training labels are sparse. Predictions tend to be 'full'. All targets 
  % are often present throughout a movie but i) this is not required and ii)
  % predictions may be done on a single or subset of targets.
  % 
  % HT labels are an edge case we do not treat as primary.
  % 
  % Manual correction of preds occurs when full preds are 
  % spot-checked/updated or "polished" manually. These corrections might tend 
  % to be sparse as well.

  properties (Constant)
    CLS_OCC = 'int8';
    CLS_MD = 'uint32';
    CLS_SPLIT = 'uint32';
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
      s.frm = zeros(n,1,Labels.CLS_MD);
      s.tgt = zeros(n,1,Labels.CLS_MD);
    end
    
    function s1 = remapLandmarks(s,new2oldpts)      
      oldnpts = s.npts;
      n = size(s.p,2);
      newnpts = numel(new2oldpts);
      isold = new2oldpts > 0;
      s1 = s;
      
      p = reshape(s.p,[oldnpts,2,n]);
      p1 = nan([newnpts,2,n],class(s.p));
      p1(isold,:,:) = p(new2oldpts(isold),:,:);
      s1.p = reshape(p1,[newnpts*2,n]);
      s1.ts = nan([newnpts,n],class(s.ts));
      s1.ts(isold,:) = s.ts(new2oldpts(isold),:);
      s1.occ = zeros([newnpts,n],class(s.occ));
      s1.occ(isold,:) = s.occ(new2oldpts(isold),:);
      s1.npts = newnpts;      
    end

    function tf = hasLbls(s)
      tf = ~isempty(s.frm);
    end

    function n = numLbls(s)
      n = numel(s.frm);
    end

    function s = setpFT(s,frm,itgt,xy)
      i = find(s.frm==frm & s.tgt==itgt);
      if isempty(i)
        % new label
        s.p(:,end+1) = xy(:);
        s.ts(:,end+1) = now;
        s.occ(:,end+1) = 0;
        s.frm(end+1,1) = frm;
        s.tgt(end+1,1) = itgt;
      else
        % updating existing label
        s.p(:,i) = xy(:);
        s.ts(:,i) = now;
        % s.occ(:,i) unchanged
        % s.frm(i) "
        % s.tgt(i) "
      end
    end

    function s = setpFTI(s,frm,itgt,ipt,xy)
      i = find(s.frm==frm & s.tgt==itgt);
      if isempty(i)
        % new label
        s.p(:,end+1) = nan;
        s.ts(:,end+1) = now;
        s.occ(:,end+1) = 0;
        s.frm(end+1,1) = frm;
        s.tgt(end+1,1) = itgt;
        i = size(s.p,2);
      end
      s.p([ipt ipt+s.npts],i) = xy(:);
      s.ts(ipt,i) = now;
    end

    function v = getFullyOccValue()
      % v = Labels.getFullyOccValue()
      % we use infinity to signify fully occluded
      % added by KB 20220202
      v = inf; 
    end

    function v = getUnlabeledValue()
      % v = Labels.getUnlabeledValue()
      % we use nan to signify that a point hasn't been labeled yet
      % added by KB 20220202
      v = nan; 
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
      % ipt: can be vector
      % val: can be scalar for scalar expansion or vec same size as ipt
      i = find(s.frm==frm & s.tgt==itgt);
      if isempty(i)
        % new label
        s.p(:,end+1) = nan;
        s.ts(:,end+1) = now;
        s.occ(:,end+1) = 0;
        s.frm(end+1,1) = frm;
        s.tgt(end+1,1) = itgt;
        i = size(s.p,2);
      end
      s.occ(ipt,i) = val;
      s.ts(:,i) = now;
    end    

    function [s,tfchanged] = rmFT(s,frm,itgt)
      % remove labels for given frm/itgt      
      i = find(s.frm==frm & s.tgt==itgt);
      tfchanged = ~isempty(i); % found our (frm,tgt)
      if tfchanged
        s.p(:,i) = [];
        s.ts(:,i) = [];
        s.occ(:,i) = [];
        s.frm(i,:) = [];
        s.tgt(i,:) = [];
      end
    end

    function [s,tfchanged] = rmFTP(s,frm,itgt,pts)
      % remove labels for given frm/itgt      
      i = find(s.frm==frm & s.tgt==itgt);
      tfchanged = ~isempty(i); % found our (frm,tgt)
      if tfchanged
        ptidx = false(size(s.occ,1),1);
        ptidx(pts) = true;
        s.p(repmat(ptidx,[2,1]),i) = Labels.getUnlabeledValue();
        s.ts(ptidx,i) = Labels.getUnlabeledValue();
        s.occ(ptidx,i) = false;
      end
    end
    
    function [s,tfchanged] = clearFTI(s,frm,itgt,ipt)
      i = find(s.frm==frm & s.tgt==itgt);
      tfchanged = ~isempty(i); % found our (frm,tgt)
      if tfchanged
        s.p([ipt ipt+s.npts],i) = nan;
        s.ts(ipt,i) = now;
        s.occ(ipt,i) = 0;
        % s.frm(ipt,1) and s.tgt(ipt,1) unchanged
      end
    end

    function [s,tfchanged,ntgts] = compact(s,frm)
      % Arbitrarily renames/remaps target indices for given frm to fall
      % into 1:ntgts. No consideration is given for continuity or
      % identification across frames.
      %
      % tfchanged: true if any change/edit was made to s
      % ntgts: number of tgts for frm
      
      tf = s.frm==frm;
      itgts = s.tgt(tf);
      ntgts = numel(itgts);
      tfchanged = ~isequal(sort(itgts),(1:ntgts)'); 
      % order currently never matters in s.frm, s.tgt
      if tfchanged
        s.tgt(tf) = (1:ntgts)';
      end
    end

    function [s,nfrmslbl,nfrmscompact] = compactall(s)
      frmsun = unique(s.frm);
      nfrmscompact = 0;
      for f=frmsun(:)'
        [s,tfchanged] = Labels.compact(s,f);
        nfrmscompact = nfrmscompact+tfchanged;
      end
      nfrmslbl = numel(frmsun);
    end

    function [tf,p,occ,ts] = isLabeledFT(s,frm,itgt)
      % Could get "getLabelsFT"
      %
      % p, occ, ts have appropriate size/vals even if tf==false
      
      i = find(s.frm==frm & s.tgt==itgt,1);
      tf = ~isempty(i);
      if tf
        p = s.p(:,i);
        occ = s.occ(:,i);
        ts = s.ts(:,i);
      else
        p = nan(2*s.npts,1);
        occ = zeros(s.npts,1,Labels.CLS_OCC);
        ts = -inf(s.npts,1);
      end
    end

    function [tf] = isLabelerPerPt(s)
      % [tf] = isLabelerPerPt(s)
      % Added by KB 20220206
      % tf(i,j) indicates whether landmark i is labeled for label j
      tf = permute(any(~isnan(reshape(s.p,[size(s.p,1)/2, 2, size(s.p,2)])),2),[1,3,2]);      
    end

    function [tf,p,occ,ts] = isLabeledPerPtFT(s,frm,itgt)
      % [tf,p,occ,ts] = isLabeledPerPtFT(s,frm,itgt)
      % Added by KB 20220202, similar to isLabeledFT
      % tf(i) indicates whether landmark i is labeled. 
      % p, occ, ts returned are the landmarks locations, occluded labels,
      % and timestamps
      i = find(s.frm==frm & s.tgt==itgt,1);
      tf = ~isempty(i);
      if tf
        p = s.p(:,i);
        tf = any(~isnan(reshape(p,[numel(p)/2 2])),2);
        occ = s.occ(:,i);
        ts = s.ts(:,i);
      else
        tf = false(s.npts,1);
        p = nan(2*s.npts,1);
        occ = zeros(s.npts,1,Labels.CLS_OCC);
        ts = -inf(s.npts,1);
      end
    end

    function itgts = isLabeledF(s,frm)
      % Find labeled targets (if any) for frame frm
      %
      % itgts: [ntgtlbled] vec of targets that are labeled in frm
      
      tf = s.frm==frm;
      itgts = s.tgt(tf);
    end

    function [tf,p,occ,ts] = isLabeledFMA(s,frm)
      % Could get "getLabelsFT"
      %
      % p, occ, ts have appropriate size/vals even if tf==false
      
      i = find(s.frm==frm);
      tf = ~isempty(i);
      if tf
        p = s.p(:,i);
        occ = s.occ(:,i);
        ts = s.ts(:,i);
      else
        p = nan(2*s.npts,1);
        occ = zeros(s.npts,1,Labels.CLS_OCC);
        ts = -inf(s.npts,1);
      end
    end

    function [frms,tgts] = isPartiallyLabeledT(s,itgt,nold)
      if isnan(itgt) || isempty(itgt),
        istgt = true(size(s.tgt));
      else
        istgt = s.tgt == itgt;
      end
      islabeled = Labels.isLabelerPerPt(s);
      ispartial = istgt' & all(islabeled(1:nold,:),1) & ~any(islabeled(nold+1:end,:),1);
      frms = s.frm(ispartial);
      tgts = s.tgt(ispartial);
    end

    function frms = isLabeledT(s,itgt)
      % Find labeled frames (if any) for target itgt
      %
      % Pass itgt==nan to mean "any target"
      %
      % frms: [nfrmslbled] vec of frames that are labeled for target itgt.
      %   Not guaranteed to be in any order
      
      if isnan(itgt)
        frms = unique(s.frm);
      else
        tf = s.tgt==itgt;
        frms = s.frm(tf);
      end
    end

    % function getLabelsFT -- see isLabeledFT
    function [p,occ] = getLabelsT_full(s,itgt,nf)
      % get labels/occ for given target.
      % nf: total number of frames for target/mov
      %
      % p: [2npts x nf]
      % occ: [npts x nf] logical
      
      p = nan(2*s.npts,nf);
      occ = false(s.npts,nf);
      tf = s.tgt==itgt;
      frms = s.frm(tf);
      p(:,frms) = s.p(:,tf);
      occ(:,frms) = s.occ(:,tf);
    end

    function [tfhasdata,p,occ,t0,t1] = getLabelsT(s,itgt)
      % get labels/occ for given target.
      %
      % p: [2npts x nf]. nf=t1-t0+1
      % occ: [npts x nf] logical
      % t0/t1: start/end frames (inclusive) labeling 2nd dims of p, occ.

      tf = s.tgt==itgt;
      frms = s.frm(tf);
      tfhasdata = ~isempty(frms);
      if tfhasdata
        t0 = min(frms);
        t1 = max(frms);
        nf = t1-t0+1;
      else
        t0 = nan;
        t1 = nan;
        nf = 0;
      end
      p = nan(2*s.npts,nf);
      occ = false(s.npts,nf);

      if tfhasdata
        idx = frms-t0+1;
        p(:,idx) = s.p(:,tf);
        occ(:,idx) = s.occ(:,tf);
      end
    end

    function [p,occ] = getLabelsF(s,frm,ntgtsmax)
      % prob rename to "getLabelsFFull" etc
      % get labels/occ for given frame, all tgts. "All tgts" here is
      % MA-style ie "all labeled tgts which often will be zero"
      %
      % p: [2npts x ntgtslbled], or [2npts x ntgtsmax] if ntgtsmax provided
      % occ: [npts x ntgtslbled], etc

      tf = s.frm==frm;
      itgts = s.tgt(tf);
      
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

    function iTgts = uniqueTgts(s)
      iTgts = unique(s.tgt);
    end

    function tf = labeledFrames(s,nfrm)
      tf = false(nfrm,1);
      tf(s.frm) = true;
      assert(numel(tf)==nfrm);
    end

    function tflbled = labeledTgts(s,nf)
      % nf: maximum number of frames
      %
      % tflbled: [nf itgtmax] tflbled(f,itgt) is true if itgt is labeled at f
      if isempty(s.tgt)
        itgtmax = 0;
      else
        itgtmax = max(s.tgt);
      end      
      tflbled = false(nf,itgtmax);
      idx = sub2ind([nf itgtmax],s.frm,s.tgt);
      tflbled(idx) = true;
      %ntgt = sum(tflbled,2);
    end

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
    end

    function t = totable(s,imov)
      frm = s.frm;
      iTgt = s.tgt;
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
      s = Labels.new(npts,n);      
      p = t.p.';
      ts = t.pTS.';
      occ = t.tfocc.';
      s.p(:) = p(:);
      s.ts(:) = ts(:);
      s.occ(:) = occ(:);
      s.frm(:) = t.frm;
      s.tgt(:) = t.iTgt;
    end

    function s = fromcoco(cocos,varargin)
      % s = fromcoco(cocos,...)
      % Create a Labels structure for one movie from input cocos struct.
      % If the coco structure contains information about which movies were
      % used to create labels, then this will create a Labels structure
      % from data corresponding only to movie imov. 
      % Input:
      % cocos: struct containing data read from COCO json file
      % Fields:
      % .images: struct array with an entry for each labeled image, with
      % the following fields:
      %   .id: Unique id for this labeled image, from 0 to
      %   numel(locs.locdata)-1
      %   .file_name: Relative path to file containing this image
      %   .movid: Id of the movie this image come from, 0-indexed. This is
      %   used iff movie information is available
      %   .frmid: Index of frame this image comes from, 0-indexed. This is
      %   used iff movie information is available
      % .annotations: struct array with an entry for each annotation, with
      % the following fields:
      %   .iscrowd: Whether this is a labeled target (0) or mask (1). If
      %   not available, it is assumed that this is a labeled target (0).
      %   .image_id: Index (0-indexed) of corresponding image
      %   .num_keypoints: Number of keypoints in this target (0 if mask)
      %   .keypoints: array of size nkeypoints*3 containing the x
      %   (keypoints(1:3:end)), y (keypoints(2:3:end)), and occlusion
      %   status (keypoints(3:3:end)). (x,y) are 0-indexed. for occlusion
      %   status, 2 means not occluded, 1 means occluded but labeled, 0
      %   means not labeled. 
      % .info:
      %   .movies: Cell containing paths to movies. If this is available,
      %   then these movies are added to the project. 
      % Optional inputs:
      % imov: If non-empty and movie information available, only
      % information corresponding to movie imov will be used to create this
      % structure. Default: []
      % tsnow: Time stamp to store in newly created labels. Default: now. 
      % Outputs:
      % s: Labels structure corresponding to input data. 
      [imov,tsnow] = myparse(varargin,'imov',[],'tsnow',now);
      s = [];
      if numel(cocos.annotations) == 0,
        return;
      end
      hasmovies = ~isempty(imov) && isfield(cocos,'info') && isfield(cocos.info,'movies');
      allnpts = [cocos.annotations.num_keypoints];
      npts = unique(allnpts(allnpts>0));
      assert(numel(npts) == 1,'All labels must have the same number of keypoints');
      if hasmovies,
        imidx = find([cocos.images.movid]==(imov-1))-1; % subtract 1 for 0-indexing
        ismov = ismember([cocos.annotations.image_id],imidx);
      else
        % assume we have created a single movie from ims, use all annotations
        ismov = true(1,numel(cocos.annotations));
      end
      if isfield(cocos.annotations,'iscrowd'),
        iskeypts = [cocos.annotations.iscrowd]==false;
        annidx = find(ismov & iskeypts);
      else
        annidx = find(ismov);
      end
      n = numel(annidx);
      if n == 0,
        return;
      end
      s = Labels.new(npts,n);
      s.ts(:) = tsnow;
      im2tgt = ones(1,numel(cocos.images));
      for i = 1:n,
        ann = cocos.annotations(annidx(i));
        px = ann.keypoints(1:3:end);
        py = ann.keypoints(2:3:end);
        s.p(:,i) = [px(:);py(:)]+1; % add 1 for 1-indexing
        s.occ(:,i) = 2-ann.keypoints(3:3:end);
        imid = ann.image_id; 
        imidxcurr = find([cocos.images.id]==imid);
        if hasmovies,
          s.frm(i) = cocos.images(imidxcurr).frm+1; % add 1 for 1-indexing
        else
          s.frm(i) = imid;
        end
        imid = ann.image_id+1; % add 1 for 1-indexing
        s.tgt(i) = im2tgt(imid);
        im2tgt(imid) = im2tgt(imid) + 1;
      end
      assert(~any(s.frm==0) && ~any(s.tgt==0));
    end

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
    end

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
    end

    function s = fromTrkfile(trk)
      if isfield(trk,'pTrkiPt')
        assert(isequal(trk.pTrkiPt(:)',1:size(trk.pTrk,1)),...
          'Unexpected point specification in .pTrkiPt.');
      end
      if trk.isfull
        s = Labels.fromarray(trk.pTrk,'lposTS',trk.pTrkTS,...
          'lpostag',trk.pTrkTag,'frms',trk.pTrkFrm,'tgts',trk.pTrkiTgt);
      else
        s = Labels.fromtable(trk.tableform('labelsColNames',true));
      end
    end

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
    end

    function s2 = indexPts(s,ipts)
      % "subsidx" given pts
      
      s2 = struct();
      ip = [ipts(:); ipts(:)+s.npts]; % xs and ys for these pts
      s2.p = s.p(ip,:);
      s2.ts = s.ts(ipts,:);
      s2.occ = s.occ(ipts,:);
      s2.frm = s.frm;
      s2.tgt = s.tgt;
    end
 
    function s = mergeviews(sarr)
      % sarr: array of Label structures

      if isscalar(sarr)
        s = sarr;
        return;
      end
      
      assert(isequal(sarr.npts),'npts must be equal across views.');
      assert(isequal(sarr.frm),'frames must be equal across views.');
      assert(isequal(sarr.tgt),'targets must be equal across views.');
      
      npts = sarr(1).npts;
      nview = numel(sarr);
      for i=1:nview
        sarr(i).p = reshape(sarr(i).p,npts,2,[]);
      end
      s = sarr(1);
      s.npts = npts*nview;
      s.p = cat(1,sarr.p);
      s.p = reshape(s.p,2*s.npts,[]);
      s.ts = cat(1,sarr.ts);
      s.occ = cat(1,sarr.occ);
      % .frm, .tgt unchanged
    end
    
    function s = rmRows(s,predicateFcn,~)
      % predicateFcn: eg @isnan, @isinf
      % rmDispStr: eg 'partially-labeled', 'fully-occluded' resp
      
      tf = any(predicateFcn(s.p),1);
      nrm = nnz(tf);
      if nrm>0
%        warningNoTrace('Labeler:nanData','Not including %d %s rows.',nrm,rmDispStr);
        s.p(:,tf) = [];
        s.ts(:,tf) = [];
        s.occ(:,tf) = [];
        s.frm(tf,:) = [];
        s.tgt(tf,:) = [];
        assert(~isfield(s,'split'));
      end
    end

    function s = replaceInfWithNan(s)      
      % Deal with full-occ rows in s in preparation from generating/writing 
      % TrnPack. infs are written as 'null' to json. match legacy SA
      % behavior by converting infs to nan. 
      
      tfinf = isinf(s.p);
      tfinfX = tfinf(1:s.npts,:);
      tfinfY = tfinf(s.npts+1:end,:);
      assert(isequal(tfinfX,tfinfY),'Label corruption: fully-occluded labels.');
      
%       tf1 = tfinf(1:s.npts,:) | tfinf(s.npts+1:end,:);
%       tf2 = s.occ>0;
%       tfInfWithoutOcc = tf1 & ~tf2;
%       % any point labeled as inf (fully-occ) should have .occ set to true 
%       assert(~any(tfInfWithoutOcc(:),'Label corruption'); 
      
      nfulloccpts = nnz(tfinfX);
      if nfulloccpts>0
        warningNoTrace('Utilizing %d fully-occluded landmarks.',nfulloccpts);
      end
      
      s.p(tfinf) = nan;
      s.occ(tfinfX) = 1;
    end
  end  % methods (Static)

  methods (Static)
    % Labeler-related utils
    function [lpos,lposTS,lpostag] = lObjGetLabeledPos(lObj,labelsfld,gt)
      [nfrms,ntgts] = lObj.getNFrmNTrx(gt);
      nfrms = num2cell(nfrms);
      ntgts = num2cell(ntgts);
      fcn = @(zs,znfrm,zntgt)Labels.toarray(zs,'nfrm',znfrm,'ntgt',zntgt);
      [lpos,lposTS,lpostag] = cellfun(fcn,lObj.(labelsfld),nfrms(:),ntgts(:),'uni',0);
    end

    function [tf] = lObjGetIsLabeled(lObj,labelsfld,tbl,gt)
      
      tf = false(height(tbl),1);
      for i = 1:numel(lObj.(labelsfld)),
        if gt,
          movi = -i;
        else
          movi = i;
        end
        idx = tbl.mov == movi;
        if ~any(idx),
          continue;
        end
        cc = Labels.CLS_MD;
        frs = eval(sprintf('%s([tbl.frm(idx),tbl.iTgt(idx)])',cc));
        [ism,j] = ismember(frs,[lObj.(labelsfld){i}.frm,lObj.(labelsfld){i}.tgt],'rows');
        idx = find(idx);
        idx = idx(ism);
        j = j(ism);
        tf(idx) = ~any(isnan(lObj.(labelsfld){i}.p(:,j)));
      end      
    end

    function n = lObjNLabeled(lObj,labelsfld,varargin)
      [movis,itgts,gt] = myparse(varargin,'movi',[],'itgt',[],'gt',[]);
      if isempty(movis),
        movis = 1:numel(lObj.(labelsfld));
        movis = reshape(movis,size(lObj.(labelsfld)));
      end
      if isempty(itgts),
        itgts = cell(size(movis));
      end
      if gt,
        movis = abs(movis);
      end
      n = cell(size(movis));
      for ii = 1:numel(movis),
        movi = movis(ii);
        if isempty(itgts{ii}),
          n{ii} = nnz(~all(isnan(lObj.(labelsfld){movi}.p),1));
        else
          n{ii} = zeros(size(itgts{ii}));
          for jj = 1:numel(itgts{ii}),
            itgt = itgts{ii}(jj);
            n{ii}(jj) = nnz(~all(isnan(lObj.(labelsfld){movi}.p(:,lObj.(labelsfld){movi}.tgt==itgt)),1));
          end
        end
      end
    end

    function verifyLObj(lObj)
      Labels.verifylObjHlp(lObj.labels,...
        lObj.labeledpos,lObj.labeledposTS,lObj.labeledpostag,'labels');
      Labels.verifylObjHlp(lObj.labelsGT,...
        lObj.labeledposGT,lObj.labeledposTSGT,lObj.labeledpostagGT,'labelsGT');
      Labels.verifylObjHlp(lObj.labels2,lObj.labeledpos2,[],[],'labels2');
      Labels.verifylObjHlp(lObj.labels2GT,lObj.labeledpos2GT,[],[],'labels2GT');      
    end

    function verifylObjHlp(labelsArr,lposArr,lposTSArr,lposTagArr,labelsField)
      for imov=1:numel(labelsArr)
        lpos0 = lposArr{imov};
        [npts,~,nfrm,ntgt] = size(lpos0);
        [lpos,lposTS,lpostag] = Labels.toarray(labelsArr{imov},...
          'nfrm',nfrm,'ntgt',ntgt);
        assert(isequaln(lpos,lpos0));
        if isempty(lposTSArr)
          lposTS0 = -inf(npts,nfrm,ntgt);
        else
          lposTS0 = lposTSArr{imov};
        end
        % TS may differ as lposTS0 can have entries where labels were
        % deleted
        d = lposTS-lposTS0;
        tfdiffer = abs(d)>1e-5 | isnan(d); % 1e-5 ~ 1s
        assert(isequal(tfdiffer,squeeze(all(isnan(lpos0),2))));
        %assert(isequaln(lposTS,lposTS0));
        if isempty(lposTagArr)
          lposTag0 = false(npts,nfrm,ntgt);
        else
          lposTag0 = lposTagArr{imov};
        end
        assert(isequaln(lpostag,lposTag0));
        fprintf(1,'Verified %s: %d\n',labelsField,imov);
      end    
    end

    function tblMF = labelAddLabelsMFTableStc(tblMF,lbls,varargin)
      % Add label/trx information to an MFTable
      %
      % tblMF (output): Same rows as tblMF, but with addnl label-related
      %   fields as in labelGetMFTableLabeledStc
      %
      % tblMF: MFTable with flds MFTable.FLDSID. tblMF.mov are 
      %   MovieIndices. tblMF.mov.get() are indices into lbls; ie lbls must
      %   be the appropriate cell-arr-of-labels to be indexed by tblMF.mov
      %
      % lbls: cell array of Labels structs. 
      
      [trxFilesAllFull,trxCache,wbObj,isma,maxanimals] = myparse(varargin,...
        'trxFilesAllFull',[],... % cellstr, indexed by tblMV.mov. if supplied, tblMF will contain .pTrx field
        'trxCache',[],... % must be supplied if trxFilesAllFull is supplied
        'wbObj',[],... % optional WaitBarWithCancel or ProgressMeter. If canceled, tblMF (output) indeterminate
        'isma',false, ...
        'maxanimals',1 ...
        );      
      tfWB = ~isempty(wbObj);
      
      assert(istable(tblMF));
      tblfldscontainsassert(tblMF,MFTable.FLDSID);
      nMov = numel(lbls);
      
      tfTrx = ~isempty(trxFilesAllFull);
      if tfTrx
        nView = size(trxFilesAllFull,2);
        szassert(trxFilesAllFull,[nMov nView]);
        tfTfafEmpty = cellfun(@isempty,trxFilesAllFull);
        % Currently, projects allowed to have some movs with trxfiles and
        % some without.
        assert(all( all(tfTfafEmpty,2) | all(~tfTfafEmpty,2) ),...
          'Unexpected trxFilesAllFull specification.');
        tfMovHasTrx = all(~tfTfafEmpty,2); % tfTfafMovEmpty(i) indicates whether movie i has trxfiles        
      else
        nView = 1;
      end
  
      nrow = height(tblMF);
      
      if tfWB && nrow>0 ,
        if isa(wbObj, 'ProgressMeter') ,
          wbObj.start('message', 'Compiling labels', ...
                      'denominator',nrow) ;
          oc = onCleanup(@()(wbObj.finish())) ;          
        else
          wbObj.startPeriod('Compiling labels','shownumden',true,...
                            'denominator',nrow);
          oc = onCleanup(@()wbObj.endPeriod);
        end
        wbtime = tic;
        maxwbtime = .1; % update waitbar every second
      end
      
      % Could also leverage Labels.totable and then do joins.
      
      npts = lbls{1}.npts;
      if isma
        pAcc = nan(nrow,maxanimals,npts*2);
        pTSAcc = -inf(nrow,maxanimals,npts);
        tfoccAcc = false(nrow,maxanimals,npts);
      else
        pAcc = nan(nrow,npts*2);
        pTSAcc = -inf(nrow,npts);
        tfoccAcc = false(nrow,npts);
      end
      pTrxAcc = nan(nrow,nView*2); % xv1 xv2 ... xvk yv1 yv2 ... yvk
      thetaTrxAcc = nan(nrow,nView);
      aTrxAcc = nan(nrow,nView);
      bTrxAcc = nan(nrow,nView);
      tfInvalid = false(nrow,1); % flags for invalid rows of tblMF encountered
      
      iMovsAll = tblMF.mov.get;
      frmsAll = tblMF.frm;
      iTgtAll = tblMF.iTgt;
      
      iMovsUnique = unique(iMovsAll);
      nRowsComplete = 0;
      
      for movIdx = 1:numel(iMovsUnique),
        iMov = iMovsUnique(movIdx);
        rowsCurr = find(iMovsAll == iMov); % absolute row indices into tblMF

        s = lbls{iMov};
               
        if tfTrx && tfMovHasTrx(iMov)
          NFRMS = [];
          % By passing [], either trxCache already contains this trxfile
          % and nfrms has been recorded for that mov/trx, or the maximum
          % trx.endFrame will be used.
          [trxI,~,frm2trxTotAnd] = Labeler.getTrxCacheAcrossViewsStc(...
            trxCache,trxFilesAllFull(iMov,:),NFRMS);
          
          assert(isscalar(trxI),'Multiview projs with trx currently unsupported.');
          trxI = trxI{1};
        end
        
        for jrow = 1:numel(rowsCurr),
          irow = rowsCurr(jrow); % absolute row index into tblMF
          
          if tfWB && nrow>0 && toc(wbtime) >= maxwbtime,
            wbtime = tic() ;
            if isa(wbObj, 'ProgressMeter') ,
              wbObj.bump(nRowsComplete) ;
              if wbObj.isCanceled ,
                return
              end
            else
              tfCancel = wbObj.updateFracWithNumDen(nRowsComplete);
              if tfCancel
                return
              end
            end
          end
          
          %tblrow = tblMF(irow,:);
          frm = frmsAll(irow);
          iTgt = iTgtAll(irow);
          
          if frm<1
            tfInvalid(irow) = true;
            continue;
          end
          
          if tfTrx && tfMovHasTrx(iMov)
            % will harderr if frm is out of bounds of frm2trxtotAnd
            tgtLiveInFrm = frm2trxTotAnd(frm,iTgt); 
            if ~tgtLiveInFrm
              tfInvalid(irow) = true;
              continue;
            end
          else
            %assert(iTgt==1);
          end
          
          if isma
            [~,p,occ,ts] = Labels.isLabeledFMA(s,frm);
            % p and occ have appropriate size/vals even if tf 
            % (first out arg) is false
            nl = size(p,2);
            pAcc(irow,1:nl,:) = p';
            pTSAcc(irow,1:nl,:) = ts';
            tfoccAcc(irow,1:nl,:) = occ'; 
          else
            [~,p,occ,ts] = Labels.isLabeledFT(s,frm,iTgt);
            % p and occ have appropriate size/vals even if tf 
            % (first out arg) is false
            pAcc(irow,:) = p;
            pTSAcc(irow,:) = ts;
            tfoccAcc(irow,:) = occ; 
          end
          
          if tfTrx && tfMovHasTrx(iMov)
            %xtrxs = cellfun(@(xx)xx(iTgt).x(frm+xx(iTgt).off),trxI);
            %ytrxs = cellfun(@(xx)xx(iTgt).y(frm+xx(iTgt).off),trxI);
            trxItgt = trxI(iTgt);
            frmabs = frm + trxItgt.off;
            xtrxs = trxItgt.x(frmabs);
            ytrxs = trxItgt.y(frmabs);
            
            pTrxAcc(irow,:) = [xtrxs(:)' ytrxs(:)']; 
            %thetas = cellfun(@(xx)xx(iTgt).theta(frm+xx(iTgt).off),trxI);
            thetas = trxItgt.theta(frmabs);
            thetaTrxAcc(irow,:) = thetas(:)'; 
            
%             as = cellfun(@(xx)xx(iTgt).a(frm+xx(iTgt).off),trxI);
%             bs = cellfun(@(xx)xx(iTgt).b(frm+xx(iTgt).off),trxI);
            as = trxItgt.a(frmabs);
            bs = trxItgt.b(frmabs);            
            aTrxAcc(irow,:) = as(:)'; 
            bTrxAcc(irow,:) = bs(:)'; 
          else
            % none; these arrays pre-initted to nan
            
%             pTrxAcc(irow,:) = nan; % singleton exp
%             thetaTrxAcc(irow,:) = nan; % singleton exp
%             aTrxAcc(irow,:) = nan; 
%             bTrxAcc(irow,:) = nan; 
          end
          nRowsComplete = nRowsComplete + 1;
        end
      end
      
      
      tLbl = table(pAcc,pTSAcc,tfoccAcc,pTrxAcc,thetaTrxAcc,aTrxAcc,bTrxAcc,...
        'VariableNames',{'p' 'pTS' 'tfocc' 'pTrx' 'thetaTrx' 'aTrx' 'bTrx'});
      tblMF = [tblMF tLbl];
      
      if any(tfInvalid)
        warningNoTrace('Removed %d invalid rows of MFTable.',nnz(tfInvalid));
        tblMF = tblMF(~tfInvalid,:);
      end      
    end

%     function tblMF = labelAddLabelsMFTableStc_Old(tblMF,lpos,lpostag,lposTS,...
%                                                   varargin)
%       % Add label/trx information to an MFTable
%       %
%       % tblMF (input): MFTable with flds MFTable.FLDSID. tblMF.mov are 
%       %   MovieIndices. tblMF.mov.get() are indices into lpos,lpostag,lposTS.
%       % lpos...lposTS: as in labelGetMFTableLabeledStc
%       %
%       % tblMF (output): Same rows as tblMF, but with addnl label-related
%       %   fields as in labelGetMFTableLabeledStc
%       
%       [trxFilesAllFull,trxCache,wbObj] = myparse(varargin,...
%         'trxFilesAllFull',[],... % cellstr, indexed by tblMV.mov. if supplied, tblMF will contain .pTrx field
%         'trxCache',[],... % must be supplied if trxFilesAllFull is supplied
%         'wbObj',[]... % optional WaitBarWithCancel. If cancel, tblMF (output) indeterminate
%         );      
%       tfWB = ~isempty(wbObj);
%       
%       assert(istable(tblMF));
%       tblfldscontainsassert(tblMF,MFTable.FLDSID);
%       nMov = size(lpos,1);
%       szassert(lpos,[nMov 1]);
%       szassert(lpostag,[nMov 1]);
%       szassert(lposTS,[nMov 1]);
%       
%       tfTrx = ~isempty(trxFilesAllFull);
%       if tfTrx
%         nView = size(trxFilesAllFull,2);
%         szassert(trxFilesAllFull,[nMov nView]);
%         tfTfafEmpty = cellfun(@isempty,trxFilesAllFull);
%         % Currently, projects allowed to have some movs with trxfiles and
%         % some without.
%         assert(all( all(tfTfafEmpty,2) | all(~tfTfafEmpty,2) ),...
%           'Unexpected trxFilesAllFull specification.');
%         tfMovHasTrx = all(~tfTfafEmpty,2); % tfTfafMovEmpty(i) indicates whether movie i has trxfiles        
%       else
%         nView = 1;
%       end
%   
%       nrow = height(tblMF);
%       
%       if tfWB
%         wbObj.startPeriod('Compiling labels','shownumden',true,...
%           'denominator',nrow);
%         oc = onCleanup(@()wbObj.endPeriod);
%         wbtime = tic;
%         maxwbtime = .1; % update waitbar every second
%       end
%       
%       % Maybe Optimize: group movies together
% 
%       npts = size(lpos{1},1);
%       
%       pAcc = nan(nrow,npts*2);
%       pTSAcc = -inf(nrow,npts);
%       tfoccAcc = false(nrow,npts);
%       pTrxAcc = nan(nrow,nView*2); % xv1 xv2 ... xvk yv1 yv2 ... yvk
%       thetaTrxAcc = nan(nrow,nView);
%       aTrxAcc = nan(nrow,nView);
%       bTrxAcc = nan(nrow,nView);
%       tfInvalid = false(nrow,1); % flags for invalid rows of tblMF encountered
%       iMovsAll = tblMF.mov.get;
%       frmsAll = tblMF.frm;
%       iTgtAll = tblMF.iTgt;
%       
%       iMovsUnique = unique(iMovsAll);
%       nRowsComplete = 0;
%       
%       for movIdx = 1:numel(iMovsUnique),
%         iMov = iMovsUnique(movIdx);
%         rowsCurr = find(iMovsAll == iMov); % absolute row indices into tblMF
%         
%         lposI = lpos{iMov};
%         lpostagI = lpostag{iMov};
%         lposTSI = lposTS{iMov};
%         [npts,d,nfrms,ntgts] = size(lposI);
%         assert(d==2);
%         szassert(lpostagI,[npts nfrms ntgts]);
%         szassert(lposTSI,[npts nfrms ntgts]);
%         
%         if tfTrx && tfMovHasTrx(iMov)
%           [trxI,~,frm2trxTotAnd] = Labeler.getTrxCacheAcrossViewsStc(...
%             trxCache,trxFilesAllFull(iMov,:),nfrms);
%           
%           assert(isscalar(trxI),'Multiview projs with trx currently unsupported.');
%           trxI = trxI{1};
%         end
%         
%         for jrow = 1:numel(rowsCurr),
%           irow = rowsCurr(jrow); % absolute row index into tblMF
%           
%           if tfWB && toc(wbtime) >= maxwbtime,
%             wbtime = tic;
%             tfCancel = wbObj.updateFracWithNumDen(nRowsComplete);
%             if tfCancel
%               return;
%             end
%           end
%           
%           %tblrow = tblMF(irow,:);
%           frm = frmsAll(irow);
%           iTgt = iTgtAll(irow);
%           
%           if frm<1 || frm>nfrms
%             tfInvalid(irow) = true;
%             continue;
%           end
%           
%           if tfTrx && tfMovHasTrx(iMov)
%             tgtLiveInFrm = frm2trxTotAnd(frm,iTgt);
%             if ~tgtLiveInFrm
%               tfInvalid(irow) = true;
%               continue;
%             end
%           else
%             assert(iTgt==1);
%           end
%           
%           lposIFrmTgt = lposI(:,:,frm,iTgt);
%           lpostagIFrmTgt = lpostagI(:,frm,iTgt);
%           lposTSIFrmTgt = lposTSI(:,frm,iTgt);
%           pAcc(irow,:) = lposIFrmTgt(:).'; % Shape.xy2vec(lposIFrmTgt);
%           pTSAcc(irow,:) = lposTSIFrmTgt'; 
%           tfoccAcc(irow,:) = lpostagIFrmTgt'; 
%           
%           if tfTrx && tfMovHasTrx(iMov)
%             %xtrxs = cellfun(@(xx)xx(iTgt).x(frm+xx(iTgt).off),trxI);
%             %ytrxs = cellfun(@(xx)xx(iTgt).y(frm+xx(iTgt).off),trxI);
%             trxItgt = trxI(iTgt);
%             frmabs = frm + trxItgt.off;
%             xtrxs = trxItgt.x(frmabs);
%             ytrxs = trxItgt.y(frmabs);
%             
%             pTrxAcc(irow,:) = [xtrxs(:)' ytrxs(:)']; 
%             %thetas = cellfun(@(xx)xx(iTgt).theta(frm+xx(iTgt).off),trxI);
%             thetas = trxItgt.theta(frmabs);
%             thetaTrxAcc(irow,:) = thetas(:)'; 
%             
% %             as = cellfun(@(xx)xx(iTgt).a(frm+xx(iTgt).off),trxI);
% %             bs = cellfun(@(xx)xx(iTgt).b(frm+xx(iTgt).off),trxI);
%             as = trxItgt.a(frmabs);
%             bs = trxItgt.b(frmabs);            
%             aTrxAcc(irow,:) = as(:)'; 
%             bTrxAcc(irow,:) = bs(:)'; 
%           else
%             % none; these arrays pre-initted to nan
%             
% %             pTrxAcc(irow,:) = nan; % singleton exp
% %             thetaTrxAcc(irow,:) = nan; % singleton exp
% %             aTrxAcc(irow,:) = nan; 
% %             bTrxAcc(irow,:) = nan; 
%           end
%           nRowsComplete = nRowsComplete + 1;
%         end
%       end
%       
%       tLbl = table(pAcc,pTSAcc,tfoccAcc,pTrxAcc,thetaTrxAcc,aTrxAcc,bTrxAcc,...
%         'VariableNames',{'p' 'pTS' 'tfocc' 'pTrx' 'thetaTrx' 'aTrx' 'bTrx'});
%       tblMF = [tblMF tLbl];
%       
%       if any(tfInvalid)
%         warningNoTrace('Removed %d invalid rows of MFTable.',nnz(tfInvalid));
%         tblMF = tblMF(~tfInvalid,:);
%       end       
%     end

    function tblMF = lblFileGetLabels(lblfile,varargin)
      % Get all labeled rows from a lblfile
      %
      % lblfile: either char/fullpath, or struct from loaded lblfile
      
      [quiet,gt] = myparse(varargin,...
        'quiet',false,...
        'gt',false ...
        );
      
      if quiet
        wbObj = [];
      else
        wbObj = WaitBarWithCancelCmdline('Reading labels');
      end
      
      if ischar(lblfile)
        lbl = loadLbl(lblfile);
      else
        lbl = lblfile;
      end
            
      if gt
        lpos = lbl.labelsGT;
      else
        lpos = lbl.labels;
      end
      tblcell = cell(size(lpos));
      nmov = numel(lpos);
      for imov=1:nmov
        tblcell{imov} = Labels.totable(lpos{imov},MovieIndex(imov,gt));
      end
      tblMF = cat(1,tblcell{:});

      % looks quite dumb, throw away label info; add it back in later
      tblMF = tblMF(:,MFTable.FLDSID);
            
      sMacro = lbl.projMacros;
      if gt
        mfa = lbl.movieFilesAllGT;
        tfa = lbl.trxFilesAllGT;
      else
        mfa = lbl.movieFilesAll;
        tfa = lbl.trxFilesAll;
      end
      mfafull = FSPath.fullyLocalizeStandardize(mfa,sMacro);
      tfafull = Labeler.trxFilesLocalize(tfa,mfafull);
            
      tblMF = Labels.labelAddLabelsMFTableStc(tblMF,lpos, ...
        'trxFilesAllFull',tfafull,...
        'trxCache',containers.Map(),...
        'wbObj',wbObj);      
    end

%     function tblMF = lblFileGetLabels_Old(lblfile,varargin)
%       % Get all labeled rows from a lblfile
%       %
%       % lblfile: either char/fullpath, or struct from loaded lblfile
%       
%       [quiet,gt] = myparse(varargin,...
%         'quiet',false,...
%         'gt',false ...
%         );
%       
%       if quiet
%         wbObj = [];
%       else
%         wbObj = WaitBarWithCancelCmdline('Reading labels');
%       end
%       
%       if ischar(lblfile)
%         lbl = loadLbl(lblfile);
%       else
%         lbl = lblfile;
%       end
%       if gt
%         lpos = lbl.labeledposGT;
%         lpostag = lbl.labeledpostagGT;
%         lposts = lbl.labeledposTSGT;
%       else
%         lpos = lbl.labeledpos;
%         lpostag = lbl.labeledpostag;
%         lposts = lbl.labeledposTS;
%       end
%       
%       tblMF = [];
%       nmov = numel(lpos);
%       for imov=1:nmov
%         lp = lpos{imov};
%         [~,~,frm,iTgt] = ind2sub(lp.size,lp.idx);
%         tblI = table(frm,iTgt);
%         tblI = unique(tblI);
%         tblI.mov = MovieIndex(repmat(imov,height(tblI),1));
%         
%         tblMF = [tblMF; tblI]; %#ok<AGROW>
%       end
%       
%       tblMF = tblMF(:,MFTable.FLDSID);
%       
%       lposfull = cellfun(@SparseLabelArray.full,lpos,'uni',0);
%       lpostagfull = cellfun(@SparseLabelArray.full,lpostag,'uni',0);
%       lpostsfull = cellfun(@SparseLabelArray.full,lposts,'uni',0);
%       
%       sMacro = lbl.projMacros;
%       if gt
%         mfa = lbl.movieFilesAllGT;
%         tfa = lbl.trxFilesAllGT;
%       else
%         mfa = lbl.movieFilesAll;
%         tfa = lbl.trxFilesAll;
%       end
%       mfafull = FSPath.fullyLocalizeStandardize(mfa,sMacro);
%       tfafull = Labeler.trxFilesLocalize(tfa,mfafull);
%             
%       tblMF = Labels.labelAddLabelsMFTableStc_Old(tblMF,...
%         lposfull,lpostagfull,lpostsfull,...
%         'trxFilesAllFull',tfafull,...
%         'trxCache',containers.Map(),...
%         'wbObj',wbObj);      
%     end
    
    function [tffound,f] = seekBigLpos(lpos,f0,df,iTgt)
      % lpos: [npts x d x nfrm x ntgt]
      % f0: starting frame
      % df: frame increment
      % iTgt: target of interest
      % 
      % tffound: logical
      % f: first frame encountered with (non-nan) label, applicable if
      %   tffound==true
      
      if isempty(lpos),
        tffound = false;
        f = f0;
        return;
      end
      
      [npts,d,nfrm,ntgt] = size(lpos); %#ok<ASGLU>
      assert(d==2);
      
      f = f0+df;
      while 0<f && f<=nfrm
        for ipt = 1:npts
          %for j = 1:2
          if ~isnan(lpos(ipt,1,f,iTgt))
            tffound = true;
            return;
          end
          %end
        end
        f = f+df;
      end
      tffound = false;
      f = nan;
    end

    function [tffound,f] = seekSmallLpos(lpos,f0,df)
      % lpos: [npts x nfrm]
      % f0: starting frame
      % df: frame increment
      % 
      % tffound: logical
      % f: first frame encountered with (non-nan) label, applicable if
      %   tffound==true
      
      [npts,nfrm] = size(lpos);
      
      f = f0+df;
      while 0<f && f<=nfrm
        for ipt=1:npts
          if ~isnan(lpos(ipt,f))
            tffound = true;
            return;
          end
        end
        f = f+df;
      end
      tffound = false;
      f = nan;
    end

    function [tffound,f] = seekSmallLposThresh(lpos,f0,df,th,cmp)
      % lpos: [npts x nfrm]
      % f0: starting frame
      % df: frame increment
      % th: threshold
      % cmp: comparitor
      % 
      % tffound: logical
      % f: first frame encountered with (non-nan) label that satisfies 
      % comparison with threshold, applicable if tffound==true
      
      switch cmp
        case '<',  cmp = @lt;
        case '<=', cmp = @le;
        case '>',  cmp = @gt;
        case '>=', cmp = @ge;
      end
          
      [npts,nfrm] = size(lpos);
      
      f = f0+df;
      while 0<f && f<=nfrm
        for ipt=1:npts
          if cmp(lpos(ipt,f),th)
            tffound = true;
            return;
          end
        end
        f = f+df;
      end
      tffound = false;
      f = nan;
    end
  
%     % Legacy meth. labelGetMFTableLabeledStc is new method but assumes
%     % .hasTrx
%     %#3DOK
%     function [I,tbl] = lblCompileContentsRaw(...
%         movieNames,lposes,lpostags,iMovs,frms,varargin)
%       % Read moviefiles with landmark labels
%       %
%       % movieNames: [NxnView] cellstr of movienames
%       % lposes: [N] cell array of labeledpos arrays [npts x 2 x nfrms x ntgts]. 
%       %   For multiview, npts=nView*NumLabelPoints.
%       % lpostags: [N] cell array of labeledpostags [npts x nfrms x ntgts]
%       % iMovs. [M] (row) indices into movieNames to read.
%       % frms. [M] cell array. frms{i} is a vector of frames to read for
%       % movie iMovs(i). frms{i} may also be:
%       %     * 'all' indicating "all frames" 
%       %     * 'lbl' indicating "all labeled frames" (currently includes partially-labeled)
%       %
%       % I: [NtrlxnView] cell vec of images
%       % tbl: [NTrl rows] labels/metadata MFTable.
%       %   MULTIVIEW NOTE: tbl.p is the 2d/projected label positions, ie
%       %   each shape has nLabelPoints*nView*2 coords, raster order is 1. pt
%       %   index, 2. view index, 3. coord index (x vs y)
%       %
%       % Optional PVs:
%       % - hWaitBar. Waitbar object
%       % - noImg. logical scalar default false. If true, all elements of I
%       % will be empty.
%       % - lposTS. [N] cell array of labeledposTS arrays [nptsxnfrms]
%       % - movieNamesID. [NxnView] Like movieNames (input arg). Use these
%       % names in tbl instead of movieNames. The point is that movieNames
%       % may be macro-replaced, platformized, etc; otoh in the MD table we
%       % might want macros unreplaced, a standard format etc.
%       % - tblMovArray. Scalar logical, defaults to false. Only relevant for
%       % multiview data. If true, use array of movies in tbl.mov. Otherwise, 
%       % use single compactified string ID.
%       
%       [hWB,noImg,lposTS,movieNamesID,tblMovArray] = myparse(varargin,...
%         'hWaitBar',[],...
%         'noImg',false,...
%         'lposTS',[],...
%         'movieNamesID',[],...
%         'tblMovArray',false);
%       assert(numel(iMovs)==numel(frms));
%       for i = 1:numel(frms)
%         val = frms{i};
%         assert(isnumeric(val) && isvector(val) || ismember(val,{'all' 'lbl'}));
%       end
%       
%       tfWB = ~isempty(hWB);
%       
%       assert(iscellstr(movieNames));
%       [N,nView] = size(movieNames);
%       assert(iscell(lposes) && iscell(lpostags));
%       assert(isequal(N,numel(lposes),numel(lpostags)));
%       tfLposTS = ~isempty(lposTS);
%       if tfLposTS
%         assert(numel(lposTS)==N);
%       end
%       for i=1:N
%         assert(size(lposes{i},1)==size(lpostags{i},1) && ...
%                size(lposes{i},3)==size(lpostags{i},2));
%         if tfLposTS
%           assert(isequal(size(lposTS{i}),size(lpostags{i})));
%         end
%       end
%       
%       if ~isempty(movieNamesID)
%         assert(iscellstr(movieNamesID));
%         szassert(movieNamesID,size(movieNames)); 
%       else
%         movieNamesID = movieNames;
%       end
%       
%       for iVw=nView:-1:1
%         mr(iVw) = MovieReader();
%       end
% 
%       I = [];
%       % Here, for multiview, mov are for the first movie in each set
%       s = struct('mov',cell(0,1),'frm',[],'p',[],'tfocc',[]);
%       
%       nMov = numel(iMovs);
%       fprintf('Reading %d movies.\n',nMov);
%       if nView>1
%         fprintf('nView=%d.\n',nView);
%       end
%       for i = 1:nMov
%         iMovSet = iMovs(i);
%         lpos = lposes{iMovSet}; % npts x 2 x nframes
%         lpostag = lpostags{iMovSet};
% 
%         [npts,d,nFrmAll] = size(lpos);
%         assert(d==2);
%         if isempty(lpos)
%           assert(isempty(lpostag));
%           lpostag = cell(npts,nFrmAll); % edge case: when lpos/lpostag are [], uninitted/degenerate case
%         end
%         szassert(lpostag,[npts nFrmAll]);
%         D = d*npts;
%         % Ordering of d is: {x1,x2,x3,...xN,y1,..yN} which for multiview is
%         % {xp1v1,xp2v1,...xpnv1,xp1v2,...xpnvk,yp1v1,...}. In other words,
%         % in decreasing raster order we have 1. pt index, 2. view index, 3.
%         % coord index (x vs y)
%         
%         for iVw=1:nView
%           movfull = movieNames{iMovSet,iVw};
%           mr(iVw).open(movfull);
%         end
%         
%         movID = MFTable.formMultiMovieID(movieNamesID(iMovSet,:));
%         
%         % find labeled/tagged frames (considering ALL frames for this
%         % movie)
%         tfLbled = arrayfun(@(x)nnz(~isnan(lpos(:,:,x)))>0,(1:nFrmAll)');
%         frmsLbled = find(tfLbled);
%         tftagged = ~cellfun(@isempty,lpostag); % [nptxnfrm]
%         ntagged = sum(tftagged,1);
%         frmsTagged = find(ntagged);
%         assert(all(ismember(frmsTagged,frmsLbled)));
% 
%         frms2Read = frms{i};
%         if strcmp(frms2Read,'all')
%           frms2Read = 1:nFrmAll;
%         elseif strcmp(frms2Read,'lbl')
%           frms2Read = frmsLbled;
%         end
%         nFrmRead = numel(frms2Read);
%         
%         ITmp = cell(nFrmRead,nView);
%         fprintf('  mov(set) %d, D=%d, reading %d frames\n',iMovSet,D,nFrmRead);
%         
%         if tfWB
%           hWB.Name = 'Reading movies';
%           wbStr = sprintf('Reading movie %s',movID);
%           waitbar(0,hWB,wbStr);
%         end
%         for iFrm = 1:nFrmRead
%           if tfWB
%             waitbar(iFrm/nFrmRead,hWB);
%           end
%           
%           f = frms2Read(iFrm);
% 
%           if noImg
%             % none; ITmp(iFrm,:) will have [] els
%           else
%             for iVw=1:nView
%               im = mr(iVw).readframe(f);
%               if size(im,3)==3 && isequal(im(:,:,1),im(:,:,2),im(:,:,3))
%                 im = rgb2gray(im);
%               end
%               ITmp{iFrm,iVw} = im;
%             end
%           end
%           
%           lblsFrmXY = lpos(:,:,f);
%           tags = lpostag(:,f);
%           
%           if tblMovArray
%             assert(false,'Unsupported codepath');
%             %s(end+1,1).mov = movieNamesID(iMovSet,:); %#ok<AGROW>
%           else
%             s(end+1,1).mov = iMovSet; %#ok<AGROW>
%           end
%           %s(end).movS = movS1;
%           s(end).frm = f;
%           s(end).p = Shape.xy2vec(lblsFrmXY);
%           s(end).tfocc = strcmp('occ',tags(:)');
%           if tfLposTS
%             lts = lposTS{iMovSet};
%             s(end).pTS = lts(:,f)';
%           end
%         end
%         
%         I = [I;ITmp]; %#ok<AGROW>
%       end
%       tbl = struct2table(s,'AsArray',true);      
%     end
  end  % methods (Static)
  
  methods (Static) % MA
    function ntgt = getNumTgts(s,frm)
      tf = s.frm==frm;
      ntgt = nnz(tf);  
    end

    function [s,itgt] = addTgtWithP(s,frm,xy)
      % itgt: new target index (also new number of tgts in frm)
      ntgt = Labels.getNumTgts(s,frm);
      itgt = ntgt+1;
      s.p(:,end+1) = xy(:);
      s.ts(:,end+1) = now;
      s.occ(:,end+1) = 0;
      s.frm(end+1,1) = frm;
      s.tgt(end+1,1) = itgt;
    end
  end  % methods (Static)
end  % classdef