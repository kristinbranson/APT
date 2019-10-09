classdef PxAssign 
  
  methods (Static)
    
    function imfg = simplebgsub(bgType,im,imbg,imbgdev)
      assert(isfloat(im));
      assert(isfloat(imbg));
      assert(isfloat(imbgdev));
      switch bgType
        case 'light on dark'
          imfg = max(im-imbg,0)./imbgdev;
        case 'dark on light'
          imfg = max(imbg-im,0)./imbgdev;
        case 'other'
          imfg = abs(imbg-im)./imbgdev;
        otherwise
          assert(false,'Unrecognized bgType.');
      end
    end 
    
    % Naturally this belongs in 'PxAssign'
    function im = imRescalePerType(im,ty)
      switch ty
        case 'uint8'
          im = double(im)/255;
        case 'uint16'
          im = double(im)/(2^16-1);
        case 'double'
          % none
        otherwise
          % Consider warning
      end
    end
    
    % Move me into TrxUtil
    function [tflive,xs,ys,ths,as,bs] = getTrxStuffAtFrm(trx,frm) 
      % trx: array of trx
      % 
      % tflive: [ntrx x 1] logical array
      % xs/ys/ths/as/bs: [ntrx x 1] double array. nan if/where tflive==false
      
      ntgt = numel(trx);
      
      tflive = true(ntgt,1);
      xs = nan(ntgt,1);
      ys = xs;
      ths = xs;
      as = xs;
      bs = xs;
      
      for itgt=1:ntgt
        trxI = trx(itgt);
        idx = trxI.off+frm;
        if idx<1 || idx>trxI.nframes
          tflive(itgt) = false;
        else
          xs(itgt) = trxI.x(idx);
          ys(itgt) = trxI.y(idx);
          ths(itgt) = trxI.theta(idx);
          as(itgt) = trxI.a(idx);
          bs(itgt) = trxI.b(idx);
        end
      end
    end
    
    function [xCtrRound,yCtrRound] = trxCtrRound(trx,f)
      assert(isscalar(trx));
      assert(trx.firstframe<=f && f<=trx.endframe);
      i = f+trx.off;
      xCtrRound = round(trx.x(i));
      yCtrRound = round(trx.y(i));
    end
    
    function [xlo,xhi,ylo,yhi] = roiTrxCtr(xc,yc,rad)
      xlo = xc-rad;
      xhi = xc+rad;
      ylo = yc-rad;
      yhi = yc+rad;
    end
    
    function idx = trxCtrLinearIdx(roi,trx,f)
      % Get the linear pixel index for the center of a trx in an roi
      %
      % roi: [xlo xhi ylo yhi]
      % trx: scalar trx
      % f: frame no
                 
      assert(isscalar(trx));
      assert(trx.firstframe<=f && f<=trx.endframe);

      [xCtrRnd,yCtrRnd] = PxAssign.trxCtrRound(trx,f);
      
      roiXlo = roi(1);
      roiXhi = roi(2);
      roiYlo = roi(3);
      roiYhi = roi(4);
      roinr = roiYhi-roiYlo+1;
      
      xCtrRoi = xCtrRnd-roiXlo+1;
      yCtrRoi = yCtrRnd-roiYlo+1;
      idx = yCtrRoi + (xCtrRoi-1)*roinr;
    end
      
    function imL = asgnCC(im,imbg,imbgdev,trx,f,varargin)
      % im, imbg, imbgdev: [nr x nc]
      % 
      % imL: label image, same size as im. All ccs contain at least one trx
      % center
      
      [bgtype,fgthresh] = myparse(varargin,...
        'bgtype','dark on light',...
        'fgthresh',4.0);
      imdiff = PxAssign.simplebgsub(bgtype,im,imbg,imbgdev);
      imL = PxAssign.asgnCCcore(imdiff,trx,f,fgthresh);      
    end
    function imL = asgnCCcore(imdiff,trx,f,fgthresh)
      bwfg = imdiff>fgthresh;
      cc = bwconncomp(bwfg);
      [nr,nc] = size(imdiff);
      
      idxTrxCtrs = zeros(0,1);
      for fly = 1:numel(trx)
        if f > trx(fly).endframe || f < trx(fly).firstframe
          continue;
        end
        idxTrxCtrs(end+1,1) = PxAssign.trxCtrLinearIdx([1 nc 1 nr],trx(fly),f); %#ok<AGROW>
      end

      tfDiscard = false(cc.NumObjects,1);
      for iCC=1:cc.NumObjects
        % keep only ccs that contain a trx center
        pxlistI = cc.PixelIdxList{iCC};
        tfDiscard(iCC) = ~any(ismember(idxTrxCtrs,pxlistI));
      end
      cc.PixelIdxList(tfDiscard) = [];
      cc.NumObjects = numel(cc.PixelIdxList);
      
      imL = labelmatrix(cc);
    end
    
    function [imL,imLpre,nfliescurr] = asgnGMMglobal(im,imbg,imbgdev,trx,f,varargin)
      [bgtype,fgthresh] = myparse(varargin,...
        'bgtype','dark on light',...
        'fgthresh',115); % in BackSub.m, n_bg_std_thresh_low
      imdiff = PxAssign.simplebgsub(bgtype,im,imbg,imbgdev);
      [imL,imLpre,nfliescurr] = PxAssign.asgnGMMglobalcore(imdiff,trx,f,fgthresh);
    end
    function [imL,imLpre,nfliescurr] = asgnGMMglobalcore(imdiff,trx,f,fgthresh)
      isfore = imdiff>=fgthresh; % in BackSub.m, n_bg_std_thresh_low  
      [imLpre,nfliescurr] = AssignPixels(isfore,imdiff,trx,f);
      imL = PxAssign.cleanupPass(imLpre,trx,f);
    end
    
    function [imL,imLpre,pdfTgts] = asgnPDF(imdiff,trx,f,...
        pdf,pdfXctr,pdfYctr,pdfamu,pdfbmu,varargin)
      
      fgthresh = myparse(varargin,...
        'fgthresh',4);

      %imdiff = PxAssign.simplebgsub(bgtype,im,imbg,imbgdev);
      imforebw = imdiff>=fgthresh;
      
      [imLpre,~,pdfTgts] = PxAssign.asgnPDFCore(imforebw,f,trx,pdf,pdfXctr,pdfYctr,...
        'scalePdf',true,...
        'scalePdfAmu',pdfamu,...
        'scalePdfBmu',pdfbmu,...
        'verbose',false); 
      imL = PxAssign.cleanupPass(imLpre,trx,f);
    end
    
    function [imforeL,splitCCnew,pdfTgts] = asgnPDFCore(...
        imforebw,frm,trx,pdf,pdfXctr,pdfYctr,varargin)
      % Assign pixels based on empirical pdf
      %
      % imforebw: b/w foreground image
      % frm: frame number
      % trx: [nTgt] trx array
      % pdf: [pdfnrxpdfnc] single-target empirical FG pdf. Canonically rotated. Also
      %   input images are scaled based on target size relative to mean.
      % pdfXctr: [pdfnc] center x-coords labeling columns of pdf
      % pdfYctr: [pdfnr] center y-coords labeling rows of pdf
      %
      % imforeL: [same as imfore] label matrix for imforebw. Each fg pixel
      %   assigned to a target/cc. All nonfg pixels are 0
      
      % splitCCnew: [nsplit] cell, splitCCnew{i} gives the new CCs that
      %   were formed by splitting a single original CC
      % pdfTgts: [nsplit] cell, pdfTgts{i} gives PDFs corresponding to each
      %   target comprising the ith original CC
      
      [scalePdf,scalePdfAmu,scalePdfBmu,verbose] = myparse(varargin,...
        'scalePdf',false,... % If true, scale/adjust pdf based on ellipse size (major/minor axes lengths)
        'scalePdfAmu',nan,... % mean ellipse 'a' value (majoraxis/2), used when scalePdf is true
        'scalePdfBmu',nan,... % etc
        'verbose',false);
      
      assert(all(imforebw(:)>=0));
      
      [pdfnr,pdfnc] = size(pdf);
      assert(numel(pdfXctr)==pdfnc);
      assert(numel(pdfYctr)==pdfnr);
      dx = pdfXctr(2)-pdfXctr(1);
      dy = pdfYctr(2)-pdfYctr(1);
      assert(isequal(pdfXctr,pdfXctr(1):dx:-pdfXctr(1)),'Unexpected pdf coordinates.');
      assert(isequal(pdfYctr,pdfYctr(1):dy:-pdfYctr(1)),'Unexpected pdf coordinates.');
      [pdfXg,pdfYg] = meshgrid(pdfXctr,pdfYctr);
      
      [imnr,imnc] = size(imforebw);
      imsz = imnr*imnc;
      [imgx,imgy] = meshgrid(1:imnc,1:imnr);
      
      [imforeL,nCC] = bwlabel(imforebw);
      
      [trxtflive,trxxs,trxys,trxths,trxas,trxbs] = ...
        PxAssign.getTrxStuffAtFrm(trx,frm);
      
      ccTrxCtrs = cell(nCC,1); % trx centers in each cc
      ntgt = numel(trx);
      for itgt=1:ntgt
        if trxtflive(itgt)
          x = round(trxxs(itgt));
          y = round(trxys(itgt));
          ccL = imforeL(y,x);
          if ccL>0
            ccTrxCtrs{ccL}(1,end+1) = itgt;
          end
        end
      end
      
      % All trx have been assigned to a CC. Some CCs are not assigned to a trx
      % (especially eg small noise pixels)
      
      % Find and deal with any CCs that have multiple trx assigned, indicating
      % two or more flies nearby+touching
      nTrxPerCC = cellfun(@numel,ccTrxCtrs);
      ccMultiTrx = find(nTrxPerCC>1);
      nCCMultiTrx = numel(ccMultiTrx);
      
      if nCCMultiTrx==0
        %   imforeLpre = imforeL;
        splitCCnew = cell(0,1);
        pdfTgts = cell(0,1);
        return;
      end
      
      splitCCnew = cell(nCCMultiTrx,1);
      pdfTgts = cell(nCCMultiTrx,1);
      maxcc = nCC;
      for iCC=1:nCCMultiTrx
        cc = ccMultiTrx(iCC);
        iTgtsCC = ccTrxCtrs{cc};
        nTgtsCC = numel(iTgtsCC);
        
        if verbose
          fprintf('CC %d has %d trx: %s\n',cc,nTgtsCC,mat2str(iTgtsCC));
        end
        
        % pdfTgtsI(:,:,j) is the likelihood of a pixel being in target iTgtsCC(j)
        % based on emp.PDF
        pdfTgtsI = nan(imnr,imnc,nTgtsCC);
        for j=1:nTgtsCC % loop over all targets assigned to this CC
          
          % Get trx info
          iTgt = iTgtsCC(j);
          assert(trxtflive(iTgt));
          trxx = trxxs(iTgt);
          trxy = trxys(iTgt);
          trxth = trxths(iTgt);
          trxa = trxas(iTgt);
          trxb = trxbs(iTgt);
          
          % Optionally rescale pdf based on ellipse size
          if scalePdf
            % trxa>MeanA => pdf is expanded in x-dir.
            % Note we re-normalize the pdf below as this changes the measure
            pdfXguse = pdfXg*trxa/scalePdfAmu;
            pdfYguse = pdfYg*trxb/scalePdfBmu;
          else
            pdfXguse = pdfXg;
            pdfYguse = pdfYg;
          end
          
          % rotate the pdf onto the x/y/th.
          % AL PERF ENHANCEMENT: don't need the whole PDF, just find it at
          % the foreground pts
          pdfItgt = readpdf(pdf,pdfXguse,pdfYguse,imgx,imgy,trxx,trxy,trxth);
          szassert(pdfItgt,[imnr imnc]);
          pdfTgtsI(:,:,j) = pdfItgt/sum(pdfItgt(:));
        end
        pdfTgts{iCC} = pdfTgtsI;
        
        % For each pixel in the CC, take the maximum
        tfcc = imforeL==cc;
        pdfTgtsI = reshape(pdfTgtsI,[imsz nTgtsCC]);
        pdfTgtsFore = pdfTgtsI(tfcc(:),:);
        [~,loc] = max(pdfTgtsFore,[],2);
        % loc is an assignment into 1..nTgtsCC for each pixel in the CC
        
        ccidx = find(tfcc); % linear indices into im for current composite cc
        szassert(loc,[numel(ccidx) 1]);
        
        % all loc==1 pixels keep their existing cc. We create new ccs for higher
        % locs
        newCCs = nan(1,nTgtsCC); % We are splitting cc up into nTgtsCC, one for each trx
        newCCs(1) = cc; % j=1 => first target => keeps cc
        for j=2:nTgtsCC % remaining targets, iTgtsCC(j)<->newCCs(j)
          maxcc = maxcc+1;
          imforeL(ccidx(loc==j)) = maxcc;
          if verbose
            fprintf('Created new cc=%d with %d px\n',maxcc,nnz(loc==j));
          end
          newCCs(j) = maxcc;
        end
        splitCCnew{iCC} = newCCs;
      end
    end    
    
    function [nmask,imtgt,imnottgt] = performMask(im,imbg,imL,trx,itgt,f,...
        varargin)
      % mask all fg pixels not assigned to itgt to imbg
      %
      % imL: label matrix. nonnegative ints, zero for nonfg pixels
      %
      % nmask: number of pixels masked (number of pixels where im differs
      % from imtgt)
      % imtgt: im, with only tgt kept
      % imnottgt: reverse
      
      imroi = myparse(varargin,...
        'imroi',[]); % Optional, [xlo xhi ylo yhi] roi specification for im, imbg, imL.
        % If im, imbg, imL are a cropped roi, then trx(itgt) in general is
        % not.
      
      if isempty(imroi)
        [xtgtctr,ytgtctr] = PxAssign.trxCtrRound(trx(itgt),f);
        lTgt = imL(ytgtctr,xtgtctr);
      else
        szassert(imroi,[1 4]);
        idx = PxAssign.trxCtrLinearIdx(imroi,trx(itgt),f);
        lTgt = imL(idx);
      end
      assert(lTgt>0);
      
      tfFGtgt = imL==lTgt;
      tfFGnottgt = imL>0 & ~tfFGtgt;
      nmask = nnz(tfFGnottgt);
      
      imtgt = im;
      imtgt(tfFGnottgt) = imbg(tfFGnottgt);
      if nargout>2
        imnottgt = im;
        imnottgt(tfFGtgt) = imbg(tfFGtgt);
      end
    end
    
    function bwl = cleanupPass(bwl,trx,f,varargin)
      % 2nd/Final pass. The breaking up of ccs into a new bwl will be 
      % imperfect. The most obvious class of imperfections is when pixels 
      % assigned to the new CCs are disjoint. We loop over all of the new 
      % CCs, and reassign any disconnected "fragments" per pass2alg. Note
      % that this procedure operates in trx-order and is order-dependent so
      % this is only a heuristic and is biased etc.
      
      pass2alg = myparse(varargin,...
        'pass2alg','touch'); % either 'dist' or 'touch'
      
      [imnr,imnc] = size(bwl);
      imsz = numel(bwl);
      [imgx,imgy] = meshgrid(1:imnc,1:imnr);
      roi = [1 imnc 1 imnr];
      
      ntrx = numel(trx);      
      [trxtflive,trxxs,trxys] = PxAssign.getTrxStuffAtFrm(trx,f);
      trxCtrLinearIdxs = nan(ntrx,1);
      for fly=1:ntrx
        if ~trxtflive(fly)
          continue;
        end
        trxCtrLinearIdxs(fly) = PxAssign.trxCtrLinearIdx(roi,trx(fly),f);
      end
      
      for fly=1:ntrx
        if ~trxtflive(fly)
          continue;
        end
        
        % for each live trx, we find its label; find all px assigned to
        % this label; break down into CCs; and try to be smart about
        % re-assigning non-central CCs.
        
        trxCtrIdx = trxCtrLinearIdxs(fly);
        lblTgt = bwl(trxCtrIdx);
        Itgt = bwl==lblTgt;
        cc = bwconncomp(Itgt,4); % cc is a breakdown of all pxs assigned to iTgt
        %     if strcmp(pass2alg,'touch')
        %       Lcc = labelmatrix(s);
        %     end
        
        % find cc containing trx center
        iCompTgt = nan;
        for iComp=1:cc.NumObjects
          if any(trxCtrIdx==cc.PixelIdxList{iComp})
            iCompTgt = iComp;
            break;
          end
        end
        assert(~isnan(iCompTgt),'trx center must be in a connected component.');
        
        % for every other cc, do something per pass2alg
        for iComp=1:cc.NumObjects
          if iComp==iCompTgt
            continue;
          end
          switch pass2alg
            case 'dist'
              % Take first pt in the pixelList; find trxcenter that is closest.
              % Use its CC
              linIdx = cc.PixelIdxList{iComp}(1);
              xPx1 = imgx(linIdx);
              yPx1 = imgy(linIdx);
              d2trx = (xPx1-trxxs).^2 + (yPx1-trxys).^2;
              szassert(d2trx,[ntrx 1]);
              [~,iTgtNearest] = min(d2trx); 
              % note, there must be at least one live target; so not all
              % els of d2trx are nan
              assert(trxtflive(iTgtNearest));
%               jNearestTgt = find(iTgtNearest==iTgtsCC);
%               assert(isscalar(jNearestTgt),'Nearest trx not found.');
              lblAssign = bwl(trxCtrLinearIdxs(iTgtNearest));
            case 'touch'
              % Take the first CC encountered that is touching/adjacent to this
              % cc.
              linidxs = cc.PixelIdxList{iComp}; % linear indices for this CC
              linidxs = linidxs(:);
              % linear indices of all pixels adjacent to CC
              linidxsadj = [linidxs+1 linidxs-1 linidxs+imnr linidxs-imnr];
              tfIB = linidxsadj>=1 & linidxsadj<=imsz;
              linidxsadj = linidxsadj(tfIB);
              bwladj = bwl(linidxsadj); % bwl values at adj pxs
              bwladj = bwladj(bwladj~=lblTgt & bwladj~=0); 
              % first try, prefer labels that are NOT the label for the 
              % current target center. b/c this CC is not connected to the
              % target center, it prob belongs to another target.
              
              if isempty(bwladj)
                % no labeled adjacent px. reassign to original label
                lblAssign = lblTgt;
              else              
                lblAssign = mode(bwladj); % most frequent adjacent bwlval; (ALT: use first adj, but filter out tiny 1-px ccs etc)
              end
            otherwise
              assert(false);
          end
          bwl(cc.PixelIdxList{iComp}) = lblAssign;
        end
      end
    end
    
  end
end