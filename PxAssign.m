classdef PxAssign 
  
  methods (Static)
    
    function imfg = simplebgsub(bgType,im,imbg,imbgdev)
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
    
    function [tflive,xs,ys,ths,as,bs] = getTrxStuffAtFrm(trx,frm) 
      % trx: array of trx
      % 
      % tflive: [ntrx x 1] logical array
      % xs/ys/ths/as/bs: [ntrx x 1] double array
      
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
      
    function bwl = asgnCC(im,imbg,imbgdev,trx,f,varargin)
      % im, imbg, imbgdev: [nr x nc]
      % 
      % imasgn: label image, same size as im. 0 for no assgn, 1 for target
      % fly, 2 for everything else
      
      fgthresh = myparse(varargin,...
        'fgthresh',4.0);

      imdiff = PxAssign.simplebgsub('dark on light',im,imbg,imbgdev);
      bwfg = imdiff>fgthresh;
      cc = bwconncomp(bwfg);
      [nr,nc] = size(im);
      
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
      
      bwl = labelmatrix(cc);
    end
    
    function [bwl,bwlpre,nfliescurr] = asgnGMMglobal(im,imbg,imbgdev,trx,f,varargin)
      [fgthresh] = myparse(varargin,...
        'fgthresh',115); % in BackSub.m, n_bg_std_thresh_low

      imdiff = PxAssign.simplebgsub('dark on light',im,imbg,imbgdev);
      isfore = imdiff>=fgthresh; % in BackSub.m, n_bg_std_thresh_low
  
      [bwlpre,nfliescurr] = AssignPixels(isfore,imdiff,trx,f);
      bwl = PxAssign.cleanupPass(bwlpre,trx,f);
    end
    
    function [bwl,bwlpre,pdfTgts] = asgnPDF(im,imbg,imbgdev,trx,f,...
        pdf,pdfXe,pdfYe,pdfamu,pdfbmu,varargin)
      
      fgthresh = myparse(varargin,...
        'fgthresh',4);

      imdiff = PxAssign.simplebgsub('dark on light',im,imbg,imbgdev);
%       isfore = imdiff>=fgthresh;

%       im2 = bg-double(im);
%       imd = abs(im2);
%       forebw = imd>FORETHRESH;
%       forebwl = bwlabel(forebw);
      
      [bwl,bwlpre,splitCC,splitCCnew,pdfTgts] = assignids(imdiff,...
        f,trx,pdf,pdfXe,pdfYe,...
        'scalePdfLeg',true,...
        'scalePdfLegMeanA',pdfamu,...
        'scalePdfLegMeanB',pdfbmu,...
        'bwthresh',fgthresh,...
        'verbose',true);      
    end
    
    function [imtgt,imnottgt] = performMask(im,imbg,imcc,trx,itgt,f)
      % imtgt: im, with only tgt kept
      % imnottgt: reverse
      
      [xtgtctr,ytgtctr] = PxAssign.trxCtrRound(trx(itgt),f);
      cctgt = imcc(ytgtctr,xtgtctr);
      assert(cctgt>0);
      tftgt = imcc==cctgt;
      imtgt = im;
      imtgt(~tftgt) = imbg(~tftgt);
      imnottgt = im;
      imnottgt(tftgt) = imbg(tftgt);
    end
    
    function bwl = cleanupPass(bwl,trx,f,varargin)
      % 2nd/Final pass. The breaking up of ccs into a new bwl will be 
      % imperfect. The most obvious class of imperfections is when pixels 
      % assigned to the new CCs are disjoint. We loop over all of the new 
      % CCs, and reassign any disconnected "fragments" per pass2alg.
      %
      
      pass2alg = myparse(varargin,...
        'pass2alg','touch'); % either 'dist' or 'touch'
      
      [imnr,imnc] = size(bwl);
      imsz = numel(bwl);
      [imgx,imgy] = meshgrid(1:imnc,1:imnr);
      roi = [1 imnc 1 imnr];
      
      ntrx = numel(trx);      
      [trxtflive,trxxs,trxys,trxths,trxas,trxbs] = ...
        PxAssign.getTrxStuffAtFrm(trx,f);
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