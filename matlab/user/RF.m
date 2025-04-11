classdef RF
  
  properties (Constant)
    
    PTMAP = [ %out  %mid   %inn
                1     7     13; % LF
                2     8     14; % LM
                3     9     15; % LH
                4     10    16; % RF
                5     11    17; % RM
                6     12    18; % RH
            ];
    PTNAMES = { ...
      'LFt' 'LFm' 'LFh'
      'LMt' 'LMm' 'LMh'
      'LHt' 'LHm' 'LHh'
      'RFt' 'RFm' 'RFh'
      'RMt' 'RMm' 'RMh'
      'RHt' 'RHm' 'RHh'};
    PTMAPCOLS = {'out' 'mid' 'inn'};
    PTMAPROWS = {'LF' 'LM' 'LH' 'RF' 'RM' 'RH'};
    PTMAPROWS_LSIDE = logical([1 1 1 0 0 0]);
    PTMAPROWS_RSIDE = logical([0 0 0 1 1 1]);
    
    PTS_LSIDE = [1 2 3 7 8 9 13 14 15];
    PTS_RSIDE = [4 5 6 10 11 12 16 17 18];
    
    COLORS = {...
      [1 0 0] [1 0 0] [1 0 0]; ...
      [1 1 0] [1 1 0] [1 1 0]; ...
      [0 1 0] [0 1 0] [0 1 0]; ...
      [0 1 1] [0 1 1] [0 1 1]; ...
      [1 0 1] [1 0 1] [1 0 1]; ...
      [1 204/255 77/255] [1 204/255 77/255] [1 204/255 77/255]};
    
    MARKERS = {...
      '^' 'o' 's';
      '^' 'o' 's';
      '^' 'o' 's';
      '^' 'o' 's';
      '^' 'o' 's';
      '^' 'o' 's'};
  end
  
  methods (Static)
    
    function tf = ptWrongSide()
      % tf: [18x3]. tf(ipt,ivw) is true if point ipt is on "far side" in view ivw
      
      tf = [ ...
        0 1 0
        0 1 0
        0 1 0
        1 0 0
        1 0 0
        1 0 0
        0 1 0
        0 1 0
        0 1 0
        1 0 0
        1 0 0
        1 0 0
        0 1 0
        0 1 0
        0 1 0
        1 0 0
        1 0 0
        1 0 0];
      tf = logical(tf);
    end
    
    function [npttot,nphyspt,nview,nfrm] = lposDim(lpos)
      [npttot,d,nfrm] = size(lpos);
      assert(d==2);
      nview = 3;
      nphyspt = npttot/nview;
    end
    
    function olFcn = olDesc2Fcn(olDesc,movInfo)
      movNr = cellfun(@(x)x.info.nr,movInfo);
      olFcn = cell(size(olDesc));
      [nphyspt,nview] = size(olDesc);
      for iphyspt=1:nphyspt
        for ivw=1:nview
          desc = olDesc{iphyspt,ivw};
          if ~isempty(desc)
            tfout = strcmp(desc,'out');
            shi = regexp(desc,'hi(?<num>[0-9]+)','names');
            slo = regexp(desc,'lo(?<num>[0-9]+)','names');
            tfhi = ~isempty(shi);
            tflo = ~isempty(slo);
            if tfhi
              yhi = movNr(ivw)-str2double(shi.num);
            end
            if tflo
              ylo = str2double(slo.num);
            end 
            if tfout
              fcn = @(xy) true(size(xy(:,1)));
            elseif tfhi && tflo
              fcn = @(xy) xy(:,2)>yhi | xy(:,2)<ylo;
            elseif tfhi
              fcn = @(xy) xy(:,2)>yhi;
            elseif tflo
              fcn = @(xy) xy(:,2)<ylo;              
            else
              fcn = [];
            end
            olFcn{iphyspt,ivw} = fcn;
          end
        end
      end
    end

    function tf = tfIsOutlier(lpos,movInfo,olDesc)
      % lpos: eg lbl.labeledpos{iMov}
      % movInfo: lbl.movieInfoAll(iMov,:)
      % olDesc: outlier description
      %
      % tf: [nfrmxnpttot]
            
      [npttot,nphyspt,nview,nfrm] = RF.lposDim(lpos);
      assert(nphyspt==18 || nphyspt==19);
       
      szassert(olDesc,[nphyspt nview]);
      olFcn = RF.olDesc2Fcn(olDesc,movInfo);
      
      tf = false(nfrm,npttot);
      for iphyspt=1:nphyspt
        for ivw=1:nview
          ipt = iphyspt+(ivw-1)*nphyspt;
          xy = squeeze(lpos(ipt,:,:))';
          assert(size(xy,2)==2);
          
          fcn = olFcn{iphyspt,ivw};
          if ~isempty(fcn)
            tf(:,ipt) = fcn(xy);
          end
        end
      end
    end
    
    function hFig = vizLabels(lpos,movieInfo,olDesc,varargin)
      
      oneBigPlot = myparse(varargin,...
        'oneBigPlot',true);
      
      [npttot,nphyspt,nview,nfrm] = RF.lposDim(lpos);

      tfOutlr = RF.tfIsOutlier(lpos,movieInfo,olDesc);
      szassert(tfOutlr,[nfrm npttot]);

      hFig = figure;
      if oneBigPlot
        axs = createsubplots(nview,nphyspt);
        axs = reshape(axs,nview,nphyspt);
      else
        axs = createsubplots(1,3,0.05);
      end

      hts = cellfun(@(x)x.info.Height,movieInfo);
      wds = cellfun(@(x)x.info.Width,movieInfo);

      for iphyspt=1:nphyspt
        for ivw=1:nview
          ipt = iphyspt+(ivw-1)*nphyspt;
          x = squeeze(lpos(ipt,:,:))';
          szassert(x,[nfrm 2]);
          tfO = tfOutlr(:,ipt);
         
          if oneBigPlot
            ax = axs(ivw,iphyspt);
          else           
            ax = axs(ivw);
            cla(ax);
          end
          scatter(ax,x(~tfO,1),x(~tfO,2),[],'b');
          hold(ax,'on');
          scatter(ax,x(tfO,1),x(tfO,2),[],'r');
          grid(ax,'on');
          axis(ax,[0 wds(ivw) 0 hts(ivw)]);
          
          if oneBigPlot
            if iphyspt==1
            else
              ax.XTickLabel = [];
              ax.YTickLabel = [];
            end            
          else
            title(ax,sprintf('vw%d pt%d',ivw,iphyspt),'fontweight','bold');
          end
        end
        if ~oneBigPlot
          input('hk');
        end
      end
    end
    
    function [lpos,nRm] = rmOutlierOcc(lpos,movieInfo,olDesc)
      % Remove "occluded with no estimate" points where labeler just
      % clicked in the corner
      % 
      % lpos (out): coords that are clearly outliers/nonsense representing 
      % "occluded with no estimate" have been replaced with NaN
      % nRm: [nphysptxnview]. Number of frames removed for each physpt/view

      [npttot,nphyspt,nview,nfrm] = RF.lposDim(lpos);
      
      tfOutlr = RF.tfIsOutlier(lpos,movieInfo,olDesc);
      szassert(tfOutlr,[nfrm npttot]);
      
      nRm = sum(tfOutlr,1); % [1xnpttot]
      nRm = reshape(nRm,[nphyspt nview]);
      
      tfOutlr = tfOutlr';
      tfOutlr = reshape(tfOutlr,[npttot 1 nfrm]);
      tfOutlr = repmat(tfOutlr,[1 2]);
      lpos(tfOutlr) = nan;
    end
    
    function tFP = FPtable(lpos,lpostag)
      % Generate Frame-Pos table
      % 
      % lpos: [npttot x d x nfrm] labeledpos
      % lpostag: [npttot x nfrm] labeledpostag
      %
      % tFrmPts: Frame-Pos table
      
      d = 2;
      [npttot,nptphys,nview,nfrm] = RF.lposDim(lpos);
      lpos4d = reshape(lpos,[nptphys nview 2 nfrm]);
      lpostag4d = reshape(lpostag,[nptphys nview nfrm]);

      
      %% Generate MD table
      %
      % Fields
      % frm
      % npts2VwLbl. scalar, number of pts that have >= 2 views labeled (regardless of occ status)
      % ipts2VwLbl. vector with npts2VwLbl els. physPt indices.
      % tfVws2VwLbl. [3xnpts2VwLbl] array, 1 if (view,physpt) is labeled.
      % occ2VwLbl. [3xnpts2VwLbl] array. 1 if (view,physpt) is occluded.
      
      frm = nan(0,1);
      npts2VwLbl = nan(0,1);
      ipts2VwLbl = cell(0,1);
      tfVws2VwLbl = cell(0,1);
      occ2VwLbl = cell(0,1); % true => occluded
      p = nan(0,npttot*d); % allx, then ally
      
      for f=1:nfrm
        tf2VwLbledAny = false(1,nptphys);
        tfVwLbledAnyPt = cell(1,nptphys);
        occStatusPt = cell(1,nptphys);
        for ippt = 1:nptphys
          lposptfrm = squeeze(lpos4d(ippt,:,:,f));
          ltagptfrm = squeeze(lpostag4d(ippt,:,f));
          ltagptfrm = ltagptfrm(:);
          szassert(lposptfrm,[nview d]);
          szassert(ltagptfrm,[nview 1]);
          tfVwLbled = ~any(isnan(lposptfrm),2);
          tfVwNotOcc = cellfun(@isempty,ltagptfrm);
          
          tf2VwLbledAny(ippt) = nnz(tfVwLbled)>=2;
          tfVwLbledAnyPt{ippt} = tfVwLbled;
          occStatusPt{ippt} = ~tfVwNotOcc;
        end
        
        if any(tf2VwLbledAny)
          frm(end+1,1) = f; %#ok<AGROW>
          lpos4dthisfrm = lpos4d(:,:,:,f);
          p(end+1,:) = lpos4dthisfrm(:); %#ok<AGROW> % raster order: pt, view, d
          npts2VwLbl(end+1,1) = nnz(tf2VwLbledAny); %#ok<AGROW>
          ipts2VwLbl{end+1,1} = find(tf2VwLbledAny); %#ok<AGROW>
          tmp = tfVwLbledAnyPt(tf2VwLbledAny);
          tmp = cat(2,tmp{:});
          szassert(tmp,[nview npts2VwLbl(end)]);
          tfVws2VwLbl{end+1,1} = tmp; %#ok<AGROW>
          tmp = occStatusPt(tf2VwLbledAny);
          tmp = cat(2,tmp{:});
          szassert(tmp,[nview npts2VwLbl(end)]);
          occ2VwLbl{end+1,1} = tmp; %#ok<AGROW>
        end
        
        if mod(f,1e3)==0
          disp(f);
        end
      end
      
      tFP = table(frm,p,npts2VwLbl,ipts2VwLbl,tfVws2VwLbl,occ2VwLbl);
    end
    
    function tFPaug = recon3D(tFP,crig2)
      % Reconstruct/err stats for pts labeled in 2 views
      %
      % For points labeled in all three views ('lrb'):
      %  * Recon 3D pt.
      %  * Use each viewpair to recon/project in 3rd view and compute error.
      % For all points labeled in only two views:
      %  * Recon 3D pt.
      %  * reproject onto 2 labeled views and compute err.
      %
      % New fields:
      %  * X [3x19]. 3D recon pt in certain frame, say 'l'.
      %  * errReconL. [npts2VwlblNOx1]. For points with 'lrb'. L2 err in L view (recon vs gt)
      %  * errReconR. etc
      %  * errReconB.
      
      np1 = numel(tFP.p(1,:));
      nview = 3;
      nphyspt = np1/nview/2;
      assert(round(nphyspt)==nphyspt);      
      assert(all(tFP.npts2VwLbl==nphyspt));      
      
      fprintf('nphyspt=%d.\n',nphyspt);
      pause(3);
      
      nRows = size(tFP,1);
      XL = cell(nRows,1);
      err = cell(nRows,1);
      % XLlr = cell(nRows,1);
      % errReconL = nan(nRows,nphyspt);
      % errReconR = nan(nRows,nphyspt);
      % errReconB = nan(nRows,nphyspt);
      % errReconL_lr = nan(nRows,nphyspt);
      % errReconR_lr = nan(nRows,nphyspt);
      % errReconB_lr = nan(nRows,nphyspt);
      for iRow = 1:nRows
        %frm = tFP.frm(iRow);
        if mod(iRow,100)==0
          disp(iRow);
        end
        
        p = tFP.p(iRow,:);
        assert(numel(p)==nphyspt*nview*2);
        p3d = reshape(p,[nphyspt nview 2]);
        XLrow = nan(3,nphyspt); % first dim is x-y-z dimensions
        errRE = nan(3,nphyspt); % first dim is L-R-B views
        %   XLlrrow = nan(3,nphyspt);
        for iPt = 1:nphyspt
          lposPt = squeeze(p3d(iPt,:,:));
          szassert(lposPt,[nview 2]);
          yL = lposPt(1,[2 1]);
          yR = lposPt(2,[2 1]);
          yB = lposPt(3,[2 1]);
          tfViewsLabeled = tFP.tfVws2VwLbl{iRow}(:,iPt);
          assert(nnz(tfViewsLabeled)>=2);
          viewsLbled = find(tfViewsLabeled);
          viewsLbled = viewsLbled(:)';
          
          yLre = nan(1,2);
          yRre = nan(1,2);
          yBre = nan(1,2);
          if isequal(viewsLbled,[1 2])
            assert(all(isnan(yB(:))));
            [XLtmp,XRtmp] = crig2.stereoTriangulateCropped(yL,yR,'L','R');
            XLrow(:,iPt) = XLtmp;
            
            [yLre(1),yLre(2)] = crig2.projectCPR(XLtmp,1);
            [yRre(1),yRre(2)] = crig2.projectCPR(XRtmp,2);
            
          elseif isequal(viewsLbled,[1 3])
            assert(all(isnan(yR(:))));
            
            [XLtmp,XBtmp] = crig2.stereoTriangulateLB(yL,yB);
            XLrow(:,iPt) = XLtmp;
            
            [yLre(1),yLre(2)] = crig2.projectCPR(XLtmp,1);
            [yBre(1),yBre(2)] = crig2.projectCPR(XBtmp,3);
            
          elseif isequal(viewsLbled,[2 3])
            assert(all(isnan(yL(:))));
            
            [XBtmp,XRtmp] = crig2.stereoTriangulateBR(yB,yR);
            XLrow(:,iPt) = crig2.camxform(XBtmp,'bl');
            
            [yBre(1),yBre(2)] = crig2.projectCPR(XBtmp,3);
            [yRre(1),yRre(2)] = crig2.projectCPR(XRtmp,2);
          elseif isequal(viewsLbled,1:3)
            [yLre,yRre,yBre,...
              ~,~,~,...
              ~,~,~,...
              XLrow(:,iPt)] = ...
              crig2.calibRoundTripFull(yL,yR,yB);
          else
            assert(false);
          end
          
          errRE(1,iPt) = sqrt(sum((yL-yLre).^2,2));
          errRE(2,iPt) = sqrt(sum((yR-yRre).^2,2));
          errRE(3,iPt) = sqrt(sum((yB-yBre).^2,2));
        end
        XL{iRow} = XLrow;
        err{iRow} = errRE;
      end
      
      tFPaug = [tFP table(XL,err)];
    end
    
    function bboxes = generateBBoxes(p)
      %
      % p: [nxnPtxd]
      %
      % bboxes: [1x2d]

      [n,npt,d] = size(p); %#ok<ASGLU>
      pcoordmins = nan(1,d);
      pcoordmaxs = nan(1,d);
      for iCoord=1:d
        z = p(:,:,iCoord); % x-, y-, or z-coords for all rows, pts
        pcoordmins(iCoord) = nanmin(z(:));
        pcoordmaxs(iCoord) = nanmax(z(:));
      end
      dels = pcoordmaxs-pcoordmins;
      % pad by 50% in every dir
      pads = dels/2;
      widths = 2*dels; % del (shapes footprint) + 2*pads (one on each side)
      bboxes = [pcoordmins-pads widths];
    end
    
    function [h,m,t,phi] = flyOrientation2D(xy)
      % 
      % xy: [18x2] x- and y-coords for landmarks
      %
      % h: [1x2] head loc (xy)
      % m: abdomen loc etc
      % t: tail loc etc
      % phi: orientation of t->h vector
      
      szassert(xy,[18 2]);
      
      FOREPTS = [1 7 13 4 10 16];
      MIDDPTS = [2 8 14 5 11 17];
      HINDPTS = [3 9 15 6 12 18];
      
      xyFore = xy(FOREPTS,:);
      xyMidd = xy(MIDDPTS,:);
      xyHind = xy(HINDPTS,:);
      
      h = nanmean(xyFore,1);
      m = nanmean(xyMidd,1);
      t = nanmean(xyHind,1);
      
      hvec = h-t;
      phi = atan2(hvec(2),hvec(1));
    end
    
    function plotShapes3D(ax,p)
      % p: [nx18x3]

      n = size(p,1);
      szassert(p,[n 18 3]);
      
      LEGCOLORS = lines(6);
      hold(ax,'on');
      for i=1:n
        for leg=1:6
          pleg = squeeze(p(i,[leg leg+6 leg+12],:)); % [3x3]
          plot3(pleg(:,1),pleg(:,2),pleg(:,3),'-','Color',LEGCOLORS(leg,:),'parent',ax);
        end
      end
    end
    
    function hLine = addLinesToLabelerAxis(lObj)
      ax = lObj.gdata.axes_curr;
      if isfield(ax.UserData,'hLine')
        deleteValidGraphicsHandles(ax.UserData.hLine);
      end
      NPTS = 18;
      hLine = gobjects(NPTS,1);
      hold(ax,'on');
      clrs = RF.COLORS;
      for iPt = 1:NPTS
        hLine(iPt) = plot(ax,nan,nan,'.',...
          'markersize',20,...
          'Color',clrs{iPt});
      end
      ax.UserData.hLine = hLine;
    end
    
  end
  
  methods (Static)
    
    function makeTrkMovie3D(movset,movout,varargin)
      
      trkRes3D = myparse(varargin,...
        'trkRes3D',[]); % struct with fields .X [3x18xnfrm], frm, crig
      
      if ~isempty(trkRes3D)
        nfrm = numel(trkRes3D.frm);
        szassert(trkRes3D.X,[3 18 nfrm]);
        assert(isa(trkRes3D.crig,'CalRig'));
      end
        
      NPTS = 18;
      FRAMERATE = 24;
      GAMMA = .3;
      mgray = gray(256);
      mgray2 = imadjust(mgray,[],[],GAMMA);
      
      assert(iscellstr(movset) && numel(movset)==3);
      for i=3:-1:1
        mr(i) = MovieReader();
        mr(i).open(movset{i});
        mr(i).forceGrayscale = true;
      end
      
      hts = [mr.nr];
      wds = [mr.nc];
      bigImHt = max(hts(1:2)) + hts(3);
      bigImWd = wds(3);
      % imL occupies cols [1,midline] inclusive
      % imR occupies cols [midline+1,wds(3)] inclusive
      midline = floor(wds(3)/2);                 
      assert(wds(1)<=midline);
      assert(wds(2)<=bigImWd-midline);
      bigim = nan(bigImHt,bigImWd);
      
      rowOffsets = {max(hts(1:2))-hts(1); max(hts(1:2))-hts(2); max(hts(1:2))};
      colOffsets = {midline-wds(1); midline; 0};
      bigimIdxL = {(1:hts(1))+rowOffsets{1},(1:wds(1))+colOffsets{1}};
      bigimIdxR = {(1:hts(2))+rowOffsets{2},(1:wds(2))+colOffsets{2}};
      bigimIdxB = {(1:hts(3))+rowOffsets{3},(1:wds(3))+colOffsets{3}};

      hFig = figure;
      ax = axes;
      hIm = imagesc(bigim,'parent',ax);
      colormap(ax,mgray2);
      truesize(hFig);
      hold(ax,'on');
      ax.XTick = [];
      ax.YTick = [];

      hLine = gobjects(3,NPTS);
      for iVw = 1:3
        for iPt = 1:NPTS
          hLine(iVw,iPt) = plot(ax,nan,nan,'.',...
            'markersize',6,...
            'marker',RF.MARKERS{iPt},...
            'Color',RF.COLORS{iPt},...
            'MarkerFaceColor',RF.COLORS{iPt});
        end
      end
      
      %trk = load(trkfile,'-mat');

      vr = VideoWriter(movout);
      vr.FrameRate = FRAMERATE;
      vr.open();
      
      hTxt = text(10,15,'','parent',ax,'Color','white','fontsize',24);
      hWB = waitbar(0,'Writing video');

      crig = trkRes3D.crig;
      frms = trkRes3D.frm;
      nfrm = numel(frms);
      for iF=1:nfrm
        f = frms(iF);
        
        imL = mr(1).readframe(f);
        imR = mr(2).readframe(f);
        imB = mr(3).readframe(f);
        bigim(bigimIdxL{:}) = imL;
        bigim(bigimIdxR{:}) = imR;
        bigim(bigimIdxB{:}) = imB;
        hIm.CData = bigim;
        
        
        Xbase = trkRes3D.X(:,:,iF);
        for iVw=1:3
          for iPt=1:18
            X = Xbase(:,iPt);
            Xvw = crig.viewXformCPR(X,1,iVw); % iViewBase==1
            [r,c] = crig.projectCPR(Xvw,iVw);
            
            radj = r + rowOffsets{iVw};
            cadj = c + colOffsets{iVw};
            
            switch iVw
              case 1
                if ~any(iPt==RF.PTS_LSIDE)
                  radj = nan;
                  cadj = nan;
                end
              case 2
                if ~any(iPt==RF.PTS_RSIDE)
                  radj = nan;
                  cadj = nan;
                end
            end
            hL = hLine(iVw,iPt);
            set(hL,'XData',cadj,'YData',radj);
          end
        end
        
        hTxt.String = sprintf('%04d',f);
        drawnow
        
        tmpFrame = getframe(ax);
        vr.writeVideo(tmpFrame);
        waitbar(iF/nfrm,hWB,sprintf('Wrote frame %d\n',f));
      end
      
      vr.close();
      delete(hTxt);
      delete(hWB);      
    end
    
    function makeTrkMovie2D(movfile,trkfile,movout,varargin)
      % Make 'results movie'
      % 
      % movfile: char or cellstr array
      % trkfile: char or cellstr array
      %
      % If cellstr arrays, movfile and trkfile must have the same size.
      % 
      % movout: char

      [trkfilefull,trkIPt] = myparse(varargin,...
        'trkfilefull',[],...
        'trkIPt',[]);   % for colors; specification of "global" points 
                        % indices for points in trkfile. if trkfile 
                        % contains n tracked points, this is an index
                        % vector of length n into 1:18
      
      FRAMERATE = 24;
      GAMMA = .2;
      mgray = gray(256);
      mgray2 = imadjust(mgray,[],[],GAMMA);
      
      movfile = cellstr(movfile);
      trkfile = cellstr(trkfile);
      szassert(trkfile,size(movfile));
      nmov = numel(movfile);
      
      % open movies, trkfiles
      for i=nmov:-1:1
        mr(i) = MovieReader();
        mr(i).open(movfile{i});
        mr(i).forceGrayscale = true;
        
        trk = load(trkfile{i},'-mat');
        ptrk{i} = trk.pTrk;
        [npts,d,nfrm] = size(ptrk{i});
        %assert(npts==18);
        assert(d==2);
        tflbl{i} = arrayfun(@(x)nnz(~isnan(ptrk{i}(:,:,x)))>0,1:nfrm);
      end      
      [subnr,subnc] = size(movfile);
      mr = reshape(mr,[subnr,subnc]);
      ptrk = reshape(ptrk,[subnr,subnc]);
      
      if ~isempty(trkIPt)        
        validateattributes(trkIPt,{'numeric'},...
          {'positive' 'integer' 'vector' 'numel' npts '<=' 18});
      else
        trkIPt = 1:npts;
      end
      
      % figure out frames for output movie
      lbledFrms = cellfun(@find,tflbl,'uni',0);
      lbledFrms = cellfun(@(x)x(:),lbledFrms,'uni',0);
      frmsMov = unique(cat(1,lbledFrms{:}));
      fprintf(1,'Making movie(s) with %d labeled frames. Movs are:\n',numel(frmsMov));
      cellfun(@(x)fprintf(1,'  %s\n',x),movfile);
      
      % create figure/axis/image
      imnr = mr(1).nr;
      imnc = mr(1).nc;
      arrayfun(@(x)assert(x.nr==imnr && x.nc==imnc),mr);
      imbig = zeros(subnr*imnr,subnc*imnc);
      hFig = figure;
      ax = axes;
      hIm = imagesc(ax,imbig);
      colormap(ax,mgray2);
      hold(ax,'on');
      ax.XTick = [];
      ax.YTick = [];
      truesize(hFig);

      hLines = gobjects(nmov,npts);
      for iMov=1:nmov
        for iPt = 1:npts
          hLines(iMov,iPt) = plot(ax,nan,nan,'.',...
            'markersize',28,...
            'Color',RF.COLORS{trkIPt(iPt)});
        end
      end
      if ~isempty(trkfilefull)
        trkfull = load(trkfilefull,'-mat');
        ptrkFull = trkfull.pTrkFull;
        nptsFull = size(ptrkFull,1);
        for iPt=1:nptsFull
          hLinesFull(iPt,1) = plot(ax,nan,nan,'.',...
            'markersize',10,'Color',RF.COLORS{trkIPt(iPt)});
        end
      end
      
      hTxtTitle = gobjects(subnr,subnc);
      for irow=1:subnr
        for icol=1:subnc
          % lower-left corner
          OFFSETPX = 18;
          rowloc = irow*imnr-OFFSETPX;
          colloc = (icol-1)*imnc+OFFSETPX;
          
          [~,trkfileS] = fileparts(trkfile{irow,icol});
          hTxtTitle(irow,icol) = text(ax,colloc,rowloc,trkfileS,...
            'FontSize',12,'Color',[1 1 1],'interpreter','none');
        end
      end
      
      vr = VideoWriter(movout);
      vr.FrameRate = FRAMERATE;
      vr.open();
      
      hTxt = text(10,15,'','parent',ax,'Color','white','fontsize',24);
      hWB = waitbar(0,'Writing video');      
      for iF=1:numel(frmsMov)
        f = frmsMov(iF);
        
        imbig = zeros(size(imbig));
        for irow=1:subnr
        for icol=1:subnc
          rowoff = (irow-1)*imnr;
          coloff = (icol-1)*imnc;
          
          imsub = mr(irow,icol).readframe(f);
          imbig((1:imnr)+rowoff,(1:imnc)+coloff) = imsub;

          ptrkSub = ptrk{irow,icol};
          xyf = ptrkSub(:,:,f); % [nptsx2]
          iMov = irow+(icol-1)*subnr;
          for iPt=1:npts
            hLines(iMov,iPt).XData = xyf(iPt,1)+coloff;
            hLines(iMov,iPt).YData = xyf(iPt,2)+rowoff;
          end
        end
        end
        
        if ~isempty(trkfilefull)
          xyf = ptrkFull(:,:,:,f); % [nptsx2xnrep]
          for iPt=1:nptsFull
            hLinesFull(iPt).XData = squeeze(xyf(iPt,1,:));
            hLinesFull(iPt).YData = squeeze(xyf(iPt,2,:));
          end
        end
        
        hIm.CData = imbig;
        hTxt.String = sprintf('%04d',f);
        drawnow
        
        tmpFrame = getframe(hFig);
        vr.writeVideo(tmpFrame);
        waitbar(iF/numel(frmsMov),hWB,sprintf('Wrote frame %d\n',f));
      end
      
      vr.close();
      delete(hTxt);
      delete(hWB);      
    end
    
  end
  
end

