classdef RF
  
  methods (Static)
    
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
            if tfhi && tflo
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
      % Generate Frame-Point md table
      % 
      % lpos: [npttot x d x nfrm] labeledpos
      % lpostag: [npttot x nfrm] labeledpostag
      %
      % tFrmPts: Frame-Pt table
      
      d = 2;
      [~,nptphys,nview,nfrm] = RF.lposDim(lpos);
      lpos = reshape(lpos,[nptphys nview 2 nfrm]);
      lpostag = reshape(lpostag,[nptphys nview nfrm]);

      
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
      
      for f=1:nfrm
        tf2VwLbledAny = false(1,nptphys);
        tfVwLbledAnyPt = cell(1,nptphys);
        occStatusPt = cell(1,nptphys);
        for ippt = 1:nptphys
          lposptfrm = squeeze(lpos(ippt,:,:,f));
          ltagptfrm = squeeze(lpostag(ippt,:,f));
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
      
      tFP = table(frm,npts2VwLbl,ipts2VwLbl,tfVws2VwLbl,occ2VwLbl);
    end
    
    function tFPaug = recon3D(tFP,lpos,crig2)
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
      
      [~,nphyspt,nview,nfrm] = RF.lposDim(lpos);
      lpos = reshape(lpos,[nphyspt nview 2 nfrm]);
      % szassert(lpos,[nptphys nview 2 nfrm]);
      
      tFP18 = tFP(tFP.npts2VwLbl==nphyspt,:);
      nRows = size(tFP18,1);
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
        frm = tFP18.frm(iRow);
        XLrow = nan(3,nphyspt); % first dim is x-y-z dimensions
        errRE = nan(3,nphyspt); % first dim is L-R-B views
        %   XLlrrow = nan(3,nphyspt);
        for iPt = 1:nphyspt
          lposPt = squeeze(lpos(iPt,:,:,frm));
          szassert(lposPt,[nview 2]);
          yL = lposPt(1,[2 1]);
          yR = lposPt(2,[2 1]);
          yB = lposPt(3,[2 1]);
          tfViewsLabeled = tFP18.tfVws2VwLbl{iRow}(:,iPt);
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
      
      tFPaug = [tFP18 table(XL,err)];
    end
    
  end
  
  methods
    
    function makeTrkMovie2D(movfile,trkfile,movout,varargin)
      % Make 'results movie'
      % 
      % movfile: char or cellstr array
      % trkfile: char or cellstr array
      %
      % If cellstr arrays, movfile and trkfile must have the same size.
      % 
      % movout: char

      trkfilefull = myparse(varargin,...
        'trkfilefull',[]);
      
      FRAMERATE = 24;
      GAMMA = .2;
      mgray = gray(256);
      mgray2 = imadjust(mgray,[],[],GAMMA);
      COLORS = {...
        [1 0 0] [1 0 0] [1 0 0]; ...
        [1 1 0] [1 1 0] [1 1 0]; ...
        [0 1 0] [0 1 0] [0 1 0]; ...
        [0 1 1] [0 1 1] [0 1 1]; ...
        [1 0 1] [1 0 1] [1 0 1]; ...
        [1 204/255 77/255] [1 204/255 77/255] [1 204/255 77/255]};
      
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
          'Color',COLORS{iPt});
      end
      end
      if ~isempty(trkfilefull)
        trkfull = load(trkfilefull,'-mat');
        ptrkFull = trkfull.pTrkFull;
        nptsFull = size(ptrkFull,1);
        for iPt=1:nptsFull
          hLinesFull(iPt,1) = plot(ax,nan,nan,'.',...
            'markersize',10,'Color',COLORS{iPt});
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

