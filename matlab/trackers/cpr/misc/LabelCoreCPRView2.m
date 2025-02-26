classdef LabelCoreCPRView2 < LabelCore
  properties
    xyRep % [nfrm x npt x 2 x nrep]    
    xyRepRed % [nfrm x npt x 2]
    nRep;
    
    hPtsRep; % [npt x nrep]
    hPtsRepRed; % [npt]
    
    tfFrmLbled;
  end
  methods
    function obj = LabelCoreCPRView2(lObj)
      obj@LabelCore(lObj);
      
      [~,nPtsLbl] = obj.labeler.labelPosLabeledFramesStats();
      tf = nPtsLbl>0;
      obj.tfFrmLbled = tf;
      fprintf(1,'%d out of %d frames labeled.\n',nnz(tf),numel(tf));
    end
    function delete(obj)
      deleteValidGraphicsHandles(obj.hPtsRep);
      deleteValidGraphicsHandles(obj.hPtsRepRed);
    end
  end
  methods
    function setPs(obj,pTst,pTstRed)
      % Set tracked/external positions
      
      [nfrm,nPtsTmp,d,obj.nRep] = size(pTst);
      assert(nPtsTmp==obj.nPts);
      assert(isequal(size(pTstRed),[nfrm,obj.nPts,d]));
      assert(d==2);
      
      obj.xyRep = pTst;
      obj.xyRepRed = pTstRed;
      
      obj.hPtsRep = gobjects(obj.nPts,obj.nRep);
      obj.hPtsRepRed = gobjects(obj.nPts,1);
      clrs = obj.labeler.labelPointsPlotInfo.Colors;
      ax = obj.hAx;
      MARKERS = repmat({'s'},1,obj.nPts);
      %PINK = [ 1.0000    0.6000    0.7843];
      for iPt = 1:obj.nPts
        %obj.hPtsGT(iPt) = plot(ax,nan,nan,MARKERS{iPt},'MarkerSize',8,'Color',clrs(iPt,:),'MarkerFaceColor',clrs(iPt,:));
        obj.hPtsRepRed(iPt) = plot(ax,nan,nan,MARKERS{iPt},'MarkerSize',12,...
          'Color',clrs(iPt,:),'LineWidth',2);  
        argsRep = {nan,nan,'o','MarkerSize',2,'Color',clrs(iPt,:)};
        for iRep = 1:obj.nRep
          obj.hPtsRep(iPt,iRep) = plot(ax,argsRep{:},'UserData',[iPt iRep]); 
        end
      end
      uistack(obj.hPtsRepRed,'top');
    end
    function newFrame(obj,iFrm0,iFrm1,iTgt)
      obj.newFrameAndTarget(iFrm0,iFrm1,iTgt,iTgt);
    end
    function newFrameAndTarget(obj,iFrm0,iFrm1,iTgt0,iTgt1)
      assert(iTgt1==1);
      [tf,lpos] = obj.labeler.labelPosIsLabeled(iFrm1,iTgt1);
      if tf
        obj.assignLabelCoords(lpos);
      end
        
      pp = squeeze(obj.xyRep(iFrm1,:,:,:)); % [nptx2xnRep]
      %hGT = obj.hPts;
      hRR = obj.hPtsRepRed;
      hR = obj.hPtsRep;
      for iPt = 1:obj.nPts
%         hGT(iPt).XData = obj.pGT(iFrm1,iPt,1);
%         hGT(iPt).YData = obj.pGT(iFrm1,iPt,2);
        hRR(iPt).XData = obj.xyRepRed(iFrm1,iPt,1);
        hRR(iPt).YData = obj.xyRepRed(iFrm1,iPt,2);
        for iRep = 1:obj.nRep
          hR(iPt,iRep).XData = pp(iPt,1,iRep);
          hR(iPt,iRep).YData = pp(iPt,2,iRep);          
        end
      end
    end
    function kpf(obj,~,evt)
      key = evt.Key;
      %modifier = evt.Modifier;
      lObj = obj.labeler;
      f = lObj.currFrame;
      nf = lObj.nframes;
      tfLbled = obj.tfFrmLbled;
      switch key
        case {'rightarrow' 'd' 'equal'}          
          f = f+1;
          while f<=nf
            if tfLbled(f)
              lObj.setFrameGUI(f);
              return;
            end
            f = f+1;
          end
        case {'leftarrow' 'a' 'hyphen'}
          f = f-1;
          while f>0
            if tfLbled(f)
              lObj.setFrameGUI(f);
              return;
            end
            f = f-1;
          end
      end
    end
  end
end