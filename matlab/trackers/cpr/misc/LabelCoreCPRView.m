classdef LabelCoreCPRView < LabelCore
  properties
    pGT % [nfrm x npt x 2]
    pRep % [nfrm x npt x 2 x nrep]    
    pRepRed % [nfrm x npt x 2]
    nRep;
    colors % [npt x 3]
    
    hPtsGT; % [npt]
    hPtsRep; % [npt x nrep]
    hPtsRepRed; % [npt]
    
    isLabeled % [nfrm], true if pGT(i,:,:) are not nan
    iLabeled % [nlabeled]
  end
  properties
    unsupportedKPFFns = {} ;  % cell array of field names for objects that have general keypressfcn 
                              % callbacks but are not supported for this LabelCore
  end  
  methods
    function obj = LabelCoreCPRView(lObj)
      obj@LabelCore(lObj);
    end
    function delete(obj)
      deleteValidGraphicsHandles(obj.hPtsGT);
      deleteValidGraphicsHandles(obj.hPtsRep);
      deleteValidGraphicsHandles(obj.hPtsRepRed);
    end
  end
  methods
    function setPs(obj,pGT,pTst,pTstRed)
      [nfrm,obj.nPts,d,obj.nRep] = size(pTst);
      assert(isequal(size(pGT),size(pTstRed),[nfrm,obj.nPts,d]));
      assert(d==2);
      
      obj.pGT = pGT;
      obj.pRep = pTst;
      obj.pRepRed = pTstRed;
      obj.colors = jet(obj.nPts);
      
      tfLbled = false(nfrm,1);
      for i = 1:nfrm
        tfLbled(i) = ~isnan(pGT(i,1,1));
      end
      obj.isLabeled = tfLbled;
      obj.iLabeled = find(tfLbled);
      
      obj.hPtsGT = gobjects(obj.nPts,1);
      obj.hPtsRep = gobjects(obj.nPts,obj.nRep);
      obj.hPtsRepRed = gobjects(obj.nPts,1);
      clrs = obj.colors;
      ax = obj.hAx;
      MARKERS = repmat({'o'},1,20);%{'o' 'o' 'o' 'v' 'o' 'v' 'o'};
      PINK = [ 1.0000    0.6000    0.7843];
      for iPt = 1:obj.nPts
        obj.hPtsGT(iPt) = plot(ax,nan,nan,MARKERS{iPt},'MarkerSize',8,'Color',clrs(iPt,:),'MarkerFaceColor',clrs(iPt,:));
        obj.hPtsRepRed(iPt) = plot(ax,nan,nan,MARKERS{iPt},'MarkerSize',8,'Color',PINK,'LineWidth',1);  
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
      pp = squeeze(obj.pRep(iFrm1,:,:,:)); % [nptx2xnRep]
      hGT = obj.hPtsGT;
      hRR = obj.hPtsRepRed;
      hR = obj.hPtsRep;
      for iPt = 1:obj.nPts
        hGT(iPt).XData = obj.pGT(iFrm1,iPt,1);
        hGT(iPt).YData = obj.pGT(iFrm1,iPt,2);
        hRR(iPt).XData = obj.pRepRed(iFrm1,iPt,1);
        hRR(iPt).YData = obj.pRepRed(iFrm1,iPt,2);
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
      tfLbled = obj.isLabeled;
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