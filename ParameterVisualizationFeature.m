classdef ParameterVisualizationFeature < ParameterVisualization
  
  properties
    initSuccessful = false;
    
    ftrVizInfo % scalar struct
    hPlot % vector of plot handles output from Features.visualize*. 
          % Set/created during init
  end
  
  methods
    
    function initBase(obj,hAx,lObj,sPrm,propFN)

      obj.initSuccessful = false;
      obj.ftrVizInfo = [];
      deleteValidHandles(obj.hPlot);
      obj.hPlot = [];
            
      if ~lObj.hasMovie
        ParameterVisualization.grayOutAxes(hAx,'No movie available.');
        return;
      end
            
      % Find a labeled frame somewhere
      frm = lObj.currFrame;
      iTgt = lObj.currTarget;
      [tffound,frm,xyLbl] = lObj.labelPosLabeledNeighbor(frm,iTgt);
      if tffound
        mIdx = lObj.currMovIdx;
        % frm,iTgt,xyLbl set
      else
        [tffound,mIdx,frm,iTgt,xyLbl] = lObj.labelFindOneLabeledFrame();
        if ~tffound
          ParameterVisualization.grayOutAxes(hAx,'No labeled frames available.');
          return;
        end
      end      
      szassert(xyLbl,[lObj.nLabelPoints 2]);
      
      % Get the image for the labeled frame, ie (mIdx,frm,iTgt)
      if lObj.currMovIdx==mIdx && lObj.currFrame==frm
        gdata = lObj.gdata;
        im = gdata.image_curr;
        im = im.CData;
      else
        IVIEW = 1;
        if lObj.currMovIdx==mIdx
          mr = lObj.movieReader(IVIEW);
        else
          mfaf = lObj.getMovieFilesAllFullMovIdx(mIdx);
          mr = MovieReader;
          lObj.movieMovieReaderOpen(mr,mfaf,IVIEW);
        end
        im = mr.readframe(frm);
      end            
      
      % We now have im and xyLbl for (mIdx,frm,iTgt)
            
      cla(hAx);
      imshow(im,'Parent',hAx);
      caxis(hAx,'auto');      
      hold(hAx,'on');
      plot(hAx,xyLbl(:,1),xyLbl(:,2),'ro');
      if lObj.hasTrx
        [xTrx,yTrx] = readtrx(lObj.trx,frm,iTgt);
        cropRadius = sPrm.ROOT.Track.MultiTarget.TargetCrop.Radius;
        [roixlo,roixhi,roiylo,roiyhi] = xyRad2roi(xTrx,yTrx,cropRadius);
        axis(hAx,[roixlo roixhi roiylo roiyhi]);
      end
      
      % Viz feature; set .hPlot
      nphyspts = lObj.nPhysPoints;
      nviews = lObj.nview;
      ifo = struct();
      prmFtr = sPrm.ROOT.CPR.Feature;
      switch prmFtr.Type
        case '1lm'
          % generate 'fake' model parameters
          prmModel = struct('nfids',nphyspts,'d',2,'nviews',1);
          ifo.xs = Features.generate1LMforSetParamViz(prmModel,...
            prmFtr.Radius);
          ifo.xLM = reshape(xyLbl(:,1),1,nphyspts,nviews);
          ifo.yLM = reshape(xyLbl(:,2),1,nphyspts,nviews);
          obj.ftrVizInfo = ifo;
          [xF,yF,iView,tmpInfo] = Features.compute1LM(ifo.xs,ifo.xLM,ifo.yLM);
          obj.hPlot = Features.visualize1LM(hAx,xF,yF,iView,tmpInfo,...
            1,1,[0 1 0]);
        otherwise
          disp('XXX TODO 2lm');
      end
      
      obj.initSuccessful = true;
    end

    function updateBase(obj,hAx,lObj,sPrm,propFN)
      if obj.initSuccessful
        ifo = obj.ftrVizInfo;
        prmFtr = sPrm.ROOT.CPR.Feature;
        switch prmFtr.Type
          case '1lm'
            ifo.xs(:,2) = prmFtr.Radius;
            [xF,yF,iView,tmpInfo] = Features.compute1LM(ifo.xs,ifo.xLM,ifo.yLM);
            Features.visualize1LM(hAx,xF,yF,iView,tmpInfo,1,1,[0 1 0],...
              'hPlot',obj.hPlot);
          otherwise
            disp('XXX TODO 2lm');
        end
      end
    end
    
  end
  
end