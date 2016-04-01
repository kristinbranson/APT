classdef CPRLabelTracker < LabelTracker
  
  properties
    doHE % logical scalar
    preProcFcn % function handle to preprocessor; caled as ppFcn(obj.td)
    trnPrmFile % training parameters file    
    
    trnData % most recent training data
    trnDataTS % timestamp for trnData
    trnRes % most recent training results
    
    % something for predicted labels on training data
  end
  
  methods 
    
    function obj = CPRLabelTracker(lObj)
      obj@LabelTracker(lObj);
    end
    
    function track(obj)
      
      lObj = obj.lObj;
      
      hWB = waitbar(0);
      hTxt = findall(hWB,'type','text');
      hTxt.Interpreter = 'none';
      
      % Create/preprocess the training data
      td = CPRData(lObj.movieFilesAll,lObj.labeledpos,lObj.labeledpostag,false,...
        'hWaitBar',hWB);
      md = td.MD;
      if obj.doHE
        gHE = categorical(md.movS);
        td.histEq('g',gHE,'hWaitBar',hWB);
      end
      if ~isempty(obj.preProcFcn)
        obj.preProcFcn(td,'hWaitBar',hWB);
      end
      obj.trnData = td;
      obj.trnDataTS = now;
      
      % Read the training parameters
      tp = ReadYaml(obj.trnPrmFile);
      
      td.iTrn = 1:td.N;
      td.summarize('movS',td.iTrn);

      [Is,nChan] = td.getTrnCombinedIs();
      tp.Ftr.nChn = nChan;
      
      delete(hWB);
      
      tr = train(td.pGTTrn,td.bboxesTrn,Is,...
          'modelPrms',tp.Model,...
          'regPrm',tp.Reg,...
          'ftrPrm',tp.Ftr,...
          'initPrm',tp.Init,...
          'prunePrm',tp.Prune,...
          'docomperr',false,...
          'singleoutarg',true);
      obj.trnRes = tr;
    end
    
  end
  
  methods (Static)
    function tdPPJan(td,varargin)
      td.computeIpp([],[],[],'iTrl',1:td.N,'jan',true,varargin{:});
    end
  end
  
end