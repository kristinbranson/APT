classdef TrainParams < handle
  
  properties
    Name
    
    USE_AL_CORRECTION = false;
    
    cpr_type = 'noocclusion';
    model_type = 'FlyBubble1';
    model_nfids = 7;
    model_d = 2;

    ftr_type = '2lm';
    ftr_gen_radius = 1;
    
    cascade_depth = 50;

    ncrossvalsets = 1;
    naugment = 50;
    nsample_std = 1000;
    nsample_cor = 5000;
    
    nftrs_test_perfern = 400;
    
    prunePrm = struct(...
      'prune',0,...
      'numInit',50,... % sometimes uesd for number of testing replicates 
      'usemaxdensity',1,... % replication-collapse during testing
      'maxIter',2,...
      'th',0.5000,...
      'tIni',10,...
      'maxdensity_sigma',5);
  end
  
  methods
    
    function pv = getPVs(obj)
      warnst = warning('off','MATLAB:structOnObject');
      s = struct(obj);
      warning(warnst);
      s = rmfield(s,'Name');
      fns = fieldnames(s);
      vals = struct2cell(s);
      
      X = [fns vals]';
      pv = X(:);      
    end
    
  end
end