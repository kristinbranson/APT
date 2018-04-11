classdef APTParameters
  properties (Constant)
    CPR_PARAMETER_FILE = lclInitCPRParameterFile();
    DEEPTRACK_PARAMETER_FILE = lclInitDeepTrackParameterFile();
  end
  methods (Static)
    function tPrm0 = defaultParamsTree
      tPrmCpr = parseConfigYaml(APTParameters.CPR_PARAMETER_FILE);
      tPrmDT = parseConfigYaml(APTParameters.DEEPTRACK_PARAMETER_FILE);
      tPrm0 = tPrmCpr;
      tPrm0.Children = [tPrm0.Children; tPrmDT.Children];
    end
    function sPrm0 = defaultParamsStruct
      % sPrm0: "new-style"
      tPrmCpr = parseConfigYaml(APTParameters.CPR_PARAMETER_FILE);
      sPrmCpr = tPrmCpr.structize();
      sPrmCpr = sPrmCpr.ROOT;
      tPrmDT = parseConfigYaml(APTParameters.DEEPTRACK_PARAMETER_FILE);
      sPrmDT = tPrmDT.structize();
      sPrmDT = sPrmDT.ROOT;
      sPrm0 = structmerge(sPrmCpr,sPrmDT);      
    end
    function ppPrm0 = defaultPreProcParamsOldStyle
      sPrm0 = APTParameters.defaultParamsOldStyle();
      ppPrm0 = sPrm0.PreProc;
    end
    function sPrm0 = defaultCPRParamsOldStyle
      sPrm0 = APTParameters.defaultParamsOldStyle();
      sPrm0 = rmfield(sPrm0,'PreProc');
    end    
  end
  methods (Static)
    function sPrm0 = defaultParamsOldStyle
      tPrm0 = parseConfigYaml(APTParameters.CPR_PARAMETER_FILE);
      sPrm0 = tPrm0.structize();
      % Use nan for npts, nviews; default parameters do not know about any
      % model
      sPrm0 = CPRParam.new2old(sPrm0,nan,nan);
    end
  end
end

function cprParamFile = lclInitCPRParameterFile()
if isdeployed
  cprParamFile = fullfile(ctfroot,'params_apt.yaml');
else
  aptroot = APT.Root;
  cprParamFile = fullfile(aptroot,'trackers','cpr','params_apt.yaml');
end
end
function dtParamFile = lclInitDeepTrackParameterFile()
if isdeployed
  dtParamFile = fullfile(ctfroot,'params_deeptrack.yaml');
else
  aptroot = APT.Root;
  dtParamFile = fullfile(aptroot,'trackers','dt','params_deeptrack.yaml');
end
end
