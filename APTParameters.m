classdef APTParameters
  properties (Constant)
    PREPROCESS_PARAMETER_FILE = lclInitPreprocessParameterFile();
    TRACK_PARAMETER_FILE = lclInitTrackParameterFile();
    CPR_PARAMETER_FILE = lclInitCPRParameterFile();
    DEEPTRACK_PARAMETER_FILE = lclInitDeepTrackParameterFile();
  end
  methods (Static)
    function tPrm0 = defaultParamsTree
%       tPrmCpr = parseConfigYaml(APTParameters.CPR_PARAMETER_FILE);
%       tPrmDT = parseConfigYaml(APTParameters.DEEPTRACK_PARAMETER_FILE);
%       tPrm0 = tPrmCpr;
%       tPrm0.Children = [tPrm0.Children; tPrmDT.Children];

      tPrmPreprocess = parseConfigYaml(APTParameters.PREPROCESS_PARAMETER_FILE);
      tPrmTrack = parseConfigYaml(APTParameters.TRACK_PARAMETER_FILE);
      tPrmCpr = parseConfigYaml(APTParameters.CPR_PARAMETER_FILE);
      tPrmDT = parseConfigYaml(APTParameters.DEEPTRACK_PARAMETER_FILE);
      tPrm0 = tPrmPreprocess;
      tPrm0.Children = [tPrm0.Children; tPrmTrack.Children;tPrmCpr.Children;tPrmDT.Children];
      tPrm0 = APTParameters.propagateLevelFromLeaf(tPrm0);
    end
    function sPrm0 = defaultParamsStruct
      % sPrm0: "new-style"
      
%       tPrmCpr = parseConfigYaml(APTParameters.CPR_PARAMETER_FILE);
%       sPrmCpr = tPrmCpr.structize();
%       sPrmCpr = sPrmCpr.ROOT;
%       tPrmDT = parseConfigYaml(APTParameters.DEEPTRACK_PARAMETER_FILE);
%       sPrmDT = tPrmDT.structize();
%       sPrmDT = sPrmDT.ROOT;
%       sPrm0 = structmerge(sPrmCpr,sPrmDT);

      tPrmPreprocess = parseConfigYaml(APTParameters.PREPROCESS_PARAMETER_FILE);
      sPrmPreprocess = tPrmPreprocess.structize();
      sPrmPreprocess = sPrmPreprocess.ROOT;
      
      tPrmTrack = parseConfigYaml(APTParameters.TRACK_PARAMETER_FILE);
      sPrmTrack = tPrmTrack.structize();
      sPrmTrack = sPrmTrack.ROOT;
      
      tPrmCpr = parseConfigYaml(APTParameters.CPR_PARAMETER_FILE);
      sPrmCpr = tPrmCpr.structize();
      sPrmCpr = sPrmCpr.ROOT;
      
      tPrmDT = parseConfigYaml(APTParameters.DEEPTRACK_PARAMETER_FILE);
      sPrmDT = tPrmDT.structize();
      sPrmDT = sPrmDT.ROOT;
      
      sPrm0 = structmerge(sPrmPreprocess,sPrmTrack,sPrmCpr,sPrmDT);
    end
    function ppPrm0 = defaultPreProcParamsOldStyle
      sPrm0 = APTParameters.defaultParamsOldStyle();
      ppPrm0 = sPrm0.PreProc;
    end
    function sPrm0 = defaultCPRParamsOldStyle
      sPrm0 = APTParameters.defaultParamsOldStyle();
      sPrm0 = rmfield(sPrm0,'PreProc');
    end
    function [tPrm,minLevel] = propagateLevelFromLeaf(tPrm)
      
      if isempty(tPrm.Children),
        minLevel = tPrm.Data.Level;
        return;
      end
      minLevel = PropertyLevelsEnum('Developer');
      for i = 1:numel(tPrm.Children),
        [tPrm.Children(i),minLevelCurr] = APTParameters.propagateLevelFromLeaf(tPrm.Children(i));
        minLevel = min(minLevel,minLevelCurr);
      end
      tPrm.Data.Level = PropertyLevelsEnum(minLevel);
      
    end
    
    function filteredtree = filterPropertiesByLevel(tree,level)
      
      if tree.Data.Level > level,
        filteredtree = [];
        return;
      end
      filteredtree = TreeNode(tree.Data);
      for i = 1:numel(tree.Children),
        filteredchild = APTParameters.filterPropertiesByLevel(tree.Children(i),level);
        if ~isempty(filteredchild),
          filteredtree.Children = [filteredtree.Children;filteredchild];
        end
      end
      
    end
    
  end
  methods (Static)
    function sPrm0 = defaultParamsOldStyle
      tPrm0 = APTParameters.defaultParamsTree;
      sPrm0 = tPrm0.structize();
      % Use nan for npts, nviews; default parameters do not know about any
      % model
      sPrm0 = CPRParam.new2old(sPrm0,nan,nan);
    end
  end
end

function preprocessParamFile = lclInitPreprocessParameterFile()
if isdeployed
  preprocessParamFile = fullfile(ctfroot,'params_preprocess.yaml');
else
  aptroot = APT.Root;
  preprocessParamFile = fullfile(aptroot,'params_preprocess.yaml');
end
end
function trackParamFile = lclInitTrackParameterFile()
if isdeployed
  trackParamFile = fullfile(ctfroot,'params_track.yaml');
else
  aptroot = APT.Root;
  trackParamFile = fullfile(aptroot,'params_track.yaml');
end
end
function cprParamFile = lclInitCPRParameterFile()
if isdeployed
  cprParamFile = fullfile(ctfroot,'params_cpr.yaml');
  %cprParamFile = fullfile(ctfroot,'params_apt.yaml');
else
  aptroot = APT.Root;
  cprParamFile = fullfile(aptroot,'trackers','cpr','params_cpr.yaml');
  %cprParamFile = fullfile(aptroot,'trackers','cpr','params_apt.yaml');
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
