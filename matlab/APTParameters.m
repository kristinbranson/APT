classdef APTParameters
  properties (Constant)
    PARAM_FILE_SPECS = lclInitParamFileSpecs();
  end
  properties (Constant, Access=private)
    % This property stores parsetrees for yamls so that yaml files only 
    % need to be parsed once.
    %
    % This property is private as these trees are handles and mutable.
    % Use getParamTrees to access copies of these trees.
    PARAM_FILES_TREES = lclInitParamFilesTrees(); 
  end
  methods (Static)
    function trees = getParamTrees(subtree)
      % Get a copy of PARAM_FILES_TREES. A copy is made for safety so the
      % user can mutate as desired.
      %
      % subtree: optional char. fieldname of PARAM_FILES_TREES. If
      % specified, the returned tree is a copy of
      % PARAM_FILES_TREES.(subtree). 
      
      if nargin==0
        trees = APTParameters.PARAM_FILES_TREES;
        for f=fieldnames(trees)',f=f{1}; %#ok<FXSET>
          trees.(f).tree = trees.(f).tree.copy();
        end
      else        
        trees = APTParameters.PARAM_FILES_TREES.(subtree).tree.copy();
      end
    end
    
    function tPrm0 = defaultParamsTree(varargin)
      incDLNetSpecific = myparse(varargin,...
        'incDLNetSpecific',true...
        );
      
      trees = APTParameters.getParamTrees;
      tPrmPreprocess = trees.preprocess.tree;
      tPrmTrack = trees.track.tree;
      tPrmCpr = trees.cpr.tree;
      tPrmDT = trees.deeptrack.tree;
      tPrmPostProc = trees.postprocess.tree;
      
      if incDLNetSpecific
        [~,nets] = enumeration('DLNetType');
        tPrmDeepNets = cellfun(@(x)trees.(x).tree,nets,'uni',0);
        tPrmDeepNets = cat(1,tPrmDeepNets{:});
        tPrmDeepNetsChildren = cat(1,tPrmDeepNets.Children);    
        tPrmDT.Children.Children = [tPrmDT.Children.Children; ...
                                    tPrmDeepNetsChildren];
      end
      
      tPrm0 = tPrmPreprocess;
      tPrm0.Children = [tPrm0.Children; tPrmTrack.Children;...
        tPrmCpr.Children; tPrmDT.Children; tPrmPostProc.Children];
      tPrm0 = APTParameters.propagateLevelFromLeaf(tPrm0);
      tPrm0 = APTParameters.propagateRequirementsFromLeaf(tPrm0);
    end
    function sPrm0 = defaultParamsStruct
      tPrm0 = APTParameters.defaultParamsTree('incDLNetSpecific',false);
      sPrm0 = tPrm0.structize();
      sPrm0 = sPrm0.ROOT;
    end
    function sPrm0 = defaultParamsStructAll
      tPrm0 = APTParameters.defaultParamsTree;
      sPrm0 = tPrm0.structize();
    end
    
    function dlNetTypes = getDLNetTypes
      dlNetTypes = enumeration('DLNetType')';
%       mc = ?DLNetType;
%       dlNetTypes = cellfun(@(x) DLNetType(x),{mc.EnumerationMemberList.Name},'uni',0);
%       dlNetTypes = cat(2,dlNetTypes{:});
    end
    
    function dlNetTypesPretty = getDLNetTypesPretty
      mc = ?DLNetType;
      dlNetTypesPretty = cellfun(@(x) DLNetType(x).prettyString,{mc.EnumerationMemberList.Name},'Uni',0);
    end
    
    function sPrmDTcommon = defaultParamsStructDTCommon
      tPrm = APTParameters.getParamTrees('deeptrack');
      sPrm = tPrm.structize();
      sPrmDTcommon = sPrm.ROOT.DeepTrack;
    end
    function sPrmDTspecific = defaultParamsStructDT(nettype)
      tPrm = APTParameters.getParamTrees(char(nettype));
      sPrmDTspecific = tPrm.structize();
      sPrmDTspecific = sPrmDTspecific.ROOT;
      fld = fieldnames(sPrmDTspecific);
      assert(isscalar(fld)); % eg {'MDN'}
      fld = fld{1};
      sPrmDTspecific = sPrmDTspecific.(fld);
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
    function [tPrm,rqts] = propagateRequirementsFromLeaf(tPrm)
      
      if isempty(tPrm.Children),
        rqts = tPrm.Data.Requirements;
        return;
      end
      for i = 1:numel(tPrm.Children),
        [tPrm.Children(i),rqts1] = APTParameters.propagateRequirementsFromLeaf(tPrm.Children(i));
        if i == 1,
          rqts = rqts1;
        else
          rqts = intersect(rqts,rqts1);
        end
      end
      tPrm.Data.Requirements = rqts;
      
    end
    function filterPropertiesByLevel(tree,level)
      
      if isempty(tree.Children),
        tree.Data.Visible = tree.Data.Visible && tree.Data.Level <= level;
        return;
      end
      
      if tree.Data.Visible,
        tree.Data.Visible = false;
        for i = 1:numel(tree.Children),
          APTParameters.filterPropertiesByLevel(tree.Children(i),level);
          tree.Data.Visible = tree.Data.Visible || tree.Children(i).Data.Visible;
        end
      end
      
    end
    
    function tree = setAllVisible(tree)
      
      tree.Data.Visible = true;
      for i = 1:numel(tree.Children),
        APTParameters.setAllVisible(tree.Children(i));
      end
      
    end
    
    function tree = filterPropertiesByCondition(tree,labelerObj,varargin)
    
      if isempty(tree.Children),
        
        [trackerAlgo,hasTrx,trackerIsDL] = myparse(varargin,...
          'trackerAlgo',[],'hasTrx',[],'trackerIsDL',[]);
        if isempty(trackerAlgo),
          trackerAlgo = labelerObj.trackerAlgo;
        end
        if isempty(hasTrx),
          hasTrx = labelerObj.hasTrx;
        end
        if isempty(trackerIsDL),
          trackerIsDL = labelerObj.trackerIsDL;
        end
      
        if ismember('isCPR',tree.Data.Requirements) && ~strcmpi(trackerAlgo,'cpr'),
          tree.Data.Visible = false;
        elseif ismember('hasTrx',tree.Data.Requirements) && ~hasTrx,
          tree.Data.Visible = false;
        elseif ismember('isMultiView',tree.Data.Requirements) && ~labelerObj.isMultiView
          tree.Data.Visible = false;
        elseif ismember('isDeepTrack',tree.Data.Requirements) && ~trackerIsDL,
          tree.Data.Visible = false;
        else
          dlnets = enumeration('DLNetType');
          for i=1:numel(dlnets)
            net = dlnets(i);
            if ismember(net,tree.Data.Requirements) && ~strcmp(trackerAlgo,net)
              tree.Data.Visible = false;
              break;
            end
          end
        end
        
        return;
        
      end
        
      if tree.Data.Visible,
        tree.Data.Visible = false;
        for i = 1:numel(tree.Children),
          APTParameters.filterPropertiesByCondition(tree.Children(i),labelerObj,varargin{:});
          tree.Data.Visible = tree.Data.Visible || tree.Children(i).Data.Visible;
        end
      end
      
    end
    
    function [tPrm] = removeFilteredProperties(tPrm)
      
      if ~tPrm.Data.Visible,
        tPrm = [];
        return;
      end
      
      if isempty(tPrm.Children),
        if ~tPrm.Data.AffectsTraining,
          tPrm = [];
        end
        return;
      end

      doremove = false(1,numel(tPrm.Children));
      for i = 1:numel(tPrm.Children),
        res = APTParameters.removeFilteredProperties(tPrm.Children(i));
        if ~isempty(res),
          tPrm.Children(i) = res;
        else
          doremove(i) = true;
        end
          
      end
      tPrm.Children(doremove) = [];
      
    end
    
    function [sPrmFilter,tPrm] = filterStructPropertiesByCondition(sPrm,varargin)
      
      [tPrm,leftovers] = myparse_nocheck(varargin,'tree',[]);
      if isempty(tPrm),
        tPrm = APTParameters.defaultParamsTree;
      end
      tPrm.structapply(sPrm);
      tPrm = APTParameters.filterPropertiesByCondition(tPrm,[],leftovers{:});
      tPrm = APTParameters.removeFilteredProperties(tPrm);
      sPrmFilter = tPrm.structize();
      
    end
    
    function [v,sPrmFilter0,sPrmFilter1] = isEqualFilteredStructProperties(sPrm0,sPrm1,varargin)
      [tPrm0,leftovers] = myparse_nocheck(varargin,'tree',[]);
      if isempty(tPrm0),
        tPrm0 = APTParameters.defaultParamsTree;
      end
      tPrm1 = tPrm0.copy();
      sPrmFilter0 = APTParameters.filterStructPropertiesByCondition(sPrm0,'tree',tPrm0,leftovers{:});
      sPrmFilter1 = APTParameters.filterStructPropertiesByCondition(sPrm1,'tree',tPrm1,leftovers{:});
      v = isequaln(sPrmFilter0,sPrmFilter1);

    end
    
    function [sPrm,tfChangeMade] = enforceConsistency(sPrm)
      % enforce constraints amongst complete NEW-style parameters
      %
      % sPrm (in): input full new-style param struct
      % 
      % sPrm (out): output params, possibly massaged. If massaged, warnings
      % thrown
      % tfChangeMade: if any changes made
      
      % TODO: reconcile with paramChecker, ParameterTreeConstraint
      
      tfChangeMade = false;
      
      if sPrm.ROOT.ImageProcessing.MultiTarget.TargetCrop.AlignUsingTrxTheta && ...
         strcmp(sPrm.ROOT.CPR.RotCorrection.OrientationType,'fixed')
        warningNoTrace('CPR OrientationType cannot be ''fixed'' if aligning target crops using trx.theta. Setting CPR OrientationType to ''arbitrary''.');
        sPrm.ROOT.CPR.RotCorrection.OrientationType = 'arbitrary';
        tfChangeMade = true;
      end
    end
    
    % convert from all parameters to various subsets 
    
    % all parameters to old preproc parameters
    function v = all2PreProcParams(sPrmAll)
      sPrmPPandCPR = sPrmAll;
      sPrmPPandCPR.ROOT = rmfield(sPrmPPandCPR.ROOT,'DeepTrack');
      [sPrmPPandCPRold] = CPRParam.new2old(sPrmPPandCPR,5,1); % we won't use npoints or nviews
      v = sPrmPPandCPRold.PreProc;
    end
    
    % all parameters to new preproc parameters
    function v = all2PreProcParamsNew(sPrmAll)
      v = sPrmAll.ROOT.ImageProcessing;
    end
    
    % all parameters to common dl parameters
    function v = all2TrackDLParams(sPrmAll)
      sPrmDT = sPrmAll.ROOT.DeepTrack;
      dlNetTypesPretty = APTParameters.getDLNetTypesPretty;
      v = rmfield(sPrmDT,intersect(fieldnames(sPrmDT),dlNetTypesPretty));
    end

%     function v = all2DLCacheDir(sPrmAll)
%       v = sPrmAll.ROOT.DeepTrack.Saving.CacheDir;
%     end
    
    % all parameters to specific dl parameters for input netType
    function v = all2DLSpecificParams(sPrmAll,netType)
      if ~ischar(netType),
        netType = netType.prettyString;
      end
      v = sPrmAll.ROOT.DeepTrack.(netType);
    end
    
    % all parameters to new cpr parameters
    function v = all2CPRParamsNew(sPrmAll)
      v = sPrmAll.ROOT.CPR;
    end

    % all parameters to old format cpr parameters
    function v = all2CPRParams(sPrmAll,nPhysPoints,nviews)
      if nargin < 2,
        nPhysPoints = 2;
      end
      if nargin < 3,
        nviews = 1;
      end
      sPrmPPandCPR = sPrmAll;
      sPrmPPandCPR.ROOT = rmfield(sPrmPPandCPR.ROOT,'DeepTrack');
      [sPrmPPandCPRold] = CPRParam.new2old(sPrmPPandCPR,nPhysPoints,nviews); % we won't use npoints or nviews
      v = rmfield(sPrmPPandCPRold,'PreProc');
    end

    % set subsets of all parameters

    % set old format preproc parameters
    function sPrmAll = setPreProcParams(sPrmAll,sPrmPPOld)
      % convert old to new format
      sPrmPPNew = CPRParam.old2newPPOnly(sPrmPPOld);
      sPrmAll = structoverlay(sPrmAll,sPrmPPNew);
    end

    % set new format preproc parameters
    function sPrmAll = setPreProcParamsNew(sPrmAll,sPrmPPNew)
      sPrmAll.ROOT.ImageProcessing = sPrmPPNew;
    end
    
    % set common dl parameters
    function sPrmAll = setTrackDLParams(sPrmAll,sPrmDT)
      sPrmAll.ROOT.DeepTrack = structoverlay(sPrmAll.ROOT.DeepTrack,sPrmDT);      
    end
    
    % set specific dl parameters for input netType
    function sPrmAll = setDLSpecificParams(sPrmAll,netType,sPrmDT)
      sPrmAll.ROOT.DeepTrack.(netType) = sPrmDT;
    end
    
    % set new cpr parameters
    function sPrmAll = setCPRParamsNew(sPrmAll,sPrmCPR)
      sPrmAll.ROOT.CPR = sPrmCPR;
    end
    
    % set old cpr parameters
    function sPrmAll = setCPRParams(sPrmAll,sPrmCPROld)
      [sPrmCPR,sPrmAll.ROOT.Track.ChunkSize] = CPRParam.old2newCPROnly(sPrmCPROld);
      sPrmAll.ROOT.CPR = sPrmCPR;
    end
    
    function sPrmAll = setNFramesTrackParams(sPrmAll,obj)      
      sPrmAll.ROOT.Track.NFramesSmall = obj.trackNFramesSmall;
      sPrmAll.ROOT.Track.NFramesLarge = obj.trackNFramesLarge;
      sPrmAll.ROOT.Track.NFramesNeighborhood = obj.trackNFramesNear;
    end
    
    function testConversions(sPrmAll)
      
      sPrmPPOld = APTParameters.all2PreProcParams(sPrmAll);
      sPrmAll1 = APTParameters.setPreProcParams(sPrmAll,sPrmPPOld);
      assert(isequaln(sPrmAll,sPrmAll1));
      sPrmPPNew = APTParameters.all2PreProcParamsNew(sPrmAll);
      sPrmAll1 = APTParameters.setPreProcParamsNew(sPrmAll,sPrmPPNew);
      assert(isequaln(sPrmAll,sPrmAll1));
      sPrmTrackDL = APTParameters.all2TrackDLParams(sPrmAll);
      sPrmAll1 = APTParameters.setTrackDLParams(sPrmAll,sPrmTrackDL);
      assert(isequaln(sPrmAll,sPrmAll1));
      sPrmTrackDLMDN = APTParameters.all2DLSpecificParams(sPrmAll,'MDN');
      sPrmAll1 = APTParameters.setDLSpecificParams(sPrmAll,'MDN',sPrmTrackDLMDN);
      assert(isequaln(sPrmAll,sPrmAll1));
      sPrmCPROld = APTParameters.all2CPRParams(sPrmAll);
      sPrmAll1 = APTParameters.setCPRParams(sPrmAll,sPrmCPROld);
      assert(isequaln(sPrmAll,sPrmAll1));
      sPrmCPRNew = APTParameters.all2CPRParamsNew(sPrmAll);
      sPrmAll1 = APTParameters.setCPRParamsNew(sPrmAll,sPrmCPRNew);
      assert(isequaln(sPrmAll,sPrmAll1));
      
    end
    
    function tfEqual = isEqualTrackDLParams(sPrm0,sPrm1)
      sPrm0 = APTParameters.all2TrackDLParams(sPrm0);
      sPrm1 = APTParameters.all2TrackDLParams(sPrm1);
      tfEqual = isequaln(sPrm0,sPrm1);
      
    end
    
    function tfEqual = isEqualDeepTrackDataAugParams(sPrm0,sPrm1)
      tfEqual = false;
      try
        sPrm0 = sPrm0.ROOT.DeepTrack.DataAugmentation;
        sPrm1 = sPrm1.ROOT.DeepTrack.DataAugmentation;
        tfEqual = isequaln(sPrm0,sPrm1);
      catch ME,
        warning('Could not find data augmentation parameters:\n%s',getReport(ME));
      end
    end
    
    function tfEqual = isEqualPreProcParams(sPrm0,sPrm1)
      sPrmPreProc0 = APTParameters.all2PreProcParams(sPrm0);
      sPrmPreProc1 = APTParameters.all2PreProcParams(sPrm1);
      tfEqual = isequaln(sPrmPreProc0,sPrmPreProc1);
    end
    
    function tfEqual = isEqualPostProcParams(sPrm0,sPrm1)
      tfEqual = isequaln(sPrm0.ROOT.PostProcess,sPrm1.ROOT.PostProcess);
    end
    
    function [tfOK,msgs] = checkParams(sPrm)
      tfOK = true;
      msgs = {};
      ppPrms = APTParameters.all2PreProcParams(sPrm);
      if ppPrms.histeq
        if ppPrms.BackSub.Use
          tfOK = false;
          msgs{end+1} = 'Histogram Equalization and Background Subtraction cannot both be enabled';
        end
        if ppPrms.NeighborMask.Use
          tfOK = false;
          msgs{end+1} = 'Histogram Equalization and Neighbor Masking cannot both be enabled';
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

function s = lclInitParamFileSpecs()
s = struct(...
  'preprocess',{{'params_preprocess.yaml'}},...
  'track',{{'params_track.yaml'}},...
  'cpr',{{'trackers','cpr','params_cpr.yaml'}},...
  'deeptrack',{{'trackers','dt','params_deeptrack.yaml'}},...
  'postprocess',{{'params_postprocess.yaml'}});

nettypes = enumeration('DLNetType');
for i=1:numel(nettypes)
  netty = nettypes(i);
  s.(char(netty)) = {'trackers','dt',netty.paramFileShort};
end
end

function s = lclInitParamFilesTrees()
%disp('APTParams init files trees');
specs = APTParameters.PARAM_FILE_SPECS;
aptroot = APT.getRoot;
[~,nets] = enumeration('DLNetType');

s = struct();
for f=fieldnames(specs)',f=f{1}; %#ok<FXSET>  
  s.(f).yaml = fullfile(aptroot,'matlab',specs.(f){:});
  yamlcontents = parseConfigYaml(s.(f).yaml);
  if any(strcmp(f,nets))
    % AL 20190711: automatically create requirements for all deep net
    %   param trees
    yamlcontents.traverse(@(x)set(x.Data,'Requirements',...
      {f,'isDeepTrack'}));
  end
  s.(f).tree = yamlcontents;
end
end
