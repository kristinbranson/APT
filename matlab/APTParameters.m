classdef APTParameters
  properties (Constant)
    % This property stores parsetrees for jsons so that json files only 
    % need to be parsed once.
    %
    % This property is private as these trees are handles and mutable.
    % Use getParamTrees to access copies of these trees.
    PARAM_FILES_TREES = APTParameters.paramFilesTrees() ;
    maDetectPath = 'ROOT.MultiAnimalDetect';
    maDetectNetworkPath = [APTParameters.maDetectPath,'.DeepTrack'];
    posePath = 'ROOT.DeepTrack';
    deepSharedPath = 'ROOT.DeepTrackShared';
    detectDataAugPath = [APTParameters.maDetectPath,'.DeepTrack.DataAugmentation'];
    poseDataAugPath = [APTParameters.posePath,'.DataAugmentation'];
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
      % tPrm0 = defaultParamsTree(...)
      % Returns a TreeNode with the default parameter hierarchy. 
      % These are constructed from concatenating together stuff in
      % PARAM_FILES_TREES. If you add new .json files, they also need to be
      % included here. 
      incDLNetSpecific = myparse(varargin,...
        'incDLNetSpecific',true...
        );
      
      trees = APTParameters.getParamTrees() ;
      tPrmPreprocess = trees.preprocess.tree;
      tPrmTrack = trees.track.tree;
      tPrmCpr = trees.cpr.tree;
      tPrmMA = trees.ma.tree;
      tPrmDetect = trees.madetect.tree;
      tPrmDTShared = trees.deeptrack_shared.tree;
      tPrmDT = trees.deeptrack.tree;
      tPrmPostProc = trees.postprocess.tree;
      
      if incDLNetSpecific
        [nettypes,nets] = enumeration('DLNetType');
        netisMA = logical([nettypes.isMultiAnimal]);
        idxisMA = find(netisMA); 
                
        tPrmDeepNets0 = cellfun(@(x)trees.(x).tree,nets,'uni',0);
        tPrmDeepNetsTopLevelDT = cat(1,tPrmDeepNets0{:});
        tPrmDeepNetsMADetect = tPrmDeepNets0(netisMA);
        % deep copy these nodes as they will be placed under .Detect and
        % will have different requirements
        tPrmDeepNetsMADetect = cellfun(@copy,tPrmDeepNetsMADetect,'uni',0);
        tPrmDeepNetsMADetect = cat(1,tPrmDeepNetsMADetect{:});
        
        % these MA nets will be added to the top-level .DeepTrack.
        % MA nets placed here apply only to bottom-up MA-tracking.
        for i=idxisMA(:)'
          tPrmDeepNetsTopLevelDT(i).traverse(...
                        @(x)x.Data.addRequirement('isBotUp'));
        end
        tPrmDeepNetsChildren = cat(1,tPrmDeepNetsTopLevelDT.Children);    
        tPrmDT.Children.Children = [tPrmDT.Children.Children; ...
                                    tPrmDeepNetsChildren];
        
        % these MA nets will be added under .Detect and apply only to 
        % top-down tracking (stage1)
        arrayfun(@(x)x.traverse(@(y)y.Data.addRequirement('isTopDown')), ...
          tPrmDeepNetsMADetect);
        tPrmDeepNetsMAChildren = cat(1,tPrmDeepNetsMADetect.Children);
        
        tPrmMADetectDT = tPrmDetect.findnode(APTParameters.maDetectNetworkPath);
        tPrmMADetectDT.Children = ...
          [tPrmMADetectDT.Children; tPrmDeepNetsMAChildren];

        tPrmDTShared = tPrmDTShared.findnode(APTParameters.deepSharedPath);
      end
      
      tPrm0 = tPrmPreprocess;
      tPrm0.Children = [tPrm0.Children; tPrmCpr.Children; tPrmMA.Children; tPrmDTShared; tPrmDetect.Children; tPrmDT.Children; tPrmPostProc.Children; tPrmTrack.Children];
      tPrm0 = APTParameters.propagateLevelFromLeaf(tPrm0);
      tPrm0 = APTParameters.propagateRequirementsFromLeaf(tPrm0);
    end

    function tPrm0 = defaultTrackParamsTree(varargin)
      % tPrm0 = defaultTrackParamsTree(...)
      % Returns the paramter tree for tracking parameters (AffectsTraining
      % == false), with default values. 
      tPrm0 = APTParameters.defaultParamsTree(varargin{:});
      APTParameters.setAllVisible(tPrm0);
      APTParameters.filterPropertiesByAffectsTraining(tPrm0,false);
    end
        
    function v = all2TrackParams(sPrmAll,varargin)
      % v = all2TrackParams(sPrmAll,'outputformat',outputformat)
      % Inputs a struct with all parameters (sPrmAll), and selects and
      % returns those related to tracking (AffectsTraining == false)
      % Optional inputs:
      % outputformat': 'struct' or 'tree'
      [outputformat,rest] = myparse_nocheck(varargin,'outputformat','struct');
      tree = APTParameters.defaultTrackParamsTree(rest{:});
      tree.structapply(sPrmAll);
      if strcmp(outputformat,'struct'),
        v = tree.structize();
      else
        v = tree;
      end
    end

    function sPrm0 = defaultParamsStruct
      % sPrm0 = defaultParamsStruct
      % Returns a struct with the default parameter hierarchy. This differs
      % from defaultParamsStructAll in that it doesn't include the
      % network-architecture specific parameters
      tPrm0 = APTParameters.defaultParamsTree('incDLNetSpecific',false);
      sPrm0 = tPrm0.structize();
      sPrm0 = sPrm0.ROOT;
    end

    function sPrm0 = defaultParamsStructAll
      % sPrm0 = defaultParamsStructAll
      % Returns a struct with the default parameter hierarchy, including
      % network-architecture specific parameters
      tPrm0 = APTParameters.defaultParamsTree;
      sPrm0 = tPrm0.structize();
    end    
 
    function dlNetTypesPretty = getDLNetTypesPretty
      % dlNetTypesPretty = getDLNetTypesPretty
      % Returns names of all DL network types
      mc = ?DLNetType;
      dlNetTypesPretty = cellfun(@APTParameters.getNetworkParamField,...
        {mc.EnumerationMemberList.Name},'Uni',0);
    end
    
    function f = getNetworkParamField(nettype)
      % f = getNetworkParamField(nettype)
      % returns parameter field name for network of type nettype
      if ~ischar(nettype)
        nettype = char(nettype);
      end
      % first non-ROOT top-level field in parameter json 
      f = APTParameters.PARAM_FILES_TREES.(nettype).tree.Children(1).Data.Field;      
    end
    
    function sPrmDT = defaultParamsStructDeepTrack(varargin)
      % sPrmDT =
      % defaultParamsStructDeepTrack('includeshared',includeshared)
      % Returns the DeepTrack parameter tree as a struct. If includeshared
      % == true, then incorporates shared parameters into result. Only
      % called by reorganizeDLParams.
      includeshared = myparse(varargin,'includeshared',true);
      if includeshared,
        sPrmAll = APTParameters.defaultParamsStruct();
        sPrmDT = APTParameters.mergeSharedDeepTrack(sPrmAll,'pose');
      else
        tPrm = APTParameters.getParamTrees('deeptrack');
        sPrm = tPrm.structize();
        sPrmDT = sPrm.ROOT.DeepTrack;
      end
    end
    function sPrmDTspecific = defaultParamsStructDT(nettype)
      % sPrmDTspecific = defaultParamsStructDT(nettype)
      % Return the parameters related to nettype networks. Only called by
      % reorganizeDLParams
      tPrm = APTParameters.getParamTrees(char(nettype));
      sPrmDTspecific = tPrm.structize();
      sPrmDTspecific = sPrmDTspecific.ROOT;
      fld = fieldnames(sPrmDTspecific);
      assert(isscalar(fld)); % eg {'MDN'}
      fld = fld{1};
      sPrmDTspecific = sPrmDTspecific.(fld);
    end
    function leaves = getVisibleLeaves(tree)
      % leaves = getVisibleLeaves(tree)
      % returns an array of all visible leaf data
      leaves = [];
      if ~tree.Data.Visible,
        return;
      end
      if isempty(tree.Children),
        leaves = tree.Data;
        return;
      end
      for i = 1:numel(tree.Children),
        leavescurr = APTParameters.getVisibleLeaves(tree.Children(i));
        if isempty(leavescurr),
          continue;
        end
        if isempty(leaves),
          leaves = leavescurr;
        else
          leaves(end+1:end+numel(leavescurr)) = leavescurr;
        end
      end
    end
    function [tPrm,minLevel] = propagateLevelFromLeaf(tPrm)
      % [tPrm,minLevel] = propagateLevelFromLeaf(tPrm)
      % Level is set in leaf nodes, set level for intermediate nodes of tree to
      % reflect minimum level of its children (Important = 1, Obsolete = 4)
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
      % [tPrm,rqts] = propagateRequirementsFromLeaf(tPrm)
      % Requirements are set in leaf nodes, set requirements for
      % intermediate nodes of tree to reflect intersection of requirements
      % from leaves
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
      % filterPropertiesByLevel(tree,level)
      % set nodes in tree to be Visible iff their Level is less than the
      % input level
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

    function filterPropertiesByAffectsTraining(tree,istrain)
      % filterPropertiesByAffectsTraining(tree,istrain)
      % set nodes in tree to be Visible iff
      % isequal(AffectsTraining,istrain)
      if isempty(tree.Children),
        tree.Data.Visible = tree.Data.Visible && isequal(tree.Data.AffectsTraining,istrain);
        return;
      end
      
      if tree.Data.Visible,
        tree.Data.Visible = false;
        for i = 1:numel(tree.Children),
          APTParameters.filterPropertiesByAffectsTraining(tree.Children(i),istrain);
          tree.Data.Visible = tree.Data.Visible || tree.Children(i).Data.Visible;
        end
      end

    end
    
    function tree = setAllVisible(tree)
      % tree = setAllVisible(tree)
      % reset all nodes to be Visible
      tree.Data.Visible = true;
      for i = 1:numel(tree.Children),
        APTParameters.setAllVisible(tree.Children(i));
      end
      
    end
    
    function stage = getStage(path)
      % stage = getStage(path)
      % based on path to node, determine if this parameter is related to
      % the first or last stage.
      % used in ParameterSetup to make it clear which parameters are
      % specific for pose and detect stage.
      if startsWith(path,[APTParameters.posePath,'.']),
        stage = 'last';
      elseif startsWith(path,[APTParameters.maDetectNetworkPath,'.']),
        stage = 'first';
      else
        stage = 'unknown';
      end
        
    end

    function tree = filterPropertiesByCondition(tree,labelerObj,varargin)
      % tree = filterPropertiesByCondition(tree,labelerObj,varargin)
      % set Visible based on Requirements and labelerObj state.
      %
      % note on netsUsed
      % Currently, topdown trackers include 2 'netsUsed'
      
      [stage,argsrest] = myparse_nocheck(varargin,'stage','last');
      if strcmp(tree.Data.Field,{'Detect'})
        stage = 'first';
      end

      if isempty(tree.Children),
        
        [netsUsed,hasTrx,trackerIsDL] = myparse(argsrest,...
          'netsUsed',[],'hasTrx',[],'trackerIsDL',[]);
        if isempty(netsUsed),
          netsUsed = labelerObj.trackerNetsUsed;
        end
        if isempty(hasTrx),
          hasTrx = labelerObj.hasTrx;
        end
        if isempty(trackerIsDL),
          trackerIsDL = labelerObj.trackerIsDL;
        end
        isma = labelerObj.maIsMA;
        is2stg = labelerObj.trackerIsTwoStage;
        isbu = labelerObj.trackerIsBotUp;
        isod = labelerObj.trackerIsObjDet;
        isht = is2stg && ~isod;
        if strcmpi(stage,'first'),
          netsUsed = netsUsed(1);
        elseif strcmpi(stage,'last')
          netsUsed = netsUsed(end);
        end
        % AL20210901: note: in parameter jsons, the 'isTopDown' requirement
        % is used; but this actually means "isTD-2stg"; vs SA-trx which is
        % conceptually TD.
      
        reqs = tree.Data.Requirements;
        if isempty(reqs),
          return;
        end
        if ismember('isCPR',reqs) && ~any(strcmpi('cpr',netsUsed)),
          tree.Data.Visible = false;
        elseif all(ismember({'hasTrx' 'isTopDown'},reqs))
          if ~hasTrx && ~is2stg
            % Special case/hack; if hasTrx and ma are both present, it's an
            % OR condition (rather than AND which is the default for 2+
            % requiremetns)
            tree.Data.Visible = false;
          end
        elseif all(ismember({'hasTrx' 'isHeadTail'},reqs))
          if ~hasTrx && ~isht
            % Special case/hack; if hasTrx and head-tail are both present, it's an
            % OR condition (rather than AND which is the default for 2+
            % requiremetns)
            tree.Data.Visible = false;
          end
        elseif ismember('isBotUp',reqs) && ~isbu
          tree.Data.Visible = false;
        elseif ismember('ma',reqs) && ~isma
          tree.Data.Visible = false;
        elseif ismember('hasTrx',reqs) && ~hasTrx,
          tree.Data.Visible = false;
        elseif ismember('isMultiView',reqs) && ~labelerObj.isMultiView
          tree.Data.Visible = false;
        elseif ismember('isDeepTrack',reqs) && ~trackerIsDL,
          tree.Data.Visible = false;
        elseif ismember('isTopDown',reqs) && ~is2stg
          tree.Data.Visible = false;          
        elseif ismember('isHeadTail',reqs) && ~isht
          tree.Data.Visible = false;
        elseif ismember('isObjDet',reqs) && ~isod
          tree.Data.Visible = false;
        elseif ismember('~isObjDet',reqs) && isod
          tree.Data.Visible = false;
        else
          dlnets = enumeration('DLNetType');
          for i=1:numel(dlnets)
            net = dlnets(i);
            if ismember(lower(char(net)),lower(reqs)) && ~any(strcmp(net,netsUsed))
              tree.Data.Visible = false;
              break;
            elseif ismember(['~',lower(char(net))],lower(reqs)) && any(strcmp(net,netsUsed)),
              tree.Data.Visible = false;
            end
          end
        end
        
        return;
        
      end
        
      if tree.Data.Visible,
        tree.Data.Visible = false;
        for i = 1:numel(tree.Children),
          APTParameters.filterPropertiesByCondition(tree.Children(i),labelerObj,'stage',stage,argsrest{:});
          tree.Data.Visible = tree.Data.Visible || tree.Children(i).Data.Visible;
        end
      end
      
    end
    
    function [tPrm] = removeFilteredProperties(tPrm)
      % [tPrm] = removeFilteredProperties(tPrm)
      % Remove nodes that re not visible from tree tPrm
      
      if ~tPrm.Data.Visible,
        tPrm = [];
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
      % [sPrmFilter,tPrm] = filterStructPropertiesByCondition(sPrm,varargin)
      % Return parameters of struct sPrm that correspond to 
      [tPrm,leftovers] = myparse_nocheck(varargin,'tree',[]);
      if isempty(tPrm),
        tPrm = APTParameters.defaultParamsTree;
      end
      tPrm.structapply(sPrm);
      tPrm = APTParameters.filterPropertiesByCondition(tPrm,[],leftovers{:});
      tPrm = APTParameters.removeFilteredProperties(tPrm);
      sPrmFilter = tPrm.structize();
      
    end
    
    function [v,sPrmFilter0,sPrmFilter1] = ...
        isEqualFilteredStructProperties(sPrm0,sPrm1,varargin)
      [tPrm0,leftovers] = myparse_nocheck(varargin,'tree',[]);
      if isempty(tPrm0),
        tPrm0 = APTParameters.defaultParamsTree;
      end
      tPrm1 = tPrm0.copy();
      sPrmFilter0 = APTParameters.filterStructPropertiesByCondition(sPrm0,'tree',tPrm0,leftovers{:});
      sPrmFilter1 = APTParameters.filterStructPropertiesByCondition(sPrm1,'tree',tPrm1,leftovers{:});
      v = isequaln(sPrmFilter0,sPrmFilter1);

    end

    function [minv,maxv] = numberRange(tPrm)
      if ~tPrm.Data.Visible,
        minv = nan;
        maxv = nan;
        return;
      end
      minv = tPrm.Data.Index;
      maxv = tPrm.Data.Index;
      for c = tPrm.Children(:)',
        [minvc,maxvc] = APTParameters.numberRange(c);
        minv = min(minv,minvc);
        maxv = max(maxv,maxvc);
      end
    end

    function addNumbers(tPrm,minv,maxv)

      if ~tPrm.Data.Visible,
        return;
      end
      if nargin < 2,
        minv = 0;
        maxv = 1;
      end
      tPrm.Data.Index = minv; %(minv+maxv)/2;
      if isempty(tPrm.Children),
        return;
      end
      n = numel(tPrm.Children);
      isvisible = false(1,n);
      for i = 1:n,
        child = tPrm.Children(i);
        isvisible(i) = child.Data.Visible;
      end
      if ~any(isvisible),
        %tPrm.Data.Index = (minv+maxv)/2;
        return;
      end
      idxvisible = find(isvisible);
      n = numel(idxvisible);
      dv = (maxv-minv)/n;
      for i = 1:n,
        child = tPrm.Children(idxvisible(i));
        APTParameters.addNumbers(child,minv+dv*(i-1),minv+dv*i);
      end

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
      
%     20211026 MA CPR no longer supported
%       if sPrm.ROOT.MultiAnimal.TargetCrop.AlignUsingTrxTheta && ...
%          strcmp(sPrm.ROOT.CPR.RotCorrection.OrientationType,'fixed')
%         warningNoTrace('CPR OrientationType cannot be ''fixed'' if aligning target crops using trx.theta. Setting CPR OrientationType to ''arbitrary''.');
%         sPrm.ROOT.CPR.RotCorrection.OrientationType = 'arbitrary';
%         tfChangeMade = true;
%       end
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
      flds2remove = {'LossFunction','Saving','DataAugmentation','GradientDescent'};
      for fndx = 1:numel(flds2remove)
        if isfield(v,flds2remove{fndx})
          v = rmfield(v,flds2remove{fndx});
        end
      end
    end

    % all parameters to specific dl parameters for input netType
    function v = all2DLSpecificParams(sPrmAll,netType)
      if ~ischar(netType),
        netType = APTParameters.getNetworkParamField(netType);
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
      [sPrmPPandCPRold] = CPRParam.n/ew2old(sPrmPPandCPR,nPhysPoints,nviews); % we won't use npoints or nviews
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
    
    % set tracking-specific parameters
    function sPrmAll = setTrackParams(sPrmAll,sPrmTrack)
      sPrmAll.ROOT.Track = sPrmTrack.ROOT.Track;
      sPrmAll.ROOT.MultiAnimal.Track = sPrmTrack.ROOT.MultiAnimal;
      sPrmAll.ROOT.PostProcess = sPrmTrack.ROOT.PostProcess;  
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
    
    function v = getTrackChunkSize(prm,varargin)
      v = APTParameters.getParam(prm,'ROOT.Track.ChunkSize',varargin{:});
    end

    function prm = setTrackChunkSize(prm,v)
      prm = APTParameters.setParam(prm,'ROOT.Track.ChunkSize',v);
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

    function prm = separateSharedDeepTrack(prm,stage)

      tshared = APTParameters.PARAM_FILES_TREES.deeptrack_shared.tree.findnode(APTParameters.deepSharedPath);
      prepath = APTParameters.DLStage2Path(stage);
      tdt = APTParameters.defaultParamsTree.findnode(prepath);
      relpaths = APTParameters.modernizeMoveSubTrees(tshared,tdt,'');
      translateflds = [cellfun(@(relpath) [prepath,'.',relpath],relpaths,'Uni',0),...
        cellfun(@(relpath) [APTParameters.deepSharedPath,'.',relpath],relpaths,'Uni',0)];
      for i = 1:size(translateflds,1),
        oldfld = translateflds{i,1};
        newfld = translateflds{i,2};
        if structisfield(prm,oldfld),
          prm = structmvfield(prm,oldfld,newfld);
        end
      end

    end

    function prm = mergeMultiStageTrackerParams(prm_stage1,prm_stage2)
      prm = [];
      if ~isempty(prm_stage1),
        prm = APTParameters.modernize(prm_stage1,'all');
      end
      if ~isempty(prm_stage2),
        prm_stage2 = APTParameters.separateSharedDeepTrack(prm_stage2,'detect');
        if isempty(prm),
          prm = prm_stage2;
        else
          copyflds = {'ROOT.MultiAnimal','ROOT.MultiAnimalDetect'};
          for i = 1:numel(copyflds),
            fld = copyflds{i};
            prm = APTParameters.setParam(prm,fld,APTParameters.getParam(prm_stage2,fld));
          end
        end
      end
      
      
    end

    function prm = toDeepTrackerParams(prm,stage)
      if strcmp(stage,'detect'),
        % stage 1 trackData; move Detect.DeepTrack to top-level
        if structisfield(prm,APTParameters.maDetectNetworkPath),
          prm = structmvfield(prm,APTParameters.maDetectNetworkPath,APTParameters.posePath);
        end
      elseif strcmp(stage,'pose')
        % remove detect/DeepTrack from stage2
        if structisfield(prm,APTParameters.maDetectNetworkPath)
          prm = structrmfield(prm,APTParameters.maDetectNetworkPath);
        end
      end
      % move shared parameters to dt
      prm = APTParameters.setParam(prm,APTParameters.posePath,APTParameters.getPoseDeepTrackParams(prm));
      % remove shared parameters
      prm = structrmfield(prm,APTParameters.deepSharedPath);
    end

    function prm = duplicateSharedDeepTrackParams(prm)
      assert(isstruct(prm));
      prm = APTParameters.setParam(prm,APTParameters.maDetectNetworkPath,APTParameters.getDetectDeepTrackParams(prm));
      prm = APTParameters.setParam(prm,APTParameters.posePath,APTParameters.getPoseDeepTrackParams(prm));
      prm = structrmfield(prm,APTParameters.deepSharedPath);
    end

    function prmDT = getPoseDeepTrackParams(prm)
      prmDT = APTParameters.mergeSharedDeepTrack(prm,'pose');
    end

    function prmMADT = getDetectDeepTrackParams(prm)
      prmMADT = APTParameters.mergeSharedDeepTrack(prm,'detect');
    end

    function tspecific = mergeSharedDeepTrack(t,varargin)
      if isstruct(t),
        tspecific = APTParameters.mergeSharedDeepTrackStruct(t,varargin{:});
      else
        tspecific = APTParameters.mergeSharedDeepTrackTree(t,varargin{:});
      end
    end

    function tspecific = mergeSharedDeepTrackTree(t,stage)

      if ischar(stage),
        tshared = t.findnode(APTParameters.deepSharedPath);
        prepath = APTParameters.DLStage2Path(stage);
        tspecific = t.findnode(prepath);
        tspecific = tspecific.copy();
      else
        tshared = t;
        tspecific = stage;
      end
      specflds = cellfun(@(x) x.Field, {tspecific.Children.Data},'Uni',0);
      for sharedchild = tshared.Children(:)',
        fld = sharedchild.Data.Field;
        i = find(strcmp(specflds,fld),1);
        if isempty(i),
          tspecific.Children(end+1) = sharedchild.copy();
        else
          tspecific.Children(i) = APTParameters.mergeSharedDeepTrackTree(sharedchild,tspecific.Children(i));
        end
      end
    end

    function sspecific = mergeSharedDeepTrackStruct(s,stage)

      if ischar(stage),
        sshared = structgetfield(s,APTParameters.deepSharedPath);
        prepath = APTParameters.DLStage2Path(stage);
        sspecific = structgetfield(s,prepath);
      else
        sshared = s;
        sspecific = stage;
      end
      specflds = fieldnames(sspecific);
      sharedflds = fieldnames(sshared);
      for j = 1:numel(sharedflds),
        fld = sharedflds{j};
        i = find(strcmp(specflds,fld),1);
        if isempty(i),
          sspecific.(fld) = sshared.(fld);
        elseif isstruct(sshared.(fld)),
          sspecific.(fld) = APTParameters.mergeSharedDeepTrackStruct(sshared.(fld),sspecific.(fld));
        else
          sspecific.(fld) = sshared.(fld);
        end
      end
    end
    
    function fps = modernizeMoveSubTrees(tnew,told,path)
      oldflds = cellfun(@(x) x.Field, {told.Children.Data},'Uni',0);
      fps = cell(0,1);
      if nargin < 3,
        path = '';
      end
      for childnew = tnew.Children(:)',
        fld = childnew.Data.Field;
        if isempty(path),
          fpcurr = fld;
        else
          fpcurr = [path,'.',fld];
        end
        i = find(strcmp(oldflds,fld),1);
        if isempty(i),
          fps{end+1,1} = fpcurr;
        else
          fpschild = APTParameters.modernizeMoveSubTrees(childnew,told.Children(i),fpcurr);
          fps = [fps;fpschild];
        end
      end
    end

    function sPrmAll = modernize(sPrmAll,allowedUnrecogFlds)
      % 20210720 param reorg MA
      if isempty(sPrmAll),
        return;
      end

      if nargin < 2,
        allowedUnrecogFlds = {};
      end

      % translation will happen in order
      % oldfld,newfld
      translateflds = {
        'ROOT.MultiAnimalDetection',APTParameters.maDetectPath
        'ROOT.MultiAnimal.Detect',APTParameters.maDetectPath
        'ROOT.ImageProcessing.MultiTarget.TargetCrop','ROOT.MultiAnimal.TargetCrop'
        'ROOT.MultiAnimal.TargetCrop.Radius','ROOT.MultiAnimal.TargetCrop.ManualRadius'
        'ROOT.MultiAnimal.TrackletStitch','ROOT.MultiAnimal.Track.TrackletStitch'
        };

      % all parameters in dtshared were in maDetectNetworkPath and posePath
      tshared = APTParameters.PARAM_FILES_TREES.deeptrack_shared.tree.findnode(APTParameters.deepSharedPath);
      tdt = APTParameters.PARAM_FILES_TREES.deeptrack.tree.findnode(APTParameters.posePath);
      tmadt = APTParameters.PARAM_FILES_TREES.madetect.tree.findnode(APTParameters.maDetectNetworkPath);

      relpaths_shared_madt = APTParameters.modernizeMoveSubTrees(tshared,tmadt,'');
      relpaths_shared_dt = APTParameters.modernizeMoveSubTrees(tshared,tdt,'');
      translateflds_shared_madt = [cellfun(@(relpath) [APTParameters.maDetectNetworkPath,'.',relpath],relpaths_shared_madt,'Uni',0),...
        cellfun(@(relpath) [APTParameters.deepSharedPath,'.',relpath],relpaths_shared_madt,'Uni',0)];
      translateflds_shared_dt = [cellfun(@(relpath) [APTParameters.posePath,'.',relpath],relpaths_shared_dt,'Uni',0),...
        cellfun(@(relpath) [APTParameters.deepSharedPath,'.',relpath],relpaths_shared_dt,'Uni',0)];
      translateflds = [translateflds;translateflds_shared_madt;translateflds_shared_dt];

      for i = 1:size(translateflds,1),
        oldfld = translateflds{i,1};
        newfld = translateflds{i,2};
        if structisfield(sPrmAll,oldfld),
          sPrmAll = structmvfield(sPrmAll,oldfld,newfld);
        end
      end

      rmflds = {'ROOT.DeepTrack.DeepPoseKit.dpk_test',
        'ROOT.DeepTrack.MMDetect.test'
        'ROOT.MultiAnimalDetect.DeepTrack.MMDetect.test'
        'ROOT.DeepTrack.MMDetect_FRCNN.test',
        'ROOT.MultiAnimalDetect.DeepTrack.MMDetect_FRCNN.test',};
      for i = 1:numel(rmflds),
        oldfld = rmflds{i};
        if structisfield(sPrmAll,oldfld),
          sPrmAll = structrmfield(sPrmAll,oldfld);
        end
      end

      sPrmDflt = APTParameters.defaultParamsStructAll;
      sPrmAll = structoverlay(sPrmDflt,sPrmAll,'allowedUnrecogFlds',allowedUnrecogFlds);%,...
        %'dontWarnUnrecog',true); % to allow removal of obsolete params
    end
    
    function [tPrm,canceled,do_update] = ...
        autosetparamsGUI(tPrm,lobj)
      
      silent = lobj.silent ;
      
      if lobj.maIsMA && lobj.trackerIsTwoStage  && ~lobj.trackerIsObjDet
          % Using head-tail for the first stage
          align_trx_theta = tPrm.findnode('ROOT.MultiAnimal.TargetCrop.AlignUsingTrxTheta').Data.Value;

          if ~align_trx_theta
            if silent
              res = 'Yes';
            else
              res = questdlg(strcatg('For head-tail based two-stage detection, align using head-tail is switched off. ', ...
                                     'Aligning animals using the head-tail direction will lead to better performance. ', ...
                                     'Align using the head-tail direction?'), ...
                             'Align using head-tail?', ...
                             'Yes','No', ...
                             'Yes');
            end
            if strcmp(res,'Yes')
              tPrm.findnode('ROOT.MultiAnimal.TargetCrop.AlignUsingTrxTheta').Data.Value = 1;
              lobj.trackParams.ROOT.MultiAnimal.TargetCrop.AlignUsingTrxTheta = 1;
            end
          end
          
          align_trx_theta = tPrm.findnode('ROOT.MultiAnimal.TargetCrop.AlignUsingTrxTheta').Data.Value;
          vert_flip = tPrm.findnode('ROOT.DeepTrack.DataAugmentation.vert_flip').Data.Value;
          if align_trx_theta && vert_flip
            if silent
              res = 'Yes';
            else
              res = questdlg(strcatg('When aligning using head-tail for 2 stage tracking, horizontal flipping and not vertical flipping is recommended ', ...
                                     'for augmentation as the animal is rotated to face up. Switch the augmentation flip augmentation type from vertical ', ...
                                     'to horizontal?'), ...
                             'Switch flipping?', ...
                             'Yes','No', ...
                             'Yes');
            end
            if strcmp(res,'Yes')
              tPrm.findnode('ROOT.DeepTrack.DataAugmentation.vert_flip').Data.Value = 0;
              tPrm.findnode('ROOT.DeepTrack.DataAugmentation.horz_flip').Data.Value = 1;
              lobj.trackParams.ROOT.DeepTrack.DataAugmentation.horz_flip = 1;
              lobj.trackParams.ROOT.DeepTrack.DataAugmentation.vert_flip = 0;
            end
          
          end
      end

      % Set default return values
      canceled = false ;
      do_update = false ;

      % Don't bother computing automatic parameters (which can take a while) if we
      % know we're not going to use them.
      if ~lobj.trackAutoSetParams ,
        return
      end

      % automatically set the parameters based on labels.
      autoparams = apt.compute_auto_params(lobj) ;
      
      kk = autoparams.keys();
      % If values have been previously updated, then check if they are
      % significantly (>10%) different now
      dstr = '';
      diff = false;
      default = true;

      for ndx = 1:numel(kk)
        nd = tPrm.findnode(kk{ndx});
        prev_val = nd.Data.Value;
        cur_val = autoparams(kk{ndx});
        reldiff = (cur_val-prev_val)/(prev_val+0.001);
        if isempty(reldiff) || abs(reldiff)>0.1 % first clause if eg prev_val empty
          diff = true;
        end
        if nd.Data.DefaultValue~=nd.Data.Value
          default = false;
        end
%         if cur_val~=prev_val
%           identical = false;
%         end
        extra_str = '';
        if ~isempty(strfind(kk{ndx},'DataAugmentation')) && lobj.maIsMA && lobj.trackerIsTwoStage
          stage = APTParameters.getStage(kk{ndx});
          extra_str = sprintf(' (%s stage)',stage);
        end
        dstr = sprintf('%s%s%s %d -> %d\n',dstr,nd.Data.DispName, extra_str,...
                    prev_val,cur_val);

      end

      if diff
        dstr = sprintf(strcatg('Auto-computed parameters have changed from earlier by more than 10%%\n', ...
                               'for some of the parameters. Update the following parameters?\n', ...
                               '%s'), ...
                       dstr);
        
        if lobj.trackAutoSetParams
          if default || silent
            res = 'Update';
          else
            res = APTParameters.raiseAcceptAutoParamsDialog(dstr,lobj.hFig);
          end
        else
          res = 'Do not update';
        end
        if strcmp(res,'Cancel')
          canceled = true;
          return
        end
      else
        % All parameters are within 10% of original values
        do_update = false;
        return
      end
      
      if strcmp(res,'Update')
        for ndx = 1:numel(kk)
          nd = tPrm.findnode(kk{ndx});
          nd.Data.Value = autoparams(kk{ndx});
        end
        do_update = true;
      end
      
    end  % function
    
    function res = raiseAcceptAutoParamsDialog(dstr, parentFig)
      % Raise a custom modal dialog that is sized to contain dstr, and looks nice,
      % and is centered on parentFig.  This function blocks until the user clicks
      % one of the dialog box buttons.  On return, res is an old-style string
      % containing the user response.  Either 'Update', 'Do not update', or
      % 'Cancel'.

      % Constants of the figure layout
      side_margin = 20;
      margin = 10;  % space between buttons, above and below buttons, and between textbox and figure boundary.
      button_width = 150;
      button_height = 30;
      button_count = 3;

      min_w = button_count*button_width+(button_count-1)*margin+2*side_margin;
      min_h = button_height + 3*side_margin;

      dstr = strtrim(dstr) ;
      fig = figure('units','pixels', ...
                   'position',[300,300,min_w,150],...
                   'toolbar','none', ...
                   'menu','none', ...
                   'name', 'Auto-update parameters?', ...
                   'Resize', 'off', ...
                   'HandleVisibility', 'off', ...
                   'NumberTitle','off', ...
                   'visible', 'off') ;
      textbox = uicontrol('style','text', ...
                          'String',cellstr(dstr), ...
                          'units','pixels',...
                          'position',[0,0,250,100], ...
                          'horizontalalignment','left',...
                          'Parent',fig);
      textbox_extent = get(textbox,'Extent');

      fig_height = min_h+textbox_extent(4);
      fig_width = max(min_w,textbox_extent(3)+2*side_margin);

      % Size the figure, center on parent
      fig.Position(4) = fig_height ;
      fig.Position(3) = fig_width ;
      centerOnParentFigure(fig,parentFig);

      % Position the textbox
      textbox.Position(2) = side_margin+margin+button_height;
      textbox.Position(1) = side_margin;
      textbox.Position(3:4) = textbox.Extent(3:4);

      % Create the buttons

      btn_center = fig.Position(3)/2;
      btn_left = btn_center - (button_count*button_width+(button_count-1)*margin) / 2;

      btn_bottom = margin; 
      btns = [];
      btns(1) = uicontrol('parent', fig, ...
                          'style','pushbutton', ...
                          'String','Update',...
                          'units','pixels', ...
                          'position',[btn_left,btn_bottom,button_width,button_height], ...
                          'Visible','on');
      btn_left = btn_left + button_width + margin;
      btns(2) = uicontrol('parent', fig, ...
                          'style','pushbutton',...
                          'String','Do not update',...
                          'units','pixels',...
                          'position',[btn_left,btn_bottom,button_width,button_height],...
                          'Visible','on');
      btn_left = btn_left + button_width + margin;
      btns(3) = uicontrol('parent', fig, ...
                          'style','pushbutton',...
                          'String','Cancel',...
                          'units','pixels',...
                          'position',[btn_left,btn_bottom,button_width,button_height],...
                          'Visible','on');

      % Define a callback to pass response back to this scope.
      res = 'Cancel' ;  % Fallback if e.g. user clicks the upper-right window close button
      function button_pressed(button_h, ~, ~)
        res = get(button_h,'String') ;
        delete(fig) ;
      end     
      set(btns,'Callback',@button_pressed)

      % Make the figure visible, and make sure it gets drawn
      fig.Visible = 'on' ;
      drawnow() ;

      % Wait for the figure to be deleted (in the button_pressed() callback)
      uiwait(fig) ;
    end
    
    function sPrm0 = defaultParamsOldStyle
      tPrm0 = APTParameters.defaultParamsTree;
      sPrm0 = tPrm0.structize();
      % Use nan for npts, nviews; default parameters do not know about any
      % model
      sPrm0 = CPRParam.new2old(sPrm0,nan,nan);
    end

    function [s,deepnets] = paramFileSpecs()
      % Specifies/finds jsons in APT tree.
      %
      % Deepnets are found dynamically to easy adding new nets.

      s.preprocess = 'params_preprocess.json';
      s.track = 'params_track.json';
      s.cpr = fullfile('trackers','cpr','params_cpr.json');
      s.deeptrack_shared = fullfile('trackers','dt','params_deeptrack_shared.json');
      s.deeptrack = fullfile('trackers','dt','params_deeptrack.json');
      s.ma = fullfile('trackers','dt','params_ma.json');
      s.madetect = fullfile('trackers','dt','params_detect.json');
      s.postprocess = 'params_postprocess.json';
      resourceFolderPath = fullfile(APT.Root, 'matlab') ;
      dd = dir(fullfile(resourceFolderPath,'trackers','dt','params_deeptrack_*.json'));
      dtjsons = {dd.name}';
      sre = regexp(dtjsons,'params_deeptrack_(?<net>[a-zA-Z_]+).json','names');
      for i=1:numel(dtjsons)
        s.(sre{i}.net) = fullfile('trackers','dt',dtjsons{i});
      end
      deepnets = cellfun(@(x)x.net,sre,'uni',0);
    end  % function
    
    function s = paramFilesTrees()
      % fprintf('APTParameters::paramFilesTrees()\n');
      [specs,deepnets] = APTParameters.paramFileSpecs() ;
      resourceFolderPath = fullfile(APT.Root, 'matlab') ;
      % fprintf('resourceFolderPath: %s\n', resourceFolderPath) ;
      
      s = struct();
      field_names = fieldnames(specs) ;
      % fprintf('field_names:\n') ;
      % fprintf('%s', formattedDisplayText(field_names)) ;
      for i = 1 : numel(field_names)
        fn = field_names{i} ;
        spec = specs.(fn) ;
        json_file_path = fullfile(resourceFolderPath, spec) ;
        assert(exist(json_file_path,'file'),sprintf('File %s does not exist',json_file_path));
        param_tree = parseConfigJson(json_file_path) ;
        if any(strcmp(fn,deepnets))
          % AL 20190711: automatically create requirements for all deep net
          %   param trees
          param_tree.traverse(@(x)set(x.Data,'Requirements',{fn,'isDeepTrack'})) ;
        end
        value = struct('json', {json_file_path}, 'tree', {param_tree}) ;
        s.(fn) = value ;
      end
      % copy structure from params_deeptrack under MultiAnimalDetect
      if isfield(s,'madetect') && isfield(s,'deeptrack'),
        t1 = s.madetect.tree.findnode(APTParameters.maDetectNetworkPath);
        t2 = s.deeptrack.tree.findnode(APTParameters.posePath);
        if isempty(t1.Children),
          t1.Children = TreeNode.empty(0,1);
        end      
        for i = 1:numel(t2.Children),
          t1.Children(end+1,1) = t2.Children(i).copy();
        end
        t1.traverse(@(node) node.Data.addRequirement('isTopDown'));
      end
      % reset full paths
      for i = 1:numel(field_names),
        fn = field_names{i};
        s.(fn).tree.setFullPath();
      end

      % % Print s
      % fprintf('s:\n') ;
      % fprintf('%s', formattedDisplayText(s)) ;
    end  % function

    function v = getParam(prm,fld,returnval)
      if nargin < 3,
        returnval = true;
      end
      if isstruct(prm),
        v = structgetfield(prm,fld);
      else
        v = prm.findnode(fld);
      end
      if returnval && isa(v,'TreeNode'),
        v = v.Data.Value;
      end
    end

    function prm = setParam(prm,fld,val)
      if isstruct(prm),
        prm = structsetfield(prm,fld,val);
      else
        node = prm.findnode(fld);
        assert(~isempty(node),sprintf('Could not find %s',fld));
        node.Data.Value = val;
      end
    end

    function v = getParamNode(prm,fld,returnval)
      if nargin < 3,
        returnval = true;
      end
      if isstruct(prm),
        v = structgetfield(prm,fld);
      else
        v = prm.findnode(fld);
      end
      if returnval && isa(v,'TreeNode')
        v = v.structize();
      end
    end

    function [horzflip,vertflip] = getDataAugmentationFlipParams(prm,varargin)
      horzflip = APTParameters.getParam(prm,[APTParameters.deepSharedPath,'.DataAugmentation.horz_flip'],varargin{:});
      vertflip = APTParameters.getParam(prm,[APTParameters.deepSharedPath,'.DataAugmentation.vert_flip'],varargin{:});
    end

    function v = getMABBoxParam(prm,varargin)
      v = APTParameters.getParam(prm,[APTParameters.maDetectPath,'.BBox'],varargin{:});
    end

    function v = getMALossMaskParam(prm,varargin)
      v = APTParameters.getParam(prm,'ROOT.MultiAnimal.LossMask',varargin{:});
    end

    function v = getScaleParam(prm,stage,varargin)
      prefix = APTParameters.DLStage2Path(stage);
      fld = [prefix,'.ImageProcessing.scale'];
      v = APTParameters.getParam(prm,fld,varargin{:});
    end

    function v = getBatchSizeParam(prm,stage,varargin)
      prefix = APTParameters.DLStage2Path(stage);
      fld = [prefix,'.GradientDescent.batch_size'];
      v = APTParameters.getParam(prm,fld,varargin{:});
    end

    function skelstr = getSkeletonString(prm,stage)
      if nargin < 2,
        stage = 'pose';
      end
      prefix = APTParameters.DLStage2Path(stage);
      skelstr = APTParameters.getParam(prm,[prefix,'.OpenPose.affinity_graph']);
    end

    function prm = setSkeletonString(prm,skelstr)
      prm = structsetfield(prm,[APTParameters.posePath,'.OpenPose.affinity_graph'],skelstr);
      prm = structsetfield(prm,[APTParameters.maDetectPath,'.DeepTrack.OpenPose.affinity_graph'],skelstr);
    end

    function v = getFlipLandmarkMatchStr(prm,varargin)
      v = APTParameter.getParam(prm,[APTParameters.deepSharedPath,'.DataAugmentation.flipLandmarkMatches']);
    end

    function prm = setFlipLandmarkMatchStr(prm,matchstr)
      prm = structsetfield(prm,[APTParameters.deepSharedPath,'.DataAugmentation.flipLandmarkMatches'],matchstr);
    end

    function s = maTargetCropRadiusManualPath()
      s = 'ROOT.MultiAnimal.TargetCrop.ManualRadius';
    end

    function v = getMATargetCropRadiusManual(prm,varargin)
      s = APTParameters.maTargetCropRadiusManualPath;
      v = APTParameters.getParam(prm,s,varargin{:});
    end

    function prm = setMATargetCropRadiusManual(prm,v)
      prm = APTparameters.setParam(prm,APTParameters.maTargetCropRadiusManualPath,v);
    end

    function v = getMATargetCropParams(prm,varargin)
      v = APTParameters.getParamNode(prm,'ROOT.MultiAnimal.TargetCrop',varargin{:});
    end

    function prm = setMATargetCropParams(prm,v)
      assert(isstruct(prm));
      prm = APTParameters.setParam(prm,'ROOT.MultiAnimal.TargetCrop',v);
    end
    
    function v = getCPRParams(prm,varargin)
      v = APTParameters.getParamNode(prm,'ROOT.CPR');
    end

    function prm = setMAIsMulti(prm,val)
      prm = APTParameters.setParam(prm,'ROOT.MultiAnimal.is_multi',val);
    end

    function val = getMAMultiCropIms(prm,varargin)
      val = APTParameters.getParam(prm,'ROOT.MultiAnimal.multi_crop_ims',varargin{:});
    end

    function prm = setMAMultiCropIms(prm,val)
      prm = APTParameters.setParam(prm,'ROOT.MultiAnimal.multi_crop_ims',val);
    end

    function v = getMAMultCropImSz(prm,varargin)
      v = APTParameters.getParam(prm,'ROOT.MultiAnimal.multi_crop_im_sz',varargin{:});
    end

    function prm = setMAMultCropImSz(prm,v)
      prm = APTParameters.setParam(prm,'ROOT.MultiAnimal.multi_crop_im_sz',v);
    end

    function prm = setMAMultiOnlyHT(prm,val)
      prm = APTParameters.setParam(prm,[APTParameters.maDetectPath,'.multi_only_ht'],val);
    end

    function s = alignTrxThetaPath()
      s = 'ROOT.MultiAnimal.TargetCrop.AlignUsingTrxTheta';
    end

    function prm = getAlignTrxTheta(prm,varargin)
      prm = APTParameters.getParam(prm,APTParameters.alignTrxThetaPath,varargin{:});
    end

    function prm = setMAAlignUsingTrxTheta(prm,val)
      prm = APTParameters.setParam(prm,APTParameters.alignTrxThetaPath,val);
    end

    function prm = setHeadTailKeypoints(prm,val)
      prm = APTParameters.setParam(prm,[APTParameters.maDetectPath,'.ht_pts'],val);
    end

    function s = multiScaleByBBoxPath()
      s = 'ROOT.MultiAnimal.TargetCrop.multi_scale_by_bbox';
    end

    function v = getMultiScaleByBBox(prm,varargin)
      v = APTParameters.getParam(prm,APTParameters.multiScaleByBBoxPath,varargin{:});
    end

    function v = getMaxNAnimals(prm,varargin)
      v = APTParameters.getParam(prm,'ROOT.MultiAnimal.Track.max_n_animals',varargin{:});
    end

    function v = getMATargetCropPadBkgd(prm,varargin)
      v = APTParameters.getParam(prm,'ROOT.MultiAnimal.TargetCrop.PadBkgd',varargin{:});
    end

    function s = getMALossMaskParams(prm)
      s = prm.ROOT.MultiAnimal.LossMask;
    end

    function v = getMAMultiLossMask(prm,varargin)
      v = APTParmaeters.getParam(prm,'ROOT.MultiAnimal.multi_loss_mask',varargin{:});
    end

    function v = getTrackNFramesSmall(prm,varargin)
      v = APTParameters.getParam(prm,'ROOT.Track.NFramesSmall',varargin{:});
    end

    function v = getTrackNFramesLarge(prm,varargin)
      v = APTParameters.getParam(prm,'ROOT.Track.NFramesLarge',varargin{:});
    end

    function v = getTrackNFramesNeighborhood(prm,varargin)
      v = APTParameters.getParam(prm,'ROOT.Track.NFramesNeighborhood',varargin{:});
    end

    function prm = setTrackNFramesSmall(prm,v)
      prm = APTParameters.setParam(prm,'ROOT.Track.NFramesSmall',v);
    end

    function prm = setTrackNFramesLarge(prm,v)
      prm = APTParameters.setParam(prm,'ROOT.Track.NFramesLarge',v);
    end

    function prm = setTrackNFramesNeighborhood(prm,v)
      prm = APTParameters.setParam(prm,'ROOT.Track.NFramesNeighborhood',v);
    end

    function prm = removeTrackNFramesParams(prm)
      prm.ROOT.Track = rmfield(prm.ROOT.Track,{'NFramesSmall','NFramesLarge','NFramesNeighborhood'});
    end

    function v = getBackSubParams(prm,varargin)
      v = APTParameters.getParamNode(prm,'ROOT.ImageProcessing.BackSub',varargin{:});
    end

    function prm = setBackSubParams(prm,v)
      assert(isstruct(prm));
      prm = APTParameters.setParam(prm,'ROOT.ImageProcessing.BackSub',v);
    end


    function prm = removeOpenPoseAffinityGraphParams(prm)
      parentpaths = {APTParameters.posePath,APTParameters.maDetectNetworkPath};
      for i = 1:numel(parentpaths),
        fld = [parentpaths{i},'.OpenPose'];
        if ~structisfield(prm,fld),
          continue;
        end
        prm = structrmfield(prm,[fld,'.affinity_graph']);
        if isempty(fieldnames(structgetfield(prm,fld))),
          structsetfield(prm,fld,'');
        end
      end
    end

    function prm = removeFlipLandmarkMatches(prm)
      fld = [APTParameters.deepSharedPath,'.DataAugmentation.flipLandmarkMatches'];
      prm = structrmfield(prm,fld);
      if isempty(fieldnames(structgetfield(prm,fld))),
        structsetfield(prm,fld,'');
      end
    end

    function s = CPRFeaturePath()
      s = 'ROOT.CPR.Feature';
    end

    function v = getImageProcessingIMax(prm,varargin)
      v = APTParameters.getParam(prm,[APTParameters.deepSharedPath,'ImageProcessing.imax'],varargin{:});
    end

    function v = getPostProcessReconcile3dType(prm,varargin)
      v = APTParameters.getParam(prm,'ROOT.PostProcess.reconcile3dType',varargin{:});
    end

    function v = getPostProcessParams(prm,varargin)
      v = APTParameters.getParam(prm,'ROOT.PostProcess',varargin{:});
    end

    function prm = setPostProcessParams(prm,v)
      prm = APTParameters.setParam(prm,'ROOT.PostProcess',v);
    end

    function prepath = DLStage2Path(stage)
      if strcmp(stage,'pose'),
        prepath = APTParameters.posePath;
      elseif ismember(stage,{'detect','maDetectNetwork'}),
        prepath = APTParameters.maDetectNetworkPath;
      else
        error('Unknown stage %s',stage);
      end
    end

    function v = getGradientDescentSteps(prm,stage,varargin)
      prepath = APTParameters.DLStage2Path(stage);
      v = APTParameters.getParam(prm,[prepath,'.GradientDescent.dl_steps'],varargin{:});
    end

    function v = getDLCOverrideGradientDescentSteps(prm,varargin)
      v = APTParameters.getParam(prm,'ROOT.DeepTrack.DeepLabCut.dlc_override_dlsteps');
    end

    function prm = setGradientDescentSteps(prm,val,stage)
      prepath = APTParameters.DLStage2Path(stage);
      prm = APTParameters.setParam(prm,[prepath,'.GradientDescent.dl_steps'],val);
    end

    function v = getTrainingSaveStep(prm,varargin)
      v = APTParameters.getParam(prm,[APTParameters.deepSharedPath,'.GradientDescent.dl_step'],varargin{:});
    end

    function prm = setTrainingSaveStep(prm,val)
      prm = APTParameters.setParam(prm,[APTParameters.deepSharedPath,'.GradientDescent.dl_step'],val);
    end

    function v = getTrainingDisplayStep(prm,varargin)
      v = APTParameters.getParam(prm,[APTParameters.deepSharedPath,'.GradientDescent.display_step'],varargin{:});
    end
    
    function prm = setTrainingDisplayStep(prm,val)
      prm = APTParameters.setParam(prm,[APTParameters.deepSharedPath,'.GradientDescent.display_step'],val);
    end

    function v = getUseBackSub(prm,varargin)
      v = APTParameters.getParam(prm,'ROOT.ImageProcessing.BackSub.Use',varargin{:});
    end
    
    function prm = setUseBackSub(prm,v)
      prm = APTParameters.setParam(prm,'ROOT.ImageProcessing.BackSub.Use',v);
    end

    function v = getBgReadFcn(prm,varargin)
      v = APTParameters.getParam(prm,'ROOT.ImageProcessing.BackSub.BgReadFcn');
    end

    function prm = setBgReadFcn(prm,v)
      prm = APTParameters.setParam(prm,'ROOT.ImageProcessing.BackSub.BgReadFcn',v);
    end

    function v = getUseHistEq(prm,varargin)
      v = APTParameters.getParam(prm,'ROOT.ImageProcessing.HistEq.Use',varargin{:});
    end

    function prm = setUseHistEq(prm,v)
      prm = APTParameters.setParam(prm,'ROOT.ImageProcessing.HistEq.Use',v);
    end

  end  % methods (Static)
end  % classdef
