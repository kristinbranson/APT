classdef APTParameters
  properties (Constant)
    % This property stores parsetrees for jsons so that json files only 
    % need to be parsed once.
    %
    % This property is private as these trees are handles and mutable.
    % Use getParamTrees to access copies of these trees.
    PARAM_FILES_TREES = APTParameters.paramFilesTrees() ;
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
      
      trees = APTParameters.getParamTrees() ;
      tPrmPreprocess = trees.preprocess.tree;
      tPrmTrack = trees.track.tree;
      tPrmCpr = trees.cpr.tree;
      tPrmMA = trees.ma.tree;
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
        
        tPrmMADetectDT = tPrmMA.findnode('ROOT.MultiAnimal.Detect.DeepTrack');
        tPrmMADetectDT.Children = ...
          [tPrmMADetectDT.Children; tPrmDeepNetsMAChildren];
      end
      
      tPrm0 = tPrmPreprocess;
      tPrm0.Children = [tPrm0.Children; tPrmTrack.Children;...
        tPrmCpr.Children; tPrmMA.Children; tPrmDT.Children; tPrmPostProc.Children];
      tPrm0 = APTParameters.propagateLevelFromLeaf(tPrm0);
      tPrm0 = APTParameters.propagateRequirementsFromLeaf(tPrm0);
    end

    function tPrm0 = defaultTrackParamsTree(varargin)
      tPrm0 = APTParameters.defaultParamsTree(varargin{:});
      tPrmTrack = tPrm0.findnode('ROOT.Track');
      tPrmMA = tPrm0.findnode('ROOT.MultiAnimal.Track');
      tPrmMA.Data.Field = 'MultiAnimal';
      tPrmPostProcess = tPrm0.findnode('ROOT.PostProcess');
      tPrm0.Children = [tPrmTrack;tPrmMA;tPrmPostProcess];
    end
        
    % all parameters to tracking (not training) parameters
    function v = all2TrackParams(sPrmAll,compress)
      if nargin < 2,
        compress = true;
      end
      v = struct() ;
      v.ROOT = struct();
      v.ROOT.Track = sPrmAll.ROOT.Track;
      if compress,
        v.ROOT.MultiAnimal = sPrmAll.ROOT.MultiAnimal.Track;
      else
        v.ROOT.MultiAnimal.Track = sPrmAll.ROOT.MultiAnimal.Track;
      end
      v.ROOT.PostProcess = sPrmAll.ROOT.PostProcess;
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
 
    function dlNetTypesPretty = getDLNetTypesPretty
      mc = ?DLNetType;
      dlNetTypesPretty = cellfun(@APTParameters.getParamField,...
        {mc.EnumerationMemberList.Name},'Uni',0);
    end
    
    function f = getParamField(nettype)
      if ~ischar(nettype)
        nettype = char(nettype);
      end
      % first non-ROOT top-level field in parameter json 
      f = APTParameters.PARAM_FILES_TREES.(nettype).tree.Children(1).Data.Field;      
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

    function filterPropertiesByAffectsTraining(tree,istrain)

      if isempty(tree.Children),
        tree.Data.Visible = tree.Data.Visible && (tree.Data.AffectsTraining == istrain);
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
      
      tree.Data.Visible = true;
      for i = 1:numel(tree.Children),
        APTParameters.setAllVisible(tree.Children(i));
      end
      
    end
    
    function tree = filterPropertiesByCondition(tree,labelerObj,varargin)
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
        netType = APTParameters.getParamField(netType);
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
    
    function sPrmAll = modernize(sPrmAll)
      % 20210720 param reorg MA
      if ~isempty(sPrmAll)
        if isfield(sPrmAll.ROOT,'MultiAnimalDetection')
          sPrmAll.ROOT.MultiAnimal.Detect = sPrmAll.ROOT.MultiAnimalDetection;
          sPrmAll.ROOT = rmfield(sPrmAll.ROOT,'MultiAnimalDetection');
        end
        if isfield(sPrmAll.ROOT,'ImageProcessing') && ... 
          isfield(sPrmAll.ROOT.ImageProcessing,'MultiTarget') && ...
           isfield(sPrmAll.ROOT.ImageProcessing.MultiTarget,'TargetCrop')
          sPrmAll.ROOT.MultiAnimal.TargetCrop = sPrmAll.ROOT.ImageProcessing.MultiTarget.TargetCrop;
          sPrmAll.ROOT.ImageProcessing.MultiTarget = ...
            rmfield(sPrmAll.ROOT.ImageProcessing.MultiTarget,'TargetCrop');
        end        
        if isfield(sPrmAll.ROOT,'MultiAnimal')
          if isfield(sPrmAll.ROOT.MultiAnimal,'TargetCrop') && ...
            isfield(sPrmAll.ROOT.MultiAnimal.TargetCrop,'Radius')
            sPrmAll.ROOT.MultiAnimal.TargetCrop.ManualRadius = ...
               sPrmAll.ROOT.MultiAnimal.TargetCrop.Radius;
            sPrmAll.ROOT.MultiAnimal.TargetCrop = ...
              rmfield(sPrmAll.ROOT.MultiAnimal.TargetCrop,'Radius');
          end
          if isfield(sPrmAll.ROOT.MultiAnimal,'Detect') && ...
            isfield(sPrmAll.ROOT.MultiAnimal.Detect,'max_n_animals')
            sPrmAll.ROOT.MultiAnimal.Track.max_n_animals = sPrmAll.ROOT.MultiAnimal.Detect.max_n_animals;
            sPrmAll.ROOT.MultiAnimal.Track.min_n_animals = sPrmAll.ROOT.MultiAnimal.Detect.min_n_animals;
            sPrmAll.ROOT.MultiAnimal.Detect = ...
              rmfield(sPrmAll.ROOT.MultiAnimal.Detect,{'max_n_animals' 'min_n_animals'});
          end
          % KB 20220516: moving tracking related parameters around
          if isfield(sPrmAll.ROOT.MultiAnimal,'max_n_animals')
            sPrmAll.ROOT.MultiAnimal.Track.max_n_animals = sPrmAll.ROOT.MultiAnimal.max_n_animals;
            sPrmAll.ROOT.MultiAnimal = rmfield(sPrmAll.ROOT.MultiAnimal,'max_n_animals');
          end
          if isfield(sPrmAll.ROOT.MultiAnimal,'min_n_animals')
            sPrmAll.ROOT.MultiAnimal.Track.min_n_animals = sPrmAll.ROOT.MultiAnimal.min_n_animals;
            sPrmAll.ROOT.MultiAnimal = rmfield(sPrmAll.ROOT.MultiAnimal,'min_n_animals');
          end
          if isfield(sPrmAll.ROOT.MultiAnimal,'TrackletStitch'),
            sPrmAll.ROOT.MultiAnimal.Track.TrackletStitch = sPrmAll.ROOT.MultiAnimal.TrackletStitch;
            sPrmAll.ROOT.MultiAnimal = rmfield(sPrmAll.ROOT.MultiAnimal,'TrackletStitch');
          end
        end
        
        sPrmDflt = APTParameters.defaultParamsStructAll;
        sPrmAll = structoverlay(sPrmDflt,sPrmAll,...
          'dontWarnUnrecog',true); % to allow removal of obsolete params
      end
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
        nd = tPrm.findnode(['ROOT.' kk{ndx}]);
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
          if strfind(kk{ndx},'MultiAnimal.Detect')
            extra_str = ' (first stage)';
          else
            extra_str = ' (second stage)';
          end
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
          nd = tPrm.findnode(['ROOT.' kk{ndx}]);
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
      s.deeptrack = fullfile('trackers','dt','params_deeptrack.json');
      s.ma = fullfile('trackers','dt','params_ma.json');
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
        param_tree = parseConfigJson(json_file_path) ;
        if any(strcmp(fn,deepnets))
          % AL 20190711: automatically create requirements for all deep net
          %   param trees
          param_tree.traverse(@(x)set(x.Data,'Requirements',{fn,'isDeepTrack'})) ;
        end
        value = struct('json', {json_file_path}, 'tree', {param_tree}) ;
        s.(fn) = value ;
      end
      % under ma.detect, we have the same deeptrack structure
      if isfield(s,'ma') && isfield(s,'deeptrack'),
        t1 = s.ma.tree.findnode('ROOT.MultiAnimal.Detect.DeepTrack');
        t2 = s.deeptrack.tree.findnode('ROOT.DeepTrack');
        if isempty(t1.Children),
          t1.Children = TreeNode.empty(0,1);
        end      
        for i = 1:numel(t2.Children),
          t1.Children(end+1,1) = t2.Children(i).copy();
        end
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
    
  end  % methods (Static)
end  % classdef
