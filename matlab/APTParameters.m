classdef APTParameters
  properties (Constant)
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
      % first non-ROOT top-level field in parameter yaml 
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
      % note on netsUsed
      % Currently, topdown trackers include 2 'netsUsed'
      
      if isempty(tree.Children),
        
        [netsUsed,hasTrx,trackerIsDL] = myparse(varargin,...
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
        
        % AL20210901: note: in parameter yamls, the 'isTopDown' requirement
        % is used; but this actually means "isTD-2stg"; vs SA-trx which is
        % conceptually TD.
      
        reqs = tree.Data.Requirements;
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
            if ismember(net,reqs) && ~any(strcmp(net,netsUsed))
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
      
      if sPrm.ROOT.MultiAnimal.TargetCrop.AlignUsingTrxTheta && ...
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
        if isfield(sPrmAll.ROOT.ImageProcessing,'MultiTarget') && ...
           isfield(sPrmAll.ROOT.ImageProcessing.MultiTarget,'TargetCrop')
          sPrmAll.ROOT.MultiAnimal.TargetCrop = sPrmAll.ROOT.ImageProcessing.MultiTarget.TargetCrop;
          sPrmAll.ROOT.ImageProcessing.MultiTarget = ...
            rmfield(sPrmAll.ROOT.ImageProcessing.MultiTarget,'TargetCrop');
        end        
        if isfield(sPrmAll.ROOT.MultiAnimal.TargetCrop,'Radius')
          sPrmAll.ROOT.MultiAnimal.TargetCrop.ManualRadius = ...
             sPrmAll.ROOT.MultiAnimal.TargetCrop.Radius;
          sPrmAll.ROOT.MultiAnimal.TargetCrop = ...
            rmfield(sPrmAll.ROOT.MultiAnimal.TargetCrop,'Radius');
        end
        if isfield(sPrmAll.ROOT.MultiAnimal,'Detect') && ...
          isfield(sPrmAll.ROOT.MultiAnimal.Detect,'max_n_animals')
          sPrmAll.ROOT.MultiAnimal.max_n_animals = sPrmAll.ROOT.MultiAnimal.Detect.max_n_animals;
          sPrmAll.ROOT.MultiAnimal.min_n_animals = sPrmAll.ROOT.MultiAnimal.Detect.min_n_animals;
          sPrmAll.ROOT.MultiAnimal.Detect = ...
            rmfield(sPrmAll.ROOT.MultiAnimal.Detect,{'max_n_animals' 'min_n_animals'});
        end        
          
        sPrmDflt = APTParameters.defaultParamsStructAll;
        sPrmAll = structoverlay(sPrmDflt,sPrmAll,...
          'dontWarnUnrecog',true); % to allow removal of obsolete params
      end
    end
    
    function [tPrm,canceled,do_update]= autosetparams(tPrm,lobj)
      % automatically set the parameters based on labels.
      autoparams = compute_auto_params(lobj);
      kk = autoparams.keys();
      
      res = 'Update';
      canceled = false;
      do_update = false;
      % If values have been previously updated, then check if they are
      % significantly (>10%) different now
      %%
      dstr = '';
      diff = false;
      identical = true;
      default = true;

      for ndx = 1:numel(kk)
        nd = tPrm.findnode(['ROOT.' kk{ndx}]);
        prev_val = nd.Data.Value;
        cur_val = autoparams(kk{ndx});
        if (cur_val-prev_val)/(prev_val+0.001)>0.1
          diff = true;
        end
        if nd.Data.DefaultValue~=nd.Data.Value
          default = false;
        end
        if cur_val~=prev_val
          identical = false;
        end
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
        dstr = sprintf('Auto-computed parameters have changed by more than 10%% \nfor some of the parameters. Update the parameters? \n%s',dstr);
      else
        dstr = sprintf('Auto-computed parameters are similar to previous parameters \nfor all the parameters. Update the parameters? \n%s',dstr);
      end

      if diff && ~default
        res = questdlg(dstr,'Update parameters?','Update','Do not update','Cancel','Update');
        if strcmp(res,'Cancel')
          canceled = true;
          return;
        end
      elseif default
        res = 'Update';
      else
        % All parameters are identical
        do_udpate = false;
        return;
      end
      
      if strcmp(res,'Update')
        for ndx = 1:numel(kk)
          nd = tPrm.findnode(['ROOT.' kk{ndx}]);
          nd.Data.Value = autoparams(kk{ndx});
        end
        do_update = true;
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

function autoparams = compute_auto_params(lobj)

  assert(lobj.nview==1, 'Auto setting of parameters not tested for multivew');
  %%
  view = 1;
  autoparams = containers.Map();
  nmov = numel(lobj.labels);
  npts = lobj.nLabelPoints;
  all_labels = [];
  all_id = [];
  all_mov = [];
  pair_labels = [];
  cur_pts = ((view-1)*2*npts+1):(view*2*npts);
  for ndx = 1:nmov        
    if ~Labels.hasLbls(lobj.labels{ndx}), continue; end

    all_labels = [all_labels lobj.labels{ndx}.p(cur_pts,:,:)];
    n_labels = size(lobj.labels{ndx}.p,2);
    all_id = [all_id 1:n_labels];
    all_mov = [all_mov ones(1,n_labels)*ndx];

    if ~lobj.maIsMA, continue, end
    pair_done = [];
    big_val = 10000000;
    for fndx = 1:numel(lobj.labels{ndx}.frm)
      f = lobj.labels{ndx}.frm(fndx);
      if nnz(lobj.labels{ndx}.frm==f)>1
        idx = find(lobj.labels{ndx}.frm==f);
        for ix = idx(:)'
          if ix==fndx, continue, end
          if any(pair_done==(f*big_val+ix))
            continue
          end
          if any(pair_done==(ix*big_val+f))
            continue
          end

          pair_done(end+1) = ix*big_val+f;
          cur_pair = [lobj.labels{ndx}.p(cur_pts,fndx);lobj.labels{ndx}.p(cur_pts,ix)];
          pair_labels = [pair_labels cur_pair];

        end
      end

    end

  end
  all_labels = reshape(all_labels,npts,2,[]);
  pair_labels = reshape(pair_labels,npts,2,2,[]);
  % animals are along the second last dimension.
  % second dim has the coordinates

  %%
  l_min = reshape(min(all_labels,[],1),size(all_labels,[2,3]));
  l_max = reshape(max(all_labels,[],1),size(all_labels,[2,3]));
  l_span = l_max-l_min;
  % l_span is labels span in x and y direction

  l_span_pc = prctile(l_span,95,2);
  l_span_max = max(l_span,[],2);

  % Check and flag outliers..
  if any( (l_span_max./l_span_pc)>2)
    outliers = zeros(0,3);
    for jj = find( (l_span_max/l_span_pc)>2)
      ix = find(l_span(jj,:)>l_span_pc(jj)*2);
      for xx = ix(:)'
        mov = all_mov(xx);
        yy = all_id(xx);
        cur_fr = lobj.labels{mov}.frm(yy);
        cur_tgt = lobj.labels{mov}.tgt(yy);
        outliers(end+1,:) = [mov,cur_fr,cur_tgt];
      end
    end
    wstr = 'Some bounding boxes have sizes much larger than normal. This suggests that they may have labeing errors\n';
    wstr = sprintf('%s The list of examples is \n',wstr);
    for zz = 1:size(outliers,1)
      wstr = sprintf('%s Movie:%d, frame:%d, target:%d\n',wstr,outliers(zz,1),outliers(zz,2),outliers(zz,3));
    end
    warning(wstr);
  end

  crop_radius = max(l_span_pc);
  crop_radius = ceil(crop_radius/32)*32;
  if lobj.trackerIsTwoStage || lobj.hasTrx
    autoparams('MultiAnimal.TargetCrop.ManualRadius') = crop_radius;
  end

  % Look at distances between labeled pairs to find what to set for
  % tranlation range for data augmentation
  min_pairs = min(pair_labels,[],1);
  max_pairs = max(pair_labels,[],1);
  ctr_pairs = (max_pairs+min_pairs)/2;
  d_pairs = sqrt(sum( (ctr_pairs(:,:,1,:)-ctr_pairs(:,:,2,:)).^2,2));
  d_pairs = squeeze(d_pairs);
  d_pairs_pc = prctile(d_pairs,5);
  d_pairs_min = min(d_pairs_pc);

  % If the distances between center of animals is much less than the
  % span then warn about using bbox based methods
  if(d_pairs_pc<min(l_span_pc/10)) && lobj.trackerIsObjDet
    wstr = 'The distances between the center of animals is much smaller than the spans of the animals';
    wstr = sprintf('%s\n Avoid using object detection based top-down methods',wstr);
    warning(wstr);
  end

  trange_frame = min(lobj.movienr(view),lobj.movienc(view))/10;
  trange_pair = d_pairs_pc/2;
  trange_crop = min(l_span_pc)/10; 
  trange_animal = min(trange_pair,trange_crop);
  rrange = 15;
  align_theta = lobj.trackParams.ROOT.MultiAnimal.TargetCrop.AlignUsingTrxTheta;

  % Estimate the translation range and rotation ranges
  if ~lobj.maIsMA
    % Single animal autoparameters.
    if lobj.hasTrx
      % If has trx then just use defaults for rrange
      % If using cropping set trange to 10 percent of the crop size.
      trange = max(5,trange_animal);
      trange = round(trange/5)*5;
      if lobj.trackParams.ROOT.MultiAnimal.TargetCrop.AlignUsingTrxTheta
        rrange = 15;
      else
        rrange = 180;
      end
      autoparams('DeepTrack.DataAugmentation.rrange') = rrange;
      autoparams('DeepTrack.DataAugmentation.trange') = trange;
    else
      % No trx. 

      % Else set it to 10 percent of the crop size
      trange = max(5,trange_frame);
      % Try to guess rotation range for single animal
      % For this look at the angles from center of the frame/crop to
      % the labels
      l_thetas = atan2(all_labels(:,1,:)-lobj.movienc(view)/2,...
        all_labels(:,2,:)-lobj.movienr(view)/2);
      ang_span = get_angle_span(l_thetas)*180/pi;
      rrange = min(180,max(10,median(ang_span)/2));
      rrange = round(rrange/10)*10;
      autoparams('DeepTrack.DataAugmentation.rrange') = rrange;
    end
  else
    % Multi-animal. Two tranges for first and second stage. Applied
    % depending on the workflow
    trange_top = max(5,trange_frame);
    trange_top = round(trange_top/5)*5;
    trange_second = min(trange_crop,trange_animal);
    trange_second = max(5,trange_second);
    trange_second = round(trange_second/5)*5;

    if lobj.trackerIsTwoStage
      autoparams('DeepTrack.DataAugmentation.trange') = trange_second;
      if lobj.trackerIsObjDet
        % If uses object detection for the first stage then the look at
        % the variation in angles relative to the center
        mid_labels = mean(all_labels,1);
        l_thetas = atan2(all_labels(:,1,:)-mid_labels(:,1,:),...
          all_labels(:,2,:)-mid_labels(:,2,:));
        ang_span = get_angle_span(l_thetas)*180/pi;
        rrange = min(180,max(10,median(ang_span)/2));
        rrange = round(rrange/10)*10;
        autoparams('DeepTrack.DataAugmentation.rrange') = rrange;
      else
        % Using head-tail for the first stage
        align_trx_theta = lobj.trackParams.ROOT.MultiAnimal.TargetCrop.AlignUsingTrxTheta;

        if align_trx_theta
        % For head-tail based, find the angle span after aligning along
        % the head-tail direction.

          hd = all_labels(lobj.skelHead,:,:);
          tl = all_labels(lobj.skelTail,:,:);
          body_ctr = (hd+tl)/2;
          l_thetas = atan2(all_labels(:,1,:)-body_ctr(:,1,:),...
            all_labels(:,2,:)-body_ctr(:,2,:));
          l_thetas_r = mod(l_thetas - l_thetas(lobj.skelHead,:,:),2*pi);
          ang_span = get_angle_span(l_thetas_r)*180/pi;
        else
          % If not aligned along the head-tail then look at the angles
          % from the center
          warning('For head-tail based two-stage detection, align using head-tail is switched off. Aligning using head-tail will lead to better performance');
          mid_labels = mean(all_labels,1);
          l_thetas = atan2(all_labels(:,1,:)-mid_labels(:,1,:),...
            all_labels(:,2,:)-mid_labels(:,2,:));
          ang_span = get_angle_span(l_thetas)*180/pi;
        end
        rrange = min(180,max(10,median(ang_span)/2));
        rrange = round(rrange/10)*10;
        autoparams('DeepTrack.DataAugmentation.rrange') = rrange;
        
        mid_labels = mean(all_labels,1);
        l_thetas = atan2(all_labels(:,1,:)-mid_labels(:,1,:),...
          all_labels(:,2,:)-mid_labels(:,2,:));
        ang_span = get_angle_span(l_thetas)*180/pi;
        ang_span = ang_span([lobj.skelHead,lobj.skelTail]);
        rrange = min(180,max(10,median(ang_span)/2));
        rrange = round(rrange/10)*10;
        autoparams('MultiAnimal.Detect.DeepTrack.DataAugmentation.rrange') = rrange;

        autoparams('MultiAnimal.Detect.DeepTrack.DataAugmentation.trange') = trange_top;
      end
    else
      % Bottom up. Just look at angles of landmarks to the center to
      % see if the animals tend to be always aligned.
      mid_labels = mean(all_labels,1);
      l_thetas = atan2(all_labels(:,1,:)-mid_labels(:,1,:),...
        all_labels(:,2,:)-mid_labels(:,2,:));
      ang_span = get_angle_span(l_thetas)*180/pi;
      rrange = min(180,max(10,median(ang_span)/2));
      rrange = round(rrange/10)*10;
      autoparams('DeepTrack.DataAugmentation.rrange') = rrange;

    end
  end

end

function ang_span = get_angle_span(theta)
  % Find the span of thetas. Hacky method that rotates the pts by
  % 10degrees and then checks the span.
  ang_span = ones(size(theta,1),1)*2*pi;
  for offset = 0:10:360
    thetao = mod(theta + offset*pi/180,2*pi);
    cur_span = prctile(thetao,98,3) - prctile(thetao,2,3);
    ang_span = min(ang_span,cur_span);
  end
end

function [s,deepnets] = lclParamFileSpecs()
% Specifies/finds yamls in APT tree.
%
% Deepnets are found dynamically to easy adding new nets.

s.preprocess = 'params_preprocess.yaml';
s.track = 'params_track.yaml';
s.cpr = fullfile('trackers','cpr','params_cpr.yaml');
s.deeptrack = fullfile('trackers','dt','params_deeptrack.yaml');
s.ma = fullfile('trackers','dt','params_ma.yaml');
s.postprocess = 'params_postprocess.yaml';
aptroot = APT.getRoot;
dd = dir(fullfile(aptroot,'matlab','trackers','dt','params_deeptrack_*.yaml'));
dtyamls = {dd.name}';
sre = regexp(dtyamls,'params_deeptrack_(?<net>[a-zA-Z_]+).yaml','names');
for i=1:numel(dtyamls)
  s.(sre{i}.net) = fullfile('trackers','dt',dtyamls{i});
end
deepnets = cellfun(@(x)x.net,sre,'uni',0);
end

function s = lclInitParamFilesTrees()
%disp('APTParams init files trees');
%fprintf(1,'APTParameters:lclInitParamFilesTrees\n');
[specs,deepnets] = lclParamFileSpecs();
aptroot = APT.getRoot;
%[~,nets] = enumeration('DLNetType');

s = struct();
for f=fieldnames(specs)',f=f{1}; %#ok<FXSET>  
  s.(f).yaml = fullfile(aptroot,'matlab',specs.(f));
  yamlcontents = parseConfigYaml(s.(f).yaml);
  if any(strcmp(f,deepnets))
    % AL 20190711: automatically create requirements for all deep net
    %   param trees
    yamlcontents.traverse(@(x)set(x.Data,'Requirements',...
      {f,'isDeepTrack'}));
  end
  s.(f).tree = yamlcontents;
end
end
