classdef InfoTimelineModel < handle
  % A domain model to hold application state that just happens to correspond
  % (roughly) to what is shown in the two APT timeline axes.  Normally the
  % Labeler object holds a reference to one of these in the infoTimelineModel
  % property.  Note that this class does not define any events, and any
  % mutation of it should happen via a suitable Labeler class method so that any
  % changes are reflected in the APT UI.

  properties (Constant)
    TLPROPFILESTR = 'landmark_features.yaml';
    TLPROPTYPES = {'Labels','Predictions','Imported','All Frames'};
  end

  properties  % Private by convention
    selectOn_  % scalar logical, if true, select "Pen" is down
    selectOnStartFrm_  % frame where selection started
    props_ % [nprop]. struct array of timeline-viewable property specs. Applicable when proptype is not 'Predictions'
    props_tracker_ % [ntrkprop]. ". Applicable when proptype is 'Predictions'
    props_allframes_ % [nallprop]. ". Applicable when proptype is All Frames
    proptypes_ % property types, eg 'Labels' or 'Predictions'
    curprop_ % row index into props, or props_tracker, depending on curproptype
    curproptype_ % row index into proptypes
    isdefault_ % whether this has been changed
    TLPROPS_  % struct array, features we can compute. Initted from yaml at construction-time
    TLPROPS_TRACKER_  % struct array, features for current tracker. Initted at setTracker time
    isSelectedFromFrameIndex_ = false(1,0)  % Internal record of what frames are shown as selected on the timeline
    custom_data_  % [1 x nframes] custom data to plot
    tldata_  % [nptsxnfrm] most recent data set/shown. NOT y-normalized
    statThresh_  % scalar, threshold value for timeline statistics
    isStatThreshVisible_  % scalar logical, whether to show threshold visualization
    tldata_ptype_  % the ptype used when tldata_ was last computed
    tldata_pcode_  % the pcode used when tldata_ was last computed
    tldata_iMov  % the iMov used when tldata_ was last computed
    tldata_iTgt  % the iTgt used when tldata_ was last computed
  end
  
  properties (Dependent)
    selectOn % scalar logical, if true, select "Pen" is down
    selectOnStartFrm % frame where selection started
    props % [nprop]. struct array of timeline-viewable property specs. Applicable when proptype is not 'Predictions'
    props_tracker % [ntrkprop]. ". Applicable when proptype is 'Predictions'
    props_allframes % [nallprop]. ". Applicable when proptype is All Frames
    proptypes % property types, eg 'Labels' or 'Predictions'
    curprop % row index into props, or props_tracker, depending on curproptype
    curproptype % row index into proptypes
    isdefault % whether this has been changed
    isSelectedFromFrameIndex
    custom_data % [1 x nframes] custom data to plot
    statThresh % scalar, threshold value for timeline statistics
    isStatThreshVisible % scalar logical, whether to show threshold visualization
  end

  methods
    function obj = InfoTimelineModel(hasTrx)
      obj.selectOn_ = false;
      obj.selectOnStartFrm_ = [];
      obj.proptypes_ = InfoTimelineModel.TLPROPTYPES(:);
      obj.curprop_ = 1;
      obj.curproptype_ = 1;
      obj.isdefault_ = true;
      obj.custom_data_ = [];
      obj.tldata_ = [];
      obj.statThresh_ = [];
      obj.isStatThreshVisible_ = false;
      obj.readTimelinePropsNew();
      obj.TLPROPS_TRACKER_ = EmptyLandmarkFeatureArray();
      obj.initializePropsEtc_(hasTrx);  % fires no events
    end
    
    function v = get.selectOn(obj)
      v = obj.selectOn_;
    end
        
    function setSelectMode(obj, newValue, currFrame)
      obj.selectOn_ = newValue ;
      if newValue
        obj.selectOnStartFrm_ = currFrame ;
      else
        obj.selectOnStartFrm_ = [] ;
      end
    end

    function v = get.selectOnStartFrm(obj)
      v = obj.selectOnStartFrm_;
    end

    function v = get.props(obj)
      v = obj.props_;
    end

    function v = get.props_tracker(obj)
      v = obj.props_tracker_;
    end

    function v = get.props_allframes(obj)
      v = obj.props_allframes_;
    end

    function v = get.proptypes(obj)
      v = obj.proptypes_;
    end

    function v = get.curprop(obj)
      v = obj.curprop_;
    end

    function v = get.curproptype(obj)
      v = obj.curproptype_;
    end

    function v = get.isdefault(obj)
      v = obj.isdefault_;
    end

    function result = get.isSelectedFromFrameIndex(obj)
      result = obj.isSelectedFromFrameIndex_ ;
    end

    function v = get.custom_data(obj)
      v = obj.custom_data_;
    end

    function v = get.statThresh(obj)
      v = obj.statThresh_;
    end

    function set.statThresh(obj, v)
      obj.statThresh_ = v;
    end

    function v = get.isStatThreshVisible(obj)
      v = obj.isStatThreshVisible_;
    end

    function toggleIsStatThreshVisible(obj)
      % Toggle threshold visualization state
      obj.isStatThreshVisible_ = ~obj.isStatThreshVisible_;
    end

    function result = getTimelineDataForCurrentMovieAndTarget(obj, labeler)
      if isempty(obj.tldata_) 
        doRecompute = true ;
      else
        [ptype,pcode] = obj.getCurPropSmart();
        iMov = labeler.currMovie;
        iTgt = labeler.currTarget;
        doRecompute = ~isequal(ptype, obj.tldata_ptype_) || ...
                      ~isequal(pcode, obj.tldata_pcode_) || ...
                      ~isequal(iMov, obj.tldata_iMov) || ...
                      ~isequal(iTgt, obj.tldata_iTgt);
      end
      if doRecompute
        obj.recomputeDataForCurrentMovieAndTarget_(labeler) ;
      end
      result = obj.tldata_;
    end
    
    function readTimelinePropsNew(obj)
      path = fullfile(APT.Root, 'matlab') ;
      tlpropfile = fullfile(path,InfoTimelineModel.TLPROPFILESTR);
      assert(logical(exist(tlpropfile,'file')), 'File %s is missing', tlpropfile);      
      obj.TLPROPS_ = ReadLandmarkFeatureFile(tlpropfile);      
    end

    function initializePropsEtc_(obj, hasTrx)
      % Set .props, .props_tracker from .TLPROPS, .TLPROPS_TRACKER
      
      % remove body features if no body tracking
      props = obj.TLPROPS_;
      if ~isempty(hasTrx) && ~hasTrx ,
        idxremove = strcmpi({props.coordsystem},'Body');
        props(idxremove) = [];
      end
      obj.props_ = props;      
      obj.props_tracker_ = cat(1,obj.props,obj.TLPROPS_TRACKER_);      
      obj.props_allframes_ = struct('name','Add custom...',...
        'code','add_custom',...
        'file','');      
    end

    function didChangeCurrentTracker(obj, propListOrEmpty)
      % Handle tracker change - update proptypes and props_tracker
      % Called by the parent Labeler.
      
      % Set .proptypes, .props_tracker
      if isempty(propListOrEmpty),
        % AL: Probably obsolete codepath
        obj.proptypes(strcmpi(obj.proptypes,'Predictions')) = [];
        obj.props_tracker_ = [];
      else
        if ~ismember('Predictions',obj.proptypes),
          obj.proptypes{end+1} = 'Predictions';
        end
        propList = propListOrEmpty ;
        obj.TLPROPS_TRACKER_ = propList ; %#ok<*PROPLC>
        obj.props_tracker_ = cat(1,obj.props,obj.TLPROPS_TRACKER_);
      end

      % Check that .curprop is in range for current .props,
      % .props_tracker, .curproptype. 
      ptype = obj.proptypes{obj.curproptype};
      switch ptype
        case 'Predictions'
          tfOOB = (obj.curprop > numel(obj.props_tracker));
        otherwise
          tfOOB = (obj.curprop > numel(obj.props)) ;
      end      
      if tfOOB
        obj.curprop_ = 1;
      end
    end  % function

    function tf = hasPredictionConfidence(obj)
      tf = ~isempty(obj.TLPROPS_TRACKER_);
    end
    
    function props = getPropsDisp(obj, ipropType)
      % Get available properties for given propType (idx)
      if nargin < 2,
        ipropType = obj.curproptype;
      end
      if strcmpi(obj.proptypes{ipropType},'Predictions'),
        props = {obj.props_tracker.name};
      elseif strcmpi(obj.proptypes{ipropType},'All Frames'),
        props = {obj.props_allframes.name};
      else
        props = {obj.props.name};
      end
    end
    
    function proptypes = getPropTypesDisp(obj)
      proptypes = obj.proptypes ;
    end  % function

    function tf = getCurPropTypeIsAllFrames(obj)
      tf = strcmpi(obj.proptypes{obj.curproptype},'All Frames');
    end

    function tf = getCurPropTypeIsLabel(obj)
      tf = strcmp(obj.proptypes{obj.curproptype},'Labels');
    end

    function initNewMovie(obj, isinit, hasMovie, nframes, hasTrx)
      if isinit || ~hasMovie || isnan(nframes)
        return
      end
      obj.clearSelection(nframes) ;
      obj.custom_data_ = [];
      obj.initializePropsEtc_(hasTrx) ;  % fires no events
      if obj.getCurPropTypeIsAllFrames()
        obj.curproptype_ = 1 ;
        obj.curprop_ = 1 ;
        obj.isdefault = true ;
      end      
    end    

    function addCustomFeature(obj, newprop)
      props_allframes = struct('name','Add custom...',...
        'code','add_custom',...
        'file','');
      obj.props_allframes_ = [newprop, props_allframes] ;
      obj.curprop_ = 1;
    end

    function addCustomFeatureGivenFileName(obj, file)
      % Add custom timeline feature from a .mat file
      % file: full path to .mat file containing variable 'x' with feature data
      %
      % Throws error if file cannot be loaded or doesn't contain required variable
      
      try
        d = load(file, 'x');
        obj.custom_data_ = d.x;
      catch ME
        error('Custom feature mat file must have a variable x which is 1 x nframes: %s', ME.message);
      end
      
      [~, f, ~] = fileparts(file);
      newprop = struct('name', ['Custom: ', f], 'code', 'custom', 'file', file);
      obj.addCustomFeature(newprop);
    end

    function bouts = selectGetSelectionAsBouts(obj)
      % Get currently selected bouts (can be noncontiguous)
      %
      % bouts: [nBout x 2]. col1 is startframe, col2 is one-past-endframe
      isSelectedFromFrameIndex = obj.isSelectedFromFrameIndex_ ;  % 1 x nframes
      [sp,ep] = get_interval_ends(isSelectedFromFrameIndex);
      bouts = [sp(:) ep(:)];
    end

    function didSetCurrFrame(obj, currFrame)
      % Called by the Labeler after currFrame is set.
      if obj.selectOn_
        f0 = obj.selectOnStartFrm_ ;
        f1 = currFrame ;
        if f1>f0
          idx = f0:f1;
        else
          idx = f1:f0;
        end
        obj.isSelectedFromFrameIndex_(:,idx) = true ;
      end
    end

    % function set.isSelectedFromFrameIndex(obj, newValue)
    %   obj.isSelectedFromFrameIndex_ = newValue ;
    % end

    function clearBout(obj, currentFrameIndex)  
      % Unselect the bout that currentFrameIndex is in.  If currentFrameIndex is not
      % in a bout, do nothing.
      isSelectedFromFrameIndex = obj.isSelectedFromFrameIndex_ ;
      bout = findBoutEdges(currentFrameIndex, isSelectedFromFrameIndex) ;
      if isempty(bout)
        return
      end
      isSelectedFromFrameIndex(:,bout(1):bout(2)) = false ;
      obj.isSelectedFromFrameIndex_ = isSelectedFromFrameIndex ;
    end  % function

    function clearSelection(obj, nframes)
      % Set the set of selected frames to the empty set.
      obj.selectOn_ = false ;
      obj.selectOnStartFrm_ = [] ;
      if isempty(nframes) || isnan(nframes) || nframes<0
        obj.isSelectedFromFrameIndex_ = false(size(obj.isSelectedFromFrameIndex_)) ;
      else
        obj.isSelectedFromFrameIndex_ = false(1, nframes) ;
      end
    end  % function

    function [ptype,prop] = getCurPropSmart(obj)
      % Get current proptype, and prop-specification-struct
      
      ptype = obj.proptypes{obj.curproptype};
      switch ptype
        case 'Predictions'
          prop = obj.props_tracker(obj.curprop);
        otherwise
          prop = obj.props(obj.curprop);
      end
    end

    function recomputeDataForCurrentMovieAndTarget_(obj, labeler)
      % Recompute the timeline data for current movie/target
      % labeler: Labeler object for accessing data sources
      
      [ptype,pcode] = obj.getCurPropSmart();
      iMov = labeler.currMovie;
      iTgt = labeler.currTarget;
      
      if isempty(iMov) || iMov==0 
        tldata = nan(labeler.nLabelPoints,1);
      else
        switch ptype
          case {'Labels','Imported'}
            needtrx = labeler.hasTrx && strcmpi(pcode.coordsystem,'Body');
            if needtrx,
              trxFile = labeler.trxFilesAllFullGTaware{iMov,1};
              bodytrx = labeler.getTrx(trxFile,labeler.movieInfoAllGTaware{iMov,1}.nframes);
              bodytrx = bodytrx(iTgt);
            else
              bodytrx = [];
            end
            
            nfrmtot = labeler.nframes;
            if strcmp(ptype,'Labels'),
              s = labeler.labelsGTaware{iMov};
              [tfhasdata,lpos,lposocc,lpost0,lpost1] = Labels.getLabelsT(s,iTgt);
              lpos = reshape(lpos,size(lpos,1)/2,2,[]);
            else
              s = labeler.labels2GTaware{iMov};
              if labeler.maIsMA
                % Use "current Tracklet" for imported data
                if ~isempty(labeler.labeledpos2trkViz)
                  iTgt = labeler.labeledpos2trkViz.currTrklet;
                  if isnan(iTgt)
                    warningNoTrace('No Tracklet currently selected; showing timeline data for first tracklet.');
                    iTgt = 1;
                  end
                else
                  iTgt = 1;
                end
              end  
              [tfhasdata,lpos,lposocc,lpost0,lpost1] = s.getPTrkTgt2(iTgt);
            end
            if tfhasdata
              tldata = ComputeLandmarkFeatureFromPos(...
                lpos,lposocc,lpost0,lpost1,nfrmtot,bodytrx,pcode);
            else
              tldata = nan(labeler.nLabelPoints,1); % looks like we don't need 2nd dim to be nfrmtot
            end
          case 'Predictions'
            % AL 20200511 hack, initialization ordering. If the timeline
            % pum has 'Predictions' selected and a new project is loaded,
            % the trackers are not updated (via
            % LabelerGUI/cbkCurrTrackerChanged) until after a movieSetGUI()
            % call which leads here.
            tracker = labeler.tracker ;
            if ~isempty(tracker) && isvalid(tracker)
              tldata = tracker.getPropValues(pcode);
            else
              tldata = nan(labeler.nLabelPoints,1);
            end
          case 'All Frames'
            %fprintf('getDataCurrMovTarg -> All Frames, %d\n',obj.curprop);
            if strcmpi(obj.props_allframes(obj.curprop).name,'Add custom...'),
              tldata = nan(labeler.nLabelPoints,1);
            else
              tldata = obj.custom_data;
            end
          otherwise
            error('Unknown data type %s',ptype);
        end
        %szassert(data,[labeler.nLabelPoints obj.nfrm]);
      end

      % Turn infs to nans
      tldata(isinf(tldata)) = nan;

      % Store the computed data for external access (e.g., navigation)
      % Also store data that allows us to know when .tldata_ needs to be recomputed
      obj.tldata_ = tldata;
      obj.tldata_ptype_ = ptype ;
      obj.tldata_pcode_ = pcode ;
      obj.tldata_iMov = iMov ;
      obj.tldata_iTgt = iTgt ;
    end  % function
    
    function setCurrentPropertyType(obj, iproptype, iprop)
      % iproptype, iprop assumed to be consistent already.
      obj.curproptype_ = iproptype;
      obj.curprop_ = iprop;
      obj.isdefault_ = false ;
    end
    
  end  % methods  
end  % classdef
