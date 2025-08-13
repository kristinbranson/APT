classdef InfoTimelineModel < handle

  properties (Constant)
    TLPROPFILESTR = 'landmark_features.yaml';
    TLPROPTYPES = {'Labels','Predictions','Imported','All Frames'};
  end

  properties
    lObj  % Labeler object that created this model
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
    TLPROPS  % struct array, features we can compute. Initted from yaml at construction-time
    TLPROPS_TRACKER  % struct array, features for current tracker. Initted at setTracker time
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
  end

  events
    updateTimelineProperties % fired when props, props_tracker, or props_allframes changes
    updateTimelinePropertyTypes % fired when proptypes changes
  end
  
  methods
    function obj = InfoTimelineModel(labeler)
      % labeler: Labeler object that owns this model
      obj.lObj = labeler;
      obj.selectOn_ = false;
      obj.selectOnStartFrm_ = [];
      obj.proptypes_ = InfoTimelineModel.TLPROPTYPES(:);
      obj.curprop_ = 1;
      obj.curproptype_ = 1;
      obj.isdefault_ = true;
      obj.readTimelinePropsNew();
      obj.TLPROPS_TRACKER = EmptyLandmarkFeatureArray();
      obj.initializePropsEtc();
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

    function set.props(obj, v)
      obj.props_ = v;
      notify(obj, 'updateTimelineProperties');
    end

    function v = get.props_tracker(obj)
      v = obj.props_tracker_;
    end

    function set.props_tracker(obj, v)
      obj.props_tracker_ = v;
      notify(obj, 'updateTimelineProperties');
    end

    function v = get.props_allframes(obj)
      v = obj.props_allframes_;
    end

    function set.props_allframes(obj, v)
      obj.props_allframes_ = v;
      notify(obj, 'updateTimelineProperties');
    end

    function v = get.proptypes(obj)
      v = obj.proptypes_;
    end

    function set.proptypes(obj, v)
      obj.proptypes_ = v;
      notify(obj, 'updateTimelinePropertyTypes');
    end

    function v = get.curprop(obj)
      v = obj.curprop_;
    end

    function set.curprop(obj, v)
      obj.curprop_ = v;
    end

    function v = get.curproptype(obj)
      v = obj.curproptype_;
    end

    function set.curproptype(obj, v)
      obj.curproptype_ = v;
    end

    function v = get.isdefault(obj)
      v = obj.isdefault_;
    end

    function set.isdefault(obj, v)
      obj.isdefault_ = v;
    end

    function selectInit(obj)
      obj.selectOn_ = false;
      obj.selectOnStartFrm_ = [];
    end  % function

    function readTimelinePropsNew(obj)
      path = fullfile(APT.Root, 'matlab') ;
      tlpropfile = fullfile(path,InfoTimelineModel.TLPROPFILESTR);
      assert(logical(exist(tlpropfile,'file')), 'File %s is missing', tlpropfile);      
      obj.TLPROPS = ReadLandmarkFeatureFile(tlpropfile);      
    end

    function initializePropsAllFrames(obj)
      obj.props_allframes = struct('name','Add custom...',...
        'code','add_custom',...
        'file','');
    end

    function initializePropsTracker(obj)
      obj.props_tracker = cat(1,obj.props,obj.TLPROPS_TRACKER);      
    end

    function initializePropsEtc(obj)
      % Set .props, .props_tracker from .TLPROPS, .TLPROPS_TRACKER
      
      % remove body features if no body tracking
      props = obj.TLPROPS;
      if ~obj.lObj.hasTrx,
        idxremove = strcmpi({props.coordsystem},'Body');
        props(idxremove) = [];
      end
      obj.props = props;      
      obj.initializePropsTracker();
      obj.initializePropsAllFrames();
    end
  end  % methods  
end  % classdef
