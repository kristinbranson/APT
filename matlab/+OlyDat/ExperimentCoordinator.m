classdef ExperimentCoordinator < handle
  %ExperimentCoordinator Coordinate experiment selection across objects/plots.
  %   An ExperimentCoordinator is a hub for two-way sending/receiving of
  %   signals. Objects and/or plots may register with the coordinator to
  %   listen to the signals. They may call a method on the coordinator to
  %   send a signal. The signal is received by all listeners.
  %
  %   Typically, the signals used by OlyDat are vectors of experimentIDs,
  %   representing the current experiments that are selected or under
  %   consideration. The typical usage is, plots send signals when clicked,
  %   and when plots receive a signal, they update themselves to highlight
  %   the selected experiments.
  %
  %   Note that listeners that register with the Coordinator must remember to
  %   unregister when those listeners cease to exist. Failure to do this will
  %   likely lead to errors as the Coordinator attempts to call code on stale
  %   objects.
  %
  %   See also OlyDat.Plot, OlyDat.Plot/doPlot, OlyDat.Browser.
  
  % TODO: Can probably make unregistration optional for registered axes if
  % this turns out to be a hassle.
  
  properties (Hidden)
    fReceiverObjectInfo = struct('obj',cell(0,1),'method',cell(0,1)); % vector of receiver objects
    fReceiverAxisInfo = struct('ax',zeros(0,1),'fcn',cell(0,1)); % vector of receiver plots
  end
  
  properties (Dependent)
    fReceiverObjects; % vector cell array of currently registered objects
    fReceiverAxes; % vector handle array of currently registered axes
  end
  
  methods
    
    function objs = get.fReceiverObjects(obj)
      objs = {obj.fReceiverObjectInfo.obj}';
    end
    
    function axes = get.fReceiverAxes(obj)
      axes = [obj.fReceiverAxisInfo.ax]';
    end
    
  end
  
  
  methods
    
    function obj = ExperimentCoordinator
      % none
    end
    
    % Send a signal to all receivers.
    % src: the object/axis doing the sending
    % signal: the signal (typically, an array of experiment ids)
    function sendSignal(obj,src,signal) %#ok<INUSL>
      
      rcvrObjs = obj.fReceiverObjectInfo;
      for c = 1:numel(rcvrObjs)
        s = obj.fReceiverObjectInfo(c);
        feval(s.method,s.obj,signal);
      end
      rcvrAxes = obj.fReceiverAxisInfo;
      for c = 1:numel(rcvrAxes)
        s = obj.fReceiverAxisInfo(c);
        feval(s.fcn,signal);
      end
      
    end
    
    % Register an object to listen to signals. When a signal is sent,
    % the ExperimentCoordinator will call the given method on the given
    % object. The method should have signature method(obj,signals).
    function registerObject(obj,receiverObj,methodName)
      assert(isscalar(receiverObj) && isa(receiverObj,'handle'));
      assert(ischar(methodName));
      tfMatch = obj.findReceiverObjectIndex(receiverObj);
      if any(tfMatch)
        error('ExperimentCoordinator:existingReceiver',...
          'The receiver is already registered.')
      end
      obj.fReceiverObjectInfo(end+1,1).obj = receiverObj;
      obj.fReceiverObjectInfo(end).method = methodName;
    end
    
    % Register an axis to listen to signals. When a signal is sent,
    % the ExperimentCoordinator will call the given function handle
    % fcn. fcn should have the signature fcn(signals).
    function registerAxis(obj,receiverAx,fcn)
      assert(ishandle(receiverAx) && strcmp(get(receiverAx,'Type'),'axes'));
      assert(isa(fcn,'function_handle'));
      currentAxes = obj.fReceiverAxes;
      if ismember(receiverAx,currentAxes)
        error('ExperimentCoordinator:existingReceiver',...
          'The receiver is already registered.')
      end
      obj.fReceiverAxisInfo(end+1,1).ax = receiverAx;
      obj.fReceiverAxisInfo(end).fcn = fcn;
    end
    
    function unregisterObject(obj,receiverObj)
      tfMatch = obj.findReceiverObjectIndex(receiverObj);
      if any(tfMatch)
        obj.fReceiverObjectInfo(tfMatch,:) = [];
      else
        warning('ExperimentCoordinator:receiverNotRegistered',...
          'The receiver was not registered.');
      end
    end
    
    function unregisterAxis(obj,receiverAx)
      assert(isscalar(receiverAx) && ishandle(receiverAx) && strcmp(get(receiverAx,'Type'),'axes'));
      tfMatch = obj.fReceiverAxes==receiverAx;
      if any(tfMatch)
        obj.fReceiverAxisInfo(tfMatch,:) = [];
      else
        warning('ExperimentCoordinator:receiverNotRegistered',...
          'The receiver was not registered.');
      end
    end
    
  end
  
  methods (Access=private)
    
    function tf = findReceiverObjectIndex(obj,receiverObj)
      currentObjs = obj.fReceiverObjects;
      tf = cell2mat(cellfun(@(x)isequal(x,receiverObj),currentObjs));
      assert(nnz(tf)<=1); % Should have at most one match
    end
    
  end
  
end
