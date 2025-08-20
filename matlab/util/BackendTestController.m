classdef BackendTestController < handle
  properties
    parent_
    labeler_
    figure_
    edit_
    listener_
  end

  methods
    function obj = BackendTestController(parent, labeler)
      obj.parent_ = parent ;  % a LabelerController
      obj.labeler_ = labeler ;
      backend = labeler.backend ;
      windowTitle = sprintf('Test %s Backend', backend.prettyName()) ;
      obj.figure_ = ...
        dialog('Name',windowTitle, ...
               'Color',[0 0 0], ...
               'WindowStyle','normal');
      obj.edit_ = ...
        uicontrol('Parent',obj.figure_,...
                  'Style','edit',...
                  'Units','normalized',...
                  'Position',[.05,.05,.9,.9],...
                  'Enable','inactive',...
                  'Min',0,...
                  'Max',10,...
                  'HorizontalAlignment','left',...
                  'BackgroundColor',[.1 .1 .1],...
                  'ForegroundColor',[0 1 0]);
      obj.listener_ = addlistener(labeler, 'updateBackendTestText', @(s,e)(obj.update())) ;      
      pause(0.05) ;  % This should be needed, but seemingly is.  
      obj.figure_.CloseRequestFcn = @(s,e)(obj.parent_.backendTestFigureCloseRequested()) ;
    end  % function

    function update(obj)
      text = obj.labeler_.backend.testText() ;
      obj.edit_.String = text ;
    end

    function delete(obj)
      if isvalid(obj.listener_) 
        delete(obj.listener_) ;
      end
      deleteValidGraphicsHandles(obj.figure_) ;
    end
  end  % methods
end  % classdef
