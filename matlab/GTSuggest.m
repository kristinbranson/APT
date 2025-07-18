function output = GTSuggest(lObj)

assert(isa(lObj,'Labeler'));
%obj.hFig = MovieManager(obj);
handles = struct;
handles.labeler = lObj;

hFig = uifigure('Units','pixels','Position',[951,1400,733,436],...
  'Name','To-Label List');
handles.figure1 = hFig;
%hFig.CloseRequestFcn = @(hObject,eventdata) CloseRequestFcn(hObject,eventdata); % todo

handles.gl = uigridlayout(hFig,[3,1],'RowHeight',{'1x','fit',40},'tag','gl');
handles.panelMovie = uipanel('Title','Movies','Parent',handles.gl);

% seems like uigridlayout and uibuttongroup don't play well together
rbh = 25;
rbs = 2;
rbw = 400;
handles.glMovie = uigridlayout(handles.panelMovie,[2,1],'RowHeight',{2*rbh+rbs+2,'1x'});
handles.btngrpMovies = uibuttongroup(handles.glMovie,...
  'SelectionChangedFcn',@btngrpMoviesSelection,...
  'BorderType','none');
handles.rbAllMovies = uiradiobutton(handles.btngrpMovies,'Tag','rbAllMovies','Text','All movies',...
  'Position',[0,rbh+2,rbw,rbh]);
handles.rbSelectedMovies = uiradiobutton(handles.btngrpMovies,'Tag','rbSelectedMovies','Text','Selected movies',...
  'Position',[0,0,rbw,rbh]);
handles.moviefiles = lObj.movieFilesAllGTFull(:,1);
handles.listMovies = uilistbox(handles.glMovie,...
  'tag','listMovies',...
  'Multiselect','on',...
  'Items',handles.moviefiles,...
  'Value',handles.moviefiles,...
  'Enable','on'); % todo

handles.panelFrames = uipanel('Title','Frames','Parent',handles.gl);
handles.glFrames = uigridlayout(handles.panelFrames,[3,2]);
handles.labelNFrames = uilabel(handles.glFrames,'Text','N. frames to label','HorizontalAlignment','right');
handles.labelNFrames.Layout.Column = 1;
handles.labelNFrames.Layout.Row = 1;
handles.etNumFrames = uieditfield(handles.glFrames,'numeric','Tag','etNumFrames',...
  "Limits",[1 inf], ...
  "LowerLimitInclusive","on");
handles.etNumFrames.Layout.Column = 2;
handles.etNumFrames.Layout.Row = 1;
handles.dropdownItems = {'In total','Per movie'};
handles.dropdownTags = {'rbInTotal','rbPerMovie'};
if lObj.hasTrx,
  handles.dropdownItems{end+1} = 'Per animal';
  handles.dropdownTags{end+1} = 'rbPerAnimal';
end
handles.dropdownFrames = uidropdown(handles.glFrames,'Items',handles.dropdownItems);
handles.dropdownFrames.Layout.Column = 2;
handles.dropdownFrames.Layout.Row = 2;

handles.labelDistTrainingFrames = uilabel(handles.glFrames,'Text','Minimum distance to training frames',...
  'HorizontalAlignment','right');
handles.labelDistTrainingFrames.Layout.Column = 1;
handles.labelDistTrainingFrames.Layout.Row = 3;
handles.etMinDistTrainingFrames = uieditfield(handles.glFrames,'numeric','Tag','etMinDistTrainingFrames',...
  "Limits",[0 inf], ...
  "LowerLimitInclusive","on");
handles.glButtons = uigridlayout(handles.gl,[1,4],'Padding',[0,0,0,0],'tag','glButtons');
handles.pbGenerateSelections = uibutton(handles.glButtons,'Text','Generate','tag','pbGenerateSelections',...
  'ButtonPushedFcn',@pbGenerateSelections_Callback);
handles.pbGenerateSelections.Layout.Column = 2;
handles.pbCancel = uibutton(handles.glButtons,'Text','Cancel','tag','pbCancel',...
  'ButtonPushedFcn',@pbCancel_Callback);
handles.pbCancel.Layout.Column = 3;

setGUIfromRC(handles);
set(handles.listMovies,'Enable',onIff(handles.rbSelectedMovies.Value==1));
handles.output = [];

uiwait(hFig);

output = handles.output;

  function btngrpMoviesSelection(src,evt)

    set(handles.listMovies,'Enable',onIff(handles.rbSelectedMovies.Value==1));
  
  end

  function setGUIfromRC(handles)

    labeler = handles.labeler;
    tag = labeler.rcGetProp('gtsuggest_btngrpMovies_selectedObject');
    if ~isempty(tag) && isfield(handles,tag)
      handles.btngrpMovies.SelectedObject = handles.(tag);
    else
      handles.btngrpMovies.SelectedObject = handles.rbAllMovies;
    end

    tag = labeler.rcGetProp('gtsuggest_btngrpFrames_selectedObject');
    if isempty(tag) || ~ismember(tag,handles.dropdownTags),
      tag = 'rbInTotal';
    end
    handles.dropdownFrames.Value = handles.dropdownItems{strcmp(handles.dropdownTags,tag)};

    val = labeler.rcGetProp('gtsuggest_numFrames');
    if ~isempty(val)
      handles.etNumFrames.Value = val;
    else
      val = 100;
      handles.etNumFrames.Value = val;
    end

    val = labeler.rcGetProp('gtsuggest_minDistTrainingFrames');
    if ~isempty(val)
      handles.etMinDistTrainingFrames.Value = val;
    else
      val = 1000;
      handles.etMinDistTrainingFrames.String = val;
    end

  end

  function setRCfromGUI(handles)

    labeler = handles.labeler;

    labeler.rcSaveProp('gtsuggest_btngrpMovies_selectedObject',...
      handles.btngrpMovies.SelectedObject.Tag);
    labeler.rcSaveProp('gtsuggest_btngrpFrames_selectedObject',...
      handles.dropdownTags{strcmp(handles.dropdownItems,handles.dropdownFrames.Value)});
    labeler.rcSaveProp('gtsuggest_numFrames',...
      handles.etNumFrames.Value);
    labeler.rcSaveProp('gtsuggest_minDistTrainingFrames',...
      handles.etMinDistTrainingFrames.Value);

  end

  function pbGenerateSelections_Callback(src,evt)

    gtsg = getCurrentConfig(handles);
    setRCfromGUI(handles);
    handles.output = gtsg;
    guidata(handles.figure1,handles);
    close(handles.figure1);

  end

  function [gtsg,tfsucc,msg] = getCurrentConfig(handles)

    gtsg = [];
    tfsucc = false;
    msg = '';

    s = struct();
    s.numFrames = ceil(handles.etNumFrames.Value);
    s.minDistTraining = ceil(handles.etMinDistTrainingFrames.Value);

    movSelObj = handles.btngrpMovies.SelectedObject;
    if movSelObj==handles.rbAllMovies
      s.movSet = MovieIndexSetVariable.AllGTMov;
      s.movIdxs = s.movSet.getMovieIndices(handles.lObj);

    elseif movSelObj==handles.rbSelectedMovies
      s.movSet = MovieIndexSetVariable.SelMov;
      moviepathsel = handles.listMovies.Value;
      if isempty(moviepathsel),
        msg = 'At least one movie must be selected';
        return;
      end
      [ism,loc] = ismember(moviepathsel,handles.labeler.movieFilesAllGTFull(:,1));
      if ~all(ism),
        warning('This should not happen, somehow selected movies are not within GT movie list');
        loc = loc(ism);
        if ~any(ism),
          warning('Setting to all GT movies');
          loc = 1:numel(handles.labeler.movieFilesAllGTFull(:,1));
        end
      end
      s.movIdxs = MovieIndex(loc,true);
    else
      assert(false);
    end

    switch lower(handles.dropdownFrames.Value)
      case 'in total'
        s.frmSpecType = GTSetNumFramesType.Total;
      case 'per movie'
        s.frmSpecType = GTSetNumFramesType.PerMovie;
      case 'per animal'
        s.frmSpecType = GTSetNumFramesType.PerTarget;
      otherwise
        assert(false);
    end

    gtsg = GTSetGenerator(s.numFrames,s.frmSpecType,s.minDistTraining,s.movIdxs);
    tfsucc = true;

  end

  function pbCancel_Callback(src, evt)

    close(handles.figure1);

  end

end