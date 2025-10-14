function SelectTrackingAlgorithm(varargin)
% SelectTrackingAlgorithm(lObj,hPar)
% GUI for selecting a deep learning tracking algorithm. After
% selection, it will create a new tracker with the function 
% lObj.trackMakeNewTrackerGivenNetTypes(nettypes)

if nargin == 2 && isa(varargin{1},'Labeler'),
  lObj = varargin{1};
  hPar = varargin{2};
else
  feval(varargin{:});
  return;
end

handles = struct;
handles.hPar = hPar;
handles.isMA = lObj.maIsMA;

handles.figure = uifigure('Name','Tracking algorithm',...
  'Units','pixels', ...
  'Position',[100,100,600,800], ...
  'WindowStyle', 'modal', ...
  'Resize', 'on', ...
  'Visible','on', ...%'CloseRequestFcn', @cancel_callback,...
  'Tag','figure_TrackingAlgorithm') ;
centerOnParentFigure(handles.figure,hPar);

[handles.maposenets,handles.mabboxnets,handles.saposenets] = Labeler.getAllTrackerTypes();

handles.trackercurr_types = lObj.trackGetCurrTrackerStageNetTypes();

% DLTrackerBottomUp: trackercurr.trnNetType
% DLTrackerTopDown: trackercurr.stage1Tracker.trnNetType +
% trackercurr.trnNetType

if handles.isMA,
  handles.gl = uigridlayout(handles.figure,[4,1],...
    'RowHeight',{'fit','1x','1x','fit'},'tag','gl');

  handles.panel_paradigm = uipanel(handles.gl,...
    'Tag','panel_paradigm');
  gl1 = uigridlayout(handles.panel_paradigm,[2,1],...
    'Tag','gl_paradigm1','RowHeight',{'fit','fit'});
  gl2 = uigridlayout(gl1,[1,2],...
    'Tag','gl_paradigm2','ColumnWidth',{'fit','fit'},'Padding',[0,0,0,0]);
  handles.label_paradigm = uilabel('Parent',gl2,'Text','Paradigm',...
    'Tag','label_paradigm','HorizontalAlignment','right');
  handles.dropdown_paradigm = uidropdown('Parent',gl2,'Items',...
    ["One-Stage/Bottom-Up","Two-Stage/Top-Down"],...
    'Tag','dropdown_paradigm',...
    'ValueChangedFcn',@cbkDropdownParadigm);
  handles.dropdown_paradigm.ValueIndex = numel(handles.trackercurr_types);
  handles.label_paradigmdesc = uilabel('Parent',gl1,'WordWrap','on',...
    'Text',{['{\bf Two-stage/Top-down} algorithms localize the animal(s) in the first stage, ',...
    'then fit the detailed pose for each animal independently in the second stage. This allows the tracker to focus its ',...
    ' attention on the parts of the image where the animals are. Two-stage algorithms work better when the animals are ',...
    'small compared to the video frame size, but may make more mistakes when there are very close interactions, as ',...
    'the second stage operates independently per animal. ',...
    'Stage-one localization may either fit a bounding box around each animal or detect where the "head" and "tail" ',...
    'of each animal is. '],...
    ['{\bf One-stage/Bottom-up} algorithms try to jointly fit the detailed pose of all animals simultaneously. ',...
    'One-stage algorithms may work better when animals have very close social interactions or ',...
    'occlude each other, but may lack resolution if the animals are very small compared to the video frame.']},...
    'Interpreter','latex',...
    'Tag','label_paradigmdesc');

  % if this is bottom-up, we will hide stage 2 panel and rename stage 1 in
  % the update function
  maxnstages = 2;
  handles.topdown_stage_names = {'Stage 1/Detection algorithm','Stage 2/Pose algorithm'};
  handles.bottomup_stage_name = 'Bottom-up pose algorithm';

  handles.bottomup_algorithms = {handles.maposenets.displayString};
  handles.bottomup_nets = handles.maposenets;

  handles.topdown_algorithms = {{},{}};
  handles.topdown_nets = {[handles.maposenets;handles.mabboxnets],handles.saposenets};
  for i = 1:numel(handles.maposenets),
    name = [strrep(handles.maposenets(i).displayString,'MultiAnimal ',''),' (head/tail)'];
    handles.topdown_algorithms{1}{end+1} = name;
  end
  for i = 1:numel(handles.mabboxnets),
    name = [strrep(handles.mabboxnets(i).displayString,' Object Detection',''),' (bounding box)'];
    handles.topdown_algorithms{1}{end+1} = name;
  end
  handles.topdown_algorithms{2} = {handles.saposenets.displayString};


  handles.nstages = numel(handles.trackercurr_types);

  if handles.nstages == 1,
    handles.algorithms = {handles.bottomup_algorithms};
    handles.nets = {handles.bottomup_nets};
  else
    handles.algorithms = handles.topdown_algorithms;
    handles.nets = handles.topdown_nets;
  end
else
  handles.gl = uigridlayout(handles.figure,[4,1],...
    'RowHeight',{'1x',40},'tag','gl');
  maxnstages = 1;
  handles.sa_stage_name = 'Pose algorithm';
  handles.stage_names = {handles.sa_stage_name};
  handles.sa_algorithms = {handles.saposenets.displayString};
  handles.algorithms = {handles.sa_algorithms};
  handles.sa_nets = handles.saposenets;
  handles.nets = {handles.sa_nets};
  handles.nstages = 1;
end

handles.last_algorithm_idx = ones(1,maxnstages);
for stage = 1:handles.nstages,
  handles.last_algorithm_idx(stage) = find(strcmp(handles.trackercurr_types(stage).shortString,{handles.nets{stage}.shortString}));
end

handles.panel_stages = gobjects(maxnstages,1);
handles.listbox_stages = gobjects(maxnstages,1);
handles.label_desc_stages = gobjects(maxnstages,1);
for stage = 1:maxnstages,

  handles.panel_stages(stage) = uipanel(handles.gl,'Tag',sprintf('panel_stage%d',stage));
  gl1 = uigridlayout(handles.panel_stages(stage),[2,1],...
    'Tag',sprintf('gl_stage%d_1',stage),'RowHeight',{'1x','fit'});
  handles.listbox_stages(stage) = uilistbox('Parent',gl1,...
    'Tag',sprintf('listbox_stage%d',stage),...
    'ValueChangedFcn',@(src,evt) cbkListboxStage(stage,src,evt));
  handles.label_desc_stages(stage)= uilabel('Parent',gl1,'WordWrap','on',...
    'Text','[Description]',...
    'Interpreter','html',...
    'Tag',sprintf('label_desc_stage%d',stage));

end

gl1 = uigridlayout(handles.gl,[1,4],'Padding',[0,0,0,0]);
handles.pb_ok = uibutton(gl1,'push',...
  'Text','OK', ....
  'ButtonPushedFcn',@cbkPbOK,...
  'Tag','pb_ok');
handles.pb_ok.Layout.Column = 2;
handles.pb_cancel = uibutton(gl1,'push',...
  'Text','Cancel', ....
  'ButtonPushedFcn',@cbkPbCancel,...
  'Tag','pb_cancel');
handles.pb_cancel.Layout.Column = 3;

if ~handles.isMA,
  handles.listbox_stages.Items = handles.algorithms{1};
  handles.listbox_stages.ValueIndex = handles.last_algorithm_idx;
end
update();
handles.figure.Visible = 'on';
uiwait(handles.figure);
% disp(handles);

  function update()

    % isMA is not expected to change
    if handles.isMA,
      if handles.dropdown_paradigm.ValueIndex == 1,
        handles.panel_stages(2).Visible = 'off';
        handles.gl.RowHeight{3} = 0;
        handles.stage_names = {handles.bottomup_stage_name};
        handles.algorithms = {handles.bottomup_algorithms};
        handles.nets = {handles.bottomup_nets};
        handles.nstages = 1;
      else % two-stage
        handles.panel_stages(2).Visible = 'on';
        handles.gl.RowHeight{3} = '1x';
        handles.stage_names = handles.topdown_stage_names;
        handles.algorithms = handles.topdown_algorithms;
        handles.nets = handles.topdown_nets;
        handles.nstages = 2;
      end

      for s = 1:handles.nstages,
        handles.panel_stages(s).Title = handles.stage_names{s};
        handles.listbox_stages(s).Items = handles.algorithms{s};
      end

    end
    updateStagePanels();

  end

  function updateStagePanels()
    for s = 1:handles.nstages,
      handles.listbox_stages(s).ValueIndex = min(numel(handles.algorithms{s}),handles.last_algorithm_idx(s));
      description = char(handles.nets{s}(handles.listbox_stages(s).ValueIndex).description);  % char() handles []
      handles.label_desc_stages(s).Text = description ;
    end
  end

  function cbkDropdownParadigm(src,evt)  %#ok<INUSD>
    update();
  end

  function cbkListboxStage(stage,src,evt)  %#ok<INUSD>
    handles.last_algorithm_idx(stage) = evt.ValueIndex;
    updateStagePanels();
  end

  function cbkPbOK(src,evt)  %#ok<INUSD>

    nettypes = handles.nets{1}(handles.last_algorithm_idx(1));
    for s = 2:handles.nstages,
      nettypes(s) = handles.nets{s}(handles.last_algorithm_idx(s));
    end
    
    lObj.trackMakeNewTrackerGivenNetTypes(nettypes);
    delete(handles.figure);
  end

  function cbkPbCancel(src,evt)  %#ok<INUSD>
    delete(handles.figure);
  end

end
