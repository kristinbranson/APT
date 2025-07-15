function SelectTrackingAlgorithm(varargin)

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

[handles.maposenets,handles.mabboxnets,handles.saposenets] = lObj.getAllTrackerTypes();

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
disp(handles);

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
      handles.label_desc_stages(s).Text = handles.nets{s}(handles.listbox_stages(s).ValueIndex).description;
    end
  end

  function cbkDropdownParadigm(src,evt)
    update();
  end

  function cbkListboxStage(stage,src,evt)
    handles.last_algorithm_idx(stage) = evt.ValueIndex;
    updateStagePanels();
  end

  function cbkPbOK(src,evt)

    nettypes = handles.nets{1}(handles.last_algorithm_idx(1));
    for s = 2:handles.nstages,
      nettypes(s) = handles.nets{s}(handles.last_algorithm_idx(s));
    end
    
    tfsucc = lObj.trackMakeNewTrackerGivenNetTypes(nettypes);
    if ~tfsucc,
      error('Something went wrong -- did not find a match for selected network');
    end
    delete(handles.figure);
  end
  function cbkPbCancel(src,evt)
    delete(handles.figure);
  end

end

% 
% 
% s1_list = uilistbox(fig,...
%   'Position',[10 60 235 200],...
%   'Items',stg1nets_str);
% s2_list = uilistbox(fig,...
%                       'Position',[255 60 235 200],...
%                       'Items',stg2nets_str);
%   uilabel(fig,...
%           'Text','Network for Stage 1 (Detection)',...
%           'Position',[10 265 235 20]);
%   uilabel(fig,...
%           'Text','Network for Stage 2 (Pose)',...
%           'Position',[255 265 235 20]);
%   pb_ok = uibutton(fig,'push',...
%                    'Position',[50 10 150 40],...
%                    'Text','OK', ....
%                    'ButtonPushedFcn',@ok_callback);  %#ok<NASGU>
%   pb_cancel = uibutton(fig,'push',...
%                        'Position',[300 10 150 40],...
%                        'Text','Cancel', ....
%                        'ButtonPushedFcn',@cancel_callback);  %#ok<NASGU>
%   fig.Visible = 'on' ;
%   uiwait(fig);
% 
% 
% 
% 
% function [stg1net, stg2net] = get_custom_two_stage_tracker_nets_ui(mainFigure, stg1nets, stg2nets)
%   stg1net = [];
%   stg2net = [];
%   stg1nets_str = cell(1, numel(stg1nets)) ;
%   for ndx = 1:numel(stg1nets)
%     stg1nets_str{ndx} = stg1nets(ndx).displayString;
%   end
%   stg2nets_str = cell(1, numel(stg2nets)) ;
%   for ndx = 1:numel(stg2nets)
%     stg2nets_str{ndx} = stg2nets(ndx).displayString;
%   end
%   fig = uifigure('Name','Custom Top-Down Tracker',...
%                  'Units','pixels', ...
%                  'Position',[100,100,500,300], ...
%                  'WindowStyle', 'modal', ...
%                  'Resize', 'off', ...
%                  'Visible','off', ...
%                  'CloseRequestFcn', @cancel_callback) ;
%   centerOnParentFigure(fig,mainFigure);
%   s1_list = uilistbox(fig,...
%                       'Position',[10 60 235 200],...
%                       'Items',stg1nets_str);
%   s2_list = uilistbox(fig,...
%                       'Position',[255 60 235 200],...
%                       'Items',stg2nets_str);
%   uilabel(fig,...
%           'Text','Network for Stage 1 (Detection)',...
%           'Position',[10 265 235 20]);
%   uilabel(fig,...
%           'Text','Network for Stage 2 (Pose)',...
%           'Position',[255 265 235 20]);
%   pb_ok = uibutton(fig,'push',...
%                    'Position',[50 10 150 40],...
%                    'Text','OK', ....
%                    'ButtonPushedFcn',@ok_callback);  %#ok<NASGU>
%   pb_cancel = uibutton(fig,'push',...
%                        'Position',[300 10 150 40],...
%                        'Text','Cancel', ....
%                        'ButtonPushedFcn',@cancel_callback);  %#ok<NASGU>
%   fig.Visible = 'on' ;
%   uiwait(fig);
% 
%   function cancel_callback(src,event)  %#ok<INUSD>
%     stg1net = [];
%     stg2net = [];
%     delete(fig);
%   end
% 
%   function ok_callback(src,event)  %#ok<INUSD>
%     stg1net_ndx = find(strcmp(s1_list.Items,s1_list.Value));
%     stg1net = stg1nets(stg1net_ndx);  %#ok<FNDSB>
%     stg2net_ndx = find(strcmp(s2_list.Items,s2_list.Value));
%     stg2net = stg2nets(stg2net_ndx);  %#ok<FNDSB>
%     delete(fig);
%   end
% 
% end  % function get_custom_two_stage_tracker_nets_ui()
% 
% 
%    function [docontinue, stg1ctorargs, stg2ctorargs] = raiseDialogsToChooseStageAlgosForCustomTopDownTracker(obj, stg1mode, stg2mode)
%       % What it says on the tin.
%       dlnets = enumeration('DLNetType') ;
%       isma = [dlnets.isMultiAnimal] ;
%       stg2nets = dlnets(~isma) ;
% 
%       is_bbox = false(1,numel(dlnets)) ;
%       for dndx = 1:numel(dlnets)          
%         is_bbox(dndx) = dlnets(dndx).isMultiAnimal && startsWith(char(dlnets(dndx)),'detect_') ;
%       end  % for
% 
%       stg1nets_ht = dlnets(isma & ~is_bbox) ;
%       stg1nets_bbox = dlnets(isma & is_bbox) ;
%       if stg1mode == DLNetMode.multiAnimalTDDetectHT
%         stg1nets = stg1nets_ht ;
%       else
%         stg1nets = stg1nets_bbox ;
%       end
%       [stg1net, stg2net] = apt.get_custom_two_stage_tracker_nets_ui(obj.mainFigure_, stg1nets, stg2nets) ;
% 
%       docontinue = ~isempty(stg1net) ;
%       if docontinue
%         stg1ctorargs = {'trnNetMode', stg1mode, 'trnNetType', stg1net} ;
%         stg2ctorargs = {'trnNetMode', stg2mode, 'trnNetType', stg2net} ;
%       else
%         stg1ctorargs = [] ;
%         stg2ctorargs = [] ;
%       end      
%     end  % function
% 
% 
% 
%     function menu_track_tracking_algorithm_item_actuated_(obj, source, event)  %#ok<INUSD> 
%       % Get the tracker index
%       trackerIndex = source.UserData;
%       labeler = obj.labeler_ ;
% 
%       % The dialog for a custom two-stage tracker takes a while to come up, so
%       % want to show the watch pointer.
%       labeler.pushBusyStatus('Creating new tracker...') ;
%       oc = onCleanup(@()(labeler.popBusyStatus())) ;
% 
%       % Validation happens inside Labeler now
%       % % Validate it
%       % trackers = labeler.trackersAll;
%       % tracker_count = numel(trackers) ;
%       % if ~is_index_in_range(tracker_index, tracker_count)
%       %   error('APT:invalidPropertyValue', 'Invalid tracker index') ;
%       % end
% 
%       % % If a custom top-down tracker, ask if we want to keep it or make a new one.
%       % previousTracker = trackers{tracker_index};
%       % if isa(previousTracker,'DeepTrackerTopDownCustom')
%       %   do_use_previous = ask_if_should_use_previous_custom_top_down_tracker(previousTracker) ;
%       % else
%       %   do_use_previous = [] ;  % value will be ignored
%       % end  % if isa(tAll{iTrk},'DeepTrackerTopDownCustom')
% 
%       % Check for a custom tracker
%       tcis = labeler.trackersAllCreateInfo ;
%       trackerCount = numel(tcis) ;
%       if ~is_index_in_range(trackerIndex, trackerCount)
%         error('No tracker at index %d.  There are %d trackers.', trackerIndex, trackerCount) ;
%       end
%       tci = tcis{trackerIndex} ;
%       trackerClassName = tci{1} ;
%       if strcmp(trackerClassName, 'DeepTrackerTopDownCustom') ,
%         stage1ModeArgs = tci{2} ;  % should itself be a two-element cell array like {'trnNetMode', DLNetMode.multiAnimalTDDetectObj}
%         stage2ModeArgs = tci{3} ;  % should itself be a two-element cell array like {'trnNetMode', DLNetMode.multiAnimalTDPoseObj}
%         stage1Mode = stage1ModeArgs{2} ;  % should be a DLNetMode
%         stage2Mode = stage2ModeArgs{2} ;  % should be a DLNetMode        
%         [docontinue, stg1ctorargs, stg2ctorargs] = obj.raiseDialogsToChooseStageAlgosForCustomTopDownTracker(stage1Mode, stage2Mode) ;
%         if ~docontinue ,
%           return
%         end
%         % Call the model method to set the tracker, providing extra args to specify
%         % the two custom stages   
%         labeler.trackMakeNewTrackerGivenIndex(trackerIndex, stg1ctorargs, stg2ctorargs) ;
%       else
%         % If not a custom tracker, our job is easier.
%         % Call the model method to set the tracker.
%         labeler.trackMakeNewTrackerGivenIndex(trackerIndex) ;
%       end
%     end
% 
% 
% 
%         function initializeTrackersAllAndFriends_(obj)
%       % Create initial values for a few Labeler props, including trackersAll.
% 
%       % Forcibly clear out any old stuff
%       cellfun(@delete, obj.trackersAll_) ;
% 
%       % Create new templates, trackers
%       trackersCreateInfo = ...
%         LabelTracker.getAllTrackersCreateInfo(obj.maIsMA) ;  % 1 x number-of-trackers
%       tAll = cellfun(@(createInfo)(LabelTracker.create(obj, createInfo)), ...
%                      trackersCreateInfo, ...
%                      'UniformOutput', false) ;  % 1 x number-of-trackers
%       obj.trackersAllCreateInfo_ = trackersCreateInfo ;
%       obj.trackersAll_ = tAll ;
%       %obj.notify('update_menu_track_tracking_algorithm') ;
%     end