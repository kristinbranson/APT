classdef TrackMonitorViz < handle
  properties
    hfig % scalar fig
    haxs % [1] axis handle, viz wait time
    haxsIDTraining % scalar axis handle for ID training loss (created when needed)
    %hannlastupdated % [1] textbox/annotation handle
    
    % Three modes. Here nmov=nMovSet*nView
    % - bulkAxsIsBulkMode: hline is [nmov]. One box/patch per mov. htext is
    % a single text label. 
    % - twoStgMode: hline/htext are [nmov*2]. One line for each stage. 
    % [nmov*2].
    % - default: hline/htext are [nmov]. 
    hline % patch handle showing fraction of frames tracked
    htext %  text handle showing fraction of frames tracked
    
    nFramesTracked = []; % same numel has hline. unused if bulkAxsIsBulkMode=true
    nFramesToTrack = 0; % same numel as hline. "
    % parttrkfileTimestamps = []; % same numel as hline. basically unused (in general) but init to 0
    jobDescs = {}; % same numel as hline. string description for hline. unused if bulkAxsIsBulkMode=true

    htrackerInfo % scalar text box handle showing information about current tracker
    wasAborted = false;  % scalar, whether tracking has been aborted
    
    resLast = []; % last contents received
    dtObj % DeepTracker Obj
    poller = [];
    backendType  % scalar DLBackEnd (a DLBackEnd enum, not a DLBackEndClass)
    actions = struct(...
      'Bsub',...
      {{'List all jobs on cluster'...
      'Show tracking jobs'' status'...
      'Update tracking monitor'...
      'Show log files'...
      'Show error messages'}},...
      'Conda',...
      {{'List all conda jobs'...
      'Show tracking jobs'' status',...
      'Update tracking monitor'...
      'Show log files'...
      'Show error messages'}},...
      'Docker',...
      {{'List all docker jobs'...
      'Show tracking jobs'' status',...
      'Update tracking monitor'...
      'Show log files'...
      'Show error messages'}},...
      'AWS',...
      {{'Update tracking monitor'...
      'Show log files'...
      'Show error messages'}});
    minFracComplete = .001;
    
    % % twostage mode
    % twoStgMode = false;

    % list tracking mode
    listMode = false;
    
    % bulk mode --- Mode used when there are too many movies being tracked to
    % comfortably display full info for all of then.
    bulkAxsIsBulkMode = false; % if true, waitbar is in "bulk mode"
    bulkIndNrow; % number of rows in bulk indicator grid
    bulkIndNcol; % number of cols in bulk indicator grid
    bulkMovTracked; % [nmov] logical indicator vec
    bulkAxLblStrArgs = {...
      'HorizontalAlignment' 'center' ...
      'FontSize' 22 ...
      'Color' [1 1 1]};
  end
  
  properties (Transient)
    parent_  % a LabelerController
    labeler_  % a Labeler
    % Widget handles, formerly reached via guidata(obj.hfig).
    axes_wait_
    edit_trackerinfo_
    text_clusterstatus_
    text_clusterinfo_
    popupmenu_actions_
    pushbutton_action_
    pushbutton_startstop_
  end

  properties (Constant)
    DEBUG = false;
    COLOR_AXSWAIT_KILLED = [0.5 0.5 0.5];
    COLOR_AXSWAIT_BULK_UNTRACKED = [0.1 0.1 0.1];
    % could have diff colors for diff views done would be fun
    COLOR_AXSWAIT_BULK_TRACKED = [0 0 1];
    COLOR_AXSWAIT_BULK_EDGE = [0.4 0.4 0.4];
    BULK_NMOV_THRESHOLD = 10; % if you are tracking more than this many movies, you get bulk mode
  end
  
  methods (Static)
    function debugfprintf(varargin)
      if TrackMonitorViz.DEBUG,
        fprintf(varargin{:});
      end
    end
  end
  
  methods
    function obj = TrackMonitorViz(parent, labeler)
      % Store a handle to the parent LabelerController, and to the labeler
      obj.parent_ = parent ;
      obj.labeler_ = labeler ;

      nview = labeler.nview ;
      dtObj = labeler.tracker ;
      poller = labeler.tracker.bgTrackPoller ;
      backendType = labeler.backend.type ;
      nFramesToTrack = labeler.tracker.nFramesToTrack ;

      % These instance variables are not really needed anymore.
      obj.dtObj = dtObj;
      obj.poller = poller;
      obj.backendType = backendType;
      
      nMovSets = numel(nFramesToTrack);
      nmov = nMovSets*nview;
      
      obj.createGui_() ;
      obj.hfig.CloseRequestFcn = @(s,e)(parent.trackMonitorVizCloseRequested()) ;
        % The figure is built with a plain CloseRequestFcn; override it here with
        % this one, which lets the LabelerController handle things in a
        % coordinated way.
      %parent.addSatellite(obj.hfig);  % Don't think we need this
      obj.updateStopButton() ;
      obj.hfig.UserData = 'running';
      obj.haxs = [obj.axes_wait_];
      %obj.hannlastupdated = obj.text_clusterstatus_;
      obj.htrackerInfo = obj.edit_trackerinfo_;

      % obj.twoStgMode = dtObj.getNumStages() > 1;
      obj.listMode = (poller.trackStyle_ == apt.TrackStyle.list);
      obj.bulkAxsIsBulkMode = ( nmov > obj.BULK_NMOV_THRESHOLD ) ;
      % if obj.twoStgMode AND .bulk* are true, twoStg will take precedence
      % for now
      
      % reset plots
      arrayfun(@(x)cla(x),obj.haxs);
      %obj.hannlastupdated.String = 'Cluster status: Initializing...';
      clusterstr = apt.monitorBackendDescription(obj.backendType) ;
      str = sprintf('%s status: Initializing...', clusterstr) ;
      obj.setStatusDisplayLine_(str, true) ;
      obj.text_clusterinfo_.String = '...';
      % set info about current tracker
      s = obj.dtObj.getTrackerInfoString();
      obj.htrackerInfo.String = s;
      obj.popupmenu_actions_.String = obj.actions.(char(backendType));
      obj.popupmenu_actions_.Value = 1;

      axwait = obj.axes_wait_;
      % if tracking movies, output size will be [poller.nMovies, poller.nViews, poller.nStages]
      % if tracking list, it will be [poller.njobs,1,1]
      pollingResultSize = obj.poller.resultSize;
      nPollingResults = prod(pollingResultSize);
      nPollingRepeats = pollingResultSize(2)*pollingResultSize(3);
      if obj.bulkAxsIsBulkMode
        pbaspect(axwait,'auto');
        axwait.DataAspectRatio = [1 1 1]; % "axis equal"
        axwait.Units = 'pixels';
        axposn = axwait.Position;
        axwait.Units = 'normalized';
        whr = axposn(3)/axposn(4);        
        [obj.bulkIndNrow,obj.bulkIndNcol] = TrackMonitorViz.getIndicatorGridSz(nmov,whr);        
        
        % Setting .DataAspectRatio and axis lims => .PlotBoxAspectRatio
        % will react
        axis(axwait,[0.5 obj.bulkIndNcol+1.5 1 obj.bulkIndNrow+1]);
        axwait.Visible = 'off';
        %axis(axwait,'equal');    
        axactual = axis(axwait);
        axxmid = sum(axactual(1:2))/2;
        axymid = sum(axactual(3:4))/2;
        lblstr = sprintf('%d movies to track',nmov);

        %obj.hline initted below
        obj.htext = text(axxmid,axymid,lblstr,'Parent',axwait,...
          obj.bulkAxLblStrArgs{:});
        obj.htext.Position(3) = 1; % Stack above patches created below
        
        obj.bulkMovTracked = false(nmov,1);
        obj.nFramesTracked = [];
        obj.nFramesToTrack = [];
        % obj.parttrkfileTimestamps = zeros(nmov,1);
        obj.jobDescs = {};
      else
        axwait.YLim = [0,nPollingResults];
        axwait.XLim = [0,1+obj.minFracComplete];
        obj.hline = gobjects(nPollingResults,1);
        obj.htext = gobjects(nPollingResults,1);
        obj.nFramesToTrack = double(repmat(nFramesToTrack,nPollingRepeats,1));
        obj.nFramesTracked = zeros(size(obj.nFramesToTrack));
        % obj.parttrkfileTimestamps = zeros(size(obj.nFramesToTrack));
        obj.jobDescs = TrackMonitorViz.initJobDescs(pollingResultSize(1),pollingResultSize(2),pollingResultSize(3),obj.listMode);        
        % ordering of hline is: movMvw1s1 ... movMvw1s1 ... mov1vw2s1 movMvwNs1 ... mov1vw1s2 ... movMvwNs2
        % for multi-stage, single view: all stage1s, then all stage2s.
        % for multi-view, single stage: all view1s, then all view2s
      end

      axwait.YDir = 'reverse';
      axwait.XTick = [];
      axwait.YTick = [];
      hold(axwait,'on');
      
      % create hline/htext
      if obj.bulkAxsIsBulkMode
        obj.hline = TrackMonitorViz.makeIndicatorPatches(nmov,...
          obj.bulkIndNrow,obj.bulkIndNcol,axwait,...
          obj.COLOR_AXSWAIT_BULK_UNTRACKED,...
          {'EdgeColor',obj.COLOR_AXSWAIT_BULK_EDGE}); 
        % obj.hline will be an nmov x 1 array of gobjects
        % obj.htext initted above
      else
        clrs = lines(nMovSets);
        for irep = 1:nPollingRepeats,
          for imovset=1:nMovSets
            itot = (irep-1)*nMovSets + imovset;
            clrI = clrs(imovset,:);
            obj.hline(itot) = patch([0,0,1,1,0]*obj.minFracComplete,...
              itot-[0,1,1,0,0],clrI,...
              'Parent',obj.axes_wait_,...
              'EdgeColor','w');
            obj.htext(itot) = text((1+obj.minFracComplete)/2,itot-.5,...
              sprintf('0/%d frames tracked%s',obj.nFramesToTrack(itot),obj.jobDescs{itot}),...
              'Color','w','HorizontalAlignment','center',...
              'VerticalAlignment','middle','Parent',obj.axes_wait_);
          end
        end
      end  % if
      
      obj.resLast = [];
      obj.wasAborted = false;
      drawnow;
    end

    function createGui_(obj)
      % Build the tracking-monitor figure and its widgets programmatically,
      % storing the widget handles as instance properties.  This replaces the
      % legacy GUIDE .fig/.m pair; the layout was lifted from GUIDE's export.
      obj.hfig = figure(...
        'Units', 'pixels', ...
        'Position', [951.6 826.153846153846 791 525], ...
        'Color', [0 0 0], ...
        'MenuBar', 'none', ...
        'ToolBar', 'none', ...
        'DockControls', 'off', ...
        'IntegerHandle', 'off', ...
        'Name', 'Tracking Monitor', ...
        'NumberTitle', 'off', ...
        'Tag', 'figure_TrackMonitor', ...
        'Visible', 'on') ;

      obj.edit_trackerinfo_ = uicontrol(...
        'Parent', obj.hfig, ...
        'Style', 'edit', ...
        'Units', 'normalized', ...
        'Min', 0, ...
        'Max', 2, ...
        'String', 'Tracker information:', ...
        'HorizontalAlignment', 'left', ...
        'Position', [0.0252844500632111 0.714285714285714 0.955752212389381 0.24], ...
        'BackgroundColor', [0.15 0.15 0.15], ...
        'ForegroundColor', [0.3 0.75 0.93], ...
        'Tag', 'edit_trackerinfo') ;

      obj.axes_wait_ = axes(...
        'Parent', obj.hfig, ...
        'Units', 'normalized', ...
        'Position', [0.0252844500632111 0.6 0.955752212389381 0.0952380952380952], ...
        'Color', [0.15 0.15 0.15], ...
        'XColor', [1 1 1], ...
        'YColor', [1 1 1], ...
        'Tag', 'axes_wait') ;

      obj.text_clusterstatus_ = uicontrol(...
        'Parent', obj.hfig, ...
        'Style', 'text', ...
        'Units', 'normalized', ...
        'String', 'Cluster status: Initializing ...', ...
        'HorizontalAlignment', 'left', ...
        'Position', [0.0252844500632111 0.516190476190476 0.955752212389381 0.0704761904761905], ...
        'BackgroundColor', [0 0 0], ...
        'ForegroundColor', [0 1 0], ...
        'FontUnits', 'normalized', ...
        'FontSize', 0.432432432432432, ...
        'Tag', 'text_clusterstatus') ;

      obj.text_clusterinfo_ = uicontrol(...
        'Parent', obj.hfig, ...
        'Style', 'edit', ...
        'Units', 'normalized', ...
        'Min', 0, ...
        'Max', 2, ...
        'String', '...', ...
        'HorizontalAlignment', 'left', ...
        'Position', [0.0252844500632111 0.121904761904762 0.955752212389381 0.396190476190476], ...
        'BackgroundColor', [0.15 0.15 0.15], ...
        'ForegroundColor', [1 1 1], ...
        'Tag', 'text_clusterinfo') ;

      obj.popupmenu_actions_ = uicontrol(...
        'Parent', obj.hfig, ...
        'Style', 'popupmenu', ...
        'Units', 'normalized', ...
        'String', {'List all jobs on cluster' ; 'Show tracking jobs'' status' ; 'Update tracking monitor' ; 'Show log files'}, ...
        'Value', 1, ...
        'Position', [0.0290771175726928 0.0457142857142857 0.571428571428572 0.0647619047619048], ...
        'BackgroundColor', [0 0 0], ...
        'ForegroundColor', [0 1 0], ...
        'FontUnits', 'normalized', ...
        'FontSize', 0.470588235294118, ...
        'Tag', 'popupmenu_actions') ;

      obj.pushbutton_action_ = uicontrol(...
        'Parent', obj.hfig, ...
        'Style', 'pushbutton', ...
        'Units', 'normalized', ...
        'String', 'Go', ...
        'Position', [0.604298356510746 0.0590476190476191 0.0922882427307207 0.0514285714285714], ...
        'BackgroundColor', [0.47 0.67 0.19], ...
        'ForegroundColor', [1 1 1], ...
        'FontUnits', 'normalized', ...
        'FontSize', 0.592592592592593, ...
        'FontWeight', 'bold', ...
        'Tag', 'pushbutton_action', ...
        'Callback', @(s,e)(obj.updateClusterInfo())) ;

      obj.pushbutton_startstop_ = uicontrol(...
        'Parent', obj.hfig, ...
        'Style', 'pushbutton', ...
        'Units', 'normalized', ...
        'String', 'Stop tracking', ...
        'Position', [0.724399494310999 0.0590476190476191 0.252844500632111 0.0514285714285714], ...
        'BackgroundColor', [0.64 0.08 0.18], ...
        'ForegroundColor', [1 1 1], ...
        'FontUnits', 'normalized', ...
        'FontSize', 0.592592592592593, ...
        'FontWeight', 'bold', ...
        'Tag', 'pushbutton_startstop', ...
        'Callback', @(s,e)(obj.abortTracking())) ;
    end  % function

    function delete(obj)
      deleteValidGraphicsHandles(obj.hfig);
      obj.hfig = [];
    end
    
    function update(obj)
      % Traditional controller update method.
      obj.resultsReceived() ;
    end

    function resultsReceived(obj, pollingResult, forceupdate)
      % Callback executed when new result received from monitor BG
      % worker
      %
      % trnComplete: scalar logical, true when all views done
      
      if ~exist('pollingResult', 'var') || isempty(pollingResult) ,
        pollingResult = obj.labeler_.tracker.bgTrkMonitor.pollingResult ;
      end
      if nargin < 3,
        forceupdate = false;
      end
      
      if isempty(obj.hfig) || ~ishandle(obj.hfig),
        TrackMonitorViz.debugfprintf('Monitor closed, results received %s\n',datestr(now()));
        return
      end

      if obj.wasAborted,
        obj.updateStopButton() ;
        TrackMonitorViz.debugfprintf('Tracking jobs killed, results received %s\n',datestr(now()));
        return
      end
      
      if isempty(pollingResult) ,
        obj.updateStopButton() ;
        return
      end

      TrackMonitorViz.debugfprintf('%s: TrackMonitorViz results received:\n',datestr(now()));
       
      if isfield(pollingResult,'parttrkfile')
        TrackMonitorViz.debugfprintf('Partial tracks exist: %d\n',exist(pollingResult.parttrkfile{1},'file'));
        TrackMonitorViz.debugfprintf('N. frames tracked: ');
      end
      TrackMonitorViz.debugfprintf('tfcomplete: %s\n',formattedDisplayText(pollingResult.tfComplete));
      nJobs = numel(pollingResult.tfComplete);
      nMovies = size(pollingResult.tfComplete, 1);

      % Check if this is an ID linking job and handle ID training axis
      if isfield(pollingResult, 'result_type') && strcmp(pollingResult.result_type, 'id_link')
        obj.handleIDTrainingUpdate(pollingResult);
      end

      % It is assumed that there is a correspondence between res and .hline
      if obj.bulkAxsIsBulkMode
        if nMovies~=numel(obj.hline)
          warningNoTrace('Unexpected monitor results size (%d); expected (%d).',...
                         nMovies,numel(obj.hline));
        end
      else
        if nJobs~=numel(obj.hline)
          warningNoTrace('Unexpected monitor results size (%d); expected (%d).',...
                         nJobs,numel(obj.hline));
        end
      end

      % always update info about current tracker, as labels may have changed
      s = obj.dtObj.getTrackerInfoString();
      obj.htrackerInfo.String = s;

      ticId = tic() ;
      isDoneFromTripleIndex = pollingResult.tfComplete ;
      if isfield(pollingResult,'parttrkfileTimestamp'),
        doesPartFileExistFromTripleIndex = ~isnan(pollingResult.parttrkfileTimestamp);
        if forceupdate
          doUpdateFromTripleIndex = true(size(isDoneFromTripleIndex)) ;
        else
          doUpdateFromTripleIndex = isDoneFromTripleIndex | doesPartFileExistFromTripleIndex ;
        end
      else
        doUpdateFromTripleIndex = false(size(isDoneFromTripleIndex)) ;
      end

      if obj.bulkAxsIsBulkMode
        for imov = 1 : nMovies
          % just update indicator based on isdone; dont try to get
          % nframes tracked, etc
          isdone = all(all(isDoneFromTripleIndex(imov,:,:), 3), 2) ;
          if isdone
            set(obj.hline(imov),'FaceColor',obj.COLOR_AXSWAIT_BULK_TRACKED);
            obj.bulkMovTracked(imov) = true;
          else
            % no nothing
          end
        end  % for imov
      else
        % if not in bulk mode
        for ijob=1:nJobs,
          doupdate = doUpdateFromTripleIndex(ijob) ;
          if doupdate,
            isdone = isDoneFromTripleIndex(ijob);
            % If not in bulk mode
            try
              if isfield(pollingResult,'parttrkfileNfrmtracked')
                % for AWS and any worker that figures this out on its own
                obj.nFramesTracked(ijob) = pollingResult.parttrkfileNfrmtracked(ijob) ;
                % if isnan(pollingResult.parttrkfileNfrmtracked(ijob)) && isfinite(pollingResult.parttrkfileTimestamp(ijob)) ,
                %   nop() ;
                %   %error('Internal error: In TrackMonitorViz instance, .nFramesTracked(%d) is nan', ijob) ;
                %     % This should be caught by the local try-catch
                % end
              else
                if isdone,
                  tfile = pollingResult.trkfile{ijob};
                else
                  tfile = pollingResult.parttrkfile{ijob};
                end
                %fprintf('TrkMonitorViz.resultsReceived: tfile = %s\n',tfile);
                try
                  [obj.nFramesTracked(ijob),didload] = TrkFile.getNFramesTracked(tfile);
                  if ~didload && isdone,
                    warning('isdone = true and could not load trk file to count nFramesTracked');
                  end
                catch ME,
                  if isdone,
                    warning('Could not compute number of frames tracked:\n%s',getReport(ME));
                  end
                end
              end
              obj.nFramesTracked(ijob) = double(obj.nFramesTracked(ijob));

              if nJobs > 1,
                sview = obj.jobDescs{ijob};
              else
                sview = '';
              end
              set(obj.htext(ijob),'String',sprintf('%d/%d frames tracked%s',...
                obj.nFramesTracked(ijob),obj.nFramesToTrack(ijob),sview));
              fracComplete = obj.minFracComplete + ...
                    (obj.nFramesTracked(ijob)/obj.nFramesToTrack(ijob));
              set(obj.hline(ijob),'XData',[0,0,1,1,0]*fracComplete);
            catch ME,
              fprintf('Could not update nFramesTracked, for whatever reason.\n');
            end
          end
          TrackMonitorViz.debugfprintf('Job %d: %d. ',ijob,obj.nFramesTracked(ijob));
        end  % for iJob
      end  % if obj.bulkAxsIsBulkMode
      TrackMonitorViz.debugfprintf('\n');
      TrackMonitorViz.debugfprintf('Update of nFramesTracked took %f s.\n',toc(ticId));

      obj.resLast = pollingResult ;

      obj.updateErrDisplay(pollingResult);
      obj.syncStatusLineToPollingResult() ;
      obj.updateStopButton() ;
    end

    function resultsReceivedLoopOverMovies_(obj, pollingResult, forceupdate)
      % Loop over the movies, updating the figure as needed.  This method should only
      % be called when in bulk mode.  Non-bulk mode requires a different approach.

      assert(~obj.bulkAxsIsBulkMode) ;

      nmov = size(pollingResult.tfComplete, 1) ;
      nJobs = numel(pollingResult.tfComplete);

      if nJobs~=numel(obj.hline)
        warningNoTrace('Unexpected monitor results size (%d); expected (%d).',...
                       nJobs,numel(obj.hline));
      end

      ticId = tic() ;
      for imov=1:nmov,
        isdone = pollingResult.tfComplete(ijob);
        if isfield(pollingResult,'parttrkfileTimestamp'),
          partFileExists = ~isnan(pollingResult.parttrkfileTimestamp(ijob));
          % isupdate = ...
          %   (partFileExists && (forceupdate || (pollingResult.parttrkfileTimestamp(ijob)>obj.parttrkfileTimestamps(ijob)))) || ...
          %   isdone ;
          isupdate = ( forceupdate || partFileExists || isdone ) ;
        else
          isupdate = false ;
        end

        if isupdate,
          % just update indicator based on isdone; dont try to get
          % nframes tracked, etc
          if isdone
            set(obj.hline(ijob),'FaceColor',obj.COLOR_AXSWAIT_BULK_TRACKED);
            obj.bulkMovTracked(ijob) = true;
          else
            % none
          end
        end  % if
      end  % for

      TrackMonitorViz.debugfprintf('\n');
      TrackMonitorViz.debugfprintf('Update of nFramesTracked took %f s.\n',toc(ticId));
    end  % method

    function resultsReceivedLoopOverJobs_(obj, pollingResult, forceupdate)
      % Loop over the jobs, updating the figure as needed.  This method should only
      % be called when *not* in bulk mode.  Bulk mode requires a different approach.

      assert(~obj.bulkAxsIsBulkMode) ;

      nJobs = numel(pollingResult.tfComplete);

      if nJobs~=numel(obj.hline)
        warningNoTrace('Unexpected monitor results size (%d); expected (%d).',...
                       nJobs,numel(obj.hline));
      end

      ticId = tic() ;
      for ijob=1:nJobs,
        isdone = pollingResult.tfComplete(ijob);
        if isfield(pollingResult,'parttrkfileTimestamp'),
          partFileExists = ~isnan(pollingResult.parttrkfileTimestamp(ijob));
          % isupdate = ...
          %   (partFileExists && (forceupdate || (pollingResult.parttrkfileTimestamp(ijob)>obj.parttrkfileTimestamps(ijob)))) || ...
          %   isdone ;
          isupdate = ( forceupdate || partFileExists || isdone ) ;
        else
          isupdate = false ;
        end

        if isupdate,
          try
            if isfield(pollingResult,'parttrkfileNfrmtracked')
              % for AWS and any worker that figures this out on its own
              obj.nFramesTracked(ijob) = pollingResult.parttrkfileNfrmtracked(ijob) ;
              % if isnan(pollingResult.parttrkfileNfrmtracked(ijob)) && isfinite(pollingResult.parttrkfileTimestamp(ijob)) ,
              %   nop() ;
              %   %error('Internal error: In TrackMonitorViz instance, .nFramesTracked(%d) is nan', ijob) ;
              %     % This should be caught by the local try-catch
              % end
            else
              if isdone,
                tfile = pollingResult.trkfile{ijob};
              else
                tfile = pollingResult.parttrkfile{ijob};
              end
              %fprintf('TrkMonitorViz.resultsReceived: tfile = %s\n',tfile);
              try
                [obj.nFramesTracked(ijob),didload] = TrkFile.getNFramesTracked(tfile);
                if ~didload && isdone,
                  warning('isdone = true and could not load trk file to count nFramesTracked');
                end
              catch ME,
                if isdone,
                  warning('Could not compute number of frames tracked:\n%s',getReport(ME));
                end
              end
            end
            obj.nFramesTracked(ijob) = double(obj.nFramesTracked(ijob));

            if nJobs > 1,
              sview = obj.jobDescs{ijob};
            else
              sview = '';
            end
            set(obj.htext(ijob),'String',sprintf('%d/%d frames tracked%s',...
              obj.nFramesTracked(ijob),obj.nFramesToTrack(ijob),sview));
            fracComplete = obj.minFracComplete + ...
                  (obj.nFramesTracked(ijob)/obj.nFramesToTrack(ijob));
            set(obj.hline(ijob),'XData',[0,0,1,1,0]*fracComplete);
          catch ME,
            fprintf('Could not update nFramesTracked, for whatever reason.\n');
          end
        end   % if isupdate
        
        TrackMonitorViz.debugfprintf('Job %d: %d. ',ijob,obj.nFramesTracked(ijob));
      end  % for iJob = 1 : nJobs

      TrackMonitorViz.debugfprintf('\n');
      TrackMonitorViz.debugfprintf('Update of nFramesTracked took %f s.\n',toc(ticId));
    end

    function syncStatusLineToPollingResult(obj)
      % Render the status line (text_clusterstatus) from the monitor's
      % accumulated poll state.  This is NOT a state-independent update method
      % -- hence the syncStatusLineToPollingResult name rather than an update*
      % one: its source of truth is mostly the monitor's own state
      % (obj.resLast, obj.wasAborted, obj.nFramesTracked,
      % obj.bulkMovTracked), which is written as poll results arrive in
      % resultsReceived().  Only a thin slice of what it reads is genuine model
      % state (labeler.bgTrkIsRunning, labeler.lastTrackEndCause).  It also
      % mutates other view state -- on completion it calls updateStatusFinal()
      % off the poll result -- so it does not merely paint the status line.  It
      % is really the tail end of the resultsReceived() pipeline, not a
      % model->view synchronizer, so calling it in isolation with a stale or
      % empty resLast need not reflect the Labeler alone.
      labeler = obj.labeler_ ;
      pollingResult = obj.resLast ;  % most recent poll result processed, or [] if none yet

      if ~labeler.bgTrkIsRunning ,
        % No tracking bout is running: reflect the authoritative outcome of the
        % last bout, as recorded by the tracker.
        switch labeler.lastTrackEndCause
          case EndCause.complete ,
            status = 'Tracking complete.' ;
            isAllGood = true ;
            if ~isempty(pollingResult) ,
              obj.updateStatusFinal(numel(pollingResult.tfComplete)) ;
            end
          case EndCause.error ,
            status = 'Error while tracking.  See error messages for details.' ;
            isAllGood = false ;
          case EndCause.abort ,
            status = 'Tracking process aborted.' ;
            isAllGood = false ;
          case EndCause.undefined ,
            status = 'No tracking jobs running.' ;
            isAllGood = true ;
          otherwise ,
            error('APT:internalError', 'Unrecognized EndCause in TrackMonitorViz.syncStatusLineToPollingResult()') ;
        end
      else
        % A tracking bout is in progress: derive the message from the most
        % recent poll result.
        if isempty(pollingResult) ,
          status = 'Initializing tracking.' ;
          isAllGood = true ;
        else
          isErr = any([pollingResult.errFileExists(:)]) ;
          isLogFile = any([pollingResult.logFileExists(:)]) ;
          if isErr ,
            status = 'Error while tracking.' ;
            isAllGood = false ;
          elseif isfield(pollingResult,'result_type') && strcmp(pollingResult.result_type,'id_link') ,
            if isLogFile && pollingResult.jsonFileExist && numel(pollingResult.idstep)>0 ,
              status = sprintf('ID Training in progress. %d iterations completed',pollingResult.idstep(end)) ;
            else
              status = 'Initializing Training for ID Linking. ' ;
            end
            isAllGood = true ;
          elseif isLogFile ,
            if obj.bulkAxsIsBulkMode ,
              status = sprintf('Tracking in progress. %d/%d movies tracked.',...
                nnz(obj.bulkMovTracked),numel(obj.bulkMovTracked)) ;
            elseif numel(pollingResult.tfComplete) > 1 ,
              status = sprintf('Tracking in progress. %s frames tracked.',mat2str(obj.nFramesTracked)) ;
            else
              status = sprintf('Tracking in progress. %d frames tracked.',obj.nFramesTracked) ;
            end
            isAllGood = true ;
          else
            status = 'Initializing tracking.' ;
            isAllGood = true ;
          end
        end
      end

      clusterstr = apt.monitorBackendDescription(obj.backendType) ;
      str = sprintf('%s status: %s (at %s)',clusterstr,status,strtrim(datestr(now(),'HH:MM:SS PM'))) ;
      obj.setStatusDisplayLine_(str, isAllGood) ;
    end  % function
    
    function updateErrDisplay(obj, pollingResult)
      isErr = any([pollingResult.errFileExists]) ;
      if ~isErr,
        return;
      end
      if any([pollingResult.errFileExists]),
        erri = find(strcmp(obj.popupmenu_actions_.String,'Show error messages'),1);
        if numel(erri) ~= 1,
          return;
        end
        obj.popupmenu_actions_.Value = erri;
      else
        erri = find(strcmp(obj.popupmenu_actions_.String,'Show log files'),1);
        if numel(erri) ~= 1,
          return;
        end
        obj.popupmenu_actions_.Value = erri;
      end
      obj.updateClusterInfo();
      obj.text_clusterinfo_.ForegroundColor = 'r';
      %TrackMonitorViz.updateStartStopButton(handles,false,false);
      drawnow;
    end  % function

    function updateStatusFinal(obj,nJobs)
      for ijob = 1:nJobs
        if nJobs > 1,
          sview = obj.jobDescs{ijob};
        else
          sview = '';
        end
        set(obj.htext(ijob),'String',sprintf('%d/%d frames tracked%s',...
          obj.nFramesToTrack(ijob),obj.nFramesToTrack(ijob),sview));
      end
      set(obj.hline,'FaceColor',obj.COLOR_AXSWAIT_BULK_TRACKED,'XData',[0,0,1,1,0]);
      obj.bulkMovTracked(:) = true;
      obj.updateStopButton() ;
    end  % function
        
    function abortTracking(obj)
      if isempty(obj.poller),
        warning('trackWorkerObj is empty -- cannot kill process');
        return;
      end
      obj.setStatusDisplayLine_('Killing tracking jobs...', false) ;
      obj.pushbutton_startstop_.String = 'Stopping tracking...';
      obj.pushbutton_startstop_.Enable = 'off';
      obj.labeler_.abortTracking() ;

      % [tfsucc,warnings] = obj.trackWorkerObj.killProcess();
      % if tfsucc,
      % 
      %   % AL: .isKilled set in resultsReceived
      %   %obj.isKilled = true;
      % else
      %   warndlg([{'Tracking processes may not have been killed properly:'},warnings],'Problem stopping tracking','modal');
      % end
      % TrackMonitorViz.updateStartStopButton(handles,false,false);
      obj.updateStopButton() ;
      obj.setStatusDisplayLine_('Tracking process killed.', false);
      drawnow;

    end
    
    function updateClusterInfo(obj)
      actions = obj.popupmenu_actions_.String; %#ok<PROP>
      v = obj.popupmenu_actions_.Value;
      action = actions{v}; %#ok<PROP>
      switch action
        case 'Show log files',
         ss = obj.getLogFilesSummary();
         obj.text_clusterinfo_.String = ss;
         drawnow;
        case 'Update tracking monitor',
          obj.updateMonitorPlots();
          drawnow;
        case {'List all jobs on cluster','List all docker jobs','List all conda jobs'},
          ss = obj.detailedStatusStringFromRegisteredJobIndex_();
          obj.text_clusterinfo_.String = ss;
          drawnow;
        case 'Show tracking jobs'' status',
          ss = obj.queryAllJobsStatus();
          obj.text_clusterinfo_.String = ss;
          drawnow;
        case 'Show error messages',
          if isempty(obj.resLast) || ~any([obj.resLast.errFileExists]),
            ss = 'No error messages.';
          else
            ss = obj.getErrorFilesSummary() ;
          end
          obj.text_clusterinfo_.String = ss;
          drawnow;
        otherwise
          fprintf('%s not implemented\n',action);
          return;
      end
      %handles.text_clusterinfo.ForegroundColor = 'w';
    end    
    
    function ss = getLogFilesSummary(obj)      
      ss = obj.dtObj.getTrackingLogFilesSummary() ;      
    end
    
    function ss = getErrorFilesSummary(obj)      
      ss = obj.dtObj.getTrackingErrorFilesSummary() ;      
    end
    
    function updateMonitorPlots(obj)      
      pollingResult = obj.poller.poll() ;
      obj.resultsReceived(pollingResult, true) ;
    end
    
    function result = queryAllJobsStatus(obj)      
      ss = obj.dtObj.queryAllJobsStatus('track') ;
      if isempty(ss) ,
        result = {'(No active jobs.)'} ;
      else
        result = ss ;
      end
    end  % function

    function result = detailedStatusStringFromRegisteredJobIndex_(obj)
      ss = obj.dtObj.detailedStatusStringFromRegisteredJobIndex('track') ;
      if isempty(ss) ,
        result = {'(No active jobs.)'} ;
      else
        result = ss ;
      end
    end  % function

        
    function updateStopButton(obj)
      % A conventional update method for the (start/)stop button.
      labeler = obj.labeler_ ;
      isRunning = labeler.bgTrkIsRunning ;
      if isRunning
        isComplete = [] ;
      else
        isComplete = (labeler.lastTrackEndCause == EndCause.complete) ;
      end
      if isRunning ,
        set(obj.pushbutton_startstop_,'String','Stop tracking','BackgroundColor',[.64,.08,.18],'Enable','on','UserData','stop');
      else
        if isComplete ,
          set(obj.pushbutton_startstop_,'String','Tracking complete','BackgroundColor',[.466,.674,.188],'Enable','off','UserData','done');
        else
          set(obj.pushbutton_startstop_,'String','Tracking stopped','BackgroundColor',[.64,.08,.18],'Enable','off','UserData','done');
        end
      end
    end  % function
        
  end  % methods
  
  methods (Static)
    
    function jobDescs = initJobDescs(nMovSets,nview,nstage,listmode)
      % jobDescs: cellstr, either [nMovSets x nview], or 
      %                           [nMovSets x 2] if tf2stg==true
      
      if nargin < 4,
        listmode = false;
      end
      
      jobDescs = cell(nMovSets,nview,nstage);
      for imovset = 1:nMovSets,
        if nMovSets > 1,
          if listmode,
            movstr = sprintf(', Job %d',imovset);
          else
            movstr = sprintf(', Mov %d',imovset);
          end
        else
          movstr = '';
        end
        for iview = 1:nview
          if nview> 1,
            viewstr = sprintf(', Vw %d',iview);
          else
            viewstr = '';
          end
          for istage = 1:nstage,
            if nstage > 1,
              stagestr = sprintf(', Stg %d',istage);
            else
              stagestr = '';
            end
            jobDescs{imovset,iview,istage} = [movstr,viewstr,stagestr];
          end
        end
      end
    end
    
    function [nrowind,ncolind] = ...
        getsizeTrackVizIndicatorGrid(nmov, width2height)
      % We create a grid of movie-is-done indicators, one ind per mov
      % [nrowind x ncolind]
      %
      % The point here is that we want square indicators with a fixed 1:1
      % w/h ratio for each indicator. So a given sized display area (eg the
      % rectangular waitbar region) can fit either one row of big squares,
      % or two rows of smaller squares, etc. The size of the grid  
      % 
      % ncolind/nrowind ~ whr
      % ncolind*nrowind >= nmov
      % => nrowind^2>=nmov/whr
      
      nrowind = ceil(sqrt(nmov/width2height));
      ncolind = ceil(nrowind*width2height);
    end

    function [gridnrow,gridncol] = getIndicatorGridSz(nmov,width2height)
      for gridnrow=1:100
        % We try using nrows 
        gridncol = ceil(nmov/gridnrow);
        widthtotal = 1; % say
        sqsz = widthtotal/gridncol; 
        heighttotal = sqsz*gridnrow;
        
        heightavail = widthtotal/width2height;
        heightextra = heightavail-heighttotal;
        tfcanfitnewrow = heightextra>sqsz;
        if tfcanfitnewrow % && (gridnrow==1 || mod(nmov,gridncol)>0)
          % none; continue, add a row try again
        else
          break;
        end
      end
    end

    function hpch = makeIndicatorPatches(nmov,gridnrow,gridncol,ax,clr,pchargs)  %#ok<INUSD> 
      hpch = gobjects(nmov,1);
      for imov = 1:nmov
        irow = ceil(imov/gridncol);
        icol = rem(imov-1,gridncol)+1;
        xpch = [icol icol icol+1 icol+1];
        ypch = [irow irow+1 irow+1 irow];
        hpch(imov) = patch(xpch,ypch,clr,'Parent',ax,pchargs{:});
      end
    end

    function mm = testIndPches(ax,n1)
      hfig = ancestor(ax,'figure');
      for nmov=1:n1
        cla(ax);
        pbaspect(ax,'auto');
        ax.DataAspectRatio = [1 1 1]; % "axis equal"
        
        %pbar = ax.PlotBoxAspectRatio;    
        ax.Units = 'pixels';
        axposn = ax.Position;
        ax.Units = 'normalized';
        whr = axposn(3)/axposn(4);
        fprintf('Axis whr is %.2f\n',whr);
        
        [gridnrow,gridncol] = TrackMonitorViz.getIndicatorGridSz(nmov,whr);
        hpch = TrackMonitorViz.makeIndicatorPatches(nmov,...
          gridnrow,gridncol,ax,[1 0 0],{});  %#ok<NASGU> 
        axis(ax,[0.5 gridncol+1.5 1 gridnrow+1]);
        
        drawnow;
        mm(nmov) = getframe(hfig);  %#ok<AGROW> 
        %input(num2str(nmov));
      end
    end  % function
  end  % methods (Static)

  methods
    function setStatusDisplayLine_(obj, str, isallgood)
      % Set the status message line and its color (green if isallgood, else red).
      text_h = obj.text_clusterstatus_ ;
      set(text_h, 'String', str) ;
      set(text_h, 'ForegroundColor', fif(isallgood, 'g', 'r')) ;
      drawnow('limitrate', 'nocallbacks') ;
    end  % function
    
    function updatePointer(obj)
      % Update the mouse pointer to reflect the Labeler state.
      labeler = obj.labeler_ ;
      is_busy = labeler.isStatusBusy ;
      pointer = fif(is_busy, 'watch', 'arrow') ;
      set(obj.hfig, 'Pointer', pointer) ;
    end  % function

    function handleIDTrainingUpdate(obj, pollingResult)
      % Handle ID training progress updates by creating/updating ID training axis
      if ~isfield(pollingResult, 'idloss') || ~isfield(pollingResult, 'idstep')
        return;
      end

      idloss = pollingResult.idloss;
      idstep = pollingResult.idstep;

      if isempty(idloss) || isempty(idstep)
        return;
      end

      % Create ID training axis if it doesn't exist
      if isempty(obj.haxsIDTraining) || ~ishandle(obj.haxsIDTraining)
        obj.createIDTrainingAxis();
      end

      % Update the plot
      if ishandle(obj.haxsIDTraining)
        %axes(obj.haxsIDTraining);
        plot(obj.haxsIDTraining,idstep, idloss, 'g-', 'LineWidth', 1.5);
        xlabel(obj.haxsIDTraining,'Training Step');
        ylabel(obj.haxsIDTraining,'Training Loss');
        title(obj.haxsIDTraining,'ID Model Training Progress','Color',[1,1,1]);
        set(obj.haxsIDTraining,'Color',[0,0,0],'XColor',[1 1 1],'YColor',[1,1,1]);
        yscale(obj.haxsIDTraining,'log');
        grid on;
        drawnow('limitrate', 'nocallbacks');
      end
    end  % function

    function createIDTrainingAxis(obj)
      % Create a new axis for ID training loss display
      % Gather the widget handles into a struct keyed by Tag, so the
      % reposition loops below can reference controls by name.
      handles = struct(...
        'edit_trackerinfo', obj.edit_trackerinfo_, ...
        'axes_wait', obj.axes_wait_, ...
        'text_clusterinfo', obj.text_clusterinfo_, ...
        'pushbutton_startstop', obj.pushbutton_startstop_, ...
        'popupmenu_actions', obj.popupmenu_actions_, ...
        'text_clusterstatus', obj.text_clusterstatus_, ...
        'pushbutton_action', obj.pushbutton_action_) ;

      % Get current figure and axes_wait positions
      figPos = get(obj.hfig, 'Position');
      pos_wait = get(handles.axes_wait, 'Position');

      % ID axis needs minimum space - use reclaimed space + additional 200px
      minIDAxisHeightPx = 100; % Minimum 200 pixels
      oldFigHeight = figPos(4);
      minIDAxisHeightNorm = max(0.25,minIDAxisHeightPx / (oldFigHeight+minIDAxisHeightPx )); % Convert to normalized units
      idAxisHeight = minIDAxisHeightNorm;

      % Calculate total additional space needed beyond reclaimed space
      additionalSpaceNeeded = idAxisHeight;
      spacing = 0.03;
      totalSpaceNeeded = additionalSpaceNeeded + 2*spacing;

      % Resize figure height to accommodate the additional space
      if totalSpaceNeeded > 0
        newFigHeight = oldFigHeight + (totalSpaceNeeded * oldFigHeight);
        figPos(4) = newFigHeight;
        set(obj.hfig, 'Position', figPos);

        % Calculate scaling factor for normalized coordinates
        heightScalingFactor = oldFigHeight / newFigHeight;
      end

      % Move all components below axes_wait down and scale their sizes
      componentsToMoveBefore = {'edit_trackerinfo', 'axes_wait'};
      componentsToMoveAfter = {'text_trackerinfo', 'text_clusterinfo', ...
                         'pushbutton_startstop', 'popupmenu_actions',...
                         'text_clusterstatus','pushbutton_action'};

      for i = 1:length(componentsToMoveBefore)
        if isfield(handles, componentsToMoveBefore{i}) && ishandle(handles.(componentsToMoveBefore{i}))
          pos = get(handles.(componentsToMoveBefore{i}), 'Position');
          % Scale the position and size to maintain proportions in new coordinate system
          pos(2) = pos(2) * heightScalingFactor + idAxisHeight; % Move down by ID axis height + spacing
          pos(4) = pos(4) * heightScalingFactor; % Scale height to maintain visual proportion
          set(handles.(componentsToMoveBefore{i}), 'Position', pos);
        end
      end

      for i = 1:length(componentsToMoveAfter)
        if isfield(handles, componentsToMoveAfter{i}) && ishandle(handles.(componentsToMoveAfter{i}))
          pos = get(handles.(componentsToMoveAfter{i}), 'Position');
          % Scale the position and size to maintain proportions in new coordinate system
          pos(2) = pos(2) * heightScalingFactor; % Move down by ID axis height + spacing
          pos(4) = pos(4) * heightScalingFactor; % Scale height to maintain visual proportion
          set(handles.(componentsToMoveAfter{i}), 'Position', pos);
        end
      end

      % Create ID training axis using the calculated space
      pos_wait_scaled = get(handles.axes_wait,'Position');
      pos_new = pos_wait_scaled;
      pos_new(1) = pos_new(1) + 0.04;
      pos_new(3) = pos_new(3) - 0.04;
      pos_new(2) = pos_wait_scaled(2) - idAxisHeight  + 1.5*spacing;  % Position below scaled axes_wait
      pos_new(4) = idAxisHeight-2.5*spacing;  % spacing for axis label and title

      obj.haxsIDTraining = axes('Parent', obj.hfig, ...
                                'Position', pos_new, ...
                                'Box', 'on');
      xlabel(obj.haxsIDTraining, 'ID Training Step');
      ylabel(obj.haxsIDTraining, 'ID Training Loss');
      title(obj.haxsIDTraining, 'ID Model Training Progress');
      grid(obj.haxsIDTraining, 'on');
      set(obj.haxsIDTraining,'Color',[0,0,0],'XColor',[1 1 1],'YColor',[1,1,1])
      yscale(obj.haxsIDTraining,'log');
    end  % function

  end  % methods
end  % classdef
