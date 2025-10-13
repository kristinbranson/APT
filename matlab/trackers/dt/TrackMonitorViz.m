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
    parttrkfileTimestamps = []; % same numel as hline. basically unused (in general) but init to 0
    jobDescs = {}; % sae numel as hline. string description for hline. unused if bulkAxsIsBulkMode=true

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
    
    % twostage mode
    twoStgMode = false;

    % list tracking mode
    listMode = false;
    
    % bulk mode 
    bulkAxsIsBulkMode = false; % if true, waitbar is in "bulk mode"
    bulkNmovThreshold = 10; % if you are tracking more than this many movies, you get bulk mode
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
  end

  properties (Constant)
    DEBUG = false;
    COLOR_AXSWAIT_KILLED = [0.5 0.5 0.5];
    COLOR_AXSWAIT_BULK_UNTRACKED = [0.1 0.1 0.1];
    % could have diff colors for diff views done would be fun
    COLOR_AXSWAIT_BULK_TRACKED = [0 0 1];
    COLOR_AXSWAIT_BULK_EDGE = [0.4 0.4 0.4];
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
      
      obj.hfig = TrackMonitorGUI(obj);
      obj.hfig.CloseRequestFcn = @(s,e)(parent.trackMonitorVizCloseRequested()) ;
        % Override the CloseRequestFcn callback in TrackMonitorGUI with this one, 
        % which lets that LabelerController handle things in a coordinated way.
      %parent.addSatellite(obj.hfig);  % Don't think we need this
      handles = guidata(obj.hfig);
      obj.updateStopButton() ;
      %TrackMonitorViz.updateStartStopButton(handles,true,false);
      %handles.pushbutton_startstop.Enable = 'on';
      obj.hfig.UserData = 'running';
      obj.haxs = [handles.axes_wait];
      %obj.hannlastupdated = handles.text_clusterstatus;
      obj.htrackerInfo = handles.edit_trackerinfo;

      obj.twoStgMode = dtObj.getNumStages() > 1;
      obj.listMode = isequal(poller.trackType_,'list');
      obj.bulkAxsIsBulkMode = nmov > obj.bulkNmovThreshold;
      % if obj.twoStgMode AND .bulk* are true, twoStg will take precedence
      % for now
      
      % reset plots
      arrayfun(@(x)cla(x),obj.haxs);
      %obj.hannlastupdated.String = 'Cluster status: Initializing...';
      clusterstr = apt.monitorBackendDescription(obj.backendType) ;
      str = sprintf('%s status: Initializing...', clusterstr) ;
      obj.setStatusDisplayLine(str, true) ;
      handles.text_clusterinfo.String = '...';
      % set info about current tracker
      s = obj.dtObj.getTrackerInfoString();
      obj.htrackerInfo.String = s;
      handles.popupmenu_actions.String = obj.actions.(char(backendType));
      handles.popupmenu_actions.Value = 1;
      
      axwait = handles.axes_wait;
      % if tracking movies, output size will be [poller.nMovies,poller.nViews,poller.nStages]
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
        obj.parttrkfileTimestamps = zeros(nmov,1);
        obj.jobDescs = {};
      else

        %nstg = 2*nmov;
        axwait.YLim = [0,nPollingResults];
        axwait.XLim = [0,1+obj.minFracComplete];
        obj.hline = gobjects(nPollingResults,1);
        obj.htext = gobjects(nPollingResults,1);
        obj.nFramesToTrack = double(repmat(nFramesToTrack,nPollingRepeats,1));
        obj.nFramesTracked = zeros(size(obj.nFramesToTrack));
        obj.parttrkfileTimestamps = zeros(size(obj.nFramesToTrack));
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
        % obj.htext initted above
      else

        clrs = lines(nMovSets);
        for irep = 1:nPollingRepeats,
          for imovset=1:nMovSets
            itot = (irep-1)*nMovSets + imovset;
            clrI = clrs(imovset,:);
            obj.hline(itot) = patch([0,0,1,1,0]*obj.minFracComplete,...
              itot-[0,1,1,0,0],clrI,...
              'Parent',handles.axes_wait,...
              'EdgeColor','w');
            obj.htext(itot) = text((1+obj.minFracComplete)/2,itot-.5,...
              sprintf('0/%d frames tracked%s',obj.nFramesToTrack(itot),obj.jobDescs{itot}),...
              'Color','w','HorizontalAlignment','center',...
              'VerticalAlignment','middle','Parent',handles.axes_wait);
          end
        end

      end
      
      obj.resLast = [];
      obj.wasAborted = false;
      drawnow;            
    end
    
    function delete(obj)
      deleteValidGraphicsHandles(obj.hfig);
      obj.hfig = [];
    end
    
    function update(obj)
      % Traditional controller update method.
      obj.resultsReceived() ;
    end

    function [tfSucc,msg] = resultsReceived(obj, pollingResult, forceupdate)
      % Callback executed when new result received from monitor BG
      % worker
      %
      % trnComplete: scalar logical, true when all views done
      
      tfSucc = false;
      msg = '';  %#ok<NASGU> 
      
      if ~exist('pollingResult', 'var') || isempty(pollingResult) ,
        pollingResult = obj.labeler_.tracker.bgTrkMonitor.pollingResult ;
      end
      if nargin < 3,
        forceupdate = false;
      end
      
      if isempty(obj.hfig) || ~ishandle(obj.hfig),
        msg = 'Monitor closed.';
        TrackMonitorViz.debugfprintf('Monitor closed, results received %s\n',datestr(now()));
        return
      end

      if obj.wasAborted,
        obj.updateStopButton() ;
        msg = 'Tracking jobs killed.';
        TrackMonitorViz.debugfprintf('Tracking jobs killed, results received %s\n',datestr(now()));
        return
      end
      
      if isempty(pollingResult) ,
        tfSucc = true ;
        msg = 'No one will read this.' ;
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

      % Check if this is an ID linking job and handle ID training axis
      if isfield(pollingResult, 'result_type') && strcmp(pollingResult.result_type, 'id_link')
        obj.handleIDTrainingUpdate(pollingResult);
      end 

      % It is assumed that there is a correspondence between res and .hline
      if nJobs~=numel(obj.hline)
        warningNoTrace('Unexpected monitor results size (%d); expected (%d).',...
                       nJobs,numel(obj.hline));
      end

      % always update info about current tracker, as labels may have changed
      s = obj.dtObj.getTrackerInfoString();
      obj.htrackerInfo.String = s;

      ticId = tic() ;
      for ijob=1:nJobs,
        isdone = pollingResult.tfComplete(ijob);
        if isfield(pollingResult,'parttrkfileTimestamp'),
          partFileExists = ~isnan(pollingResult.parttrkfileTimestamp(ijob));
          isupdate = ...
            (partFileExists && (forceupdate || (pollingResult.parttrkfileTimestamp(ijob)>obj.parttrkfileTimestamps(ijob)))) || ...
            isdone ;
        else
          isupdate = false ;
        end

        if isupdate,
          if obj.bulkAxsIsBulkMode
            % just update indicator based on isdone; dont try to get
            % nframes tracked, etc            
            if isdone
              set(obj.hline(ijob),'FaceColor',obj.COLOR_AXSWAIT_BULK_TRACKED);
              obj.bulkMovTracked(ijob) = true;
            else
              % none
            end            
          else
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
        end
        
        % if res(ijob).killFileExists,
        %   obj.isKilled = true;
        %   set(obj.hline(ijob),'FaceColor',obj.COLOR_AXSWAIT_KILLED);
        %   obj.hfig.UserData = 'killed';
        % end
        if ~obj.bulkAxsIsBulkMode
          TrackMonitorViz.debugfprintf('Job %d: %d. ',ijob,obj.nFramesTracked(ijob));
        end
      end
      TrackMonitorViz.debugfprintf('\n');
      TrackMonitorViz.debugfprintf('Update of nFramesTracked took %f s.\n',toc(ticId));
      
      obj.resLast = pollingResult ;
      
      obj.updateErrDisplay(pollingResult);
      [tfSucc,msg] = obj.updateStatusDisplayLine_(pollingResult);      
      obj.updateStopButton() ;
    end
    
    function [tfSucc,status] = updateStatusDisplayLine_(obj, pollingResult)
      % pollsuccess: [nview] logical
      % pollts: [nview] timestamps
      
      tfSucc = true;
      nJobs = numel(pollingResult.tfComplete);  % nJobs == nmovies * nviews * nstages
      pollsuccess = true(1,nJobs);
      isTrackComplete = false;
      isErr = false;
      isLogFile = false;
      if ~isempty(pollingResult),
        isTrackComplete = all([pollingResult.tfComplete]);
        isErr = any([pollingResult.errFileExists]) ;
        isLogFile = any([pollingResult.logFileExists]);
      end
      
      if ~isempty(pollingResult) && isfield(pollingResult,'isRunning')
        isRunning = any([pollingResult.isRunning]);
      else
        isRunning = true ;
      end
      
      if obj.wasAborted,
        status = 'Tracking process aborted.';
        tfSucc = false;
      elseif isTrackComplete
        status = 'Tracking complete.';
        obj.updateStatusFinal(nJobs)
      elseif ~isRunning,
        if isErr,
          status = 'Error while tracking.';
        else
          status = 'No tracking jobs running.';
        end
        % handles = guidata(obj.hfig);
        % TrackMonitorViz.updateStartStopButton(handles,false,false);        
        obj.updateStopButton() ;
        tfSucc = false;
      elseif isErr,
        status = 'Error while tracking.';
        tfSucc = false;
      elseif isfield(pollingResult,'result_type')&& strcmp(pollingResult.result_type,'id_link') 
        if isLogFile && pollingResult.jsonFileExist && numel(pollingResult.idstep)>0
            status = sprintf('ID Training in progress. %d iterations completed',pollingResult.idstep(end));
        else
          status = 'Initializing Training for ID Linking. ';            
        end
      elseif isLogFile,
        if obj.bulkAxsIsBulkMode
          status = sprintf('Tracking in progress. %d/%d movies tracked.',...
            nnz(obj.bulkMovTracked),numel(obj.bulkMovTracked));
        elseif nJobs > 1,
          status = sprintf('Tracking in progress. %s frames tracked.',mat2str(obj.nFramesTracked));
        else
          status = sprintf('Tracking in progress. %d frames tracked.',obj.nFramesTracked);
        end
      else
        status = 'Initializing tracking.';
      end
      
      clusterstr = apt.monitorBackendDescription(obj.backendType) ;
      str = sprintf('%s status: %s (at %s)',clusterstr,status,strtrim(datestr(now(),'HH:MM:SS PM'))) ;
      isAllGood = all(pollsuccess) && ~isErr ;
      obj.setStatusDisplayLine(str, isAllGood) ;
    end  % function
    
    function updateErrDisplay(obj, pollingResult)
      isErr = any([pollingResult.errFileExists]) ;
      if ~isErr,
        return;
      end
      handles = guidata(obj.hfig);
      if any([pollingResult.errFileExists]),
        erri = find(strcmp(handles.popupmenu_actions.String,'Show error messages'),1);
        if numel(erri) ~= 1,
          return;
        end
        handles.popupmenu_actions.Value = erri;
      else
        erri = find(strcmp(handles.popupmenu_actions.String,'Show log files'),1);
        if numel(erri) ~= 1,
          return;
        end
        handles.popupmenu_actions.Value = erri;        
      end
      obj.updateClusterInfo();
      handles.text_clusterinfo.ForegroundColor = 'r';
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
      obj.setStatusDisplayLine('Killing tracking jobs...', false) ;
      handles = guidata(obj.hfig);
      handles.pushbutton_startstop.String = 'Stopping tracking...';
      handles.pushbutton_startstop.Enable = 'off';
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
      obj.setStatusDisplayLine('Tracking process killed.', false);
      drawnow;

    end
    
    function updateClusterInfo(obj)
      
      handles = guidata(obj.hfig);
      actions = handles.popupmenu_actions.String; %#ok<PROP>
      v = handles.popupmenu_actions.Value;
      action = actions{v}; %#ok<PROP>
      switch action
        case 'Show log files',
         ss = obj.getLogFilesSummary();
         handles.text_clusterinfo.String = ss;
         drawnow;
        case 'Update tracking monitor',
          obj.updateMonitorPlots();
          drawnow;
        case {'List all jobs on cluster','List all docker jobs','List all conda jobs'},
          ss = obj.detailedStatusStringFromRegisteredJobIndex_();
          handles.text_clusterinfo.String = ss;
          drawnow;
        case 'Show tracking jobs'' status',
          ss = obj.queryAllJobsStatus();
          handles.text_clusterinfo.String = ss;
          drawnow;
        case 'Show error messages',
          if isempty(obj.resLast) || ~any([obj.resLast.errFileExists]),
            ss = 'No error messages.';
          else
            ss = obj.getErrorFilesSummary() ;
          end
          handles.text_clusterinfo.String = ss;
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
      handles = guidata(obj.hfig) ;
      labeler = obj.labeler_ ;
      isRunning = labeler.bgTrkIsRunning ;
      if isRunning
        isComplete = [] ;
      else
        isComplete = (labeler.lastTrackEndCause == EndCause.complete) ;
      end      
      if isRunning ,
        set(handles.pushbutton_startstop,'String','Stop tracking','BackgroundColor',[.64,.08,.18],'Enable','on','UserData','stop');
      else
        if isComplete ,
          set(handles.pushbutton_startstop,'String','Tracking complete','BackgroundColor',[.466,.674,.188],'Enable','off','UserData','done');
        else
          set(handles.pushbutton_startstop,'String','Tracking stopped','BackgroundColor',[.64,.08,.18],'Enable','off','UserData','done');
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
    function setStatusDisplayLine(obj, str, isallgood)
      % Set either or both of the status message line and the color of the status
      % message.  Any of the two (non-obj) args can be empty, in which case that
      % aspect is not changed.  obj.hfig's guidata must have a text_clusterstatus
      % field containing the handle of an 'text' appropriate graphics object.

      hfig = obj.hfig ;
      handles = guidata(hfig);
      text_h = handles.text_clusterstatus ;
      if ~exist('str', 'var') ,
        str = [] ;
      end
      if ~exist('isallgood', 'var') ,
        isallgood = [] ;
      end
      if isempty(str) ,
        % do nothing
      else
        set(text_h, 'String', str) ;
      end
      if isempty(isallgood) ,
        % do nothing
      else
        color = fif(isallgood, 'g', 'r') ;
        set(text_h, 'ForegroundColor',color) ;
      end
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
      handles = guidata(obj.hfig);

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
