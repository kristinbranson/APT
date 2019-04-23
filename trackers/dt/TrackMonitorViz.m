classdef TrackMonitorViz < handle
  properties
    hfig % scalar fig
    haxs % [1] axis handle, viz wait time
    hannlastupdated % [1] textbox/annotation handle
    hline % [nviewx1] patch handle showing fraction of frames tracked
    htext % [nviewx1] text handle showing fraction of frames tracked
    htrackerInfo % scalar text box handle showing information about current tracker
    isKilled = false; % scalar, whether tracking has been halted
    
    resLast = []; % last contents received
    dtObj % DeepTracker Obj
    trackWorkerObj = [];
    backEnd % scalar DLBackEnd
    parttrkfileTimestamps = [];
    nFramesTracked = [];
    nFramesToTrack = 0;
    jobDescs = {};
    actions = struct(...
      'Bsub',...
      {{'List all jobs on cluster'...
      'Show tracking jobs'' status'...
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
  end
  
  
  properties (Constant)
    DEBUG = false;
  end
  
  methods (Static)
    function debugfprintf(varargin)
      if TrackMonitorViz.DEBUG,
        fprintf(varargin{:});
      end
    end
  end
  
  methods
    function obj = TrackMonitorViz(nview,dtObj,trackWorkerObj,backEnd,nFramesToTrack,jobDescs)
      obj.dtObj = dtObj;
      obj.trackWorkerObj = trackWorkerObj;
      obj.backEnd = backEnd;
      nMovies = numel(nFramesToTrack);
      obj.nFramesToTrack = repmat(nFramesToTrack,nview);
      nJobs = nMovies*nview;
      if ~exist('jobDescs','var'),
        jobDescs = cell(nMovies,nview);
        for imov = 1:nMovies,
          if nMovies > 1,
            movstr = sprintf(', Mov %d',imov);
          else
            movstr = '';
          end
          for ivw = 1:nview,
            if nview > 1,
              vwstr = sprintf(', Vw %d',ivw);
            else
              vwstr = '';
            end
            jobDescs{imov,ivw} = [movstr,vwstr];
          end
        end
      end
      obj.jobDescs = jobDescs;
      
      obj.hfig = TrackMonitorGUI(obj);
      handles = guidata(obj.hfig);
      TrackMonitorViz.updateStartStopButton(handles,true,false);
      %handles.pushbutton_startstop.Enable = 'on';
      obj.hfig.UserData = 'running';
      obj.haxs = [handles.axes_wait];
      obj.hannlastupdated = handles.text_clusterstatus;
      obj.htrackerInfo = handles.edit_trackerinfo;

      % reset plots
      arrayfun(@(x)cla(x),obj.haxs);
      obj.hannlastupdated.String = 'Cluster status: Initializing...';
      handles.text_clusterinfo.String = '...';
	  % set info about current tracker
      s = obj.dtObj.getTrackerInfoString();
      obj.htrackerInfo.String = s;
      handles.popupmenu_actions.String = obj.actions.(char(backEnd));
      handles.popupmenu_actions.Value = 1;
      handles.axes_wait.YLim = [0,nJobs];
      handles.axes_wait.XLim = [0,1+obj.minFracComplete];
      handles.axes_wait.XTick = [];
      handles.axes_wait.YTick = [];
      hold(handles.axes_wait,'on');

      clrs = lines(nJobs);
      obj.hline = gobjects(nJobs,1);
      obj.htext = gobjects(nJobs,1);
      for ijob = 1:nJobs,
        obj.hline(ijob) = patch([0,0,1,1,0]*obj.minFracComplete,...
          ijob-[0,1,1,0,0],clrs(ijob,:),...
          'Parent',handles.axes_wait,...
          'EdgeColor','w');
        if nJobs > 1,
          sview = jobDescs{ijob};
        else
          sview = '';
        end
        obj.htext(ijob) = text((1+obj.minFracComplete)/2,ijob-.5,...
          sprintf('0/%d frames tracked%s',obj.nFramesToTrack(ijob),sview),...
          'Color','w','HorizontalAlignment','center',...
          'VerticalAlignment','middle','Parent',handles.axes_wait);
      end
      
      obj.resLast = [];
      obj.parttrkfileTimestamps = zeros(1,nJobs);
      obj.nFramesTracked = zeros(1,nJobs);
      obj.isKilled = false;
      drawnow;
            
    end
    
    function delete(obj)
      deleteValidHandles(obj.hfig);
      obj.hfig = [];
%       obj.haxs = [];
    end
        
    function [tfSucc,msg] = resultsReceived(obj,sRes,forceupdate)
      % Callback executed when new result received from monitor BG
      % worker
      %
      % trnComplete: scalar logical, true when all views done
      
      tfSucc = false;
      msg = '';
      
      if nargin < 3,
        forceupdate = false;
      end
      
      if isempty(obj.hfig) || ~ishandle(obj.hfig),
        msg = 'Monitor closed.';
        TrackMonitorViz.debugfprintf('Monitor closed, results received %s\n',datestr(now));
        return;
      end

      if obj.isKilled,
        msg = 'Tracking jobs killed.';
        TrackMonitorViz.debugfprintf('Tracking jobs killed, results received %s\n',datestr(now));
        return;
      end

      
      TrackMonitorViz.debugfprintf('%s: TrackMonitorViz results received:\n',datestr(now));
      res = sRes.result;      
            
      TrackMonitorViz.debugfprintf('Partial tracks exist: %d\n',exist(res(1).parttrkfile,'file'));
      TrackMonitorViz.debugfprintf('N. frames tracked: ');
      nJobs = numel(res);

      % always update info about current tracker, as labels may have changed
      s = obj.dtObj.getTrackerInfoString();
      obj.htrackerInfo.String = s;

      tic;
      for ijob=1:nJobs,
        isdone = res(ijob).tfComplete;
        partFileExists = ~isnan(res(ijob).parttrkfileTimestamp); % maybe unnec since parttrkfileTimestamp will be nan otherwise
        isupdate = ...
          (partFileExists && (forceupdate || (res(ijob).parttrkfileTimestamp>obj.parttrkfileTimestamps(ijob)))) ...
           || isdone;

        if isupdate,          
          try
            if isfield(res(ijob),'parttrkfileNfrmtracked')
              % for AWS and any worker that figures this out on its own
              obj.nFramesTracked(ijob) = nanmax(res(ijob).parttrkfileNfrmtracked,...
                res(ijob).trkfileNfrmtracked);
            else
              didload = false;
              if isdone,
                try
                  ptrk = load(res(ijob).trkfile,'pTrk','-mat');
                  didload = true;
                catch,
                  warning('isdone = true and could not load pTrk');
                end
              else
                try
                  ptrk = load(res(ijob).parttrkfile,'pTrk','-mat');
                  didload = true;
                catch,
                end
              end
              if didload && isfield(ptrk,'pTrk'),
                try
                  obj.nFramesTracked(ijob) = nnz(~isnan(ptrk.pTrk(1,1,:,:)));
                catch ME
                  warning(getReport(ME));
                end
              end
            end
            
            if nJobs > 1,
              sview = obj.jobDescs{ijob};
            else
              sview = '';
            end
            set(obj.htext(ijob),'String',sprintf('%d/%d frames tracked%s',obj.nFramesTracked(ijob),obj.nFramesToTrack(ijob),sview));
            fracComplete = obj.minFracComplete + (obj.nFramesTracked(ijob)/obj.nFramesToTrack(ijob));
            set(obj.hline(ijob),'XData',[0,0,1,1,0]*fracComplete);
            
          catch ME,
            fprintf('Could not update nFramesTracked:\n%s',getReport(ME));
          end

        end
        
        if res(ijob).killFileExists,
          obj.isKilled = true;
          set(obj.hline(ijob),'FaceColor',[.5,.5,.5]);
          obj.hfig.UserData = 'killed';
        end
        TrackMonitorViz.debugfprintf('Job %d: %d. ',ijob,obj.nFramesTracked(ijob));
      end
      TrackMonitorViz.debugfprintf('\n');
      TrackMonitorViz.debugfprintf('Update of nFramesTracked took %f s.\n',toc);
      
      if isstruct(sRes) && isfield(sRes,'result'),
        obj.resLast = sRes.result;
      end
      
      obj.updateErrDisplay(res);
      [tfSucc,msg] = obj.updateAnn(res);
      
    end
    
    function [tfSucc,status] = updateAnn(obj,res)
      % pollsuccess: [nview] logical
      % pollts: [nview] timestamps
      
      tfSucc = true;
      nJobs = numel(res);
      pollsuccess = true(1,nJobs);
      
      clusterstr = 'Cluster';
      switch obj.backEnd
        case DLBackEnd.Bsub
          clusterstr = 'JRC cluster';
        case DLBackEnd.Docker
          clusterstr = 'Local';
        case DLBackEnd.AWS,
          clusterstr = 'AWS';
        otherwise
          warning('Unknown back end type');
      end
          
      isTrackComplete = false;
      isErr = false;
      isLogFile = false;
      if ~isempty(res),
        isTrackComplete = all([res.tfComplete]);
        isErr = any([res.errFileExists]) || any([res.logFileErrLikely]);
        % to-do: figure out how to make this robust to different file
        % systems
        isLogFile = any([res.logFileExists]);
      end
      
      if ~isempty(res) && isfield(res,'isRunning')
        isRunning = any([res.isRunning]);
      else
        isRunning = true;
      end
      
      if obj.isKilled,
        status = 'Tracking process killed.';
        tfSucc = false;
      elseif isTrackComplete,
        status = 'Tracking complete.';
        handles = guidata(obj.hfig);
        TrackMonitorViz.updateStartStopButton(handles,false,true);
      elseif ~isRunning,
        if isErr,
          status = 'Error while tracking.';
        else
          status = 'No tracking jobs running.';
        end
        handles = guidata(obj.hfig);
        TrackMonitorViz.updateStartStopButton(handles,false,false);        
        tfSucc = false;
      elseif isErr,
        status = 'Error while tracking.';
        tfSucc = false;
      elseif isLogFile,
        if nJobs > 1,
          status = sprintf('Tracking in progress. %s frames tracked.',mat2str(obj.nFramesTracked));
        else
          status = sprintf('Tracking in progress. %d frames tracked.',obj.nFramesTracked);
        end
      else
        status = 'Initializing tracking.';
      end
      
      str = {sprintf('%s status: %s',clusterstr,status),sprintf('Monitor updated %s.',datestr(now,'HH:MM:SS PM'))};
      hAnn = obj.hannlastupdated;
      hAnn.String = str;
      
      isok = all(pollsuccess) && ~isErr;
      if all(isok)
        hAnn.ForegroundColor = [0 1 0];
      else
        hAnn.ForegroundColor = [1 0 0];
      end
      
      %       ax = obj.haxs(1);
      %       hAnn.Position(1) = ax.Position(1)+ax.Position(3)-hAnn.Position(3);
      %       hAnn.Position(2) = ax.Position(2)+ax.Position(4)-hAnn.Position(4);
    end
    
    function updateErrDisplay(obj,res)
      isErr = any([res.errFileExists]) || any([res.logFileErrLikely]);
      if ~isErr,
        return;
      end
      handles = guidata(obj.hfig);
      if any([res.errFileExists]),
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
      TrackMonitorViz.updateStartStopButton(handles,false,false);
      drawnow;
    end
        
    function stopTracking(obj)
      
%       warning('not implemented');
%       return;
      
      if isempty(obj.trackWorkerObj),
        warning('trackWorkerObj is empty -- cannot kill process');
        return;
      end
      obj.SetBusy('Killing tracking jobs...',true);
      handles = guidata(obj.hfig);
      handles.pushbutton_startstop.String = 'Stopping tracking...';
      handles.pushbutton_startstop.Enable = 'off';
      [tfsucc,warnings] = obj.trackWorkerObj.killProcess();
      if tfsucc,
        obj.isKilled = true;
      else
        warndlg([{'Tracking processes may not have been killed properly:'},warnings],'Problem stopping tracking','modal');
      end
      TrackMonitorViz.updateStartStopButton(handles,false,false);
      obj.ClearBusy('Tracking process killed');
      drawnow;

    end
    
    function updateClusterInfo(obj)
      
      handles = guidata(obj.hfig);
      actions = handles.popupmenu_actions.String; %#ok<PROP>
      v = handles.popupmenu_actions.Value;
      action = actions{v}; %#ok<PROP>
      switch action
        case 'Show log files',
         ss = obj.getLogFilesContents();
         handles.text_clusterinfo.String = ss;
         drawnow;
        case 'Update tracking monitor',
          obj.updateMonitorPlots();
          drawnow;
        case {'List all jobs on cluster','List all docker jobs'},
          ss = obj.queryAllJobsStatus();
          handles.text_clusterinfo.String = ss;
          drawnow;
        case 'Show tracking jobs'' status',
          ss = obj.queryTrackJobsStatus();
          handles.text_clusterinfo.String = ss;
          drawnow;
        case 'Show error messages',
          if isempty(obj.resLast) || ~any([obj.resLast.errFileExists]),
            ss = 'No error messages.';
          else
            ss = obj.getErrorFileContents();
          end
          handles.text_clusterinfo.String = ss;
        otherwise
          fprintf('%s not implemented\n',action);
          return;
      end
      %handles.text_clusterinfo.ForegroundColor = 'w';
    end
    
    
    
    function ss = getLogFilesContents(obj)
      
      ss = obj.trackWorkerObj.getLogfilesContent;
      
    end
    
    function ss = getErrorFileContents(obj)
      
      ss = obj.trackWorkerObj.getErrorfileContent;
      
    end
    
    function updateMonitorPlots(obj)
      
      sRes.result = obj.trackWorkerObj.compute();
      obj.resultsReceived(sRes,true);
      
    end
    
    function ss = queryAllJobsStatus(obj)
      
      ss = obj.trackWorkerObj.queryAllJobsStatus();
      ss = strsplit(ss,'\n');
      
    end
    
    function ss = queryTrackJobsStatus(obj)
      
      ss = {};
      raw = obj.trackWorkerObj.queryMyJobsStatus();
      nview = numel(raw);
      for i = 1:nview,
        snew = strsplit(raw{i},'\n');
        ss(end+1:end+numel(snew)) = snew;
      end

    end
    
    function ClearBusy(obj,s)

      obj.SetBusy(s,false);
    
    end

    function SetBusy(obj,s,isbusy)

      handles = guidata(obj.hfig);
      
      if nargin < 3
        isbusy = true;
      end
      
      if isbusy,
        set(obj.hfig,'Pointer','watch');
        if ~isempty(s),
          set(handles.text_clusterstatus,'String',s,'ForegroundColor','r');
        end
        
      else
        set(obj.hfig,'Pointer','arrow');
        set(handles.text_clusterstatus,'ForegroundColor','g');
      end

      drawnow('limitrate');
    end
    
  end
  
  methods (Static)
    
    function updateStartStopButton(handles,isStop,isDone,msg)
      
      if nargin < 3,
        isDone = false;
      end
      if nargin < 4,
        msg = 'Tracking complete';
      end
      
      if isDone == 1,
        set(handles.pushbutton_startstop,'String',msg,'BackgroundColor',[.466,.674,.188],'Enable','off','UserData','done');
      else
        if isStop,
          set(handles.pushbutton_startstop,'String','Stop tracking','BackgroundColor',[.64,.08,.18],'Enable','on','UserData','stop');
        else
          set(handles.pushbutton_startstop,'String','Tracking stopped','BackgroundColor',[.64,.08,.18],'Enable','off','UserData','done');
        end
      end
      
    end
    
  end
    
end
