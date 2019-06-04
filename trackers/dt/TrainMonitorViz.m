classdef TrainMonitorViz < handle
  properties
    hfig % scalar fig
    haxs % [2] axis handle, viz training loss, dist
    hannlastupdated % [1] textbox/annotation handle
    hline % [nviewx2] line handle, one loss curve per view
    hlinekill % [nviewx2] line handle, killed marker per view
    
    isKilled = false; % scalar, whether training has been halted
    lastTrainIter; % [nview] last iteration of training
    
    axisXRange = 2e3; % show last (this many) iterations along x-axis
    
    resLast % last training json contents received
    dtObj % DeepTracker Obj
    trainWorkerObj = [];
    backEnd % scalar DLBackEnd
    actions = struct(...
      'Bsub',...
        {{'List all jobs on cluster'...
        'Show training jobs'' status'...
        'Update training monitor plots'...
        'Show log files'...
        'Show error messages'}},...
      'Conda',...
        {{'List all conda jobs'...
        'Show training jobs'' status',...
        'Update training monitor plots'...
        'Show log files'...
        'Show error messages'}},...
      'Docker',...
        {{'List all docker jobs'...
        'Show training jobs'' status',...
        'Update training monitor plots'...
        'Show log files'...
        'Show error messages'}},...
      'AWS',...
        {{'Update training monitor plots'...
        'Show log files'...
        'Show error messages'}});
  end
  properties (Dependent)
    nview 
  end
  
  properties (Constant)
    DEBUG = false;
  end
  
  methods (Static)
    function debugfprintf(varargin)
      if TrainMonitorViz.DEBUG,
        fprintf(varargin{:});
      end
    end
  end
  methods
    function v = get.nview(obj)
      v = size(obj.hline,1);
    end
  end
  
  methods
    
    function obj = TrainMonitorViz(nview,dtObj,trainWorkerObj,backEnd)
      obj.dtObj = dtObj;
      obj.trainWorkerObj = trainWorkerObj;
      obj.backEnd = backEnd;
      obj.hfig = TrainMonitorGUI(obj);
      handles = guidata(obj.hfig);
      TrainMonitorViz.updateStartStopButton(handles,true,false);
      handles.pushbutton_startstop.Enable = 'on';
      obj.haxs = [handles.axes_loss,handles.axes_dist];
      obj.hannlastupdated = handles.text_clusterstatus;
      
      % reset
      arrayfun(@(x)cla(x),obj.haxs);
      obj.hannlastupdated.String = 'Cluster status: Initializing...';
      handles.text_clusterinfo.String = '...';
      handles.popupmenu_actions.String = obj.actions.(char(backEnd));
      handles.popupmenu_actions.Value = 1;
      
      arrayfun(@(x)grid(x,'on'),obj.haxs);
      arrayfun(@(x)hold(x,'on'),obj.haxs);
      title(obj.haxs(1),'Training Monitor','fontweight','bold');
      xlabel(obj.haxs(2),'Iteration');
      ylabel(obj.haxs(1),'Loss');
      ylabel(obj.haxs(2),'Dist');
      linkaxes(obj.haxs,'x');
      set(obj.haxs(1),'XTickLabel',{});
      
      %obj.hannlastupdated = TrainMonitorViz.createAnnUpdate(obj.haxs(1));
      
      clrs = lines(nview)*.9+.1;
      h = gobjects(nview,2);
      hkill = gobjects(nview,2);
      for ivw=1:nview
        for j=1:2
          h(ivw,j) = plot(obj.haxs(j),nan,nan,'.-','color',clrs(ivw,:),'LineWidth',2);
          hkill(ivw,j) = plot(obj.haxs(j),nan,nan,'rx','markersize',12,'linewidth',2);
        end
      end
      if nview > 1,
        viewstrs = arrayfun(@(x)sprintf('view%d',x),(1:nview)','uni',0);
        legend(obj.haxs(2),h(:,1),viewstrs,'TextColor','w');
      end
      set(obj.haxs,'XLimMode','manual','YScale','log');
      obj.hline = h;
      obj.hlinekill = hkill;
      obj.resLast = [];
      obj.isKilled = false;
      obj.lastTrainIter = zeros(1,nview);      
    end
    
    function delete(obj)
      deleteValidHandles(obj.hfig);
      obj.hfig = [];
%       obj.haxs = [];
    end
        
    function [tfSucc,msg] = resultsReceived(obj,sRes,forceupdate)
      % Callback executed when new result received from training monitor BG
      % worker
      %
      % trnComplete: scalar logical, true when all views done
      
      tfSucc = false;
      msg = ''; %#ok<NASGU>
      
      if isempty(obj.hfig) || ~ishandle(obj.hfig),
        msg = 'Monitor closed';
        TrainMonitorViz.debugfprintf('Monitor closed, results received %s\n',datestr(now));
        return;
      end
      
      res = sRes.result;
      tfAnyLineUpdate = false;
      lineUpdateMaxStep = zeros(1,numel(res));
      
      h = obj.hline;
      
      if nargin < 3,
        forceupdate = false;
      end
      
      for ivw=1:numel(res)
        if res(ivw).pollsuccess
          if res(ivw).jsonPresent && (forceupdate || res(ivw).tfUpdate)
            contents = res(ivw).contents;
            set(h(ivw,1),'XData',contents.step,'YData',contents.train_loss);
            set(h(ivw,2),'XData',contents.step,'YData',contents.train_dist);
            tfAnyLineUpdate = true;
            lineUpdateMaxStep(ivw) = contents.step(end);
          end

          if res(ivw).killFileExists, 
            obj.isKilled = true;
            if res(ivw).jsonPresent,
              contents = res(ivw).contents;
              hkill = obj.hlinekill;
              % hmm really want to mark the last 2k interval when model is
              % actually saved
              set(hkill(ivw,1),'XData',contents.step(end),'YData',contents.train_loss(end));
              set(hkill(ivw,2),'XData',contents.step(end),'YData',contents.train_dist(end));
            end
            handles = guidata(obj.hfig);
            handles.pushbutton_startstop.Enable = 'on';

          end
        
          if res(ivw).tfComplete
            contents = res(ivw).contents;
            if ~isempty(contents)
              hkill = obj.hlinekill;
              % re-use kill marker 
              set(hkill(ivw,1),'XData',contents.step(end),'YData',contents.train_loss(end),...
                'color',[0 0.5 0],'marker','o');
              set(hkill(ivw,2),'XData',contents.step(end),'YData',contents.train_dist(end),...
                'color',[0 0.5 0],'marker','o');
            end
          end
        end
      end
      
      if any([res.errFileExists]),
        handles = guidata(obj.hfig);
        i = find(strcmp(handles.popupmenu_actions.String,'Show error messages'));
        if ~isempty(i),
          handles.popupmenu_actions.Value = i;
        end
      end
      
      if tfAnyLineUpdate
        obj.lastTrainIter = max(obj.lastTrainIter,lineUpdateMaxStep);
        obj.adjustAxes(max(obj.lastTrainIter));
        %obj.dtObj.setTrackerInfo('iterCurr',obj.lastTrainIter);
      end
      
      if isempty(obj.resLast) || tfAnyLineUpdate
        obj.resLast = res;
      end

      [tfSucc,msg] = obj.updateAnn(res);
      TrainMonitorViz.debugfprintf('resultsReceived - tfSucc = %d, msg = %s\n',tfSucc,msg);
    end
    
    function [tfSucc,status] = updateAnn(obj,res)
      % pollsuccess: [nview] logical
      % pollts: [nview] timestamps
      
      tfSucc = true;
      pollsuccess = [res.pollsuccess];
      
      clusterstr = 'Cluster';
      switch obj.backEnd        
        case DLBackEnd.Bsub
          clusterstr = 'JRC cluster';
        case DLBackEnd.Conda
          clusterstr = 'Local';
        case DLBackEnd.Docker
          clusterstr = 'Local';
        case DLBackEnd.AWS,
          clusterstr = 'AWS';
        otherwise
          warning('Unknown back end type');
      end
            
      isTrainComplete = false;
      isErr = false;
      isLogFile = false;
      if ~isempty(res),
        isTrainComplete = all([res.tfComplete]);
        isErr = any([res.errFileExists]) || any([res.logFileErrLikely]);
        % to-do: figure out how to make this robust to different file
        % systems
        isLogFile = cellfun(@(x) exist(x,'file'),{res.logFile});
        isJsonFile = [res.jsonPresent]>0;
      end
      
      isRunning0 = obj.trainWorkerObj.getIsRunning();
      if isempty(isRunning0),
        isRunning = true;
      else
        isRunning = any(isRunning0);
      end

      TrainMonitorViz.debugfprintf('updateAnn: isRunning = %d, isTrainComplete = %d, isErr = %d, isKilled = %d\n',isRunning,isTrainComplete,isErr,obj.isKilled);
      
      if obj.isKilled,
        status = 'Training process killed.';
        tfSucc = false;
      elseif isErr,
        status = sprintf('Error while training after %s iterations',mat2str(obj.lastTrainIter));
        tfSucc = false;
      elseif isTrainComplete,
        status = 'Training complete.';
        handles = guidata(obj.hfig);
        TrainMonitorViz.updateStartStopButton(handles,false,true);
      elseif ~isRunning,
        status = 'No training jobs running.';
        tfSucc = false;
      elseif any(isLogFile) && all(~isJsonFile),
        status = 'Training in progress. Building training image database.';
      elseif any(isLogFile) && any(isJsonFile),
        status = sprintf('Training in progress. %s iterations completed.',mat2str(obj.lastTrainIter));
      else
        status = 'Initializing training.';
      end
      
      str = {sprintf('%s status: %s',clusterstr,status),sprintf('Monitor updated %s.',datestr(now,'HH:MM:SS PM'))};
      hAnn = obj.hannlastupdated;
      hAnn.String = str;
      
      tfsucc = all(pollsuccess);
      if all(tfsucc)
        hAnn.ForegroundColor = [0 1 0];
      else
        hAnn.ForegroundColor = [1 0 0];
      end
      
%       ax = obj.haxs(1);
%       hAnn.Position(1) = ax.Position(1)+ax.Position(3)-hAnn.Position(3);
%       hAnn.Position(2) = ax.Position(2)+ax.Position(4)-hAnn.Position(4);
    end
    
    function adjustAxes(obj,lineUpdateMaxStep)
      for i=1:numel(obj.haxs)
        ax = obj.haxs(i);
        xlim = ax.XLim;
        x0 = max(0,lineUpdateMaxStep-obj.axisXRange);
        xlim(2) = max(1,lineUpdateMaxStep+0.5*(lineUpdateMaxStep-x0));
        ax.XLim = xlim;
        %ylim(ax,'auto');
      end
    end
    
    function stopTraining(obj)
      if isempty(obj.trainWorkerObj),
        warning('trainWorkerObj is empty -- cannot kill process');
        return;
      end
      obj.SetBusy('Killing training jobs...',true);
      handles = guidata(obj.hfig);
      handles.pushbutton_startstop.String = 'Stopping training...';
      handles.pushbutton_startstop.Enable = 'off';
      drawnow;
      [tfsucc,warnings] = obj.trainWorkerObj.killProcess();
      obj.isKilled = true;
      if tfsucc,
        
        startTime = tic;
        maxWaitTime = 30;
        while true,
          if toc(startTime) > maxWaitTime,
            warndlg([{'Training processes may not have been killed properly:'},warnings],'Problem stopping training','modal');
            break;
          end
          if ~obj.dtObj.bgTrnIsRunning,
            break;
          end
          pause(1);
        end        
      else
        warndlg([{'Training processes may not have been killed properly:'},warnings],'Problem stopping training','modal');
      end
      obj.ClearBusy('Training process killed');
      TrainMonitorViz.updateStartStopButton(handles,false,false);
    end
    
    function startTraining(obj)
      % Placeholder meth AL 20190108
      % - Always do a regular restart for now; if project is updated might
      % want RestartAug.
      % - If the training has reached final iter, training will immediately 
      % end
      
      % Kills and creates new TrainMonitorViz, maybe that's fine
      
      obj.dtObj.retrain('dlTrnType',DLTrainType.Restart);
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
        case 'Update training monitor plots',
          obj.updateMonitorPlots();
          drawnow;
        case {'List all jobs on cluster','List all docker jobs','List all conda jobs'}
          ss = obj.queryAllJobsStatus();
          handles.text_clusterinfo.String = ss;
          drawnow;
        case 'Show training jobs'' status',
          ss = obj.queryTrainJobsStatus();
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
    end
    
    function ss = getLogFilesContents(obj)
      
      ss = obj.trainWorkerObj.getLogfilesContent;
      
    end
    
    function ss = getErrorFileContents(obj)
      
      ss = obj.trainWorkerObj.getErrorfileContent;
      
    end
    
    function updateMonitorPlots(obj)
      
      sRes.result = obj.trainWorkerObj.compute();
      obj.resultsReceived(sRes,true);
      
    end
    
    function ss = queryAllJobsStatus(obj)
      
      ss = obj.trainWorkerObj.queryAllJobsStatus();
      if ischar(ss),
        ss = strsplit(ss,'\n');
      end
      
    end
    
    function ss = queryTrainJobsStatus(obj)
      
      ss = {};
      raw = obj.trainWorkerObj.queryMyJobsStatus();
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
        set(handles.figure_TrainMonitor,'Pointer','watch');
        if ~isempty(s),
          set(handles.text_clusterstatus,'String',s,'ForegroundColor','r');
        end
        
      else
        set(handles.figure_TrainMonitor,'Pointer','arrow');
        set(handles.text_clusterstatus,'ForegroundColor','g');
      end

      drawnow('limitrate');
    end
    
  end
  
  methods (Static)
    function hAnn = createAnnUpdate(ax)
      hfig = ax.Parent;
      ax.Units = 'normalized';
      hfig.Units = 'normalized';
      str = sprintf('last updated: %s',datestr(now,'HH:MM:SS PM'));
      hAnn = annotation(hfig,'textbox',[1 1 0.1 0.1],...
        'String',str,'FitBoxToText','on','EdgeColor',[1 1 1],...
        'FontAngle','italic','HorizontalAlignment','right');
      drawnow;
      hAnn.Position(1) = ax.Position(1)+ax.Position(3)-hAnn.Position(3);
      hAnn.Position(2) = ax.Position(2)+ax.Position(4)-hAnn.Position(4);
    end   
    
    function updateStartStopButton(handles,isStop,isDone)
      
      if nargin < 3,
        isDone = false;
      end
      
      if isDone,
        set(handles.pushbutton_startstop,'String','Training complete','BackgroundColor',[.466,.674,.188],'Enable','off','UserData','done');
      else
        if isStop,
          set(handles.pushbutton_startstop,'String','Stop training','BackgroundColor',[.64,.08,.18],'Enable','on','UserData','stop');
        else
          set(handles.pushbutton_startstop,'String','Restart training','BackgroundColor',[.3,.75,.93],'Enable','on','UserData','start');
        end
      end
      
    end
    
  end
  
end
