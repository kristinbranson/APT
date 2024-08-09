classdef TrainMonitorViz < handle
  properties
    % 'sets' are groups of related trains that may be spawned in parallel
    % or serially. example is top-down trackers which have nset=2,
    % stage1=detect, stage2=pose.
    
    hfig % scalar fig
    haxs % [2xnset] axis handle, viz training loss, dist
    hannlastupdated % [1] textbox/annotation handle
    hline % [nmodel x 2] line handle, one loss curve per view
    hlinekill % [nmodel x 2] line handle, killed marker per view
    trainMontageFigs = []; % figure handles for showing training image montages
    setidx % [1 x nmodel], which set each line belongs to
    
    isKilled = []; % scalar, whether any training has been halted
    lastTrainIter; % [nset x nview] last iteration of training
    
    axisXRange = 2e3; % [nset] show last (this many) iterations along x-axis

    % AL 20220526. Testing MA/XV with 3 folds on bsub. Finding jobs are
    % ending before xv results MATs are done writing to disk (and visible
    % over NFS etc).
    %
    % Adding counter/delay so that jobs are considered "stopped" only once
    % they read as stopped a certain number of times in resultsReceived().
    % (Default polling time is 20-30seconds). 
    jobStoppedRepeatsReqd = 2; 
    
    resLast % last training json contents received
    dtObj % DeepTracker Obj
    trainWorkerObj = [];
    backEnd % scalar DLBackEnd
    actions = struct(...
      'Bsub',...
        {{...
        'Show sample training images' ...
        'List all jobs on cluster'...
        'Show training jobs'' status'...
        'Update training monitor plots'...
        'Show log files'...
        'Show error messages'}},...
      'Conda',...
        {{...
        'Show sample training images' ...
        'List all conda jobs'...
        'Show training jobs'' status',...
        'Update training monitor plots'...
        'Show log files'...
        'Show error messages'}},...
      'Docker',...
        {{...
        'Show sample training images' ...
        'List all docker jobs'...
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
    nmodels
    nset
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
    function v = get.nmodels(obj)
      v = size(obj.hline,1);
    end
    function v = get.nset(obj)
      v = size(obj.haxs,2);
    end
  end
  
  methods
    
    function obj = TrainMonitorViz(dmc,dtObj,trainWorkerObj,backEnd,...
        varargin)
            
      stage = dmc.getStages();
      view = dmc.getViews();
      splitidx = dmc.getSplits();
      nmodels = dmc.n;
      % sets currently correspond to stages
      [unique_stages,~,obj.setidx] = unique(stage);
      nsets = numel(unique_stages);
      if nsets > 1,
        set_names = arrayfun(@(x) sprintf(', Stage %d',x),unique_stages,'Uni',0);
      else
        set_names = {''};
      end

      lObj = dtObj.lObj;
      obj.dtObj = dtObj;
      obj.trainWorkerObj = trainWorkerObj;
      obj.backEnd = backEnd;
      obj.hfig = TrainMonitorGUI(obj);
      lObj.addDepHandle(obj.hfig);
      
      handles = guidata(obj.hfig);
      TrainMonitorViz.updateStartStopButton(handles,true,false);
      handles.pushbutton_startstop.Enable = 'on';
            
      obj.haxs = [handles.axes_loss;handles.axes_dist];
      obj.hannlastupdated = handles.text_clusterstatus;
      tfMultiSet = nsets>1;
      if tfMultiSet
        obj.splitaxs(nsets);
      end
      
      % reset
      arrayfun(@(x)cla(x),obj.haxs);
      obj.hannlastupdated.String = 'Cluster status: Initializing...';
      handles.text_clusterinfo.String = '...';
      handles.popupmenu_actions.String = obj.actions.(char(backEnd));
      handles.popupmenu_actions.Value = 1;
      
      arrayfun(@(x)grid(x,'on'),obj.haxs);
      arrayfun(@(x)hold(x,'on'),obj.haxs);
      %title(obj.haxs(1),'Training Monitor','fontweight','bold');
      for j = 1:nsets,
        xlabel(obj.haxs(2,j),['Iteration',set_names{j}]);
      end
      ylabel(obj.haxs(1),'Loss');
      ylabel(obj.haxs(2),'Dist');
      for j=1:size(obj.haxs,2)
        linkaxes(obj.haxs(:,j),'x');
      end
      set(obj.haxs(1,:),'XTickLabel',{});
      
      %obj.hannlastupdated = TrainMonitorViz.createAnnUpdate(obj.haxs(1));
      
      clrs = lines(nmodels)*.9+.1;
      h = gobjects(nmodels,2);
      hkill = gobjects(nmodels,2);
      for i=1:nmodels,
        iset = obj.setidx(i);
        for j=1:2,
          h(i,j) = plot(obj.haxs(j,iset),nan,nan,'.-','color',clrs(i,:),'LineWidth',2);
          hkill(i,j) = plot(obj.haxs(j,iset),nan,nan,'rx','markersize',12,'linewidth',2);
        end
      end
      ismultiview = numel(unique(view)) > 1;
      ismultisplit = numel(unique(splitidx(splitidx>0))) > 1;
      islegend = ismultiview || ismultisplit;
      if islegend,
        legstrs = repmat({''},[1,nmodels]);
        if ismultisplit,
          for i = 1:nmodels,
            legstrs{i} = [legstrs{i},sprintf('split %d ',splitidx(i))];
          end
        end
        if ismultiview,
          for i = 1:nmodels,
            legstrs{i} = [legstrs{i},sprintf('view %d ',view(i))];
          end
        end
        legend(obj.haxs(2,nsets),h(:,nsets),legstrs,'TextColor','w');
      end
      set(obj.haxs,'XLimMode','manual','YScale','log');
      obj.hline = h;
      obj.hlinekill = hkill;
      obj.resLast = [];
      obj.isKilled = false(1,nmodels);
      obj.lastTrainIter = zeros(1,nmodels);
      obj.axisXRange = repmat(obj.axisXRange,[1 nsets]);

      obj.jobStoppedRepeatsReqd = 2; 
    end
    
    function delete(obj)
      deleteValidHandles(obj.hfig);
      obj.hfig = [];
    end
    
    function splitaxs(obj,nsets)
      hax = obj.haxs;
      szassert(hax,[2 1]);
      haxnew = gobjects(2,nsets);
      SPACERFAC = 0.98;
      for i=1:numel(hax)
        posn = hax(i).Position;
        w0 = posn(3);
        h = posn(4);
        x0 = posn(1);
        y = posn(2);
        w = w0/nsets*SPACERFAC;
        gap = w0/nsets*(1-SPACERFAC);
        x = x0;
        for j=1:nsets,
          if j == 1,
            hnew = hax(i);
          else
            hnew = copyobj(hax(i),hax(i).Parent);
          end
          hnew.Position = [x,y,w,h];
          haxnew(i,j) = hnew;
          x = x + w+gap;
        end

      end
      obj.haxs = haxnew;
    end
        
    function [tfSucc,msg] = resultsReceived(obj,sRes,forceupdate)
      % Callback executed when new result received from training monitor BG
      % worker
      %
      % trnComplete: scalar logical, true when all views done
      
      if nargin < 3,
        forceupdate = false;
      end

      tfSucc = false;
      msg = '';
      
      if isempty(obj.hfig) || ~ishandle(obj.hfig),
        msg = 'Monitor closed';
        TrainMonitorViz.debugfprintf('Monitor closed, results received %s\n',datestr(now));
        return;
      end
      
      res = sRes.result;
      if ~res.pollsuccess,
        return;
      end
      nres = numel(res.contents);
      assert(nres==obj.nmodels);

      % for each axes, record if any line got updated and max xlim
      tfAnyLineUpdate = false(1,obj.nset);
      lineUpdateMaxStep = zeros(1,obj.nmodels);

      for i = 1:obj.nmodels,
        if res.jsonPresent(i) && (forceupdate || res.tfUpdate(i)),
          contents = res.contents{i};
          set(obj.hline(i,1),'XData',contents.step,'YData',contents.train_loss);
          set(obj.hline(i,2),'XData',contents.step,'YData',contents.train_dist);
          iset = obj.setidx(i);
          tfAnyLineUpdate(iset) = true;
          lineUpdateMaxStep(i) = max(lineUpdateMaxStep(i),contents.step(end));
        end

        if res.killFileExists(i),
          obj.isKilled(i) = true;
          if res.jsonPresent,
            contents = res.contents{i};
            % hmm really want to mark the last 2k interval when model is
            % actually saved
            set(obj.hlinekill(i,1),'XData',contents.step(end),'YData',contents.train_loss(end));
            set(obj.hlinekill(i,2),'XData',contents.step(end),'YData',contents.train_dist(end));
          end
          handles = guidata(obj.hfig);
          handles.pushbutton_startstop.Enable = 'on';
        end
        
        if res.tfComplete(i)
          contents = res.contents{i};
          if ~isempty(contents)
            % re-use kill marker
            set(obj.hlinekill(i,1),'XData',contents.step(end),'YData',contents.train_loss(end),...
              'color',[0 0.5 0],'marker','o');
            set(obj.hlinekill(i,2),'XData',contents.step(end),'YData',contents.train_dist(end),...
              'color',[0 0.5 0],'marker','o');
          end
        end
      end
      
      if any(res.errFileExists),
        handles = guidata(obj.hfig);
        i = find(strcmp(handles.popupmenu_actions.String,'Show error messages'));
        if ~isempty(i),
          handles.popupmenu_actions.Value = i;
        end
      end

      for i = 1:obj.nmodels,
        obj.lastTrainIter(i) = max(obj.lastTrainIter(i),lineUpdateMaxStep(i));
      end
      for iset = 1:obj.nset,
        if tfAnyLineUpdate(iset),
          obj.adjustAxes(max(obj.lastTrainIter(obj.setidx==iset)),iset);
        end
      end
      
      if isempty(obj.resLast) || any(tfAnyLineUpdate)
        obj.resLast = res;
      end

      [tfSucc,msg] = obj.updateAnn(res);
      TrainMonitorViz.debugfprintf('resultsReceived - tfSucc = %d, msg = %s\n',tfSucc,msg);
    end
    
    function [tfSucc,status] = updateAnn(obj,res)
      % pollsuccess: [nview] logical
      % pollts: [nview] timestamps
      
      tfSucc = true;
      pollsuccess = res.pollsuccess;
      
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
                  
      if ~isempty(res),
        isTrainComplete = res.tfComplete;
        isErr = res.errFileExists | res.logFileErrLikely;
        isLogFile = res.logFileExists;
        isJsonFile = res.jsonPresent;
      else
        isTrainComplete = false(1,obj.nmodels);
        isErr = false(1,obj.nmodels);
        isLogFile = false(1,obj.nmodels);
        isJsonFile = false(1,obj.nmodels);
      end
      
      isRunning0 = obj.trainWorkerObj.getIsRunning();
      if isempty(isRunning0),
        isRunning = true;
      else
        isRunning = any(isRunning0);
      end
      if ~isRunning
        if obj.jobStoppedRepeatsReqd>=1
          obj.jobStoppedRepeatsReqd = obj.jobStoppedRepeatsReqd-1;
          isRunning = true;
        end
      end

      TrainMonitorViz.debugfprintf('updateAnn: isRunning = %d, isTrainComplete = %d/%d, isErr = %d/d, isKilled = %d/%d\n',...
        isRunning,nnz(isTrainComplete),obj.nmodels,nnz(isErr),obj.nmodels,nnz(obj.isKilled),obj.nmodels);
      
      if any(obj.isKilled),
        status = sprintf('Training process killed (%d/%d models).',nnz(obj.isKilled),obj.nmodels);
        tfSucc = false;
      elseif isErr,
        status = sprintf('Error (%d/%d models) while training after %s iterations',nnz(isErr),obj.nmodels,mat2str(obj.lastTrainIter));
        tfSucc = false;
      elseif all(isTrainComplete),
        status = 'Training complete.';
        handles = guidata(obj.hfig);
        TrainMonitorViz.updateStartStopButton(handles,false,true);
      elseif ~isRunning,
        status = 'No training jobs running.';
        tfSucc = false;
      elseif any(isLogFile) && all(~isJsonFile),
        status = 'Training in progress. Preprocessing.';
      elseif any(isLogFile) && any(isJsonFile),
        status = sprintf('Training in progress. %s iterations completed.',mat2str(obj.lastTrainIter));
      else
        status = 'Initializing training.';
      end
      
      str = {sprintf('%s status: %s',clusterstr,status),sprintf('Monitor updated %s.',datestr(now,'HH:MM:SS PM'))};
      hAnn = obj.hannlastupdated;
      hAnn.String = str;
      
      tfsucc = pollsuccess;
      if tfsucc,
        hAnn.ForegroundColor = [0 1 0];
      else
        hAnn.ForegroundColor = [1 0 0];
      end
      
%       ax = obj.haxs(1);
%       hAnn.Position(1) = ax.Position(1)+ax.Position(3)-hAnn.Position(3);
%       hAnn.Position(2) = ax.Position(2)+ax.Position(4)-hAnn.Position(4);
    end
    
    function adjustAxes(obj,lineUpdateMaxStep,iset)
      for i=1:size(obj.haxs,1)
        ax = obj.haxs(i,iset);
        xlim = ax.XLim;
        x0 = max(0,lineUpdateMaxStep-obj.axisXRange(iset));
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
      handles.pushbutton_startstop.Enable = 'inactive';
      drawnow;
      [tfsucc,warnings] = obj.trainWorkerObj.killProcess();
      obj.isKilled(:) = tfsucc;
      obj.SetBusy('Checking that training jobs are killed...',true);
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
        case 'Show sample training images' 
          obj.showTrainingImages();          
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
    
    function showTrainingImages(obj)
      trnImgIfo = obj.trainWorkerObj.loadTrainingImages();
      obj.trainMontageFigs = obj.dtObj.trainImageMontage(trnImgIfo,'hfigs',obj.trainMontageFigs);
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
      for i = 1:numel(raw),
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
        set(handles.pushbutton_startstop,'String','Training complete','BackgroundColor',[.466,.674,.188],...
          'Enable','inactive','UserData','done');
      else
        if isStop,
          set(handles.pushbutton_startstop,'String','Stop training','BackgroundColor',[.64,.08,.18],'Enable','on','UserData','stop');
        else
          set(handles.pushbutton_startstop,'String','Training stopped',...
            'Enable','inactive','UserData','start');
          %set(handles.pushbutton_startstop,'String','Restart training','BackgroundColor',[.3,.75,.93],'Enable','on','UserData','start');
        end
      end
      
    end
    
  end
  
end
