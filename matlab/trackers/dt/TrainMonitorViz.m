classdef TrainMonitorViz < handle
  properties
    % 'sets' are groups of related trains that may be spawned in parallel
    % or serially. example is top-down trackers which have nset=2,
    % stage1=detect, stage2=pose.
    
    hfig % scalar fig
    haxs % [2xnset] axis handle, viz training loss, dist
    hannlastupdated % [1] textbox/annotation handle
    hline % [nviewx2xnset] line handle, one loss curve per view
    hlinekill % [nviewx2xnset] line handle, killed marker per view
    
    isKilled = false; % scalar, whether training has been halted
    lastTrainIter; % [nset x nview] last iteration of training
    
    axisXRange = 2e3; % [nset] show last (this many) iterations along x-axis
    
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
    nview
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
    function v = get.nview(obj)
      v = size(obj.hline,1);
    end
    function v = get.nset(obj)
      v = size(obj.hline,3);
    end
  end
  
  methods
    
    function obj = TrainMonitorViz(nview,dtObj,trainWorkerObj,backEnd,...
        varargin)
      
      [trainSplits,nsets] = myparse(varargin,...
        'trainSplits',false, ...
        'nsets',1 ...
        );
      
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
        assert(nsets==2,'Only two sets supported');
        obj.splitaxs();
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
      if tfMultiSet
        xlabel(obj.haxs(2,1),'Iteration -- Detect');
        xlabel(obj.haxs(2,2),'Iteration -- Pose');
      else
        xlabel(obj.haxs(2),'Iteration');
      end
      ylabel(obj.haxs(1),'Loss');
      ylabel(obj.haxs(2),'Dist');
      for j=1:size(obj.haxs,2)
        linkaxes(obj.haxs(:,j),'x');
      end
      set(obj.haxs(1,:),'XTickLabel',{});
      
      %obj.hannlastupdated = TrainMonitorViz.createAnnUpdate(obj.haxs(1));
      
      clrs = lines(nview)*.9+.1;
      h = gobjects(nview,2,nsets);
      hkill = gobjects(nview,2,nsets);
      for ivw=1:nview
        for j=1:2
          for iset=1:nsets
            h(ivw,j,iset) = plot(obj.haxs(j,iset),nan,nan,...
                                 '.-','color',clrs(ivw,:),'LineWidth',2);
            hkill(ivw,j,iset) = plot(obj.haxs(j,iset),nan,nan,...
                                     'rx','markersize',12,'linewidth',2);
          end
        end
      end
      if nview > 1,
        if trainSplits
          legstrs = arrayfun(@(x)sprintf('split%d',x),(1:nview)','uni',0);
        else
          legstrs = arrayfun(@(x)sprintf('view%d',x),(1:nview)','uni',0);
        end
        legend(obj.haxs(2,nsets),h(:,nsets),legstrs,'TextColor','w');
      end
      set(obj.haxs,'XLimMode','manual','YScale','log');
      obj.hline = h;
      obj.hlinekill = hkill;
      obj.resLast = [];
      obj.isKilled = false;
      obj.lastTrainIter = zeros(nsets,nview);
      obj.axisXRange = repmat(obj.axisXRange,[1 nsets]);
    end
    
    function delete(obj)
      deleteValidHandles(obj.hfig);
      obj.hfig = [];
    end
    
    function splitaxs(obj)
      h = obj.haxs;
      szassert(h,[2 1]);
      SPACERFAC = 0.98;
      for i=1:numel(h)
        posn = h(i).Position;
        x0 = posn(1);
        %y0 = posn(2);
        w = posn(3);
        %h = posn(4);
        h(i).Position(3) = w/2;
        h(i).Position(3:4) = h(1).Position(3:4)*SPACERFAC;        
        hnew = copyobj(h(i),h(i).Parent);
        hnew.Position(1) = x0+w/2;
        h(i,2) = hnew;
      end
      obj.haxs = h;
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
      % for each set, record if any line got updated and max xlim
      tfAnyLineUpdate = false(1,obj.nset);
      lineUpdateMaxStep = zeros(1,obj.nset);
      
      h = obj.hline; % [nview x 2 x nset]
      hkill = obj.hlinekill;
      nres = numel(res);
      assert(nres==numel(h)/2);
      
      %tfIResAreSets = false; % if true, res indexes sets; otherwise, views      
      if nres==1
        ires2set = 1;
      elseif size(h,1)==nres
        % multiview case
        ires2set = ones(nres,1); % every res is a different view in one set
      elseif size(h,3)==nres
        % multiset case
        assert(size(h,1)==1);
        h = reshape(h,2,[])';
        hkill = reshape(hkill,2,[])';
        ires2set = (1:nres)'; % every res is view 1 in a different set
        %tfIResAreSets = true;
      else
        assert(false);
      end
      % h/hkill are now [nres 2] where the first dim is either a view or 
      % set idx for view indices, h(:,1) are plotted on same axes as are 
      % h(:,2) for set indices, h(:,1) are plotted on diff axes
      
      if nargin < 3,
        forceupdate = false;
      end
            
      for ires=1:numel(res)
        if res(ires).pollsuccess
          if res(ires).jsonPresent && (forceupdate || res(ires).tfUpdate)
            contents = res(ires).contents;
            set(h(ires,1),'XData',contents.step,'YData',contents.train_loss);
            set(h(ires,2),'XData',contents.step,'YData',contents.train_dist);
            
            iset = ires2set(ires);
            tfAnyLineUpdate(iset) = true;
            lineUpdateMaxStep(iset) = max(lineUpdateMaxStep(iset),contents.step(end));
          end

          if res(ires).killFileExists, 
            obj.isKilled = true;
            if res(ires).jsonPresent,
              contents = res(ires).contents;
              % hmm really want to mark the last 2k interval when model is
              % actually saved
              set(hkill(ires,1),'XData',contents.step(end),'YData',contents.train_loss(end));
              set(hkill(ires,2),'XData',contents.step(end),'YData',contents.train_dist(end));
            end
            handles = guidata(obj.hfig);
            handles.pushbutton_startstop.Enable = 'on';
          end
        
          if res(ires).tfComplete
            contents = res(ires).contents;
            if ~isempty(contents)
              % re-use kill marker 
              set(hkill(ires,1),'XData',contents.step(end),'YData',contents.train_loss(end),...
                'color',[0 0.5 0],'marker','o');
              set(hkill(ires,2),'XData',contents.step(end),'YData',contents.train_dist(end),...
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
      
      for iset=1:obj.nset
        if tfAnyLineUpdate(iset)
          obj.lastTrainIter(iset,:) = ...
                max(obj.lastTrainIter(iset,:),lineUpdateMaxStep(iset));
          obj.adjustAxes(max(obj.lastTrainIter(iset,:)),iset);
          %obj.dtObj.setTrackerInfo('iterCurr',obj.lastTrainIter);
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
      pppi = obj.dtObj.lObj.labelPointsPlotInfo;
      mrkrProps = struct2paramscell(pppi.MarkerProps);
      montageArgs = {'nr',3,'nc',3,'maskalpha',0.3,...
        'framelblscolor',[1 1 0],...
        'colors',pppi.Colors ...
        'pplotargs',mrkrProps ...  
        };
      for i=1:numel(trnImgIfo)
        tii = trnImgIfo{i};
        if isempty(tii)
          continue;
        end
        dam = DataAugMontage();
        dam.init(tii);  
        dam.show(montageArgs);
      end
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
