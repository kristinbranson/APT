classdef TrainMonitorViz < handle
  properties
    % 'sets' are groups of related trains that may be spawned in parallel
    % or serially. example is top-down trackers which have nset=2,
    % stage1=detect, stage2=pose.
    
    hfig % scalar fig
    haxs % [2xnset] axis handle, viz training loss, dist
    %hannlastupdated % [1] textbox/annotation handle
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
    
    resLast  % last training json contents received
    dtObj  % DeepTracker Obj
    poller = []
    backendType  % scalar DLBackEnd (a DLBackEnd enum, not a DLBackEndClass)
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

  properties (Transient) 
    parent_
    labeler_
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
    
    function obj = TrainMonitorViz(parent, labeler)

      obj.parent_ = parent ;
      obj.labeler_ = labeler ;

      dmc = labeler.tracker.trnLastDMC ;
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

      obj.dtObj = labeler.tracker ;
      obj.poller = labeler.tracker.bgTrainPoller ;
      obj.backendType = labeler.backend.type ;
      obj.hfig = TrainMonitorGUI(obj);
      % parent.addSatellite(obj.hfig);  % Don't think we need this
      
      handles = guidata(obj.hfig);
      TrainMonitorViz.updateStartStopButton(handles,true,false);
      handles.pushbutton_startstop.Enable = 'on';
            
      obj.haxs = [handles.axes_loss;handles.axes_dist];
      %obj.hannlastupdated = handles.text_clusterstatus;
      tfMultiSet = nsets>1;
      if tfMultiSet
        obj.splitaxs(nsets);
      end
      
      % reset
      arrayfun(@(x)cla(x),obj.haxs);
      clusterstr = apt.monitorBackendDescription(obj.backendType) ;
      str = sprintf('%s status: Initializing...', clusterstr) ;
      apt.setStatusDisplayLineBang(obj.hfig, str, true) ;
      %obj.hannlastupdated.String = 'Cluster status: Initializing...';
      handles.text_clusterinfo.String = '...';
      handles.popupmenu_actions.String = obj.actions.(char(obj.backendType));
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
      deleteValidGraphicsHandles(obj.hfig);
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
      msg = '';  %#ok<NASGU> 
      
      if isempty(obj.hfig) || ~ishandle(obj.hfig),
        msg = 'Monitor closed';
        TrainMonitorViz.debugfprintf('Monitor closed, results received %s\n',datestr(now()));
        return
      end
      
      res = sRes.result;

      % This early exit seems to prevent user from seeing an error that occurs before
      % any training iterations.
%       if ~res.pollsuccess,
%         % Even if the poll failed, if .resLast is empty then populate it, since maybe there was an error or
%         % something.
%         if isempty(obj.resLast)
%           obj.resLast = res;
%         end
%         return
%       end
%       nres = numel(res.contents);
%       assert(nres==obj.nmodels);

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

        % if res.killFileExists(i),
        %   obj.isKilled(i) = true;
        %   if res.jsonPresent,
        %     contents = res.contents{i};
        %     % hmm really want to mark the last 2k interval when model is
        %     % actually saved
        %     set(obj.hlinekill(i,1),'XData',contents.step(end),'YData',contents.train_loss(end));
        %     set(obj.hlinekill(i,2),'XData',contents.step(end),'YData',contents.train_dist(end));
        %   end
        %   handles = guidata(obj.hfig);
        %   handles.pushbutton_startstop.Enable = 'on';
        % end
        
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

      [tfSucc,msg] = obj.updateStatusDisplayLine_(res);
      TrainMonitorViz.debugfprintf('resultsReceived - tfSucc = %d, msg = %s\n',tfSucc,msg);
    end  % function resultsReceived()
    
    function [tfSucc,status] = updateStatusDisplayLine_(obj,res)
      % pollsuccess: [nview] logical
      % pollts: [nview] timestamps
      
      tfSucc = true;
      
      if ~isempty(res),
        pollsuccess = res.pollsuccess;
        isTrainComplete = res.tfComplete;
        isErr = res.errFileExists ;
        isLogFile = res.logFileExists;
        isJsonFile = res.jsonPresent;
      else
        pollsuccess = false ;  % is this right?  -- ALT, 2024-06-27
        isTrainComplete = false(1,obj.nmodels);
        isErr = false(1,obj.nmodels);
        isLogFile = false(1,obj.nmodels);
        isJsonFile = false(1,obj.nmodels);
      end
      
      isRunning0 = obj.dtObj.isAliveFromRegisteredJobIndex('train') ;
      %isRunning0 = obj.trainWorkerObj.getIsRunning();
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
      elseif any(isErr),
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

      clusterstr = apt.monitorBackendDescription(obj.backendType) ;
      str = sprintf('%s status: %s (at %s)',clusterstr,status,strtrim(datestr(now(),'HH:MM:SS PM'))) ;
      isAllGood = pollsuccess && ~any(isErr) ;
      apt.setStatusDisplayLineBang(obj.hfig, str, isAllGood) ;
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
    
    function abortTraining(obj)
      % if isempty(obj.trainWorkerObj),
      %   warning('trainWorkerObj is empty -- cannot kill process');
      %   return
      % end
      apt.setStatusDisplayLineBang(obj.hfig, 'Killing training jobs...', false);
      handles = guidata(obj.hfig);
      handles.pushbutton_startstop.String = 'Stopping training...';
      handles.pushbutton_startstop.Enable = 'inactive';
      drawnow;

      obj.labeler_.abortTraining() ;

      obj.isKilled(:) = true ;
      apt.setStatusDisplayLineBang(obj.hfig, 'Training process killed.', true);

      % [tfsucc,warnings] = obj.trainWorkerObj.killProcess();
      % obj.isKilled(:) = tfsucc;
      % apt.setStatusDisplayLineBang(obj.hfig, 'Checking that training jobs were killed...', false);
      % wereTrainingProcessesKilledForSure = false ;
      % if tfsucc ,        
      %   startTime = tic() ;
      %   maxWaitTime = 30;
      %   while true,
      %     if toc(startTime) > maxWaitTime,
      %       fprintf('Stopping training processes is taking too long, giving up.\n') ;
      %       if isempty(warnings) ,
      %         fprintf('But there were no warnings while trying to stop training processes.\n') ;
      %       else
      %         fprintf('Warning(s) while trying to stop training processes:\n') ;
      %         cellfun(@(warning)(fprintf('%s\n', warning)), warnings) ;
      %         fprintf('\n') ;
      %       end
      %       warndlg('Stopping training processes took too long.  See console for details.', 'Problem stopping training', 'modal') ;
      %       break
      %     end
      %     if ~obj.dtObj.bgTrnIsRunning,
      %       wereTrainingProcessesKilledForSure = true ;
      %       break
      %     end
      %     pause(1);
      %   end        
      % else
      %   %warndlg([{'Training processes may not have been killed properly:'},warnings],'Problem stopping training','modal');
      %   fprintf('There was a problem stopping training processes.\n') ;
      %   fprintf('Training processes may not have been killed properly.\n') ;
      %   if isempty(warnings) ,
      %     fprintf('But there were no warnings while trying to stop training processes.\n') ;
      %   else
      %     fprintf('Warning(s) while trying to stop training processes:\n') ;
      %     cellfun(@(warning)(fprintf('%s\n', warning)), warnings) ;
      %     fprintf('\n') ;
      %   end
      %   warndlg('There was a problem while stopping training processes.  See console for details.', 'Problem stopping training', 'modal') ;
      % end
      % if wereTrainingProcessesKilledForSure ,
      %   str = 'Training process killed.' ;
      % else
      %   str = 'Tried to kill training process, but there were issues.' ;
      % end        
      % apt.setStatusDisplayLineBang(obj.hfig, str, true);


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
          ss = obj.getLogFilesSummary();
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
          ss = obj.detailedStatusStringFromRegisteredJobIndex_();
          handles.text_clusterinfo.String = ss;
          drawnow;
        case 'Show error messages',
          obj.displayErrorMessages() ;
        otherwise
          fprintf('%s not implemented\n',action);
          return;
      end
    end

    function displayErrorMessages(obj)
      handles = guidata(obj.hfig);
      if isempty(obj.resLast) || ~any([obj.resLast.errFileExists]),
        ss = 'No error messages.';
      else
        ss = obj.getErrorFilesSummary();
      end
      handles.text_clusterinfo.String = ss;
      drawnow('limitrate', 'nocallbacks') ;
    end      

    % function ss = getLogFilesContents(obj)
    %   ss = obj.trainWorkerObj.getLogFilesContent();
    % end  % function
    % 
    % function ss = getErrorFileContents(obj)
    %   ss = obj.trainWorkerObj.getErrorfileContent();
    % end  % function
    
    function ss = getLogFilesSummary(obj)      
      ss = obj.dtObj.getTrainingLogFilesSummary() ;      
    end
    
    function ss = getErrorFilesSummary(obj)      
      ss = obj.dtObj.getTrainingErrorFilesSummary() ;      
    end
    
    function updateMonitorPlots(obj)      
      sRes.result = obj.poller.poll() ;
      obj.resultsReceived(sRes,true);      
    end  % function
    
    function showTrainingImages(obj)
      trnImgIfo = obj.dtObj.loadTrainingImages();
      obj.trainMontageFigs = obj.dtObj.trainImageMontage(trnImgIfo,'hfigs',obj.trainMontageFigs);
    end  % function
    
    function result = queryAllJobsStatus(obj)      
      ss = obj.dtObj.queryAllJobsStatus('train') ;
      if isempty(ss) ,
        result = {'(No active jobs.)'} ;
      else
        result = ss ;
      end
    end  % function
    
    function result = detailedStatusStringFromRegisteredJobIndex_(obj)      
      ss = obj.dtObj.detailedStatusStringFromRegisteredJobIndex('train') ;
      if isempty(ss) ,
        result = {'(No active jobs.)'} ;
      else
        result = ss ;
      end
    end  % function
    
  end  % methods
  
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

  end  % methods (Static)
  
end
