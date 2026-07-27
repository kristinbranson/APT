classdef TrainMonitorController < handle
  properties
    % 'sets' are groups of related trains that may be spawned in parallel
    % or serially. example is top-down trackers which have nset=2,
    % stage1=detect, stage2=pose.
    
    hfig % scalar fig
    haxs % [2xnset] axis handle, viz training loss, disteit
    %hannlastupdated % [1] textbox/annotation handle
    hline % [nmodel x 2] line handle, one loss curve per view
    hlinekill % [nmodel x 2] line handle, killed marker per view
    setidx % [1 x nmodel], which set each line belongs to
    
    wasAborted = []  % 1 x nmodel, whether training has been aborted
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
    trainMontageFigures_ = []  % figure handles for showing training image montages
    % Widget handles, formerly reached via guidata(obj.hfig).
    axes_loss_
    axes_dist_
    text_clusterstatus_
    text_clusterinfo_
    popupmenu_actions_
    pushbutton_action_
    pushbutton_startstop_
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
      if TrainMonitorController.DEBUG,
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
    
    function obj = TrainMonitorController(parent, labeler)

      obj.parent_ = parent ;  % parent a LabelerController
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
      obj.createGui_() ;
      % parent.addSatellite(obj.hfig);  % Don't think we need this
      obj.hfig.CloseRequestFcn = @(s,e)(parent.trainMonitorVizCloseRequested()) ;
        % The figure is built with a plain CloseRequestFcn; override it here with
        % this one, which lets the LabelerController handle things in a
        % coordinated way.

      TrainMonitorController.updateStartStopButton(obj.pushbutton_startstop_, false, []) ;
      obj.pushbutton_startstop_.Enable = 'on';

      obj.haxs = [obj.axes_loss_ ; obj.axes_dist_];
      %obj.hannlastupdated = handles.text_clusterstatus;
      tfMultiSet = nsets>1;
      if tfMultiSet
        obj.splitaxs(nsets);
      end
      
      % reset
      arrayfun(@(x)cla(x),obj.haxs);
      clusterstr = apt.monitorBackendDescription(obj.backendType) ;
      str = sprintf('%s status: Initializing...', clusterstr) ;
      obj.setStatusDisplayLine_(str, true) ;
      %obj.hannlastupdated.String = 'Cluster status: Initializing...';
      obj.text_clusterinfo_.String = '...';
      obj.popupmenu_actions_.String = obj.actions.(char(obj.backendType));
      obj.popupmenu_actions_.Value = 1;
      
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
      
      %obj.hannlastupdated = TrainMonitorController.createAnnUpdate(obj.haxs(1));
      
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
      obj.wasAborted = false(1,nmodels);
      obj.lastTrainIter = zeros(1,nmodels);
      obj.axisXRange = repmat(obj.axisXRange,[1 nsets]);

      obj.jobStoppedRepeatsReqd = 2;
    end

    function createGui_(obj)
      % Build the training-monitor figure and its widgets programmatically,
      % storing the widget handles as instance properties.  This replaces the
      % legacy GUIDE .fig/.m pair; the layout was lifted from GUIDE's export.
      obj.hfig = figure(...
        'Units', 'pixels', ...
        'Position', [951.6 603.153846153846 792 748], ...
        'Color', [0 0 0], ...
        'MenuBar', 'none', ...
        'ToolBar', 'none', ...
        'DockControls', 'off', ...
        'IntegerHandle', 'off', ...
        'Name', 'Training Monitor', ...
        'NumberTitle', 'off', ...
        'Tag', 'figure_TrainMonitor', ...
        'Visible', 'on') ;

      obj.axes_loss_ = axes(...
        'Parent', obj.hfig, ...
        'Units', 'normalized', ...
        'Position', [0.0653535353535353 0.732723159193747 0.924545454545454 0.25], ...
        'Color', [0.15 0.15 0.15], ...
        'XColor', [1 1 1], ...
        'YColor', [1 1 1], ...
        'Tag', 'axes_loss') ;

      obj.axes_dist_ = axes(...
        'Parent', obj.hfig, ...
        'Units', 'normalized', ...
        'Position', [0.0653535353535353 0.472027972027972 0.924545454545454 0.25], ...
        'Color', [0.15 0.15 0.15], ...
        'XColor', [1 1 1], ...
        'YColor', [1 1 1], ...
        'Tag', 'axes_dist') ;

      obj.text_clusterinfo_ = uicontrol(...
        'Parent', obj.hfig, ...
        'Style', 'edit', ...
        'Units', 'normalized', ...
        'Min', 0, ...
        'Max', 2, ...
        'String', '...', ...
        'HorizontalAlignment', 'left', ...
        'Position', [0.0353535353535354 0.070855614973262 0.954545454545455 0.27807486631016], ...
        'BackgroundColor', [0.15 0.15 0.15], ...
        'ForegroundColor', [1 1 1], ...
        'Tag', 'text_clusterinfo') ;

      obj.text_clusterstatus_ = uicontrol(...
        'Parent', obj.hfig, ...
        'Style', 'text', ...
        'Units', 'normalized', ...
        'String', 'Cluster status: Initializing ...', ...
        'HorizontalAlignment', 'left', ...
        'Position', [0.0366161616161616 0.347593582887701 0.953282828282828 0.0494652406417112], ...
        'BackgroundColor', [0 0 0], ...
        'ForegroundColor', [0 1 0], ...
        'FontUnits', 'normalized', ...
        'FontSize', 0.432432432432432, ...
        'Tag', 'text_clusterstatus') ;

      obj.popupmenu_actions_ = uicontrol(...
        'Parent', obj.hfig, ...
        'Style', 'popupmenu', ...
        'Units', 'normalized', ...
        'String', {'List all jobs on cluster' ; 'Show training jobs'' status' ; 'Update training monitor plots' ; 'Show log files'}, ...
        'Value', 1, ...
        'Position', [0.0391414141414141 0.017379679144385 0.570707070707071 0.0454545454545455], ...
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
        'Position', [0.613636363636364 0.0267379679144385 0.0921717171717172 0.036096256684492], ...
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
        'String', 'Stop training', ...
        'Position', [0.733585858585859 0.0267379679144385 0.252525252525252 0.036096256684492], ...
        'BackgroundColor', [0.64 0.08 0.18], ...
        'ForegroundColor', [1 1 1], ...
        'FontUnits', 'normalized', ...
        'FontSize', 0.592592592592593, ...
        'FontWeight', 'bold', ...
        'Tag', 'pushbutton_startstop', ...
        'Callback', @(s,e)(obj.abortTraining())) ;
    end  % function

    function delete(obj)
      deleteValidGraphicsHandles(obj.trainMontageFigures_) ;
      obj.trainMontageFigures_ = [] ;
      deleteValidGraphicsHandles(obj.hfig) ;
      obj.hfig = [] ;
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
    
    function update(obj)
      % Traditional controller update method.
      obj.resultsReceived() ;
    end
    
    function resultsReceived(obj,pollingResult,forceupdate)
      % Callback executed when new result received from training monitor BG
      % worker
      %
      % trnComplete: scalar logical, true when all views done
      
      if nargin < 3,
        forceupdate = false;
      end

      if ~exist('pollingResult', 'var') || isempty(pollingResult) ,
        pollingResult = obj.labeler_.tracker.bgTrnMonitor.pollingResult ;
      end      
      if isempty(obj.hfig) || ~ishandle(obj.hfig),
        TrainMonitorController.debugfprintf('Monitor closed, results received %s\n',datestr(now()));
        return
      end

      % If there is no pollingResult, just update the stop button.
      % May add more here in the future.
      if isempty(pollingResult) ,
        obj.updateStopButton() ;
        return
      end
      
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
        if pollingResult.jsonPresent(i) && (forceupdate || pollingResult.tfUpdate(i)),
          contents = pollingResult.contents{i};
          if isempty(contents)
            continue
          end
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
        
        if pollingResult.tfComplete(i)
          contents = pollingResult.contents{i};
          if ~isempty(contents)
            % re-use kill marker
            set(obj.hlinekill(i,1),'XData',contents.step(end),'YData',contents.train_loss(end),...
              'color',[0 0.5 0],'marker','o');
            set(obj.hlinekill(i,2),'XData',contents.step(end),'YData',contents.train_dist(end),...
              'color',[0 0.5 0],'marker','o');
          end
        end
      end
      
      if any(pollingResult.errFileExists),
        i = find(strcmp(obj.popupmenu_actions_.String,'Show error messages'));
        if ~isempty(i),
          obj.popupmenu_actions_.Value = i;
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
        obj.resLast = pollingResult;
      end

      obj.syncStatusLineToPollingResult() ;
      obj.updateStopButton() ;
    end  % function resultsReceived()
    
    function syncStatusLineToPollingResult(obj)
      % Render the status line (text_clusterstatus) from the monitor's
      % accumulated poll state.  This is NOT a state-independent update method
      % -- hence the syncStatusLineToPollingResult name rather than an update*
      % one: its source of truth is mostly the monitor's own state
      % (obj.resLast, obj.wasAborted, obj.lastTrainIter),
      % which is written as poll results arrive in resultsReceived().  Only a
      % thin slice of what it reads is genuine model state
      % (labeler.bgTrnIsRunning, labeler.lastTrainEndCause).  It is really the
      % tail end of the resultsReceived() pipeline, not a model->view
      % synchronizer, so calling it in isolation with a stale or empty resLast
      % need not reflect the Labeler alone.
      labeler = obj.labeler_ ;
      pollingResult = obj.resLast ;  % most recent poll result received, or [] if none yet

      if any(obj.wasAborted) ,
        % The user stopped training during this bout.
        status = sprintf('Training process killed (%d/%d models).',nnz(obj.wasAborted),obj.nmodels) ;
        isAllGood = false ;
      elseif ~labeler.bgTrnIsRunning ,
        % No training bout is running: reflect the authoritative outcome of the
        % last bout, as recorded by the tracker.
        switch labeler.lastTrainEndCause
          case EndCause.complete ,
            status = 'Training complete.' ;
            isAllGood = true ;
          case EndCause.error ,
            status = 'Error while training.  See error messages for details.' ;
            isAllGood = false ;
          case EndCause.abort ,
            status = 'Training process killed.' ;
            isAllGood = false ;
          case EndCause.undefined ,
            status = 'No training jobs running.' ;
            isAllGood = true ;
          otherwise ,
            error('APT:internalError', 'Unrecognized EndCause in TrainMonitorController.syncStatusLineToPollingResult()') ;
        end
      elseif isempty(pollingResult) ,
        status = 'Initializing training.' ;
        isAllGood = true ;
      else
        % A training bout is in progress: derive the message from the most
        % recent poll result.
        isErr = pollingResult.errFileExists ;
        isLogFile = pollingResult.logFileExists ;
        isJsonFile = pollingResult.jsonPresent ;
        if any(isErr) ,
          status = sprintf('Error (%d/%d models) while training after %s iterations',nnz(isErr),obj.nmodels,mat2str(obj.lastTrainIter)) ;
          isAllGood = false ;
        elseif any(isLogFile) && all(~isJsonFile) ,
          status = 'Training in progress. Preprocessing.' ;
          isAllGood = pollingResult.pollsuccess ;
        elseif any(isLogFile) && any(isJsonFile) ,
          status = sprintf('Training in progress. %s iterations completed.',mat2str(obj.lastTrainIter)) ;
          isAllGood = pollingResult.pollsuccess ;
        else
          status = 'Initializing training.' ;
          isAllGood = pollingResult.pollsuccess ;
        end
      end

      clusterstr = apt.monitorBackendDescription(obj.backendType) ;
      str = sprintf('%s status: %s (at %s)',clusterstr,status,strtrim(datestr(now(),'HH:MM:SS PM'))) ;
      obj.setStatusDisplayLine_(str, isAllGood) ;
    end  % function
    
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
      % Called in response to the user pressing the stop button
      obj.setStatusDisplayLine_('Killing training jobs...', false);
      obj.pushbutton_startstop_.String = 'Stopping training...';
      obj.pushbutton_startstop_.Enable = 'inactive';
      drawnow();

      obj.labeler_.abortTraining() ;

      obj.wasAborted(:) = true ;
      obj.setStatusDisplayLine_('Training process killed.', true);

      TrainMonitorController.updateStartStopButton(obj.pushbutton_startstop_,false,false);
    end
    
    % function startTraining(obj)
    %   % Placeholder meth AL 20190108
    %   % - Always do a regular restart for now; if project is updated might
    %   % want RestartAug.
    %   % - If the training has reached final iter, training will immediately 
    %   % end
    % 
    %   % Kills and creates new TrainMonitorController, maybe that's fine
    % 
    %   obj.dtObj.retrain('dlTrnType',DLTrainType.Restart);
    % end
    
    function updateClusterInfo(obj)
      actions = obj.popupmenu_actions_.String; %#ok<PROP>
      v = obj.popupmenu_actions_.Value;
      action = actions{v}; %#ok<PROP>
      switch action
        case 'Show sample training images'
          obj.showTrainingImages();
        case 'Show log files',
          ss = obj.getLogFilesSummary();
          obj.text_clusterinfo_.String = ss;
          drawnow;
        case 'Update training monitor plots',
          obj.updateMonitorPlots();
          drawnow;
        case {'List all jobs on cluster','List all docker jobs','List all conda jobs'}
          ss = obj.queryAllJobsStatus();
          obj.text_clusterinfo_.String = ss;
          drawnow;
        case 'Show training jobs'' status',
          ss = obj.detailedStatusStringFromRegisteredJobIndex_();
          obj.text_clusterinfo_.String = ss;
          drawnow;
        case 'Show error messages',
          obj.displayErrorMessages() ;
        otherwise
          fprintf('%s not implemented\n',action);
          return;
      end
    end

    function displayErrorMessages(obj)
      if isempty(obj.resLast) || ~any([obj.resLast.errFileExists]),
        ss = 'No error messages.';
      else
        ss = obj.getErrorFilesSummary();
      end
      obj.text_clusterinfo_.String = ss;
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
      pollingResult = obj.poller.poll() ;
      obj.resultsReceived(pollingResult,true);      
    end  % function
    
    function showTrainingImages(obj)
      trnImgIfo = obj.dtObj.loadTrainingImages() ;
      obj.trainMontageFigures_ = obj.trainImageMontage(trnImgIfo, 'hfigs', obj.trainMontageFigures_) ;
    end  % function

    function hfigs = trainImageMontage(obj, trnImgMats, varargin)
      % Show montage of training images with data augmentation.

      pppi = obj.labeler_.labelPointsPlotInfo ;
      mrkrProps = struct2paramscell(pppi.MarkerProps) ;
      margs0 = {'nr', 3, 'nc', 3, 'maskalpha', 0.3, ...
        'framelblscolor', [1 1 0], ...
        'pplotargs', mrkrProps} ;

      hfigs = myparse(varargin, 'hfigs', []) ;

      if isempty(hfigs),
        hfigs = nan(1, numel(trnImgMats)) ;
      end

      for i = 1:numel(trnImgMats)
        ti = trnImgMats{i} ;
        if isempty(ti)
          continue ;
        end

        dam = DataAugMontage() ;
        dam.init(ti) ;
        npts = size(dam.locs, 2) ;
        colors = pppi.Colors(1:npts, :) ;
        margs = [margs0 {'colors' colors}] ;
        if numel(hfigs) >= i && hfigs(i) > 0 && ishandle(hfigs(i)) && ~any(hfigs(1:i-1)==hfigs(i)),
          hfig = hfigs(i) ;
        else
          hfig = [] ;
        end
        hfigs(i) = dam.show(margs, 'hfig', hfig) ;
      end
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
    
    function updateStartStopButton(pushbutton_startstop, isRunning, isComplete)
      if isRunning || isempty(isComplete),
        set(pushbutton_startstop,'String','Stop training','BackgroundColor',[.64,.08,.18],'Enable','on','UserData','stop');
      else
        if isComplete,
          set(pushbutton_startstop,'String','Training complete','BackgroundColor',[.466,.674,.188],...
              'Enable','off','UserData','done');
        else
          set(pushbutton_startstop,'String','Training incomplete',...
              'Enable','off','UserData','done');
        end
      end
    end  % function

  end  % methods (Static)
  
  methods
    function updateStopButton(obj)
      % A conventional update method for the (start/)stop button.
      labeler = obj.labeler_ ;
      isRunning = labeler.bgTrnIsRunning ;
      if isRunning
        isComplete = [] ;
      else
        isComplete = (labeler.lastTrainEndCause == EndCause.complete) ;
      end
      TrainMonitorController.updateStartStopButton(obj.pushbutton_startstop_, isRunning, isComplete) ;
    end  % function

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

  end  % methods    
  
end  % classdef
