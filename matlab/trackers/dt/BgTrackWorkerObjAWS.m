classdef BgTrackWorkerObjAWS < BgWorkerObjAWS & BgTrackWorkerObj 
  methods    
    function obj = BgTrackWorkerObjAWS(nviews,varargin)
      obj@BgWorkerObjAWS(varargin{:});
      obj.nviews = nviews;
    end
    
    function sRes = compute(obj)
      % sRes: [nviews] struct array
      %
      % Production/external tracking not handled yet

      aws = obj.awsEc2;
      
      % Handle err/log files etc; there are nViewJobs of these

      errFileExists = num2cell(false(size(obj.artfctErrFiles)));
      logFileExists = num2cell(false(size(obj.artfctLogfiles)));
      logFileErrLikely = num2cell(false(size(obj.artfctLogfiles)));
      killFileExists = num2cell(false(size(obj.killFiles)));
      for ivwjb=1:obj.nViewJobs
        errFile = obj.artfctErrFiles{ivwjb};
        logFile = obj.artfctLogfiles{ivwjb};
        killFile = obj.killFiles{ivwjb};
        fspollargs = ...
          sprintf('existsNE %s existsNE %s existsNEerr %s exists %s',...
            errFile,logFile,logFile,killFile);
        cmdremote = sprintf('~/APT/matlab/misc/fspoll.py %s',fspollargs);
        
        fprintf('The time is %s\n',datestr(now,'yyyymmddTHHMMSS'));
        [tfpollsucc,res] = aws.cmdInstance(cmdremote,'dispcmd',true);
        if tfpollsucc
          reslines = regexp(res,'\n','split');
          tfpollsucc = iscell(reslines) && numel(reslines)==4+1; % last cell is {0x0 char}
        end        
        if tfpollsucc
          errFileExists{ivwjb} = strcmp(reslines{1},'y');
          logFileExists{ivwjb} = strcmp(reslines{2},'y');
          logFileErrLikely{ivwjb} = strcmp(reslines{3},'y');
          killFileExists{ivwjb} = strcmp(reslines{4},'y');          
          fprintf('The poll succeeded. Time is %s\n',datestr(now,'yyyymmddTHHMMSS'));
        else
          % none; errFileExists, etc initted to false
        end
      end

      nVwPerJob = obj.nViewsPerJob;

      sRes = struct(...
        'tfComplete',[],...
        'errFile',repmat(obj.artfctErrFiles,[1 nVwPerJob]),... % char, full path to DL err file
        'errFileExists',repmat(errFileExists,[1 nVwPerJob]),... % true of errFile exists and has size>0
        'logFile',repmat(obj.artfctLogfiles,[1 nVwPerJob]),... % char, full path to Bsub logfile
        'logFileExists',repmat(logFileExists,[1 nVwPerJob]),...
        'logFileErrLikely',repmat(logFileErrLikely,[1 nVwPerJob]),... % true if bsub logfile looks like err
        'iview',num2cell((1:obj.nviews)),...
        'movfile',obj.movfiles,...
        'trkfile',obj.artfctTrkfiles,...
        'parttrkfile',obj.artfctPartTrkfiles,...
        'parttrkfileTimestamp',[],...
        'parttrkfileNfrmtracked',[],... % 'extra' in BgTrackWorkerObjAWS 
        'trkfileNfrmtracked',[],... % 'extra' in BgTrackWorkerObjAWS 
        'killFile',repmat(obj.killFiles,[1 nVwPerJob]),...
        'killFileExists',repmat(killFileExists,[1 nVwPerJob]),...
        'isexternal',obj.isexternal);
      
      assert(numel(sRes)==obj.nviews);
      
      % Handle trk/part files etc; these are one per view
      for ivw=1:obj.nviews
        trkfile = obj.artfctTrkfiles{ivw};
        partFile = obj.artfctPartTrkfiles{ivw};
                
        % See AWSEC2 convenience meth
        fspollargs = ...
          sprintf('exists %s lastmodified %s nfrmtracked %s nfrmtracked %s',...
            trkfile,partFile,partFile,trkfile);
        cmdremote = sprintf('~/APT/matlab/misc/fspoll.py %s',fspollargs);

        fprintf('The time is %s\n',datestr(now,'yyyymmddTHHMMSS'));
        [tfpollsucc,res] = aws.cmdInstance(cmdremote,'dispcmd',true);
        if tfpollsucc
          reslines = regexp(res,'\n','split');
          tfpollsucc = iscell(reslines) && numel(reslines)==4+1; % last cell is {0x0 char}
        end
        
        if tfpollsucc          
          sRes(ivw).tfComplete = strcmp(reslines{1},'y');
          sRes(ivw).parttrkfileTimestamp = str2double(reslines{2}); % includes nan for 'DNE'
          sRes(ivw).parttrkfileNfrmtracked = str2double(reslines{3}); % includes nan for 'DNE'
          sRes(ivw).trkfileNfrmtracked = str2double(reslines{4}); % includes nan for 'DNE'
          
          fprintf('The poll succeeded. Time is %s\n',datestr(now,'yyyymmddTHHMMSS'));
        else
          % trackWorkerObj results don't have a 'pollsuccess' for some
          % reason
          sRes(ivw).tfComplete = false;
          sRes(ivw).parttrkfileTimestamp = nan;
          sRes(ivw).parttrkfileNfrmtracked = nan;
          sRes(ivw).trkfileNfrmtracked = nan;
          
          fprintf('The poll failed. Time is %s\n',datestr(now,'yyyymmddTHHMMSS'));
        end
      end
      
    end
  end
end