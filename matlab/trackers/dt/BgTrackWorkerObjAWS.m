classdef BgTrackWorkerObjAWS < BgWorkerObjAWS & BgTrackWorkerObj 
  methods    
    function obj = BgTrackWorkerObjAWS(varargin)
      obj@BgWorkerObjAWS(varargin{:});
    end
    
    function sRes = compute(obj)
      % sRes: [nviews] struct array
      
      sRes = struct(...
        'tfComplete',[],...
        'errFile',obj.artfctErrFiles,... % char, full path to DL err file
        'errFileExists',[],... % true of errFile exists and has size>0
        'logFile',obj.artfctLogfiles,... % char, full path to Bsub logfile
        'logFileExists',[],...
        'logFileErrLikely',[],... % true if bsub logfile looks like err
        'mIdx',obj.mIdx,...
        'iview',num2cell((1:obj.nviews)),...
        'movfile',obj.movfiles,...
        'trkfile',obj.artfctTrkfiles,...
        'parttrkfile',obj.artfctPartTrkfiles,...
        'parttrkfileTimestamp',[],...
        'parttrkfileNfrmtracked',[],... % 'extra' in BgTrackWorkerObjAWS 
        'trkfileNfrmtracked',[],... % 'extra' in BgTrackWorkerObjAWS 
        'killFile',obj.killFiles,...
        'killFileExists',[]);      
      
      aws = obj.awsEc2;
      
      for ivw=1:obj.nviews
        trkfile = obj.artfctTrkfiles{ivw};
        errFile = obj.artfctErrFiles{ivw};
        logFile = obj.artfctLogfiles{ivw};
        partFile = obj.artfctPartTrkfiles{ivw};
        killFile = obj.killFiles{ivw};
                
        % See AWSEC2 convenience meth
        fspollargs = ...
          sprintf('exists %s existsNE %s existsNE %s existsNEerr %s exists %s lastmodified %s nfrmtracked %s nfrmtracked %s',...
            trkfile,errFile,logFile,logFile,killFile,partFile,partFile,trkfile);
        cmdremote = sprintf('~/APT/matlab/misc/fspoll.py %s',fspollargs);

        fprintf('The time is %s\n',datestr(now,'yyyymmddTHHMMSS'));
        [tfpollsucc,res] = aws.cmdInstance(cmdremote,'dispcmd',true);
        if tfpollsucc
          reslines = regexp(res,'\n','split');
          tfpollsucc = iscell(reslines) && numel(reslines)==8+1; % last cell is {0x0 char}
        end
        
        if tfpollsucc          
          sRes(ivw).tfComplete = strcmp(reslines{1},'y');
          sRes(ivw).errFileExists = strcmp(reslines{2},'y');
          sRes(ivw).logFileExists = strcmp(reslines{3},'y');
          sRes(ivw).logFileErrLikely = strcmp(reslines{4},'y');
          sRes(ivw).killFileExists = strcmp(reslines{5},'y');
          sRes(ivw).parttrkfileTimestamp = str2double(reslines{6}); % includes nan for 'DNE'
          sRes(ivw).parttrkfileNfrmtracked = str2double(reslines{7}); % includes nan for 'DNE'
          sRes(ivw).trkfileNfrmtracked = str2double(reslines{8}); % includes nan for 'DNE'
          
          fprintf('The poll succeeded. Time is %s\n',datestr(now,'yyyymmddTHHMMSS'));
        else
          % trackWorkerObj results don't have a 'pollsuccess' for some
          % reason
          sRes(ivw).tfComplete = false;
          sRes(ivw).errFileExists = false;
          sRes(ivw).logFileExists = false;
          sRes(ivw).logFileErrLikely = false;
          sRes(ivw).killFileExists = false;
          sRes(ivw).parttrkfileTimestamp = nan;
          sRes(ivw).parttrkfileNfrmtracked = nan;
          sRes(ivw).trkfileNfrmtracked = nan;
          
          fprintf('The poll failed. Time is %s\n',datestr(now,'yyyymmddTHHMMSS'));
        end
      end
      
%       sRes = struct(...
%         'tfComplete',cellfun(@obj.fileExists,obj.artfctTrkfiles,'uni',0),...
%         'errFile',obj.artfctErrFiles,... % char, full path to DL err file
%         'errFileExists',tfErrFileErr,... % true of errFile exists and has size>0
%         'logFile',obj.artfctLogfiles,... % char, full path to Bsub logfile
%         'logFileErrLikely',bsuberrlikely,... % true if bsub logfile looks like err
%         'mIdx',obj.mIdx,...
%         'iview',num2cell((1:obj.nviews)'),...
%         'movfile',obj.movfiles,...
%         'trkfile',obj.artfctTrkfiles,...
%         'parttrkfile',obj.artfctPartTrkfiles,...
%         'parttrkfileTimestamp',partTrkFileTimestamps,...
%         'killFile',obj.killFiles,...
%         'killFileExists',killFileExists);
    end
  end
end