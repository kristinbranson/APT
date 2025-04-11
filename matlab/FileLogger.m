classdef FileLogger < handle
  properties
    filename_  % name of log file, or 1 for stdout, or 2 for stderr
    tag_  % string added to start of each log message
    is_fid_valid_  % if true, fid_ is valid.  If not, fid_ is empty, and obj is a "null logger".
    fid_  % a file_object of the log file
  end

  methods
    function obj = FileLogger(filename, tag)
      if nargin==0 ,
        % Returned FileLogger will be a "null logger" such that obj.log() does
        % nothing, not even write to /dev/null.
        filename = '' ;
        tag = '' ;
        is_fid_valid = false ;
        fid = [] ;
      else
        if isnumeric(filename) ,
          if filename == 1 ,
            fid = 1 ;
            is_fid_valid = true ;
          elseif filename == 2 ,
            fid = 2 ;
            is_fid_valid = true ;
          else
            error('Numeric filename must be 1 (for stdout) or 2 (for stderr)') ;
          end
        else
          if isempty(filename) || strcmp(filename, '/dev/null') ,
            % Returned FileLogger will be a "null logger" such that obj.log() does
            % nothing, not even write to /dev/null.
            is_fid_valid = false ;
            fid = [] ;
          else
            is_fid_valid = true ;
            fid = file_object(filename, 'wt') ;
          end
        end
      end
      obj.filename_ = filename ;
      obj.is_fid_valid_ = is_fid_valid ;
      obj.fid_ = fid ;
      obj.tag_ = tag ;
    end

    function log(obj, varargin)
      if obj.is_fid_valid_ ,
        str = sprintf(varargin{:}) ;
        fprintf(obj.fid_, '%s (%s): %s\n', obj.tag_, datestr(now(),'yyyymmddTHHMMSS'), str) ;
      end
    end    
  end
end
