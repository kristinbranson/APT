classdef FileLogger < handle
  properties
    filename_  % name of log file, or 1 for stdout, or 2 for stderr
    tag_  % string added to start of each log message
    fid_  % a file_object of the log file
  end

  methods
    function obj = FileLogger(filename, tag)
      if isnumeric(filename) ,        
        if filename == 1 ,
          fid = 1 ;
        elseif filename == 2 ,
          fid = 2 ;
        else
          error('Numeric filename must be 1 (for stdout) or 2 (for stderr)') ;
        end
      else
        fid = file_object(filename, 'wt') ;
      end
      obj.filename_ = filename ;
      obj.fid_ = fid ;
      obj.tag_ = tag ;
    end

    function log(obj, varargin)
      str = sprintf(varargin{:}) ;
      fprintf(obj.fid_, '%s (%s): %s\n', obj.tag_, datestr(now(),'yyyymmddTHHMMSS'), str) ;
    end    
  end
end
