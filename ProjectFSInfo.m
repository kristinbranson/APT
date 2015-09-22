classdef ProjectFSInfo
  % Project filesystem interaction info
  properties
    timestamp
    action
    filename
  end
  methods
    function obj = ProjectFSInfo(act,fname)
      obj.timestamp = now();
      obj.action = act;
      obj.filename = fname;      
    end
  end
end