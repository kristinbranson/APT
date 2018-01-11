classdef ProjectFSInfo
% Project/filesystem interaction metadata

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