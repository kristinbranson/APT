classdef CameraSet < handle
  properties
    NumCameras
    CameraNames
    Cameras
  end

  methods 
    function obj = CameraSet(s)
      if ischar(s)
        s = ReadYaml(s);
      end
      obj.NumCameras = s.NumCameras;
      obj.CameraNames = s.CameraNames;

      for icam=1:s.NumCameras
        cam = s.CameraNames{icam};
        camObjs(icam) = CalibratedCamera(s.(cam)); %#ok<AGROW> 
      end
      fprintf(1,'Read information for %d cameras: %s.\n',s.NumCameras, ...
        String.cellstr2CommaSepList(s.CameraNames));

      obj.Cameras = camObjs;
    end

    function [R,t] = getXformCams(obj,cam1,cam2)
      % Get/compute matrices R,T that convert from cam1 coordsys to
      % cam2coordsys; defined by 
      %
      %   [x2;y2;z2] = R*[x1;y1;z1] + t

      c1 = obj.Cameras(cam1);
      c2 = obj.Cameras(cam2);

      R1 = rodrigues(c1.Rotation);
      t1 = c1.Translation(:);
      R2 = rodrigues(c2.Rotation);
      t2 = c2.Translation(:);

      R = R2/R1;
      t = -R*t1 + t2;
    end

    function sp = getMatlabStereoParameters(obj)
      if obj.NumCameras~=2
        error('stereoParameters conversion only avaiable when NumCameras==2.');
      end

      cams = obj.Cameras;
      sp = struct();
      
      sp.CameraParameters1 = cams(1).getMatlabCameraIntrinsics;
      %sp.CameraParameters1.Intrinsics = sp.CameraParameters1;
      sp.CameraParameters2 = cams(2).getMatlabCameraIntrinsics;
      %sp.CameraParameters2.Intrinsics = sp.CameraParameters2;

      [R,t] = obj.getXformCams(1,2);
      sp.RotationOfCamera2 = R';
      sp.TranslationOfCamera2 = t(:)';
    end
  end
end