classdef CalibratedCamera < handle
  properties
    ImageSize % [nr nc]
    FocalLengths % [fx fy]
    PrincipalPoint % [px py]
    Skew % scalar
    RadialDistortions % [k2 k4]
    Rotation % [rx ry rz] rodrigues vec
    Translation % [tx ty tz]
  end
  methods
    function obj = CalibratedCamera(s)
      for fn = fieldnames(s)', f=fn{1};
        try
          obj.(f) = s.(f);
        catch
          warning('Ignoring unrecognized field: %s', f);
        end
      end
    end

    function camInts = getMatlabCameraIntrinsics(obj)
      camInts = cameraIntrinsics(...
        obj.FocalLengths,obj.PrincipalPoint,obj.ImageSize,...
        'Skew',obj.Skew,...
        'RadialDistortion',obj.RadialDistortions...
        );
    end
  end
end