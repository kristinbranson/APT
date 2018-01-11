classdef MovieReader < handle
% Like VideoReader, but wraps get_readframe_fcn
  
  properties (SetAccess=private)
    filename = '';
    
    readFrameFcn = [];
    nframes = nan;
    info = [];
    nr = nan;
    nc = nan;
    
    fid = nan;
  end
  
  properties (SetObservable)
    % AL June 2016. Note on these SetObservable props.
    %
    % Our view of Labeler as a client of MovieReader is slightly unusual.
    % We posit that Labeler.movieReader is part of Labeler's public API, 
    % rather than .movieReader being purely private implementation. These
    % MovieReader props are Public+SetObservable b/c we consider them part
    % of Labeler's public API; users or the UI can/will set them directly, 
    % rather than being forwarded through eg a dummy public property on 
    % Labeler.
    %
    % The two main reasons for this decision are i) convenience and ii) to
    % keep a dummy/forwarding prop on Labeler in sync, a listener would 
    % need to be attached to the MovieReader prop anyway. I kind of like 
    % the idea of having a select few properties of a large object like 
    % Labeler (that contain 'permanent' scalar subobjects) to be considered 
    % public API. It's a convenient and clean way of dividing 
    % code/responsibility a little without creating a bunch of forwarding 
    % props/methods.

    forceGrayscale = false; % if true, [MxNx3] images are run through rgb2gray
    flipVert = false; % if true, images are flipud-ed on read
  end
  
  properties (Dependent)
    isOpen
  end
  
  methods
    function v = get.isOpen(obj)
      v = ~isnan(obj.fid);
    end
  end
  
  methods
    
    function obj = MovieReader
      % none
    end
        
    function open(obj,fname)
      assert(exist(fname,'file')>0,'Movie ''%s'' not found.',fname);
      
      if obj.isOpen
        obj.close();
      end
      
      obj.filename = fname;      
      [obj.readFrameFcn,obj.nframes,obj.fid,obj.info] = get_readframe_fcn(obj.filename);
      
      ifo = obj.info;
      if isfield(ifo,'nr') && isfield(ifo,'nc')
        obj.nr = ifo.nr;
        obj.nc = ifo.nc;
      else
        im = obj.readFrameFcn(1);
        [obj.nr,obj.nc] = size(im);
      end
    end
    
    function varargout = readframe(obj,i)
      assert(obj.isOpen,'Movie is not open.');
      [varargout{1:nargout}] = obj.readFrameFcn(i);

      if obj.flipVert
        varargout{1} = flipud(varargout{1});
      end
      if obj.forceGrayscale
        if size(varargout{1},3)==3 % doesn't have to be RGB but convert anyway
          varargout{1} = rgb2gray(varargout{1});
        end
      end
    end
    
    function close(obj)
      if obj.fid>0
        fclose(obj.fid);
      end
      
      obj.readFrameFcn = [];
      obj.nframes = nan;
      obj.info = [];
      obj.nr = nan;
      obj.nc = nan;
      
      obj.fid = nan;
      obj.filename = '';
    end    
  
    function delete(obj)
      obj.close();
    end
    
  end
  
end

