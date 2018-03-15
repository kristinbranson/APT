classdef MovieReader < handle
% Like VideoReader, but wraps get_readframe_fcn
  
  properties (SetAccess=private)
    filename = '';
    
    readFrameFcn = [];
    nframes = nan;
    info = [];
    nr = nan;
    nc = nan;
    
    fid = nan; % file handle/resource to movie
    
    % bgsub
    bgType % see PxAssign.simplebgsub 
    bgIm % [nr x nc] background im
    bgDevIm % [nr x nc] background dev im
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
        
    function open(obj,fname,varargin)
      
      [bgTy,bgReadFcn] = myparse(varargin,...
        'bgType',[],... % optional, string enum
        'bgReadFcn',[]); % optional, fcn handle to compute [bg,bgdev] = bgReadFcn(movfile,movifo)
      
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
      
      tfHasBG = ~isempty(bgTy) && ~isempty(bgReadFcn);
      if tfHasBG
        obj.bgType = bgTy;
        [obj.bgIm,obj.bgDevIm] = feval(bgReadFcn,fname,ifo);
      else
        obj.bgType = [];
        obj.bgIm = [];
        obj.bgDevIm = [];        
      end
    end
    
    function [im,imOrigType] = readframe(obj,i,varargin)
      % im: image
      % imOrigType: type of original/raw image; this may differ from the
      % type of im when doBGsub is on.
      %
      % Currently, when doBGsub is on, im is forced to a double and an
      % attempt is made to rescale im to [0,1] based on imOrigType. If
      % imOrigType is "unusual" then no rescaling is performed. So, the
      % output type depends on:
      % 0. Movie format
      % 1. If 'doBGsub' is on or off
      % 2. If on, then if imOrigType is an expected uint* type
      %
      % Note: im==varargout{1} may be of different type if doBGsub is on 
      % (double) vs off.
      
      doBGsub = myparse(varargin,...
        'doBGsub',false);
      
      assert(obj.isOpen,'Movie is not open.');
      im = obj.readFrameFcn(i);
      imOrigType = class(im);

      if obj.flipVert
        im = flipud(im);
      end
      if obj.forceGrayscale
        if size(im,3)==3 % doesn't have to be RGB but convert anyway
          im = rgb2gray(im);
        end
      end
      
      if doBGsub
        assert(size(im,3)==1,'Background subtraction supported only on grayscale images.');
        assert(~isempty(obj.bgType),...
          'Cannot perform background subtraction. Background type and/or read function unspecified.');
        
        % Note, bgReadFcn should be returning bg images with same
        % scaling as im.

        im = PxAssign.simplebgsub(obj.bgType,double(im),obj.bgIm,obj.bgDevIm);
        
        % For now we attempt to rescale im based on imOrigType. This
        % behavior is consistent with shapeGt, but only works for certain
        % uint* types.
        im = PxAssign.imRescalePerType(im,imOrigType);
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
      
      obj.bgType = [];
      obj.bgIm = [];
      obj.bgDevIm = [];
    end    
  
    function delete(obj)
      obj.close();
    end
    
  end
  
end

