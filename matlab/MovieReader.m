classdef MovieReader < handle
% Like VideoReader, but wraps get_readframe_fcn
  
  properties (SetAccess=private)
    filename = '';
    
    readFrameFcn = [];
    nframes = nan;
    info = [];
    nr = nan; % numrows in raw/orig movie
    nc = nan; % numcols in raw/orig movie
    nchan = nan; % numchans "
    fid = nan; % file handle/resource to movie
    
    % bgsub
    bgType % see PxAssign.simplebgsub 
    bgIm % [nr x nc] background im
    bgDevIm % [nr x nc] background dev im
    
    % crop
    cropInfo % Either empty array, or scalar CropInfo. CropInfo is a handle 
      % so this is subject to external mutations. Used only when 'docrop' 
      % flag is true in read()
  end
  
  properties (SetAccess=public)
    
%     neednframes = true; % whether nframes needs to be exact
    preload = false; % only used at open() time, could be passed as option at that time

  end
  
  properties (Dependent)
    nrread % numrows in image-as-read, post-crop (if any)
    ncread % numcols in "
    roiread % [xlo xhi ylo yhi] of image-as-read. If there is no cropping, 
      % this is just [1 nc 1 nr]. 
    hascrop % logical scalar, true if cropInfo is set
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
    function v = get.nrread(obj)
      ci = obj.cropInfo; 
      if isempty(ci)
        v = obj.nr;
      else
        v = ci.roi(4)-ci.roi(3)+1;
      end
    end
    function v = get.ncread(obj)
      ci = obj.cropInfo; 
      if isempty(ci)
        v = obj.nc;
      else
        v = ci.roi(2)-ci.roi(1)+1;
      end
    end
    function v = get.roiread(obj)
      ci = obj.cropInfo;
      if ~isempty(ci)
        v = ci.roi;
      else
        v = [1 obj.nc 1 obj.nr];
      end
    end
    function v = get.hascrop(obj)
      v = ~isempty(obj.cropInfo);
    end
  end
  
  methods
    
    function obj = MovieReader
      % none
    end
        
    function open(obj,fname,varargin)
      
      [bgTy,bgReadFcn] = myparse(varargin,...
        'bgType',[],... % optional, string enum
        'bgReadFcn',[]... % optional, fcn handle to compute [bg,bgdev] = bgReadFcn(movfile,movifo)
         ... % 'preload',false... % if true, frames pre-read upfront. passed thru to get_readframe_fcn
        ); 
      
      assert(exist(fname,'file')>0,'Movie ''%s'' not found.',fname);
      
      if obj.isOpen
        obj.close();
      end
      
      obj.filename = fname;      
      [obj.readFrameFcn,obj.nframes,obj.fid,obj.info] = ...
        get_readframe_fcn(obj.filename,'preload',obj.preload);%,'neednframes',obj.neednframes);
      
      if isfield(obj.info,'readerobj')
        obj.info = rmfield(obj.info,'readerobj');
      end
      
      ifo = obj.info;
      if isfield(ifo,'nr') && isfield(ifo,'nc')
        obj.nr = ifo.nr;
        obj.nc = ifo.nc;
        obj.nchan = nan;
      else
        im = obj.readFrameFcn(1);
        [obj.nr,obj.nc,obj.nchan] = size(im);
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
    
    function setCropInfo(obj,cInfo)
      assert(isempty(cInfo) || isscalar(cInfo) && isa(cInfo,'CropInfo'));
      obj.cropInfo = cInfo;
    end
    
    function [im,imOrigType,imroi] = readframe(obj,i,varargin)
      % im: image
      % imOrigType: type of original/raw image; this may differ from the
      % type of im when doBGsub is on.
      % imroi: [1x4] [xlo xhi ylo yhi] roi of im-as-read. Usually, just 
      %  [1 nc 1 nr]. If docrop, then the roi used to crop.
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
      
      [doBGsub,docrop] = myparse(varargin,...
        'doBGsub',false,...
        'docrop',false ... % if true, .cropInfo is used if avail. Note, 
                       ... % cropping occurs AFTER flipvert, if that is on
        );
      
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

        % Note: we do NOT apply .flipVert to bgIm, bgDevIm here...
        
        im = PxAssign.simplebgsub(obj.bgType,double(im),obj.bgIm,obj.bgDevIm);
        
%         % For now we attempt to rescale im based on imOrigType. This
%         % behavior is consistent with shapeGt, but only works for certain
%         % uint* types.
%         im = PxAssign.imRescalePerType(im,imOrigType);
      end
      
      if docrop && obj.hascrop
        imroi = obj.cropInfo.roi; % .cropInfo must be set
        im = im(imroi(3):imroi(4),imroi(1):imroi(2));        
      else
        imroi = [1 size(im,2) 1 size(im,1)];
      end
    end
    
    function nchan = getreadnchan(obj)
      % nchan: number of channels in raw/orig movie
      
      if ~isnan(obj.nchan)
        nchan = obj.nchan;
      else
        assert(obj.isOpen,'Movie is not open.');
        im = obj.readFrameFcn(1);
        nchan = size(im,3);
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
      
      obj.cropInfo = [];
    end    
  
    function delete(obj)
      obj.close();
    end
    
    function openForLabeler(obj,labeler,mIdx,iView)
      % Take a Labeler object and open a movie for movieset mIdx and view iView, being 
      % faithful to labeler as per:
      %   - .movieForceGrayScale 
      %   - .movieInvert(iView)
      %   - .preProcParams.BackSub
      %   - .cropInfo for (mIdx,iView) as appropriate
      %
      % labelerr: scalar Labeler object (not mutated)
      % mIdx: scalar MovieIndex
      % iView: view index; used for .movieInvert

      ppPrms = labeler.preProcParams;
      if ~isempty(ppPrms)
        bgsubPrms = ppPrms.BackSub;
        bgArgs = {'bgType',bgsubPrms.BGType,'bgReadFcn',bgsubPrms.BGReadFcn};
      else
        bgArgs = {};
      end
      
      movfname = labeler.getMovieFilesAllFullMovIdx(mIdx);
      obj.preload = labeler.movieReadPreLoadMovies; % must occur before .open()
      obj.open(movfname{iView},bgArgs{:});
      obj.forceGrayscale = labeler.movieForceGrayscale;
      obj.flipVert = labeler.movieInvert(iView);      
      cInfo = labeler.getMovieFilesAllCropInfoMovIdx(mIdx);
      if ~isempty(cInfo)
        obj.setCropInfo(cInfo(iView));
      else
        obj.setCropInfo([]);
      end      
    end  % method    
  end  % methods
  
  methods (Static)
    
    function s = getInfo(movfile)
      
      obj = MovieReader();
      obj.open(movfile);
      s = obj.info;
      s.nframes = obj.nframes;
      s.nr = obj.nr;
      s.nc = obj.nc;
      s.nchan = obj.nchan;
      delete(obj);
      
    end
    
    function nframes = getNFrames(movfile)
      
      s = MovieReader.getInfo(movfile);
      nframes = s.nframes;
      
    end
    
    function imsz = getFrameSize(movfile)
      
      s = MovieReader.getInfo(movfile);
      imsz = [s.nr,s.nc,s.nchan];

    end    

  end
  
end

