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
  
  properties
    forceGrayscale = false;  % if true, [MxNx3] images are run through rgb2gray
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
        
    function open(obj,fname)

      assert(exist(fname,'file')>0,'Movie ''%s'' not found.',fname);
      
      if obj.isOpen
        obj.close();
      end
      
      obj.filename = fname;      
      [obj.readFrameFcn,obj.nframes,obj.fid,obj.info] = ...
        get_readframe_fcn(obj.filename,'preload',obj.preload);%,'neednframes',obj.neednframes);
      
      im = obj.readFrameFcn(1);
      [obj.nr,obj.nc,obj.nchan] = size(im);

      if isfield(obj.info,'readerobj')
        obj.info = rmfield(obj.info,'readerobj');
      end
      obj.info.nr = obj.nr;
      obj.info.nc = obj.nc;
    end
    
    function setCropInfo(obj,cInfo)
      assert(isempty(cInfo) || isscalar(cInfo) && isa(cInfo,'CropInfo'));
      obj.cropInfo = cInfo;
    end
    
    function [im,imOrigType,imroi] = readframe(obj,i,varargin)
      % im: image
      % imOrigType: type of original/raw image.
      % imroi: [1x4] [xlo xhi ylo yhi] roi of im-as-read. Usually, just
      %  [1 nc 1 nr]. If docrop, then the roi used to crop.
      %
      % 'doBGsub' is accepted for backward compatibility but must be false;
      % background subtraction is no longer supported.

      [doBGsub,docrop] = myparse(varargin,...
        'doBGsub',false,...
        'docrop',false ... % if true, .cropInfo is used if avail
        );
      assert(~doBGsub,'MovieReader:noBGsub',...
             'Background subtraction is no longer supported.') ;

      assert(obj.isOpen,'Movie is not open.');
      im = obj.readFrameFcn(i);
      imOrigType = class(im);

      if obj.forceGrayscale
        if size(im,3)==3 % doesn't have to be RGB but convert anyway
          im = rgb2gray(im);
        end
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
      %   - .cropInfo for (mIdx,iView) as appropriate
      %
      % labelerr: scalar Labeler object (not mutated)
      % mIdx: scalar MovieIndex
      % iView: view index; used for .movieInvert

      movfname = labeler.getMovieFilesAllFullMovIdx(mIdx);
      obj.preload = labeler.movieReadPreLoadMovies; % must occur before .open()
      obj.open(movfname{iView});
      obj.forceGrayscale = labeler.movieForceGrayscale;
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

