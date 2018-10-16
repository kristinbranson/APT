classdef HeatmapReader < handle
  
  properties
    dir % fullpath heatmap dir
    
    % sprintf-style pattern; tgt,frm,part, all 1-based
    filepat = 'hmap_trx_%d_t_%d_part_%d.jpg'; 
    
    imnr
    imnc
    nfrm
    npt
    ntgt
    
    hmcls = 'uint8';
    hmdat % [imnr x imnc x nfrm x npt x ntgt]; preload only; very mem expensive needless to say 
  end
  
  methods
    
    function obj = HeatmapReader
    end
    
    function init(obj,hmdir,imnr,imnc,nfrm,npt,ntgt)
      obj.dir = hmdir;
      obj.imnr = imnr;
      obj.imnc = imnc;
      obj.nfrm = nfrm;
      obj.npt = npt;
      obj.ntgt = ntgt;
      
      obj.hmdat = [];
      
      dd = dir(fullfile(hmdir,'*.jpg')); %#ok<CPROPLC>
      nhmaps = nfrm*npt*ntgt;
      if numel(dd)<nhmaps
        warningNoTrace('Heatmap dir contains %d jpgs; expected %d.',...
          numel(dd),nhmaps);
      end
    end
    
    function preloadfull(obj,varargin)
      % Assumes your WS has mem
      % Good luck
      
      wbObj = myparse(varargin,...
        'wbObj',[]... % waitbarwithcancel
        );
      tfWB = ~isempty(wbObj);
      
      hmdir = obj.dir;
      pat = obj.filepat;
      
      if tfWB
        %nhm = obj.nfrm*obj.npt*obj.ntgt;
        wbObj.startPeriod('Preloading heatmaps',...
          'shownumden',true,'denominator',obj.nfrm);
      end

      dat = zeros(obj.imnr,obj.imnc,obj.nfrm,obj.npt,obj.ntgt,obj.hmcls);
      for f=1:obj.nfrm
        if tfWB
          wbObj.updateFracWithNumDen(f);
        end
        for ipt=1:obj.npt
          for itgt=1:obj.ntgt
            fname = sprintf(pat,itgt,f,ipt);
            fname = fullfile(hmdir,fname);
            dat(:,:,f,ipt,itgt) = imread(fname);
          end
        end
      end
      
      obj.hmdat = dat;
    end
    
    function hm = read(obj,f,ipt,itgt,varargin)
      % ipt can be vec
      % 
      % hm: [imnr x imnc x numel(ipt)]
      
      normalize = myparse(varargin,...
        'normalize',false... % if true, hm is of class double in [0,1]
        );

      if ~isempty(obj.hmdat)
        hm = squeeze(obj.hmdat(:,:,f,ipt,itgt));
        return;
      end
      
      nptcurr = numel(ipt);
      if normalize
        hm = zeros(obj.imnr,obj.imnc,nptcurr);
      else
        hm = zeros(obj.imnr,obj.imnc,nptcurr,obj.hmcls);
      end
        
      hmdir = obj.dir;
      for iipt=1:nptcurr
        fname = sprintf(obj.filepat,itgt,f,ipt(iipt));
        fname = fullfile(hmdir,fname);
        hm0 = imread(fname);
        if normalize
          hm0 = HistEq.normalizeGrayscaleIm(hm0);
        end
        hm(:,:,iipt) = hm0;
      end
    end
    
  end
  
end