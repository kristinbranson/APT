classdef HeatmapReader < handle

  properties (Constant)
    % sprintf-style pattern; tgt,frm,part, all 1-based
    filepat = 'hmap_trx_%d_t_%d_part_%d.jpg';
    filepatre = 'hmap_trx_(?<iTgt>[0-9]+)_t_(?<t>[0-9]+)_part_(?<ipt>[0-9]+).jpg';
  end
  
  properties
    dir % fullpath heatmap dir
    
    
    imnr
    imnc
    nfrm
    npt
    ntgt
    
    hmcls = 'uint8';
    hmdat % [imnr x imnc x nfrm x npt x ntgt]; preload only; very mem expensive needless to say 
    hmnormalizeType % 'none','lohi','bitdepth'
    hmlothresh
    hmhithresh
  end
  
  methods
    
    function obj = HeatmapReader
    end
    
    function init(obj,hmdir,imnr,imnc,nfrm,npt,ntgt)
      % imnr/imnc allowed to be empty for 'unknown'
      
      obj.dir = hmdir;
      obj.imnr = imnr;
      obj.imnc = imnc;
      obj.nfrm = nfrm;
      obj.npt = npt;
      obj.ntgt = ntgt;
      
      obj.hmdat = [];
      
%       dd = dir(fullfile(hmdir,'*.jpg')); %#ok<CPROPLC>
%       nhmaps = nfrm*npt*ntgt;
%       if numel(dd)<nhmaps
%         warningNoTrace('Heatmap dir contains %d jpgs; expected %d.',...
%           numel(dd),nhmaps);
%       end
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
      
      assert(~isempty(obj.imnr));

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
    
    function [hm,fname] = read(obj,f,ipt,itgt,varargin)
      % ipt can be vec
      % 
      % hm: [imnr x imnc x numel(ipt)]
      
      normalizeType = obj.hmnormalizeType;
      
      if ~isempty(obj.hmdat)
        hm = squeeze(obj.hmdat(:,:,f,ipt,itgt));
        return;
      end
      
      hmdir = obj.dir;
      
      if isempty(obj.imnr)
        assert(~isempty(ipt));
        fname = sprintf(obj.filepat,itgt,f,ipt(1));
        fname = fullfile(hmdir,fname);
        hm0 = imread(fname);
        [obj.imnr,obj.imnc] = size(hm0);
        fprintf(1,'Setting heatmap size: [nr nc]=[%d %d]\n',obj.imnr,obj.imnc);
      end
      
      nptcurr = numel(ipt);
      switch normalizeType
        case {'lohi' 'bitdepth'}
          hm = zeros(obj.imnr,obj.imnc,nptcurr);
        case 'none'
          hm = zeros(obj.imnr,obj.imnc,nptcurr,obj.hmcls);
        otherwise
          assert(false);
      end
        
      for iipt=1:nptcurr
        fname = sprintf(obj.filepat,itgt,f,ipt(iipt));
        fname = fullfile(hmdir,fname);
        hm0 = imread(fname);
        switch normalizeType
          case 'lohi'
            hm0 = double(hm0);
            hm0 = min(max( (hm0-obj.hmlothresh)/(obj.hmhithresh-obj.hmlothresh), 0), 1);
          case 'bitdepth'
            hm0 = HistEq.normalizeGrayscaleIm(hm0);
        end
        hm(:,:,iipt) = hm0;
      end
    end
    
  end

  methods (Static)
    
    function s = parsehmapjpgname(n)
      s = regexp(n,HeatmapReader.filepatre,'names');
      s = structfun(@str2double,s,'uni',0);
    end
    
    function [t0,t1] = findFirstLastFrameHmapDir(hmdir)
      dd = dir(fullfile(hmdir,'*.jpg'));
      nm = {dd.name}';
      nm = fullfile(hmdir,nm);
      sMD = cellfun(@HeatmapReader.parsehmapjpgname,nm);
      ts = cat(1,sMD.t);
      t0 = min(ts);
      t1 = max(ts);
    end
    
  end
  
end