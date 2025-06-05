classdef DataAugMontage < handle
    
  properties 
    idx % [nx3] (m,f,t)
    ims % [n imnr imnc 3]
    locs % [n x npts x d x ntgt]. 
    mask % [n x imnr x nc]
    isMA
    name = '' % character description
    maskrgb = [0 1 1];
  end
  
  methods
    function obj = DataAugMontage
    end
    function init(obj,mat)
      if ischar(mat)
        mat = load(mat);
      end
      fns = fieldnames(mat);
      for f=fns(:)',f=f{1}; %#ok<FXSET>
        obj.(f) = mat.(f);
      end
      obj.isMA = ndims(obj.locs)==4;
      if obj.isMA
        obj.locs = permute(obj.locs,[1 3 4 2]);
      end
    end
    function hfig = show(obj,montageargs,varargin)
      hfig = myparse(varargin,'hfig',[]);
      
      [n,imnr,imnc,nch] = size(obj.ims);
      [m,npts,d,ntgts] = size(obj.locs);
      I = mat2cell(obj.ims,ones(n,1),imnr,imnc,nch);
      I = cellfun(@squeeze,I,'uni',0);
      I = cellfun(@(x)x(:,:,1),I,'uni',0);
      I = cellfun(@DataAugMontage.convertIm2Double,I,'uni',0);
      p = reshape(obj.locs,n,npts*d,ntgts);
      if isempty(obj.mask)
        masks = [];
      else
        masks = mat2cell(obj.mask,ones(n,1),imnr,imnc,1);
        masks = cellfun(@squeeze,masks,'uni',0);
        maskbase = reshape(obj.maskrgb(:),1,1,3);
        maskbase = repmat(maskbase,imnr,imnc);
        masks = cellfun(@(x)x.*maskbase,masks,'uni',0);
      end
      
      mft = obj.idx;
      didcreate = false;
      if isempty(hfig) || ~ishandle(hfig),
        hfig = figure('Name',sprintf('%s Training Examples',obj.name));
        didcreate = true;
      end
      lbls = arrayfun(@(x)sprintf('%d.%d.%d',mft(x,1),mft(x,2),mft(x,3)),...
        (1:n)','uni',0);
      title = 'Training Data (augmented)';
      if nch == 3
        title = sprintf('%s. Colored images are used for training but displayed as grayscale',title);
      end
      Shape.montage(I,p,...
        'masks',masks,...
        'titlestr',title,...
        'framelbls',lbls, ...
        'idxs',1:n,...
        montageargs{:},...
        'fig',hfig...
        );
      if didcreate,
        truesize(hfig);
      end
    end
    
  end
  
  methods (Static)
    function img = convertIm2Double(img)
      if isa(img,'uint8')
        img = double(img)/255;
      elseif isa(img,'uint16')
        img = double(img)/(2^16-1);
      elseif max(img(:))>1.0
        % double, but assume in range [0,255]
        img = img/255;
      else
        % none
      end
    end
  end
  
end