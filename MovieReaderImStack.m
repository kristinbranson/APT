classdef MovieReaderImStack < handle
  % MovieReaderImStack
  % Like MovieReader, but just read from in-mem images
  
  properties
    I; % [N] cell vec of images
    forceGrayscale = false; % if true, [MxNx3] images are run through rgb2gray
  end
  
  properties (Dependent)
    isOpen
    nframes
  end
  
  methods
    function v = get.isOpen(obj)
      v = ~isempty(obj.I);
    end
    function v = get.nframes(obj)
      v = numel(obj.I);
    end
  end
  
  methods
    
    function obj = MovieReaderImStack()
      % none
    end
        
    function open(obj,I)
      obj.I = I;
    end
    
    function varargout = readframe(obj,i)
      varargout{1} = obj.I{i};      
      if obj.forceGrayscale
        if nargout==1
          if size(varargout{1},3)==3 % doesn't have to be RGB but convert anyway
            varargout{1} = rgb2gray(varargout{1});
          end
        else
          warning('MovieReader:grayscale','Do not know how to convert to grayscale.');          
        end
      end
    end
    
    function close(obj)
      obj.I = [];
    end    
  
    function delete(obj)
      obj.close();
    end
    
  end
  
end

