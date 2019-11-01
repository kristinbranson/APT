classdef SparseLabelArray
%     size   % size() of array
%     type % Either 'nan', 'ts', or 'log'. Corresponding base element: nan, 
%          % -inf, false
%     idx  % [nval] linear indices 
%     val % [nval] values at idx. In case of type=='log' this is a scalar
  methods (Static)
    function s = create(x,ty)
      switch ty
        case 'nan'
          i = find(~isnan(x));
          v = x(i);
        case 'ts'
          i = find(~isinf(x));
          v = x(i);
        case 'log'
          i = find(x);
          v = true; % just a placeholder
        otherwise
          assert(false,'Unrecognized type.');
      end      
      s = struct();
      s.size = size(x);
      s.type = ty;
      s.idx = i;
      s.val = v;
    end
    function s = createEmpty(sz,ty)
      switch ty
        case {'nan' 'ts' 'log'}
          % none
        otherwise
          assert(false,'Unrecognized type.');
      end      
      s = struct();
      s.size = sz;
      s.type = ty;
      s.idx = zeros(0,1);
      s.val = zeros(0,1);
    end
    function x = full(s)
      assert(isstruct(s));
      switch s.type
        case 'nan'
          x = nan(s.size);
        case 'ts'
          x = -inf(s.size);
        case 'log'
          x = false(s.size);
        otherwise
          assert(false,'Unrecognized type.');
      end
      x(s.idx) = s.val; % scalar expansion for 'log'
    end
  end
  methods (Static) 
    % Methods that act on full arrays; maybe this class should be
    % 'LabelArray'
    
    function x = fullExpandPts(x,ty,nptsadd)
      % Add points to labelarrays
      szadd = size(x);
      szadd(1) = nptsadd;
      switch ty
        case 'nan', xadd = nan(szadd);          
        case 'ts', xadd = -inf(szadd);
        case 'log', xadd = false(szadd);
        otherwise, assert(false);
      end
      x = cat(1,x,xadd);
    end
    function x = fullRmPts(x,iptsrm)
      colons = repmat({':'},1,ndims(x)-1);
      x(iptsrm,colons{:}) = [];      
    end
  end
end