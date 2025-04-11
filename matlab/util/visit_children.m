function visit_children(h, f, varargin)

feval(f,h,varargin{:}) ;
if isprop(h, 'Children') ,
  kids = h.Children ;
  arrayfun(@(kid)(visit_children(kid, f, varargin{:})), kids) ;
end

end
