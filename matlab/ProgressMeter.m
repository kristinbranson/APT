classdef ProgressMeter < handle
  % This is a model class for a waitbar.  

  properties
%     parent_  % the parent object, the target of message sends when the ProgressMeter is cancelled or closed
%     cancelMethodName_
%     closeMethodName_
    title_ = ''
    message_ = ''
    denominator_ = nan
    numerator_ = nan
    canCancel_ = true
    doShowFraction_
    isActive_ = false
    wasCanceled_ = false
  end

  properties (Dependent)
    title
    fraction
    message
    canCancel
    wasCanceled
    isActive
  end

  events
    update
    willBeDeleted
  end

  methods
    function obj = ProgressMeter(varargin)
      [title, canCancel, doShowFraction] = ...
        myparse(varargin , ...
                'title', 'Progress', ...
                'canCancel', true, ...
                'doShowFraction', true) ;
%       if isempty(parent) ,
%         error('parent argument is required') ;
%       end
%       if isempty(cancelMethodName) ,
%         error('cancelMethodName argument is required') ;
%       end
%       if isempty(closeMethodName) ,
%         error('closeMethodName argument is required') ;
%       end
      %obj.parent_ = parent ;
      obj.title_ = title ;
      obj.canCancel_ = canCancel ;
      obj.doShowFraction_ = doShowFraction ;
      %obj.cancelMethodName_ = cancelMethodName ;
      %obj.closeMethodName_ = closeMethodName ;
    end

    function delete(obj)
      obj.notify('willBeDeleted') ;      
    end

    function start(obj, varargin)
      [message, numerator, denominator] = ...
        myparse(varargin , ...
                'message', 'Fraction completed', ...
                'numerator', 0, ...
                'denominator', []) ;
      if isempty(denominator) ,
        error('denominator argument is required') ;
      end
      obj.message_ = message ;
      obj.numerator_ = numerator ;
      obj.denominator_ = denominator ;
      obj.wasCanceled_ = false ;
      obj.isActive_ = true ;
      obj.notify('update') ;
    end  % function
    
    function bump(obj, numerator)
      obj.numerator_ = numerator ;
      obj.notify('update') ;
    end  % function

    function finish(obj)
      obj.isActive_ = false ;
      obj.notify('update') ;
    end  % function

    function cancel(obj)
      obj.isActive_ = false ;
      obj.wasCanceled_ = true ;
      obj.notify('update') ;
    end

%     function close(obj)
%       obj.isActive_ = false ;
%       obj.wasCanceled_ = true ;  % is this what we want?
%       obj.notify('update') ;
%       feval(obj.closeMethodName_, obj.parent_) ;            
%     end

    function result = get.fraction(obj) 
      protoresult = obj.numerator_ / obj.denominator_ ;
      result = fif(isfinite(protoresult), protoresult, 0) ;
    end

    function result = get.message(obj) 
      fraction = obj.fraction ;
      if obj.doShowFraction_ && isfinite(fraction) ,
        result = sprintf('%s (%d/%d)', obj.message_, obj.numerator_ , obj.denominator_) ;
      else
        result = obj.message_ ;
      end
    end

    function result = get.canCancel(obj)
      result = obj.canCancel_ ;
    end

    function result = get.wasCanceled(obj)
      result = obj.wasCanceled_ ;
    end

    function result = get.title(obj)
      result = obj.title_ ;
    end

    function result = get.isActive(obj)
      result = obj.isActive_ ;
    end

  end  % methods
end  % classdef
