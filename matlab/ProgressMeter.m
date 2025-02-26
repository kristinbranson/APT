classdef ProgressMeter < handle
  % This is a model class for a waitbar.  Designed to be used with a Labeler and
  % (optionaly) a LabelerController.

  properties
    title_ = ''
    message_ = ''
    denominator_ = nan
    numerator_ = nan
    doShowFraction_ = true
    isActive_ = false  % maps 1-to-1 to whether the figure is Visible
    wasCanceled_ = false
  end

  properties (Dependent)
    title
    fraction
    message
    wasCanceled
    isActive
  end

  events
    didArm
    update
  end

  methods
    function obj = ProgressMeter()
    end

    function delete(~)
    end

    function arm(obj, varargin)
      [title, doShowFraction] = ...
        myparse(varargin , ...
                'title', 'Progress', ...
                'doShowFraction', true) ;
      obj.title_ = title ;
      obj.doShowFraction_ = doShowFraction ;
      obj.denominator_ = nan ;
      obj.numerator_ = nan ;
      obj.isActive_ = false ;
      obj.wasCanceled_ = false ;      
      obj.notify('didArm') ;
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

    function disarm(obj, varargin)
      obj.title_ = '' ;
      obj.doShowFraction_ = true ;
      obj.denominator_ = nan ;
      obj.numerator_ = nan ;
      obj.isActive_ = false ;
      obj.wasCanceled_ = false ;      
      obj.notify('update') ;
    end
    
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

    function result = get.wasCanceled(obj)
      result = obj.wasCanceled_ ;
    end

    function result = get.title(obj)
      result = obj.title_ ;
    end

    function result = get.isActive(obj)
      result = obj.isActive_ ;
    end  % function

  end  % methods
end  % classdef
