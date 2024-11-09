classdef ProgressMeter < handle
  properties
    title_
    baseMessage_
    denominator_
    numerator_
    canCancel_
    doShowFraction_
  end

  properties (Dependent)
    fraction
    message
  end

  events
    update
    done
  end

  methods
    function obj = ProgressMeter(varargin)
      % Reset the progress meter with the given denominator
      [title, baseMessage, denominator, numerator, canCancel, doShowFraction] = ...
        myparse(varargin , ...
                'title', 'Progress', ...
                'baseMessage', 'Fraction completed', ...
                'denominator', nan, ...  % not really an obvious default here
                'numerator', 0, ...
                'canCancel', false, ...
                'doShowFraction', true) ;
      obj.title_ = title ;
      obj.baseMessage_ = baseMessage ;
      obj.numerator_ = numerator ;
      obj.denominator_ = denominator ;
      obj.canCancel_ = canCancel ;
      obj.doShowFraction_ = doShowFraction ;
    end

    function delete(obj)
      obj.notify('done') ;      
    end

    function bump(numerator)
      obj.numerator_ = numerator ;
      obj.notify('update') ;
    end  % function

    function result = get.fraction(obj) 
      protoresult = obj.numerator_ / obj.denominator_ ;
      result = fif(isfinite(protoresult), protoresult, 0) ;
    end

    function result = get.message(obj) 
      fraction = obj.fraction ;
      if obj.doShowFraction_ && isfinite(fraction) ,
        result = sprintf('%s (%d/%d)', obj.baseMessage_, obj.numerator_ , obj.denominator_) ;
      else
        result = obj.baseMessage_ ;
      end
    end
  end  % methods
end  % classdef
