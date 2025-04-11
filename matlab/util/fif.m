function y = fif(test, y_true, y_false)
    % "Functional if" --- Like the ternary operator in C.
    % Probably only good to use if the 2nd and 3rd args are cheap to compute,
    % since both will be computed, regardless of the value of test.
    
    if isscalar(test) ,
        if test,
            y = y_true ;
        else
            y = y_false ;
        end
    else
        % Promote y_true, y_false to be the same size as test, if needed
        if isscalar(y_true) ,
            y_true = repmat(y_true, size(test)) ;
        end
        if isscalar(y_false) ,
            y_false = repmat(y_false, size(test)) ;
        end
        y = y_false ;
        y(test) = y_true(test) ;
    end
end
