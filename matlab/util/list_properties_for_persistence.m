function property_names = list_properties_for_persistence(object)
    % Return a list of the properties of object that you *generally* want to
    % persist.  I.e. those properties that are not dependent, not transient, and
    % not constant.  Of course, some classes will contain properties that fit this
    % description, but you do not want to persist.  So this is simply a utility
    % function to be used in a general method, that may need to be overridden for
    % some subclasses.

    % Define a helper function
    should_property_be_persisted = @(x)(~x.Dependent && ~x.Transient && ~x.Constant) ;

    % Actually get the prop names that satisfy the predicate
    property_names = property_names_satisfying_predicate(object, should_property_be_persisted) ;
end
