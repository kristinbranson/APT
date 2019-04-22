function result=isControlLike(modifiers)

result= ( ( ~ismac() && any(strcmp('control',modifiers)) ) ...
          || ...
          (  ismac() && any(strcmp('command',modifiers)) ) ) ;

end