function y = identity_except_EditInteractions_to_empty(x) 

if isa(x, 'matlab.graphics.interaction.interactions.EditInteraction') ,
  y = [] ;
else
  y = x ;
end
