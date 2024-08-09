function name = get_user_name()
    if ispc()
        name = getenv('USERNAME') ;
    else
        name = getenv('USER') ;
    end        
end
