function ensureNativeFolderExists(raw_folder_path)
    % Ensure the folder indicated exists.  Will create parent fodlers if
    % needed.  raw_folder_path should be a valid folder name/path in the
    % native filesystem (Linux/Windows/Mac).
    folder_path = absolutifyFileName(raw_folder_path) ;
    ensureNativeFolderExistsHelper(folder_path) ;
end



function ensureNativeFolderExistsHelper(folder_path)
    if exist(folder_path, 'file') ,
        if exist(folder_path, 'dir') ,
            % do nothing, all is well, return
        else
            error('Want to create folder %s, but a file (not a folder) already exists at that location', folder_path) ;
        end
    else
        parent_folder_path = fileparts(folder_path) ;
        ensureNativeFolderExistsHelper(parent_folder_path) ;
        mkdir(folder_path) ;
    end        
end
