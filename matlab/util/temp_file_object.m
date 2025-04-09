classdef temp_file_object < file_object
    % Simple temp file object that fcloses() the fid when no longer referenced,
    % and also deletes the temp file.
    % Can be used just like a fid in most cases.
    
    properties (Access = protected)
        abs_file_path_  % absolute path to temp file
    end

    properties (Dependent)
        abs_file_path
    end

    methods
        function obj = temp_file_object(varargin)
            abs_file_path = tempname() ;
            obj = obj@file_object(abs_file_path, varargin{:}) ;
            obj.abs_file_path_ = abs_file_path ;
        end

        function result = get.abs_file_path(obj) 
            result = obj.abs_file_path_ ;
        end
        
        function delete(obj)
            delete@file_object(obj) ;  % close the file if open
            delete(obj.abs_file_path_) ;  % delete the temp file
        end
    end
end
