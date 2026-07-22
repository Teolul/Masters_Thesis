function isValid = checkFileValidity(targetFile, referenceFile)
    % Initialize validity flag
    isValid = false;
    
    % Check if target file exists
    if ~exist(targetFile, 'file')
        disp('Target file does not exist.');
        return;
    end
    
    % Check if target file has the correct name format <model>_<Sx>
    [~, targetName, targetExt] = fileparts(targetFile);
    targetExt = lower(targetExt);
    
    if ~ismember(targetExt, {'.h5', '.hdf5'})
        disp('Target file is not a .h5 or .hdf5 file.');
        return;
    end
    
    % Regular expression for checking the file name <model>_<Sx>
    namePattern = '^[a-zA-Z0-9]+_(A|B)[12]$';
    if isempty(regexp(targetName, namePattern, 'once'))
        disp('Target file name is not in the required format <model>_<Sx>.');
        return;
    end
    
    % Open the target h5 file and check for 'LUTdata'
    try
        targetLUTdata = h5read(targetFile, '/LUTdata');
    catch
        disp('LUTdata variable not found in the target file.');
        return;
    end
    
    % Check if 'LUTdata' is a 2D matrix
    if ndims(targetLUTdata) ~= 2
        disp('LUTdata in the target file is not a 2D matrix.');
        return;
    end
    
    % Check if reference file exists
    if ~exist(referenceFile, 'file')
        disp('Reference file does not exist.');
        return;
    end
    
    % Open the reference h5 file and check for 'LUTdata'
    try
        referenceLUTdata = h5read(referenceFile, '/LUTdata');
    catch
        disp('LUTdata variable not found in the reference file.');
        return;
    end
    
    % Check if 'LUTdata' in reference file is 2D
    if ndims(referenceLUTdata) ~= 2
        disp('LUTdata in the reference file is not a 2D matrix.');
        return;
    end
    
    % Check if the second dimension of LUTdata in target file matches reference file
    if size(targetLUTdata, 2) ~= size(referenceLUTdata, 2)
        disp('Second dimension of LUTdata in target file does not match reference file.');
        return;
    end
    
    % If all checks pass, the file is valid
    isValid = true;
    disp('Target file is valid.');
end
