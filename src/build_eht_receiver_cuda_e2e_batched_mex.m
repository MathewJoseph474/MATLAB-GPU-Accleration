function build_eht_receiver_cuda_e2e_batched_mex()
% Build the E2E/batched development receiver CUDA MEX.

    thisDir = fileparts(mfilename("fullpath"));
    source = fullfile(thisDir,'eht_receiver_cuda_e2e_batched_mex.cu');
    assert(isfile(source),'Missing source:\n%s',source);

    % Regenerate packet-size-dependent CUDA/LDPC constants from project config.
    projectDir = fileparts(thisDir);
    generate_eht_post_equalizer_fixed_config(projectDir);

    try
        eht_receiver_cuda_e2e_batched_mex('reset');
    catch
    end

    clear eht_receiver_cuda_e2e_batched_mex
    clear mex
    rehash

    fprintf('Building E2E/batched development receiver...\n');

    % Always place the rebuilt MEX beside this source file.  Without
    % -outdir, mexcuda writes to MATLAB's current working directory, which
    % can leave an older src\*.mexw64 earlier on the project path.
    mexcuda( ...
        '-R2018a', ...
        '-outdir',thisDir, ...
        '-output','eht_receiver_cuda_e2e_batched_mex', ...
        source, ...
        '-lcufft');

    clear eht_receiver_cuda_e2e_batched_mex
    rehash

    expectedMex = fullfile(thisDir, ...
        ['eht_receiver_cuda_e2e_batched_mex.' mexext]);

    assert(isfile(expectedMex), ...
        'E2E/batched receiver MEX was not created at:\n%s',expectedMex);

    resolvedMex = which('eht_receiver_cuda_e2e_batched_mex');

    fprintf('Created MEX:\n%s\n',expectedMex);
    fprintf('MATLAB resolves MEX to:\n%s\n',resolvedMex);

    if ~strcmpi(char(resolvedMex),char(expectedMex))
        warning(['MATLAB is resolving a different copy of the MEX. ' ...
            'Run: which eht_receiver_cuda_e2e_batched_mex -all']);
    end
end
