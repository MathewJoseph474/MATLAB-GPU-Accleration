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

    mexcuda( ...
        '-R2018a', ...
        '-output','eht_receiver_cuda_e2e_batched_mex', ...
        source, ...
        '-lcufft');

    rehash
    assert(exist('eht_receiver_cuda_e2e_batched_mex','file') == 3, ...
        'E2E/batched receiver MEX was not created.');

    fprintf('Created:\n%s\n',which('eht_receiver_cuda_e2e_batched_mex'));
end
