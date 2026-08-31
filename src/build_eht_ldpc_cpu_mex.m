function build_eht_ldpc_cpu_mex()
%BUILD_EHT_LDPC_CPU_MEX Build the independent CPU-only LDPC decoder MEX.
%
% Requires a configured C++ MEX compiler. On Windows with Visual Studio:
%   mex -setup C++
%   build_eht_ldpc_cpu_mex

thisDir = fileparts(mfilename('fullpath'));
source = fullfile(thisDir,'eht_ldpc_cpu_mex.cpp');

fprintf('Building CPU-only LDPC MEX...\n');
mex('-R2018a','-O',source,'-outdir',thisDir,'-output','eht_ldpc_cpu_mex');
rehash;

assert(exist('eht_ldpc_cpu_mex','file') == 3, ...
    'The MEX build did not produce eht_ldpc_cpu_mex.');
fprintf('Built: %s\n',which('eht_ldpc_cpu_mex'));
fprintf('This MEX contains no CUDA code and executes only on the CPU.\n');
end
