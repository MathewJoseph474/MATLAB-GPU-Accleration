%% RUN_WIFI_PROJECT
% One-click launcher for the Wi-Fi 7 CPU/GPU receiver benchmark.
clc;
close all;
projectRoot = fileparts(mfilename("fullpath"));
addpath(fullfile(projectRoot,"src"));
addpath(fullfile(projectRoot,"config"));
addpath(fullfile(projectRoot,"functions"));
parallel.gpu.enableCUDAForwardCompatibility(true);
dev = gpuDevice();
fprintf('GPU               : %s\n',dev.Name);
fprintf('Compute capability: %s\n',dev.ComputeCapability);
fprintf('Available memory  : %.2f GB\n',dev.AvailableMemory / 1e9);
cfg = project_config(projectRoot);
packetCounts = build_packet_sweep(cfg);
fprintf('\n============================================================\n');
fprintf(' Wi-Fi 7 CPU/GPU Complete Batched Receiver Benchmark\n');
fprintf('============================================================\n');
fprintf('Project root       : %s\n',projectRoot);
fprintf('CPU workers        : %d\n',cfg.CPU.NumWorkers);
if cfg.CPU.NumWorkers == 1, fprintf('CPU mode           : sequential\n'); else, fprintf('CPU mode           : parallel\n'); end
fprintf('GPU batch size     : %d packets\n',cfg.GPU.BatchSize);
fprintf('Validation packets : %d\n',cfg.Execution.ValidationPackets);
fprintf('Warmup packets     : %d\n',cfg.Execution.WarmupPackets);
fprintf('LLR scale          : %.2f\n',cfg.Execution.LLRScale);
fprintf('Packet mode        : %s\n',cfg.Packets.Mode);
fprintf('Packet sweep       : '); fprintf('%d ',packetCounts); fprintf('\n');
fprintf('============================================================\n\n');
results = main_eht_receiver_sweep_8k_16k_32k;
fprintf('\n============================================================\n');
fprintf(' BENCHMARK COMPLETE\n');
fprintf('============================================================\n');
fprintf('Results saved to:\n%s\n',results.OutputDirectory);
fprintf('============================================================\n');
