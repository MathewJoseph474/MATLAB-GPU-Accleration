function cfg = project_config(projectRoot)

% PROJECT PATHS
cfg.Project.Root = projectRoot;

cfg.Project.DatabaseFolder = ...
    fullfile(projectRoot, "database");

cfg.Project.ResultsFolder = ...
    fullfile(projectRoot, "results");

%% CPU EXECUTION
% Number of CPU workers requested by the user.
% 1  = sequential CPU execution
% less than 1 = parallel CPU execution
cfg.CPU.NumWorkers = 8;

cfg.CPU.PoolType = "Processes";
cfg.CPU.MaximumSafeWorkers = 20;

%% GPU / RECEIVER EXECUTION

cfg.Execution.RunCPU = true;
cfg.Execution.RunGPU = true;

% before the timed benchmark begins.
cfg.Execution.ValidationPackets = 128; % Number of packets used for correctness validation

% Warm-up is excluded from the timed region.
cfg.Execution.WarmupPackets = 2; % Number of packets used to warm up the receiver.

% LLR scaling used by the CUDA receiver.
cfg.Execution.LLRScale = single(3.5);

%% PACKET SWEEP
% "manual"
% "power2"
% "baseMultiples"

cfg.Packets.Mode = "manual";

% Manual mode
cfg.Packets.Values = [
   8192
   16384
   32768
];

% Power-of-two mode

cfg.Packets.Min = 1024;
cfg.Packets.Max = 65536;

% Base-multiple mode

cfg.Packets.Base = 8192;

cfg.Packets.Multipliers = [
    1
];


%% PACKET SIZE

% Payload size of each Wi-Fi packet in bits.
% Default preserves the current 32000-bit packet size.
cfg.PacketSize.BitsPerPacket = 32000; 

%% DATA REPRESENTATION

% Receiver database storage format.
cfg.Data.InputType = "int16";

% Receiver processing precision.
cfg.Data.ComputeType = "single";

%% PLOTTING

cfg.Plot.Enable = true;
cfg.Plot.PacketThroughput = true;
cfg.Plot.PayloadMbps = true;
cfg.Plot.Latency = true;
cfg.Plot.Speedup = true;

% Figure output

cfg.Plot.SaveFIG = false;
cfg.Plot.SavePDF = true;
cfg.Plot.SavePNG = false;

cfg.Plot.ShowFigures = true;

cfg.Plot.CloseExistingFigures = true;

%% RESULTS

cfg.Results.SaveMAT = true;
cfg.Results.SaveCSV = true;

cfg.Results.SaveIndividualBenchmarkResults = true;



%% GPU ASYNCHRONOUS EXECUTION

cfg.GPU.MaxMemoryFraction = 0.95;

% Runtime packet batch size for the complete GPU E2E receiver.
cfg.GPU.BatchSize = 1024;

cfg.GPU.Async.Enable = true;

% Number of packets allowed in flight concurrently.
cfg.GPU.Async.NumBuffers = 4;

%% VALIDATION

validateattributes( ...
    cfg.CPU.NumWorkers, ...
    {'numeric'}, ...
    {'scalar','integer','positive','finite'});

validateattributes( ...
    cfg.CPU.MaximumSafeWorkers, ...
    {'numeric'}, ...
    {'scalar','integer','positive','finite'});

validateattributes( ...
    cfg.Execution.ValidationPackets, ...
    {'numeric'}, ...
    {'scalar','integer','nonnegative','finite'});

validateattributes( ...
    cfg.Execution.WarmupPackets, ...
    {'numeric'}, ...
    {'scalar','integer','nonnegative','finite'});

validateattributes( ...
    cfg.PacketSize.BitsPerPacket, ...
    {'numeric'}, ...
    {'scalar','integer','positive','finite'});

if mod(cfg.PacketSize.BitsPerPacket, 8) ~= 0
    error("PacketSize.BitsPerPacket must be divisible by 8.");
end

if cfg.CPU.NumWorkers > cfg.CPU.MaximumSafeWorkers

    error( ...
        "Requested CPU workers (%d) exceeds MaximumSafeWorkers (%d).", ...
        cfg.CPU.NumWorkers, ...
        cfg.CPU.MaximumSafeWorkers);

end

end