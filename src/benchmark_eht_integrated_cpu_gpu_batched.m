% Complete CPU receiver versus complete batched GPU E2E receiver.
% GPU timing uses e2eBatchRun, which does not transfer reference payload
% bits during the timed region. 

function results = benchmark_eht_integrated_cpu_gpu_batched(userOpts)

arguments
    userOpts struct = struct()
end

opts = struct;
opts.ProjectDir = fileparts(fileparts(mfilename("fullpath")));
opts.NumPackets = 8192;
opts.CPUExecutionMode = "sequential";
opts.NumWorkers = 0;
opts.PoolType = "Processes";
opts.MaximumSafeWorkers = 8;
opts.ValidationPackets = 4;
opts.WarmupPackets = 2;
opts.LLRScale = single(3.5);
opts.GPUBatchSize = 1024;
opts.SaveIndividualResult = true;
opts.PrintHeader = true;

names = fieldnames(userOpts);
for k = 1:numel(names)
    name = names{k};
    if ~isfield(opts,name)
        error('wifi7cuda:UnknownOption','Unknown benchmark option: %s',name);
    end
    opts.(name) = userOpts.(name);
end

opts.CPUExecutionMode = lower(string(opts.CPUExecutionMode));
opts.PoolType = string(opts.PoolType);
validateattributes(opts.NumPackets,{'numeric'},{'scalar','integer','positive','finite'});
validateattributes(opts.GPUBatchSize,{'numeric'},{'scalar','integer','positive','finite'});

projectDir = opts.ProjectDir;
addpath(fullfile(projectDir,'config'));
projectCfg = project_config(projectDir);
databasePath = fullfile(projectDir,'database','eht_receiver_database_128.mat');
assert(isfile(databasePath),'Database not found:\n%s',databasePath);
assert(exist('eht_receiver_cuda_e2e_batched_mex','file') == 3, ...
    'Missing eht_receiver_cuda_e2e_batched_mex. Build it first.');

database = load(databasePath);
cfgEHT = database.cfgEHT;
fieldIndices = wlanFieldIndices(cfgEHT);
sampleRate = wlanSampleRate(cfgEHT);
channelBandwidth = char(cfgEHT.ChannelBandwidth);
packetLengthSamples = fieldIndices.EHTData(2);

numPackets = double(opts.NumPackets);
payloadBitsPerPacket = double(projectCfg.PacketSize.BitsPerPacket);
databasePayloadBits = 8*double(cfgEHT.User{1}.APEPLength);
assert(databasePayloadBits == payloadBitsPerPacket, ...
    'Configured packet size (%d bits) does not match the receiver database (%d bits). Rebuild the database.', ...
    payloadBitsPerPacket,databasePayloadBits);
uniquePackets = size(database.RxI,2);
sourceMap = mod(0:numPackets-1,uniquePackets)+1;

% Fixed receiver metadata/reference signals. Not timed.
cleanWaveform = single(wlanWaveformGenerator(database.TxBits(:,1),cfgEHT));
lltfReference = cleanWaveform(fieldIndices.LLTF(1):fieldIndices.LLTF(2),:);
lltfReference = complex(single(real(lltfReference)),single(imag(lltfReference)));

ltfInfo = wlanEHTOFDMInfo('EHT-LTF',cfgEHT);
dataInfo = wlanEHTOFDMInfo('EHT-Data',cfgEHT);
ltfFFTStart = round(0.75*double(ltfInfo.CPLength))+1;
dataCP = double(dataInfo.CPLength);
dataSymbolLength = double(dataInfo.FFTLength)+dataCP;
dataFFTStart = round(0.75*dataCP)+1;
[commonParams,~] = wlan.internal.ehtCodingParameters(cfgEHT,1);
numDataSymbols = double(commonParams.NSYM);
dataFFTStarts = dataFFTStart + (0:numDataSymbols-1).' * dataSymbolLength;
ltfFFTIndices = double(ltfInfo.ActiveFFTIndices(:));
dataFFTIndices = double(dataInfo.ActiveFFTIndices(:));
pilotIndices = double(dataInfo.PilotIndices(:));
dataIndices = double(dataInfo.DataIndices(:));

cleanEHTLTF = cleanWaveform(fieldIndices.EHTLTF(1):fieldIndices.EHTLTF(2),:);
cleanEHTData = cleanWaveform(fieldIndices.EHTData(1):fieldIndices.EHTData(2),:);
cleanLTFDemod = single(wlanEHTDemodulate(cleanEHTLTF,'EHT-LTF',cfgEHT));
cleanChannel = single(wlanEHTLTFChannelEstimate(cleanLTFDemod,cfgEHT));
cleanDataDemod = single(wlanEHTDemodulate(cleanEHTData,'EHT-Data',cfgEHT));
knownLTF = cleanLTFDemod(:)./cleanChannel(:);
knownLTF = knownLTF./abs(knownLTF);
knownLTF = complex(single(real(knownLTF)),single(imag(knownLTF)));
pilotReference = cleanDataDemod(pilotIndices,:)./cleanChannel(pilotIndices);
pilotReference = pilotReference./abs(pilotReference);
pilotReference = complex(single(real(pilotReference)),single(imag(pilotReference)));

checksumWeights = (1:payloadBitsPerPacket).';
expectedUniqueChecksums = zeros(uniquePackets,1);
for k = 1:uniquePackets
    expectedUniqueChecksums(k) = sum(checksumWeights .* ...
        double(database.TxBits(1:payloadBitsPerPacket,k)));
end
expectedAggregateChecksum = sum(expectedUniqueChecksums(sourceMap));

if opts.PrintHeader
    fprintf('%d-PACKET COMPLETE CPU VS BATCHED GPU E2E BENCHMARK\n',numPackets);
    fprintf('CPU mode                 : %s\n',opts.CPUExecutionMode);
    fprintf('GPU batch size           : %d packets\n',opts.GPUBatchSize);
    fprintf('GPU submissions          : %d\n',ceil(numPackets/opts.GPUBatchSize));
    fprintf('Unique source packets    : %d\n',uniquePackets);
    fprintf('Payload bits per packet  : %d\n',payloadBitsPerPacket);
end

%% CPU pool
pool = gcp('nocreate');
if opts.CPUExecutionMode == "parallel"
    if opts.NumWorkers == 0
        opts.NumWorkers = min(6,opts.MaximumSafeWorkers);
    end
    opts.NumWorkers = min(opts.NumWorkers,opts.MaximumSafeWorkers);
    requestedType = lower(opts.PoolType);
    recreate = isempty(pool);
    if ~isempty(pool)
        currentType = "processes";
        if contains(lower(class(pool)),"thread"), currentType = "threads"; end
        if currentType ~= requestedType || pool.NumWorkers ~= opts.NumWorkers
            delete(pool); pool = []; recreate = true;
        end
    end
    if recreate, pool = parpool(char(opts.PoolType),opts.NumWorkers); end
    actualWorkers = pool.NumWorkers;
else
    actualWorkers = 0;
end

%% Small correctness validation outside timing
validationCount = min(double(opts.ValidationPackets),numPackets);
if validationCount > 0
    idx = sourceMap(1:validationCount);
    refBits = int8(database.TxBits(1:payloadBitsPerPacket,idx));
    gpuValidation = eht_receiver_cuda_e2e_batched_mex( ...
        'e2eBatch',database.RxI(:,idx),database.RxQ(:,idx),database.RxScale(idx), ...
        opts.LLRScale,lltfReference,ltfFFTIndices,dataFFTIndices,ltfFFTStart, ...
        dataFFTStarts,knownLTF,pilotIndices,pilotReference,dataIndices,refBits);
    assert(gpuValidation.TotalBitErrors == 0 && gpuValidation.FailedPackets == 0, ...
        'GPU validation failed before timed benchmark.');
end

%% Warm-up run which isoutside timing
warmCount = min(double(opts.WarmupPackets),numPackets);
if warmCount > 0
    idx = sourceMap(1:warmCount);
    eht_receiver_cuda_e2e_batched_mex( ...
        'e2eBatchRun',database.RxI(:,idx),database.RxQ(:,idx),database.RxScale(idx), ...
        opts.LLRScale,lltfReference,ltfFFTIndices,dataFFTIndices,ltfFFTStart, ...
        dataFFTStarts,knownLTF,pilotIndices,pilotReference,dataIndices);
end

%% CPU complete receiver timed path
fprintf('Running CPU complete receiver for %d packets...\n',numPackets);
cpuChecksums = zeros(numPackets,1);

cpuStageTiming_s = zeros(numPackets,12);

cpuProcessingStart = datetime('now');
fprintf('\n[CPU PROCESSING START] %s | Packets: %d | Mode: %s | Workers: %d\n', ...
    timestampTextLocal(cpuProcessingStart),numPackets,opts.CPUExecutionMode,actualWorkers);

cpuWallTimer = tic;
if opts.CPUExecutionMode == "parallel"
    parfor logicalIndex = 1:numPackets
        sourceIndex = sourceMap(logicalIndex);
        [rxSync,~,~,noiseVar,frontTiming] = synchronizePacketLocal( ...
            database.RxI(:,sourceIndex),database.RxQ(:,sourceIndex), ...
            database.RxScale(sourceIndex),cfgEHT,sampleRate,fieldIndices, ...
            packetLengthSamples,channelBandwidth);

        [bits,backendTiming] = recoverCPUFromSynchronizedLocal( ...
            rxSync,noiseVar,cfgEHT,fieldIndices);

        cpuStageTiming_s(logicalIndex,:) = [ ...
            frontTiming.PacketDetection_s, ...
            frontTiming.CoarseCFO_s, ...
            frontTiming.TimingSynchronization_s, ...
            frontTiming.FineCFO_s, ...
            backendTiming.OFDMDemodulation_s, ...
            backendTiming.ChannelEstimation_s, ...
            backendTiming.Equalization_s, ...
            backendTiming.QAM4096Demapping_s, ...
            backendTiming.LLRGeneration_s, ...
            backendTiming.LDPCDecoding_s, ...
            backendTiming.Descrambling_s, ...
            backendTiming.PayloadRecovery_s];

        cpuChecksums(logicalIndex) = checksumLocal(bits,payloadBitsPerPacket);
    end
else
    for logicalIndex = 1:numPackets
        sourceIndex = sourceMap(logicalIndex);
        [rxSync,~,~,noiseVar,frontTiming] = synchronizePacketLocal( ...
            database.RxI(:,sourceIndex),database.RxQ(:,sourceIndex), ...
            database.RxScale(sourceIndex),cfgEHT,sampleRate,fieldIndices, ...
            packetLengthSamples,channelBandwidth);

        [bits,backendTiming] = recoverCPUFromSynchronizedLocal( ...
            rxSync,noiseVar,cfgEHT,fieldIndices);

        cpuStageTiming_s(logicalIndex,:) = [ ...
            frontTiming.PacketDetection_s, ...
            frontTiming.CoarseCFO_s, ...
            frontTiming.TimingSynchronization_s, ...
            frontTiming.FineCFO_s, ...
            backendTiming.OFDMDemodulation_s, ...
            backendTiming.ChannelEstimation_s, ...
            backendTiming.Equalization_s, ...
            backendTiming.QAM4096Demapping_s, ...
            backendTiming.LLRGeneration_s, ...
            backendTiming.LDPCDecoding_s, ...
            backendTiming.Descrambling_s, ...
            backendTiming.PayloadRecovery_s];

        cpuChecksums(logicalIndex) = checksumLocal(bits,payloadBitsPerPacket);
    end
end
cpuWall_s = toc(cpuWallTimer);
cpuProcessingEnd = datetime('now');
fprintf('[CPU PROCESSING END]   %s | Elapsed: %.6f s\n', ...
    timestampTextLocal(cpuProcessingEnd),cpuWall_s);

cpuAggregateChecksum = sum(cpuChecksums);
assert(cpuAggregateChecksum == expectedAggregateChecksum,'Timed CPU checksum mismatch.');

cpuStageMean_s = mean(cpuStageTiming_s,1);
cpuStageTotal_s = sum(cpuStageTiming_s,1);
cpuStageMean_ms = 1000*cpuStageMean_s;

fprintf('\nCPU RECEIVER STAGE TIMING \n');
fprintf('Packet Detection       : %.6f ms/packet\n',cpuStageMean_ms(1));
fprintf('Coarse CFO             : %.6f ms/packet\n',cpuStageMean_ms(2));
fprintf('Timing Synchronization : %.6f ms/packet\n',cpuStageMean_ms(3));
fprintf('Fine CFO               : %.6f ms/packet\n',cpuStageMean_ms(4));
fprintf('OFDM Demodulation      : %.6f ms/packet\n',cpuStageMean_ms(5));
fprintf('Channel Estimation     : %.6f ms/packet\n',cpuStageMean_ms(6));
fprintf('Equalization           : %.6f ms/packet\n',cpuStageMean_ms(7));
fprintf('4096-QAM Demapping     : %.6f ms/packet\n',cpuStageMean_ms(8));
fprintf('LLR Generation         : %.6f ms/packet\n',cpuStageMean_ms(9));
fprintf('LDPC Decoding (CPU)    : %.6f ms/packet\n',cpuStageMean_ms(10));
fprintf('Descrambling           : %.6f ms/packet\n',cpuStageMean_ms(11));
fprintf('Payload Recovery       : %.6f ms/packet\n',cpuStageMean_ms(12));
fprintf('Stages 1-12 subtotal   : %.6f ms/packet\n',sum(cpuStageMean_ms));
fprintf('LDPC backend           : eht_ldpc_cpu_mex\n');

%% GPU complete receiver timed with 4 async buffers
batchSize = min(double(opts.GPUBatchSize),numPackets);
numBatches = ceil(numPackets/batchSize);
fprintf('Running GPU complete receiver: %d packets in %d batch(es), max batch=%d...\n', ...
    numPackets,numBatches,batchSize);

% Allocate the persistent asynchronous batched slots before timing.
% Batch size is configurable; the CUDA MEX fixes only the
% number of buffers/streams to four.
gpuPrepareInfo = eht_receiver_cuda_e2e_batched_mex( ...
    'e2eBatchPrepare',size(database.RxI,1),batchSize);

numGPUBuffers = double(gpuPrepareInfo.BufferCount);
maxInFlight = min(numGPUBuffers,numBatches);

gpuChecksum = 0;
processed = 0;

gpuStageTotal_ms = zeros(1,12);
% Column order:
%   1 = Packet Detection
%   2 = Coarse CFO
%   3 = Timing Synchronization
%   4 = Fine CFO
%   5 = OFDM Demodulation
%   6 = Channel Estimation
%   7 = Equalization
%   8 = 4096-QAM Demapping
%   9 = LLR Generation
%  10 = LDPC Decoding
%  11 = Descrambling
%  12 = Payload Recovery

% Terminal-only timing/overlap diagnostics. These are  not
% added to any CSV.
gpuLoopStart = NaT(numBatches,1);
gpuLoopEnd = NaT(numBatches,1);
gpuBufferEnter = NaT(numBatches,1);
gpuBufferExit = NaT(numBatches,1);
gpuBufferForBatch = zeros(numBatches,1);

% FIFO metadata for batches currently in flight.
activeTickets = zeros(1,maxInFlight);
activeBatchNumbers = zeros(1,maxInFlight);
activeBufferNumbers = zeros(1,maxInFlight);
activeCount = 0;
peakInFlight = 0;

gpuProcessingStart = datetime('now');
fprintf('\n[GPU PROCESSING START] %s | Packets: %d | Batches: %d | Batch size: %d | Buffers: %d\n', ...
    timestampTextLocal(gpuProcessingStart),numPackets,numBatches,batchSize,numGPUBuffers);

gpuWallTimer = tic;

for b = 1:numBatches

    % If every GPU buffer is occupied, collect the oldest batch. This waits
    % only for that one buffer; work already queued in the other streams is
    % allowed to continue.
    if activeCount == maxInFlight
        completedBatch = activeBatchNumbers(1);
        expectedBuffer = activeBufferNumbers(1);

        stats = eht_receiver_cuda_e2e_batched_mex( ...
            'e2eBatchCollect',activeTickets(1));

        actualBuffer = double(stats.BufferIndex);
        assert(actualBuffer == expectedBuffer, ...
            'GPU buffer assignment mismatch for batch %d.',completedBatch);

        completionTime = datetime('now');
        gpuBufferExit(completedBatch) = completionTime;
        gpuLoopEnd(completedBatch) = completionTime;

        fprintf('[BUFFER %d EXIT]      %s | GPU loop %d/%d | packets %d-%d\n', ...
            actualBuffer,timestampTextLocal(completionTime),completedBatch,numBatches, ...
            (completedBatch-1)*batchSize+1,min(completedBatch*batchSize,numPackets));
        fprintf('[GPU LOOP %d/%d END]   %s | Buffer %d | elapsed in buffer: %.3f ms\n', ...
            completedBatch,numBatches,timestampTextLocal(completionTime),actualBuffer, ...
            1000*seconds(gpuBufferExit(completedBatch)-gpuBufferEnter(completedBatch)));

        gpuChecksum = gpuChecksum + sum(double(stats.Checksums(:)));
        processed = processed + double(stats.PacketsProcessed);

        gpuStageTotal_ms = gpuStageTotal_ms + [ ...
            double(stats.PacketDetectionTimeMs), ...
            double(stats.CoarseCFOTimeMs), ...
            double(stats.TimingSynchronizationTimeMs), ...
            double(stats.FineCFOTimeMs), ...
            double(stats.OFDMDemodulationTimeMs), ...
            double(stats.ChannelEstimationTimeMs), ...
            double(stats.EqualizationTimeMs), ...
            double(stats.QAMDemappingTimeMs), ...
            double(stats.LLRGenerationTimeMs), ...
            double(stats.LDPCDecodingTimeMs), ...
            double(stats.DescramblingTimeMs), ...
            double(stats.PayloadRecoveryTimeMs)];

        if activeCount > 1
            activeTickets(1:activeCount-1) = activeTickets(2:activeCount);
            activeBatchNumbers(1:activeCount-1) = activeBatchNumbers(2:activeCount);
            activeBufferNumbers(1:activeCount-1) = activeBufferNumbers(2:activeCount);
        end
        activeTickets(activeCount) = 0;
        activeBatchNumbers(activeCount) = 0;
        activeBufferNumbers(activeCount) = 0;
        activeCount = activeCount - 1;
    end

    first = (b-1)*batchSize + 1;
    last = min(b*batchSize,numPackets);
    idx = sourceMap(first:last);

    % Batch assembly remains identical to the existing benchmark.
    rxI = database.RxI(:,idx);
    rxQ = database.RxQ(:,idx);
    rxScale = database.RxScale(idx);

    % Because the MEX selects the first free slot and MATLAB frees slots in
    % FIFO order, the physical buffers rotate 1,2,3,4,1,2,3,4 and so on
    expectedBuffer = mod(b-1,numGPUBuffers) + 1;

    startTime = datetime('now');
    gpuLoopStart(b) = startTime;
    gpuBufferEnter(b) = startTime;
    gpuBufferForBatch(b) = expectedBuffer;

    fprintf('\n[GPU LOOP %d/%d START] %s | packets %d-%d | Buffer %d\n', ...
        b,numBatches,timestampTextLocal(startTime),first,last,expectedBuffer);
    fprintf('[BUFFER %d ENTER]     %s | GPU loop %d/%d | packets %d-%d\n', ...
        expectedBuffer,timestampTextLocal(startTime),b,numBatches,first,last);

    ticket = eht_receiver_cuda_e2e_batched_mex( ...
        'e2eBatchSubmit',rxI,rxQ,rxScale,opts.LLRScale,lltfReference, ...
        ltfFFTIndices,dataFFTIndices,ltfFFTStart,dataFFTStarts,knownLTF, ...
        pilotIndices,pilotReference,dataIndices);

    activeCount = activeCount + 1;
    activeTickets(activeCount) = double(ticket);
    activeBatchNumbers(activeCount) = b;
    activeBufferNumbers(activeCount) = expectedBuffer;
    peakInFlight = max(peakInFlight,activeCount);

    submitReturnTime = datetime('now');
    fprintf('[BUFFER %d SUBMITTED] %s | ticket %.0f | in flight: %d/%d\n', ...
        expectedBuffer,timestampTextLocal(submitReturnTime),double(ticket), ...
        activeCount,numGPUBuffers);
end

% Drain any final outstanding batches.
while activeCount > 0
    completedBatch = activeBatchNumbers(1);
    expectedBuffer = activeBufferNumbers(1);

    stats = eht_receiver_cuda_e2e_batched_mex( ...
        'e2eBatchCollect',activeTickets(1));

    actualBuffer = double(stats.BufferIndex);
    assert(actualBuffer == expectedBuffer, ...
        'GPU buffer assignment mismatch for batch %d.',completedBatch);

    completionTime = datetime('now');
    gpuBufferExit(completedBatch) = completionTime;
    gpuLoopEnd(completedBatch) = completionTime;

    fprintf('[BUFFER %d EXIT]      %s | GPU loop %d/%d | packets %d-%d\n', ...
        actualBuffer,timestampTextLocal(completionTime),completedBatch,numBatches, ...
        (completedBatch-1)*batchSize+1,min(completedBatch*batchSize,numPackets));
    fprintf('[GPU LOOP %d/%d END]   %s | Buffer %d | elapsed in buffer: %.3f ms\n', ...
        completedBatch,numBatches,timestampTextLocal(completionTime),actualBuffer, ...
        1000*seconds(gpuBufferExit(completedBatch)-gpuBufferEnter(completedBatch)));

    gpuChecksum = gpuChecksum + sum(double(stats.Checksums(:)));
    processed = processed + double(stats.PacketsProcessed);

    gpuStageTotal_ms = gpuStageTotal_ms + [ ...
        double(stats.PacketDetectionTimeMs), ...
        double(stats.CoarseCFOTimeMs), ...
        double(stats.TimingSynchronizationTimeMs), ...
        double(stats.FineCFOTimeMs), ...
        double(stats.OFDMDemodulationTimeMs), ...
        double(stats.ChannelEstimationTimeMs), ...
        double(stats.EqualizationTimeMs), ...
        double(stats.QAMDemappingTimeMs), ...
        double(stats.LLRGenerationTimeMs), ...
        double(stats.LDPCDecodingTimeMs), ...
        double(stats.DescramblingTimeMs), ...
        double(stats.PayloadRecoveryTimeMs)];

    if activeCount > 1
        activeTickets(1:activeCount-1) = activeTickets(2:activeCount);
        activeBatchNumbers(1:activeCount-1) = activeBatchNumbers(2:activeCount);
        activeBufferNumbers(1:activeCount-1) = activeBufferNumbers(2:activeCount);
    end
    activeTickets(activeCount) = 0;
    activeBatchNumbers(activeCount) = 0;
    activeBufferNumbers(activeCount) = 0;
    activeCount = activeCount - 1;
end

gpuWall_s = toc(gpuWallTimer);
gpuProcessingEnd = datetime('now');

fprintf('\n[GPU PROCESSING END]   %s | Elapsed: %.6f s\n', ...
    timestampTextLocal(gpuProcessingEnd),gpuWall_s);

fprintf('\n============================================================\n');
fprintf('GPU 4-BUFFER OVERLAP CHECK\n');
fprintf('============================================================\n');
fprintf('Peak batches simultaneously in flight : %d\n',peakInFlight);
fprintf('Configured GPU buffers                : %d\n',numGPUBuffers);

for b = 1:numBatches-1
    overlap = gpuLoopStart(b+1) < gpuLoopEnd(b);
    fprintf('Loop %d START before Loop %d END       : %s\n', ...
        b+1,b,yesNoLocal(overlap));
end

if numBatches >= numGPUBuffers
    fprintf('All four buffers simultaneously in flight: %s\n', ...
        yesNoLocal(peakInFlight >= numGPUBuffers));
else
    fprintf('All available batches simultaneously in flight: %s (%d batch(es))\n', ...
        yesNoLocal(peakInFlight >= numBatches),numBatches);
end
fprintf('============================================================\n');

assert(processed == numPackets,'GPU packet count mismatch.');
assert(gpuChecksum == expectedAggregateChecksum,'Timed GPU checksum mismatch.');

gpuStageMean_ms_per_packet = gpuStageTotal_ms / numPackets;

%% Metrics
payloadMegabits = numPackets*payloadBitsPerPacket/1e6;
cpuLatencyMs = 1000*cpuWall_s/numPackets;
gpuLatencyMs = 1000*gpuWall_s/numPackets;
cpuPacketsPerSecond = numPackets/cpuWall_s;
gpuPacketsPerSecond = numPackets/gpuWall_s;
cpuPayloadMbps = payloadMegabits/cpuWall_s;
gpuPayloadMbps = payloadMegabits/gpuWall_s;
speedup = cpuWall_s/gpuWall_s;
timeReduction = 100*(1-gpuWall_s/cpuWall_s);

results = struct;
results.Packets = numPackets;
results.UniquePackets = uniquePackets;
results.PayloadBitsPerPacket = payloadBitsPerPacket;
results.CPUExecutionMode = opts.CPUExecutionMode;
results.NumWorkers = actualWorkers;
results.GPUBatchSize = batchSize;
results.GPUNumBatches = numBatches;
results.CPUWallSeconds = cpuWall_s;
results.GPUWallSeconds = gpuWall_s;
results.CPULatencyMsPerPacket = cpuLatencyMs;
results.GPULatencyMsPerPacket = gpuLatencyMs;
results.CPUPacketsPerSecond = cpuPacketsPerSecond;
results.GPUPacketsPerSecond = gpuPacketsPerSecond;
results.CPUPayloadMbps = cpuPayloadMbps;
results.GPUPayloadMbps = gpuPayloadMbps;
results.Speedup = speedup;
results.TimeReductionPercent = timeReduction;
results.CPUAggregateChecksum = cpuAggregateChecksum;
results.GPUAggregateChecksum = gpuChecksum;
results.ChecksumMatch = cpuAggregateChecksum == gpuChecksum;

% STEP 5 CPU profiling results. Kept out of the existing summary CSV for
% now; the dedicated stage-timing CSV will be added after GPU profiling.
results.CPUStageNames = [ ...
    "Packet Detection","Coarse CFO","Timing Synchronization","Fine CFO", ...
    "OFDM Demodulation","Channel Estimation","Equalization", ...
    "4096-QAM Demapping","LLR Generation","LDPC Decoding", ...
    "Descrambling","Payload Recovery"];
results.CPUStageTimingPerPacket_s = cpuStageTiming_s;
results.CPUStageMean_ms_per_packet = cpuStageMean_ms;
results.CPUStageTotal_s = cpuStageTotal_s;
results.CPULDPCStageUsesCUDAMEX = false;

% GPU full receiver CUDA-event profiling results.
results.GPUStageNames = [ ...
    "Packet Detection","Coarse CFO","Timing Synchronization","Fine CFO", ...
    "OFDM Demodulation","Channel Estimation","Equalization", ...
    "4096-QAM Demapping","LLR Generation","LDPC Decoding", ...
    "Descrambling","Payload Recovery"];
results.GPUStageTotal_ms = gpuStageTotal_ms;
results.GPUStageMean_ms_per_packet = gpuStageMean_ms_per_packet;

% Dedicated 12-stage timing table which will be exported to csv.
% One row per locked receiver stage. Graphs and existing summary tables are
% intentionally unchanged.
stageNames = results.CPUStageNames(:);
cpuTotal_ms = 1000*cpuStageTotal_s(:);
cpuMean_ms = cpuStageMean_ms(:);
gpuTotal_ms_col = gpuStageTotal_ms(:);
gpuMean_ms = gpuStageMean_ms_per_packet(:);
ldpcUsesCudaMex = false(12,1);
ldpcUsesCudaMex(10) = results.CPULDPCStageUsesCUDAMEX;

stageTimingTable = table( ...
    repmat(numPackets,12,1), ...
    repmat(string(opts.CPUExecutionMode),12,1), ...
    repmat(actualWorkers,12,1), ...
    repmat(batchSize,12,1), ...
    repmat(numBatches,12,1), ...
    stageNames, ...
    cpuTotal_ms, ...
    cpuMean_ms, ...
    gpuTotal_ms_col, ...
    gpuMean_ms, ...
    ldpcUsesCudaMex, ...
    'VariableNames',{ ...
    'Packets','CPUExecutionMode','NumWorkers','GPU_BatchSize','GPU_NumBatches', ...
    'Stage','CPU_Total_ms','CPU_ms_per_packet','GPU_Total_ms','GPU_ms_per_packet', ...
    'CPU_Stage_Uses_CUDA_MEX'});
results.StageTimingTable = stageTimingTable;

resultTable = table(numPackets,opts.CPUExecutionMode,actualWorkers,batchSize,numBatches, ...
    cpuWall_s,gpuWall_s,cpuLatencyMs,gpuLatencyMs,cpuPacketsPerSecond, ...
    gpuPacketsPerSecond,cpuPayloadMbps,gpuPayloadMbps,speedup,timeReduction, ...
    results.ChecksumMatch,'VariableNames',{ ...
    'Packets','CPUExecutionMode','NumWorkers','GPU_BatchSize','GPU_NumBatches', ...
    'CPU_Wall_s','GPU_Wall_s','CPU_Latency_ms_per_packet','GPU_Latency_ms_per_packet', ...
    'CPU_Packets_per_s','GPU_Packets_per_s','CPU_Payload_Mbps','GPU_Payload_Mbps', ...
    'Speedup_x','TimeReduction_pct','ChecksumMatch'});
results.ResultTable = resultTable;

fprintf('\n%d-packet result: CPU %.3f s, GPU %.3f s, speedup %.3fx\n', ...
    numPackets,cpuWall_s,gpuWall_s,speedup);
fprintf('GPU batch size: %d | submissions: %d | GPU throughput: %.3f Mbps\n', ...
    batchSize,numBatches,gpuPayloadMbps);

if opts.SaveIndividualResult
    resultsDir = fullfile(projectDir,'results');
    if ~isfolder(resultsDir), mkdir(resultsDir); end
    timestamp = char(datetime('now','Format','yyyyMMdd_HHmmss'));
    stem = sprintf('complete_batched_e2e_%d_B%d_%s_%s',numPackets,batchSize, ...
        char(opts.CPUExecutionMode),timestamp);
    matPath = fullfile(resultsDir,[stem '.mat']);
    csvPath = fullfile(resultsDir,[stem '.csv']);
    stageCsvPath = fullfile(resultsDir,[stem '_stage_timings.csv']);
    save(matPath,'results','resultTable','stageTimingTable','opts','-v7.3');
    writetable(resultTable,csvPath);
    writetable(stageTimingTable,stageCsvPath);
    results.ResultsMATPath = matPath;
    results.ResultsCSVPath = csvPath;
    results.StageTimingCSVPath = stageCsvPath;
end
end

function [rxSynchronized,rxEHTLTF,rxEHTData,noiseVariance,stageTiming] = synchronizePacketLocal( ...
    rxI,rxQ,rxScale,cfgEHT,sampleRate,fieldIndices,packetLengthSamples,channelBandwidth)
%SYNCHRONIZE PACKET LOCAL CPU front-end with Step-2 stage profiling.

stageTiming = struct( ...
    'PacketDetection_s',0, ...
    'CoarseCFO_s',0, ...
    'TimingSynchronization_s',0, ...
    'FineCFO_s',0);

% INT16 I/Q -> single-complex conversion remains outside the 12 locked
% receiver timing stages.
rxWaveform = complex(single(rxI),single(rxQ))./single(rxScale);

%% 1. Packet Detection
tStage = tic;
coarsePacketOffset = wlanPacketDetect(rxWaveform,channelBandwidth);
stageTiming.PacketDetection_s = toc(tStage);

if isempty(coarsePacketOffset)
    error('wifi7cuda:PacketDetectionFailure','Packet detection failed.');
end
rxDetected = rxWaveform(coarsePacketOffset+1:end,:);
rxLSTF = rxDetected(fieldIndices.LSTF(1):fieldIndices.LSTF(2),:);

%% 2. Coarse CFO - estimation + first correction
tStage = tic;
coarseCFOHz = wlanCoarseCFOEstimate(rxLSTF,channelBandwidth);
rxCoarseCorrected = applyFrequencyOffsetLocal(rxDetected,sampleRate,-coarseCFOHz);
stageTiming.CoarseCFO_s = toc(tStage);

%% 3. Timing Synchronization
tStage = tic;
legacyPreamble = rxCoarseCorrected(fieldIndices.LSTF(1):fieldIndices.LSIG(2),:);
fineTimingOffset = wlanSymbolTimingEstimate(legacyPreamble,channelBandwidth);
finalPacketOffset = coarsePacketOffset+fineTimingOffset;
rxTimed = rxWaveform(finalPacketOffset+1:end,:);
if size(rxTimed,1) < packetLengthSamples
    error('wifi7cuda:InsufficientSamples','Insufficient samples after synchronization.');
end
rxTimed = rxTimed(1:packetLengthSamples,:);
stageTiming.TimingSynchronization_s = toc(tStage);

% The original receiver performs a second coarse-CFO correction after the
% final timing offset is known. Count that work in the locked Coarse CFO
% category without changing execution order.
tStage = tic;
rxTimed = applyFrequencyOffsetLocal(rxTimed,sampleRate,-coarseCFOHz);
stageTiming.CoarseCFO_s = stageTiming.CoarseCFO_s + toc(tStage);

%% 4. Fine CFO - estimation + correction
tStage = tic;
rxLLTF = rxTimed(fieldIndices.LLTF(1):fieldIndices.LLTF(2),:);
fineCFOHz = wlanFineCFOEstimate(rxLLTF,channelBandwidth);
rxSynchronized = applyFrequencyOffsetLocal(rxTimed,sampleRate,-fineCFOHz);
stageTiming.FineCFO_s = toc(tStage);

%% Existing post-synchronization work
% Noise estimation and field extraction are intentionally not assigned to
% a Step-2 stage yet. They remain exactly where they were in the receiver.
rxLLTF = rxSynchronized(fieldIndices.LLTF(1):fieldIndices.LLTF(2),:);
lltfDemod = wlanEHTDemodulate(rxLLTF,'L-LTF',cfgEHT);
noiseVariance = single(wlanLLTFNoiseEstimate(lltfDemod));
rxEHTLTF = single(rxSynchronized(fieldIndices.EHTLTF(1):fieldIndices.EHTLTF(2),:));
rxEHTData = single(rxSynchronized(fieldIndices.EHTData(1):fieldIndices.EHTData(2),:));
end

function [recoveredBits,stageTiming] = recoverCPUFromSynchronizedLocal( ...
    rxSynchronized,noiseVariance,cfgEHT,fieldIndices)
%RECOVERCPUFROMSYNCHRONIZEDLOCAL Corrected Step-5 backend profiling.
% Stages 5-7 retain the validated Step-4 receiver operations. Stages 8-12
% use the project's existing external-LLR profiled continuation rather than
% replacing the decoder algorithm.

stageTiming = struct( ...
    'OFDMDemodulation_s',0, ...
    'ChannelEstimation_s',0, ...
    'Equalization_s',0, ...
    'QAM4096Demapping_s',0, ...
    'LLRGeneration_s',0, ...
    'LDPCDecoding_s',0, ...
    'Descrambling_s',0, ...
    'PayloadRecovery_s',0);

%% 5. OFDM Demodulation - EHT-LTF extraction + demodulation
tStage = tic;
rxEHTLTF = rxSynchronized(fieldIndices.EHTLTF(1):fieldIndices.EHTLTF(2),:);
ehtLTFDemod = wlanEHTDemodulate(rxEHTLTF,'EHT-LTF',cfgEHT);
stageTiming.OFDMDemodulation_s = toc(tStage);

%% 6. Channel Estimation
tStage = tic;
channelEstimate = wlanEHTLTFChannelEstimate(ehtLTFDemod,cfgEHT);
stageTiming.ChannelEstimation_s = toc(tStage);

%% 5. OFDM Demodulation - EHT-Data extraction + demodulation
tStage = tic;
rxEHTData = rxSynchronized(fieldIndices.EHTData(1):fieldIndices.EHTData(2),:);
dataDemod = wlanEHTDemodulate(rxEHTData,'EHT-Data',cfgEHT);
stageTiming.OFDMDemodulation_s = stageTiming.OFDMDemodulation_s + toc(tStage);

%% 7. Equalization - pilot tracking + equalization
tStage = tic;
trackedData = wlanEHTTrackPilotError(dataDemod,channelEstimate,cfgEHT,'EHT-Data');
[equalizedAll,csiAll] = wlanEHTEqualize( ...
    trackedData,channelEstimate,noiseVariance,cfgEHT,'EHT-Data',1);
stageTiming.Equalization_s = toc(tStage);

dataInfo = wlanEHTOFDMInfo('EHT-Data',cfgEHT);
equalizedData = equalizedAll(dataInfo.DataIndices,:,:);
csiData = csiAll(dataInfo.DataIndices,:);

% Build the exact EHT demapper/coding metadata used by the project's
% external-LLR continuation. This metadata preparation is deliberately
% outside the locked 4096-QAM constellation-demapping timer.
prep = wlanEHTPrepareDemapperInput( ...
    equalizedData,noiseVariance,csiData,cfgEHT,1, ...
    LDPCDecodingMethod='norm-min-sum', ...
    MinSumScalingFactor=0.75, ...
    MaximumLDPCIterationCount=12, ...
    EarlyTermination=false);

%% 8. 4096-QAM Demapping
% Generate soft max-log LLRs from the prepared EHT data symbols. CSI
% weighting is intentionally excluded here and counted in Stage 9.
tStage = tic;
externalLLR = cell(1,prep.NumSegments);
for l = 1:prep.NumSegments
    % Use the WLAN Toolbox constellation demapper, not generic qamdemod.
    % wlanConstellationDemap reproduces that WLAN-specific soft-LLR ordering.
    externalLLR{l} = wlanConstellationDemap( ...
        prep.DataSymbols{l},noiseVariance, ...
        prep.UserParameters.NBPSCS,'soft', ...
        OutputDataType='single');
end
stageTiming.QAM4096Demapping_s = toc(tStage);

%% 9-12. Use the project's existing validated profiled continuation
[recoveredBits,postTiming] = wlanEHTRecoverFromExternalLLRProfiled( ...
    prep,externalLLR);

% Locked Stage 9 = all LLR-domain processing between soft demapping and
% channel decode in the helper.
stageTiming.LLRGeneration_s = ...
    postTiming.CSIWeighting_s + ...
    postTiming.BCCDeinterleave_s + ...
    postTiming.SegmentDeparse_s + ...
    postTiming.StreamDeparse_s + ...
    postTiming.RemovePostFECPadding_s;

% Locked Stages 10-12 map directly onto existing profiled helper fields.
stageTiming.LDPCDecoding_s = postTiming.ChannelDecode_s;
stageTiming.Descrambling_s = postTiming.Descramble_s;
stageTiming.PayloadRecovery_s = postTiming.FinalBitExtraction_s;
end

function checksum = checksumLocal(bits,payloadBitsPerPacket)
bits = double(bits(1:payloadBitsPerPacket));
checksum = sum((1:payloadBitsPerPacket)'.*bits);
end

function y = applyFrequencyOffsetLocal(x,sampleRate,frequencyOffsetHz)
n = single((0:size(x,1)-1).');
phase = single(2*pi*frequencyOffsetHz/sampleRate).*n;
y = x .* complex(cos(phase),sin(phase));
end

function s = timestampTextLocal(t)
t.Format = 'HH:mm:ss.SSSSSS';
s = char(t);
end

function s = yesNoLocal(tf)
if tf
    s = 'YES';
else
    s = 'NO';
end
end

