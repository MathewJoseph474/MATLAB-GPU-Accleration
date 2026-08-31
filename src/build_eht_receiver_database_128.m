function databasePath = build_eht_receiver_database_128()
%BUILD_EHT_RECEIVER_DATABASE_128
% Create a permanent database of 128 independently generated Wi-Fi 7 EHT
% packets for CPU/GPU receiver performance benchmarking.
%
% Each packet includes the complete validated transmitter/channel chain:
%   Random PSDU bits
%   EHT waveform generation
%   Scrambling, LDPC, 4096-QAM, OFDM and preamble generation
%   Timing offset
%   Carrier-frequency offset
%   AWGN
%   INT16 I/Q quantization
%
% Larger receiver workloads reuse these 128 packets logically:
%   sourceIndex = mod(logicalPacketIndex-1,128) + 1;
%
% The database is generated in small batches and written to a Version 7.3
% MAT-file. Parallel CPU generation is used when available.

    clc;

   %% Project location

srcDir = fileparts(mfilename("fullpath"));
projectDir = fileparts(srcDir);
configDir = fullfile(projectDir,'config');
addpath(configDir);
cfg = project_config(projectDir);

databaseDir = fullfile(projectDir,'database');
databasePath = fullfile(databaseDir,'eht_receiver_database_128.mat');

    if ~isfolder(databaseDir)
        mkdir(databaseDir);
    end

    if isfile(databasePath)
        messageText = sprintf([ ...
            'Database already exists:\n%s\n\n' ...
            'It was not overwritten. Delete or rename it deliberately ' ...
            'before regenerating.'],databasePath);
        error('wifi7cuda:DatabaseAlreadyExists','%s',messageText);
    end

    %% Generation settings
    opts = struct;
    opts.NumPackets = 128;
    opts.GenerationBatchPackets = 8;
    opts.UseParallel = true;
    opts.RequestedWorkers = 4;

    opts.ChannelBandwidth = 'CBW320';
    opts.MCS = 13;
    opts.APEPLengthBytes = cfg.PacketSize.BitsPerPacket / 8;
    opts.ChannelCoding = 'LDPC';
    opts.NumSpaceTimeStreams = 1;

    opts.SNRdB = 42;
    opts.InsertedTimingOffsetSamples = 256;
    opts.InsertedCFOHz = 80e3;
    opts.RandomSeed = 20260731;

    opts.InputStorageClass = 'int16';
    opts.ReceiverProcessingClass = 'single';
    opts.DatabaseVersion = '1.0';

    %% EHT configuration
    cfgEHT = wlanEHTMUConfig(opts.ChannelBandwidth);
    cfgEHT.User{1}.MCS = opts.MCS;
    cfgEHT.User{1}.APEPLength = opts.APEPLengthBytes;
    cfgEHT.User{1}.ChannelCoding = opts.ChannelCoding;
    cfgEHT.User{1}.NumSpaceTimeStreams = opts.NumSpaceTimeStreams;

    sampleRate = wlanSampleRate(cfgEHT);
    fieldIndices = wlanFieldIndices(cfgEHT);
    psduBytes = psduLength(cfgEHT);
    bitsPerPacket = 8*psduBytes(1);

    %% Configure optional parallel generation
    useParallel = false;
    pool = [];

    if opts.UseParallel && license('test','Distrib_Computing_Toolbox')
        try
            pool = gcp('nocreate');

            if isempty(pool)
                pool = parpool('Processes',opts.RequestedWorkers);
            elseif ~isa(pool,'parallel.ProcessPool')
                delete(pool);
                pool = parpool('Processes',opts.RequestedWorkers);
            end

            useParallel = true;
        catch parallelError
            warning('wifi7cuda:ParallelDisabled', ...
                'Parallel generation unavailable. Using serial generation.\n%s', ...
                parallelError.message);
            useParallel = false;
            pool = [];
        end
    end

    fprintf('128-packet EHT database generation\n');
    fprintf('Bandwidth           : %s\n',opts.ChannelBandwidth);
    fprintf('MCS                 : %d (4096-QAM)\n',opts.MCS);
    fprintf('APEP length         : %d bytes\n',opts.APEPLengthBytes);
    fprintf('Channel coding      : %s\n',opts.ChannelCoding);
    fprintf('Parallel generation : %s\n',onOffLocal(useParallel));

    if useParallel
        fprintf('Parallel workers    : %d\n',pool.NumWorkers);
    end

    fprintf('\n');

    %% Generate first packet to establish fixed dimensions
    fprintf('Generating first packet to establish dimensions...\n');

    firstPacket = generateOnePacketLocal( ...
        1,cfgEHT,sampleRate,bitsPerPacket,opts);

    samplesPerPacket = numel(firstPacket.RxI);

    %% Metadata
    metadata = struct;
    metadata.DatabaseVersion = opts.DatabaseVersion;
    metadata.Created = char(datetime( ...
        'now','Format','yyyy-MM-dd HH:mm:ss Z'));
    metadata.NumUniquePackets = opts.NumPackets;
    metadata.SamplesPerPacket = samplesPerPacket;
    metadata.BitsPerPacket = bitsPerPacket;
    metadata.SampleRateHz = sampleRate;
    metadata.ChannelBandwidth = opts.ChannelBandwidth;
    metadata.MCS = opts.MCS;
    metadata.Modulation = '4096-QAM';
    metadata.APEPLengthBytes = opts.APEPLengthBytes;
    metadata.PSDULengthBytes = psduBytes(1);
    metadata.ChannelCoding = opts.ChannelCoding;
    metadata.NumSpaceTimeStreams = opts.NumSpaceTimeStreams;
    metadata.SNRdB = opts.SNRdB;
    metadata.InsertedTimingOffsetSamples = ...
        opts.InsertedTimingOffsetSamples;
    metadata.InsertedCFOHz = opts.InsertedCFOHz;
    metadata.RandomSeed = opts.RandomSeed;
    metadata.InputStorageClass = opts.InputStorageClass;
    metadata.ReceiverProcessingClass = ...
        opts.ReceiverProcessingClass;
    metadata.FieldIndices = fieldIndices;
    metadata.GenerationComplete = false;
    metadata.CompletedPackets = 0;
    metadata.ReuseFormula = ...
        'sourceIndex = mod(logicalPacketIndex-1,128) + 1';

    save(databasePath,'cfgEHT','opts','metadata','-v7.3');
    database = matfile(databasePath,'Writable',true);

    %% Disk-backed preallocation
    database.RxI(samplesPerPacket,opts.NumPackets) = int16(0);
    database.RxQ(samplesPerPacket,opts.NumPackets) = int16(0);
    database.RxScale(1,opts.NumPackets) = single(0);
    database.TxBits(bitsPerPacket,opts.NumPackets) = int8(0);
    database.PacketSignalPower(1,opts.NumPackets) = single(0);
    database.PacketNoiseVariance(1,opts.NumPackets) = single(0);

    writePacketLocal(database,1,firstPacket);

    %% Generate remaining packets in batches
    generationTimer = tic;
    firstPendingPacket = 2;

    for batchStart = firstPendingPacket: ...
            opts.GenerationBatchPackets:opts.NumPackets

        batchEnd = min( ...
            batchStart+opts.GenerationBatchPackets-1,opts.NumPackets);

        packetIndices = batchStart:batchEnd;
        packets = cell(numel(packetIndices),1);

        batchTimer = tic;

        if useParallel
            parfor localIndex = 1:numel(packetIndices)
                packetIndex = packetIndices(localIndex);
                packets{localIndex} = generateOnePacketLocal( ...
                    packetIndex,cfgEHT,sampleRate,bitsPerPacket,opts);
            end
        else
            for localIndex = 1:numel(packetIndices)
                packetIndex = packetIndices(localIndex);
                packets{localIndex} = generateOnePacketLocal( ...
                    packetIndex,cfgEHT,sampleRate,bitsPerPacket,opts);
            end
        end

        % Write one contiguous block per variable.
        blockCount = numel(packetIndices);
        blockI = zeros(samplesPerPacket,blockCount,'int16');
        blockQ = zeros(samplesPerPacket,blockCount,'int16');
        blockScale = zeros(1,blockCount,'single');
        blockBits = zeros(bitsPerPacket,blockCount,'int8');
        blockSignalPower = zeros(1,blockCount,'single');
        blockNoiseVariance = zeros(1,blockCount,'single');

        for localIndex = 1:blockCount
            packet = packets{localIndex};

            if numel(packet.RxI) ~= samplesPerPacket
                messageText = sprintf([ ...
                    'Packet %d contains %d samples; expected %d. ' ...
                    'The fixed-size database cannot continue.'], ...
                    packetIndices(localIndex), ...
                    numel(packet.RxI),samplesPerPacket);
                error('wifi7cuda:VariablePacketLength','%s',messageText);
            end

            blockI(:,localIndex) = packet.RxI;
            blockQ(:,localIndex) = packet.RxQ;
            blockScale(localIndex) = packet.RxScale;
            blockBits(:,localIndex) = packet.TxBits;
            blockSignalPower(localIndex) = packet.SignalPower;
            blockNoiseVariance(localIndex) = packet.NoiseVariance;
        end

        database.RxI(:,packetIndices) = blockI;
        database.RxQ(:,packetIndices) = blockQ;
        database.RxScale(1,packetIndices) = blockScale;
        database.TxBits(:,packetIndices) = blockBits;
        database.PacketSignalPower(1,packetIndices) = blockSignalPower;
        database.PacketNoiseVariance(1,packetIndices) = blockNoiseVariance;

        elapsed = toc(generationTimer);
        batchElapsed = toc(batchTimer);
        completedPackets = batchEnd;
        generatedAfterFirst = completedPackets-1;
        cumulativeRate = generatedAfterFirst/max(elapsed,eps);
        batchRate = blockCount/max(batchElapsed,eps);
        remaining = opts.NumPackets-completedPackets;
        etaSeconds = remaining/max(cumulativeRate,eps);

        metadata.CompletedPackets = completedPackets;
        metadata.GenerationElapsedSeconds = elapsed;
        database.metadata = metadata;

        fprintf([ ...
            'Packet %3d / %3d | batch %.2f pkt/s | average %.2f pkt/s | ' ...
            'ETA %.1f min\n'], ...
            completedPackets,opts.NumPackets,batchRate,cumulativeRate, ...
            etaSeconds/60);

        clear packets blockI blockQ blockScale blockBits
        clear blockSignalPower blockNoiseVariance
    end

    %% Finalize
    metadata.GenerationComplete = true;
    metadata.CompletedPackets = opts.NumPackets;
    metadata.GenerationElapsedSeconds = toc(generationTimer);
    metadata.DatabasePath = databasePath;
    database.metadata = metadata;

    fprintf('\n128-packet EHT database completed successfully.\n');
    fprintf('Location           : %s\n',databasePath);
    fprintf('Unique packets     : %d\n',opts.NumPackets);
    fprintf('Samples per packet : %d\n',samplesPerPacket);
    fprintf('Bits per packet    : %d\n',bitsPerPacket);
    fprintf('Generation time    : %.2f minutes\n', ...
        metadata.GenerationElapsedSeconds/60);
    fprintf('\nLogical 8192-packet workload:\n');
    fprintf('128 unique packets repeated 64 times.\n');
end


function packet = generateOnePacketLocal( ...
        packetIndex,cfgEHT,sampleRate,bitsPerPacket,opts)

    % Packet-specific deterministic random stream.
    rng(opts.RandomSeed+packetIndex-1,'twister');

    %% Complete transmitter
    txBits = randi([0 1],bitsPerPacket,1,'int8');
    txWaveform = single(wlanWaveformGenerator(txBits,cfgEHT));

    %% Timing offset
    timingOffset = opts.InsertedTimingOffsetSamples;

    delayedWaveform = [
        complex(zeros(timingOffset,size(txWaveform,2),'single'));
        txWaveform
    ];

    %% Carrier-frequency offset
    impairedWaveform = applyFrequencyOffsetLocal( ...
        delayedWaveform,sampleRate,opts.InsertedCFOHz);

    %% AWGN
    usefulWaveform = impairedWaveform(timingOffset+1:end,:);
    signalPower = mean(abs(usefulWaveform).^2,'all');
    noiseVariance = signalPower/10^(opts.SNRdB/10);

    noise = sqrt(noiseVariance/2).*complex( ...
        randn(size(impairedWaveform),'single'), ...
        randn(size(impairedWaveform),'single'));

    rxWaveform = impairedWaveform+noise;

    %% Device-style INT16 I/Q storage
    [rxI,rxQ,rxScale] = quantizeComplexInt16Local(rxWaveform);

    packet = struct;
    packet.RxI = rxI(:);
    packet.RxQ = rxQ(:);
    packet.RxScale = rxScale;
    packet.TxBits = txBits(:);
    packet.SignalPower = single(signalPower);
    packet.NoiseVariance = single(noiseVariance);
end


function writePacketLocal(database,packetIndex,packet)
    database.RxI(:,packetIndex) = packet.RxI;
    database.RxQ(:,packetIndex) = packet.RxQ;
    database.RxScale(1,packetIndex) = packet.RxScale;
    database.TxBits(:,packetIndex) = packet.TxBits;
    database.PacketSignalPower(1,packetIndex) = packet.SignalPower;
    database.PacketNoiseVariance(1,packetIndex) = packet.NoiseVariance;
end


function [i16,q16,scale] = quantizeComplexInt16Local(x)
    peak = max( ...
        max(abs(real(x)),[],'all'), ...
        max(abs(imag(x)),[],'all'));

    if peak == 0
        scale = single(1);
    else
        scale = single(0.90*double(intmax('int16'))) / single(peak);
    end

    lowerLimit = single(intmin('int16'));
    upperLimit = single(intmax('int16'));

    i16 = int16(min(max( ...
        round(real(x).*scale),lowerLimit),upperLimit));

    q16 = int16(min(max( ...
        round(imag(x).*scale),lowerLimit),upperLimit));
end


function y = applyFrequencyOffsetLocal( ...
        x,sampleRate,frequencyOffsetHz)

    sampleIndices = single((0:size(x,1)-1).');
    phase = single(2*pi*frequencyOffsetHz/sampleRate).*sampleIndices;
    y = x.*exp(1j*phase);
end


function textValue = onOffLocal(tf)
    if tf
        textValue = 'ON';
    else
        textValue = 'OFF';
    end
end
