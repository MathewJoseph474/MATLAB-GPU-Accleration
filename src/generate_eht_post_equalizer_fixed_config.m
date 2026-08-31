function headerPath = generate_eht_post_equalizer_fixed_config(projectDir)
%GENERATE_EHT_POST_EQUALIZER_FIXED_CONFIG
% Regenerate the CUDA post-equalizer/LDPC fixed-configuration header from
% the packet size selected in config/project_config.m.
%
% Only packet-size-dependent receiver constants/maps are regenerated. The
% modulation, bandwidth, coding method, LDPC decoder settings, and receiver

    if nargin < 1 || strlength(string(projectDir)) == 0
        thisDir = fileparts(mfilename("fullpath"));
        projectDir = fileparts(thisDir);
    else
        projectDir = char(projectDir);
        thisDir = fullfile(projectDir,'src');
    end

    addpath(fullfile(projectDir,'config'));
    cfgProject = project_config(projectDir);

    payloadBits = double(cfgProject.PacketSize.BitsPerPacket);
    apepBytes = payloadBits/8;

    %% Match the receiver's existing fixed PHY configuration.
    cfgEHT = wlanEHTMUConfig('CBW320');
    cfgEHT.User{1}.MCS = 13;
    cfgEHT.User{1}.APEPLength = apepBytes;
    cfgEHT.User{1}.ChannelCoding = 'LDPC';
    cfgEHT.User{1}.NumSpaceTimeStreams = 1;

    actualPayloadBits = 8*double(cfgEHT.User{1}.APEPLength);
    if actualPayloadBits ~= payloadBits
        error('wifi7cuda:PacketSizeMismatch', ...
            ['Configured packet size is %d bits, but the WLAN configuration ' ...
             'contains %d APEP payload bits.'],payloadBits,actualPayloadBits);
    end

    %% Build the same demapper metadata used by the MATLAB receiver.
    [commonParams,~] = wlan.internal.ehtCodingParameters(cfgEHT,1);
    dataInfo = wlanEHTOFDMInfo('EHT-Data',cfgEHT);
    fieldIndices = wlanFieldIndices(cfgEHT);
    numDataSubcarriers = numel(dataInfo.DataIndices);
    numOFDMSymbols = double(commonParams.NSYM);
    numSpatialStreams = double(cfgEHT.User{1}.NumSpaceTimeStreams);
    dataSymbolLength = double(dataInfo.FFTLength)+double(dataInfo.CPLength);
    ehtLTFStart = double(fieldIndices.EHTLTF(1))-1;
    ehtLTFLength = double(fieldIndices.EHTLTF(2)-fieldIndices.EHTLTF(1)+1);
    ehtDataStart = double(fieldIndices.EHTData(1))-1;
    ehtDataLength = double(fieldIndices.EHTData(2)-fieldIndices.EHTData(1)+1);
    packetLengthSamples = double(fieldIndices.EHTData(2));
    if ehtDataLength ~= numOFDMSymbols*dataSymbolLength
        error('wifi7cuda:EHTDataLengthMismatch', ...
            'EHT-Data length does not match NSYM times the OFDM symbol length.');
    end

    dummyRx = complex(ones( ...
        numDataSubcarriers,numOFDMSymbols,numSpatialStreams,'single'));
    dummyCSI = ones(numDataSubcarriers,numSpatialStreams,'single');

    prep = wlanEHTPrepareDemapperInput( ...
        dummyRx,single(1),dummyCSI,cfgEHT,1, ...
        LDPCDecodingMethod='norm-min-sum', ...
        MinSumScalingFactor=0.75, ...
        MaximumLDPCIterationCount=12, ...
        EarlyTermination=false);

    userParams = prep.UserParameters;
    cfgLDPC = wlan.internal.heLDPCParameters(userParams);

    vecPayloadBits = double(cfgLDPC.VecPayloadBits(:));
    vecShortenBits = double(cfgLDPC.VecShortenBits(:));
    vecPunctureBits = double(cfgLDPC.VecPunctureBits(:));
    vecRepeatBits = double(cfgLDPC.VecRepeatBits(:));

    numCodewords = numel(vecPayloadBits);
    totalDecodedBits = sum(vecPayloadBits);

    %% Trace the encoded-data source map with unique integer labels.
    % This reproduces the exact segment deparse -> stream deparse ->
    % post-FEC-padding removal path without changing receiver math.
    L = double(prep.NumSegments);
    nsym = double(prep.NumOFDMSymbols);
    nss = double(prep.NumSpatialStreams);
    bitsPerSymbol = double(userParams.NBPSCS);

    interleavedBits = cell(1,L);
    externalCounts = zeros(1,L);
    globalOffset = 0;

    for l = 1:L
        tonesThisSegment = size(prep.DataSymbols{l},1);
        countThisSegment = bitsPerSymbol*tonesThisSegment*nsym*nss;
        externalCounts(l) = countThisSegment;

        labels = globalOffset + (1:countThisSegment);
        interleavedBits{l} = reshape(labels,[],nsym,nss);
        globalOffset = globalOffset + countThisSegment;
    end

    if any(externalCounts ~= externalCounts(1))
        error('wifi7cuda:UnequalSegmentSizes', ...
            'CUDA fixed configuration expects equal external segment sizes.');
    end

    if sum(prep.RUSize) >= 1480
        streamParsedData = wlan.internal.ehtSegmentDeparseBits( ...
            interleavedBits,nsym,nss,userParams.NBPSCS, ...
            prep.RUSize,userParams.DCM);
    else
        streamParsedData = reshape(interleavedBits{1}(:), ...
            userParams.NSD*nsym*userParams.NBPSCS,nss);
    end

    postFECpaddedData = wlanStreamDeparse( ...
        streamParsedData,1,userParams.NCBPS,userParams.NBPSCS);

    encodedSourceLabels = wlan.internal.heRemovePostFECPadding( ...
        postFECpaddedData,prep.ChannelCoding,userParams);

    encodedSourceGlobalIndex = uint32(encodedSourceLabels(:)-1);
    encodedLength = numel(encodedSourceGlobalIndex);
    totalExternalElements = sum(externalCounts);
    nPadPostFEC = totalExternalElements-encodedLength;

    tonesPerSegment = size(prep.DataSymbols{1},1);
    externalElementsPerSegment = externalCounts(1);

    %% Sanity checks before replacing the header.
    if any(double(encodedSourceGlobalIndex) < 0) || ...
            any(double(encodedSourceGlobalIndex) >= totalExternalElements)
        error('wifi7cuda:InvalidEncodedSourceMap', ...
            'Generated encoded-source map is outside the external LLR range.');
    end

    if numel(vecShortenBits) ~= numCodewords || ...
            numel(vecPunctureBits) ~= numCodewords || ...
            numel(vecRepeatBits) ~= numCodewords
        error('wifi7cuda:LDPCVectorSizeMismatch', ...
            'Generated LDPC vectors do not have matching codeword counts.');
    end

    %% Write CUDA header.
    headerPath = fullfile(thisDir,'eht_post_equalizer_fixed_config.h');
    tempPath = [headerPath '.tmp'];
    fid = fopen(tempPath,'wt');
    if fid < 0
        error('wifi7cuda:HeaderOpenFailed','Cannot create %s.',tempPath);
    end
    fprintf(fid,'#pragma once\n');
    fprintf(fid,'#include <stdint.h>\n\n');
    fprintf(fid,'// Auto-generated by generate_eht_post_equalizer_fixed_config.m\n');
    fprintf(fid,'// Packet payload: %d bits (%d-byte APEP).\n',payloadBits,apepBytes);
    fprintf(fid,'// EHT 320 MHz, MCS 13, 4096-QAM, LDPC rate 5/6, one spatial stream.\n');
    fprintf(fid,'// Array indices are zero-based for CUDA use.\n\n');
    fprintf(fid,'namespace eht_fixed {\n\n');

    writeScalarIntLocal(fid,'kNumSegments',L);
    writeScalarIntLocal(fid,'kTonesPerSegment',tonesPerSegment);
    writeScalarIntLocal(fid,'kNumOFDMSymbols',nsym);
    writeScalarIntLocal(fid,'kDataSymbolLength',dataSymbolLength);
    writeScalarIntLocal(fid,'kEHTLTFStart',ehtLTFStart);
    writeScalarIntLocal(fid,'kEHTLTFLength',ehtLTFLength);
    writeScalarIntLocal(fid,'kEHTDataStart',ehtDataStart);
    writeScalarIntLocal(fid,'kEHTDataLength',ehtDataLength);
    writeScalarIntLocal(fid,'kPacketLengthSamples',packetLengthSamples);
    writeScalarIntLocal(fid,'kBitsPerSymbol',bitsPerSymbol);
    writeScalarIntLocal(fid,'kExternalElementsPerSegment',externalElementsPerSegment);
    writeScalarIntLocal(fid,'kTotalExternalElements',totalExternalElements);
    writeScalarIntLocal(fid,'kEncodedLength',encodedLength);
    writeScalarIntLocal(fid,'kNumCodewords',numCodewords);
    writeScalarIntLocal(fid,'kPayloadBits',payloadBits);
    writeScalarIntLocal(fid,'kTotalDecodedBits',totalDecodedBits);
    writeScalarIntLocal(fid,'kNPadPostFEC',nPadPostFEC);
    fprintf(fid,'static constexpr float kAlpha = 0.75f;\n');
    writeScalarIntLocal(fid,'kMaximumIterations',12);
    fprintf(fid,'\n');

    writeArrayLocal(fid,'uint32_t','kEncodedSourceGlobalIndex', ...
        double(encodedSourceGlobalIndex),12);
    writeArrayLocal(fid,'int','kVecPayloadBits',vecPayloadBits,12);
    writeArrayLocal(fid,'int','kVecShortenBits',vecShortenBits,12);
    writeArrayLocal(fid,'int','kVecPunctureBits',vecPunctureBits,12);
    writeArrayLocal(fid,'int','kVecRepeatBits',vecRepeatBits,12);

    fprintf(fid,'} // namespace eht_fixed\n');
    fclose(fid);
    fid = -1;

    movefile(tempPath,headerPath,'f');

    fprintf('Regenerated CUDA fixed configuration:\n');
    fprintf('  Payload bits       : %d\n',payloadBits);
    fprintf('  APEP bytes         : %d\n',apepBytes);
    fprintf('  EHT data symbols   : %d\n',nsym);
    fprintf('  LDPC codewords     : %d\n',numCodewords);
    fprintf('  Encoded LLR values : %d\n',encodedLength);
    fprintf('  Decoded bits       : %d\n',totalDecodedBits);
    fprintf('  Header             : %s\n',headerPath);
end

function writeScalarIntLocal(fid,name,value)
    fprintf(fid,'static constexpr int %s = %d;\n',name,round(double(value)));
end

function writeArrayLocal(fid,typeName,name,values,perLine)
    values = values(:).';
    fprintf(fid,'static constexpr %s %s[%d] = {\n', ...
        typeName,name,numel(values));
    for first = 1:perLine:numel(values)
        last = min(first+perLine-1,numel(values));
        chunk = values(first:last);
        fprintf(fid,'    ');
        for k = 1:numel(chunk)
            if k < numel(chunk) || last < numel(values)
                fprintf(fid,'%d, ',round(double(chunk(k))));
            else
                fprintf(fid,'%d',round(double(chunk(k))));
            end
        end
        fprintf(fid,'\n');
    end
    fprintf(fid,'};\n\n');
end

