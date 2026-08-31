function [dataBits,timing] = wlanEHTRecoverFromExternalLLRProfiled(prep,externalLLR)
% LDPC decoding is CPU-only through eht_ldpc_cpu_mex (no CUDA/GPU calls).

totalTimer = tic;
timing = struct('CSIWeighting_s',0,'BCCDeinterleave_s',0, ...
    'SegmentDeparse_s',0,'StreamDeparse_s',0, ...
    'RemovePostFECPadding_s',0,'ChannelDecode_s',0, ...
    'Descramble_s',0,'FinalBitExtraction_s',0,'Total_s',0);

userParams = prep.UserParameters;
nsym = prep.NumOFDMSymbols;
nss = prep.NumSpatialStreams;
L = prep.NumSegments;
channelCoding = prep.ChannelCoding;
ruSize = prep.RUSize;
ruSize80MHzSubblock = prep.RUSize80MHzSubblock;
ldpcParams = prep.LDPCParameters;
csiToneMapperOut = prep.CSIToneMapperOutput;

if ~iscell(externalLLR) || numel(externalLLR) ~= L
    error("externalLLR must contain one cell per EHT segment.");
end

stageTimer = tic;
interleavedBits = cell(1,L);
parsedData = cell(1,L);
for l = 1:L
    interleavedSym = externalLLR{l};
    expectedElements = userParams.NBPSCS * ...
        size(prep.DataSymbols{l},1) * nsym * nss;
    if numel(interleavedSym) ~= expectedElements
        error("External LLR size mismatch for segment %d.",l);
    end
    interleavedBitsScaled = reshape( ...
        interleavedSym,userParams.NBPSCS,[],nsym,nss) .* ...
        reshape(csiToneMapperOut{l},1,[],1,nss);
    interleavedBits{l} = reshape(interleavedBitsScaled,[],nsym,nss);
    parsedData{l} = cast(0,prep.InputClass);
end
timing.CSIWeighting_s = toc(stageTimer);

stageTimer = tic;
if channelCoding == wlan.type.ChannelCoding.bcc
    interleavedBitsBCC = reshape(interleavedBits{1},[],nss,L);
    assert(L == 1);
    NCBPSSI = userParams.NCBPS/userParams.NSS;
    parsedData{1} = wlan.internal.heBCCDeinterleave( ...
        interleavedBitsBCC,ruSize80MHzSubblock, ...
        userParams.NBPSCS,NCBPSSI,userParams.DCM, ...
        userParams.NCBPSLAST);
else
    parsedData = interleavedBits;
end
timing.BCCDeinterleave_s = toc(stageTimer);

stageTimer = tic;
if sum(ruSize) >= 1480
    streamParsedData = wlan.internal.ehtSegmentDeparseBits( ...
        parsedData,nsym,nss,userParams.NBPSCS,ruSize,userParams.DCM);
else
    streamParsedData = reshape(parsedData{1}(:), ...
        userParams.NSD*nsym*userParams.NBPSCS,nss);
end
timing.SegmentDeparse_s = toc(stageTimer);

stageTimer = tic;
Nes = 1;
postFECpaddedData = wlanStreamDeparse( ...
    streamParsedData,Nes,userParams.NCBPS,userParams.NBPSCS);
timing.StreamDeparse_s = toc(stageTimer);

stageTimer = tic;
encodedData = wlan.internal.heRemovePostFECPadding( ...
    postFECpaddedData,channelCoding,userParams);
timing.RemovePostFECPadding_s = toc(stageTimer);

stageTimer = tic;
if channelCoding == wlan.type.ChannelCoding.bcc
    numTailBits = 6;
    scrambData = wlanBCCDecode(encodedData,userParams.Rate);
else
    numTailBits = 0;
    cfgLDPC = wlan.internal.heLDPCParameters(userParams);
    scrambData = eht_ldpc_cpu_mex( ...
    single(encodedData(:)), ...
    cfgLDPC.VecPayloadBits, ...
    cfgLDPC.VecShortenBits, ...
    cfgLDPC.VecPunctureBits, ...
    cfgLDPC.VecRepeatBits, ...
    single(ldpcParams.alphaBeta), ...
    ldpcParams.MaximumLDPCIterationCount);
end
timing.ChannelDecode_s = toc(stageTimer);

stageTimer = tic;
scrambData = scrambData(:);
scramInitBits = wlan.internal.ehtScramblerInitialState(scrambData(1:11));
if all(scramInitBits == 0)
    preFECPaddedData = scrambData;
else
    preFECPaddedData = wlan.internal.ehtScramble(scrambData,scramInitBits);
end
timing.Descramble_s = toc(stageTimer);

stageTimer = tic;
dataBits = preFECPaddedData( ...
    (7+9+1):(end-userParams.NPADPreFECPHY-numTailBits));
timing.FinalBitExtraction_s = toc(stageTimer);
timing.Total_s = toc(totalTimer);
end
