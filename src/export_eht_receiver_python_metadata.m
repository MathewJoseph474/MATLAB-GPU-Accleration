function outputPath = export_eht_receiver_python_metadata(outputPath)
%EXPORT_EHT_RECEIVER_PYTHON_METADATA Export fixed receiver constants.
%   The packet database is a MATLAB v7.3 (HDF5) file and can be read from
%   Python with h5py.  The WLAN Toolbox objects stored in that file are not
%   portable, so this function evaluates them once and writes only plain
%   numeric arrays to a MATLAB v7 file that scipy.io.loadmat can read.

srcDir = fileparts(mfilename("fullpath"));
projectDir = fileparts(srcDir);
databasePath = fullfile(projectDir,"database", ...
    "eht_receiver_database_128.mat");

if nargin < 1 || strlength(string(outputPath)) == 0
    outputPath = fullfile(projectDir,"database", ...
        "eht_receiver_python_metadata.mat");
end

assert(isfile(databasePath),'Database not found:\n%s',databasePath);
database = load(databasePath,'cfgEHT','TxBits');
cfgEHT = database.cfgEHT;

fieldIndices = wlanFieldIndices(cfgEHT);
sampleRateHz = double(wlanSampleRate(cfgEHT));
packetLengthSamples = int32(fieldIndices.EHTData(2));
payloadBits = int32(8*double(cfgEHT.User{1}.APEPLength));

cleanWaveform = single(wlanWaveformGenerator(database.TxBits(:,1),cfgEHT));
lltfReference = cleanWaveform( ...
    fieldIndices.LLTF(1):fieldIndices.LLTF(2),:);
lltfReference = complex(single(real(lltfReference)), ...
    single(imag(lltfReference)));

ltfInfo = wlanEHTOFDMInfo('EHT-LTF',cfgEHT);
dataInfo = wlanEHTOFDMInfo('EHT-Data',cfgEHT);
ltfFFTStart = int32(round(0.75*double(ltfInfo.CPLength))+1);
dataCP = double(dataInfo.CPLength);
dataSymbolLength = double(dataInfo.FFTLength)+dataCP;
dataFFTStart = round(0.75*dataCP)+1;
[commonParams,~] = wlan.internal.ehtCodingParameters(cfgEHT,1);
numDataSymbols = double(commonParams.NSYM);
dataFFTStarts = int32(dataFFTStart + ...
    (0:numDataSymbols-1).' * dataSymbolLength);

ltfFFTIndices = int32(ltfInfo.ActiveFFTIndices(:));
dataFFTIndices = int32(dataInfo.ActiveFFTIndices(:));
pilotIndices = int32(dataInfo.PilotIndices(:));
dataIndices = int32(dataInfo.DataIndices(:));

cleanEHTLTF = cleanWaveform( ...
    fieldIndices.EHTLTF(1):fieldIndices.EHTLTF(2),:);
cleanEHTData = cleanWaveform( ...
    fieldIndices.EHTData(1):fieldIndices.EHTData(2),:);
cleanLTFDemod = single(wlanEHTDemodulate( ...
    cleanEHTLTF,'EHT-LTF',cfgEHT));
cleanChannel = single(wlanEHTLTFChannelEstimate(cleanLTFDemod,cfgEHT));
cleanDataDemod = single(wlanEHTDemodulate( ...
    cleanEHTData,'EHT-Data',cfgEHT));

knownLTF = cleanLTFDemod(:)./cleanChannel(:);
knownLTF = knownLTF./abs(knownLTF);
knownLTF = complex(single(real(knownLTF)),single(imag(knownLTF)));

pilotReference = cleanDataDemod(pilotIndices,:)./ ...
    cleanChannel(pilotIndices);
pilotReference = pilotReference./abs(pilotReference);
pilotReference = complex(single(real(pilotReference)), ...
    single(imag(pilotReference)));

schemaVersion = int32(1);
indexBase = int32(1);
save(outputPath,'schemaVersion','indexBase','sampleRateHz', ...
    'packetLengthSamples','payloadBits','lltfReference', ...
    'ltfFFTIndices','dataFFTIndices','ltfFFTStart','dataFFTStarts', ...
    'knownLTF','pilotIndices','pilotReference','dataIndices','-v7');

fprintf('Python receiver metadata saved to:\n%s\n',outputPath);
end
