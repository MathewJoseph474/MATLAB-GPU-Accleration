function prep = wlanEHTPrepareDemapperInput(rx,noiseVarEst,varargin)

    narginchk(3,15);
    [nsd,nsym,nss] = size(rx);
    [userIdx,numUsers,ldpcParams,csi,cfg,isEHTMU] = parseInput(nargin,nsd,nss,varargin{:});
    validateattributes(userIdx,{'numeric'},{'integer','scalar','>=',1,'<=',numUsers},mfilename,'user number');

    % Validate input symbols and noise variance estimates
    validateattributes(rx,{'single','double'},{'3d','finite'},mfilename,'EHT-Data equalized symbol(s)');
    validateattributes(noiseVarEst,{'single','double'},{'finite'},mfilename,'noise variance estimate');

    % Validate coding parameters
    validateConfig(cfg,'Coding');

    % Validate MCS-15
    validateConfig(cfg,'EHTMCS15');

    % Get the appropriate RU and user properties
    if isa(cfg,'wlanEHTRecoveryConfig')
        wlan.internal.mustBeDefined(char(cfg.PPDUType),'PPDUType');
        ruSize = cfg.RUSize;
        isEHTDUPMode = cfg.EHTDUPMode;
        % For code generation, to convert wlan.type.RecoveredChannelCoding
        % to a wlan.type.ChannelCoding type, first cast to uint8 then
        % wlan.type.ChannelCoding
        channelCoding = wlan.type.ChannelCoding(uint8(cfg.ChannelCoding));
        numSTS = cfg.NumSpaceTimeStreams;
        if any(cfg.MCS==[14 15])
            DCM = true;
        else
            DCM = false;
        end
        ldpcExtraSymbol = cfg.LDPCExtraSymbol;
        s = validateConfig(cfg);
        nsymCalc = s.NumDataSymbols;
        stbc = false; % No STBC in EHT
        userParams = wlan.internal.heRecoverCodingParameters(nsymCalc,cfg.PreFECPaddingFactor,sum(ruSize),cfg.MCS,numSTS,channelCoding,stbc,DCM,ldpcExtraSymbol,isEHTDUPMode);
    else
        if isEHTMU
            % Validate EHT DUP mode
            validateConfig(cfg,'EHTDUPMode');
            ruIdx = cfg.User{userIdx}.RUNumber;
            channelCoding = cfg.User{userIdx}.ChannelCoding;
            numSTS = cfg.User{userIdx}.NumSpaceTimeStreams;
            ruSize = cfg.RU{ruIdx}.Size;
        else % EHT TB
            allocInfo = ruInfo(cfg);
            ruIdx = 1; % Only one RU/MRU in EHT TB
            channelCoding = cfg.ChannelCoding;
            numSTS = cfg.NumSpaceTimeStreams;
            ruSize = allocInfo.RUSizes{ruIdx};
        end
        % Get coding parameters for the user of interest
        [commonParams,userParams] = wlan.internal.ehtCodingParameters(cfg,userIdx);
        nsymCalc = commonParams.NSYM;
        isEHTDUPMode = cfg.EHTDUPMode;
    end

    if isEHTDUPMode
        % Halve the RU size defined for CBW320 for EHT DUP
        % mode. Segment parsing and constellation mapping is performed
        % on bits required for half the RU size for the given channel
        % bandwidth. The remaining half of the RU has symbols generated from
        % frequency domain duplication as defined in Section 36.3.13.10 of IEEE
        % P802.11be/D5.0.
        ruSize = ruSize/2;

        % Average lower and upper subcarriers due to frequency domain
        % duplication in EHT DUP mode.
        halfNumSubcarriers = size(rx,1)/2;
        % Remove scaling by multiplying the upper subcarriers by -1
        upperDataSubcarriers = [rx(1+halfNumSubcarriers:1.5*halfNumSubcarriers,:,:)*-1; rx(1+1.5*halfNumSubcarriers:end,:,:)];
        rxSym = (rx(1:halfNumSubcarriers,:,:)+upperDataSubcarriers)/2; 

        csiSym = (csi(1:halfNumSubcarriers,:,:)+csi(halfNumSubcarriers+1:end,:,:))/2;

        nsd = nsd/2;
    else
        rxSym = rx;
        csiSym = csi;
    end

    expectedNss = numSTS;
    % Test we have a correct number of data subcarriers, corresponding to
    % an RU size, OFDM symbols, and spatial streams
    tac = wlan.internal.heRUToneAllocationConstants(sum(ruSize));
    if any(nsd ~= tac.NSD)
        coder.internal.error('wlan:shared:IncorrectSC',tac.NSD,nsd);
    end
    if any(nsym < nsymCalc)
        coder.internal.error('wlan:shared:IncorrectNumOFDMSym',nsymCalc,nsym);
    end
    nsym = nsymCalc; % Use the required number of symbols
    rxSym = rxSym(:,1:nsym,:); % Extract the minimum input signal length required
    if any(nss ~= expectedNss)
        coder.internal.error('wlan:shared:IncorrectNumSS',expectedNss,nss);
    end

    % Validate size of CSI
    validateattributes(csiSym,{'single','double'},{'real','3d','finite'},mfilename,'CSI');
    if any(size(csiSym) ~= [tac.NSD nss])
        coder.internal.error('wlan:he:InvalidCSISize',tac.NSD,nss);
    end

    % Inverse frequency segment deparsing
    if sum(ruSize)>=1480 % Deparsing is applicable for MRU/RU size >= 996+484
        p = wlan.internal.ehtSegmentParserParameters(ruSize,userParams.NBPSCS,userParams.DCM);
        L = p.L; % Number of 80 MHz frequency segments
        if userParams.DCM
            % Double the number of coded bits per OFDM symbol per spatial stream per segment for DCM
            Ncbpssl = p.Ncbpssl*2;
        else
            Ncbpssl = p.Ncbpssl;
        end
        mappedData = ehtSegmentParseSymbols(rxSym,Ncbpssl/userParams.NBPSCS); % 
        csiParserOut = ehtSegmentParseSymbols(reshape(csiSym,[],1,nss),Ncbpssl/userParams.NBPSCS);
        ruSize80MHzSubblock = p.RUSizePer80MHz; % RU/MRU size per 80 MHz segment
    else
        L = 1;
        mappedData = {rxSym};
        csiParserOut = {csiSym};
        ruSize80MHzSubblock = sum(ruSize); % Sum RUs if it is an MRU
    end

    % Inverse LDPC tone mapping (if applicable)
    if channelCoding==wlan.type.ChannelCoding.bcc
        dataSym = mappedData; % [Nsd,Nsym,Nss,L]
        csiToneMapperOut = csiParserOut;
    else % LDPC tone mapping
        dataSym = cell(1,L);
        csiToneMapperOut = cell(1,L);
        for l=1:L
            mappingInd = wlan.internal.ehtLDPCToneMappingIndices(ruSize80MHzSubblock(l),userParams.DCM); % Get LDPC mapping index for an 80 MHz subblock
            dataSym{l} = mappedData{l}(mappingInd,:,:);
            csiToneMapperOut{l} = csiParserOut{l}(mappingInd,:,:);
        end
    end

    if userParams.DCM
        for l=1:L
           
            csiToneMapperOut{l} = (csiToneMapperOut{l}(1:end/2,:,:,:)+csiToneMapperOut{l}(end/2+1:end,:,:,:))/2;
        end
    end

    prep = struct;
    prep.DataSymbols = dataSym;
    prep.CSIToneMapperOutput = csiToneMapperOut;
    prep.NoiseVarianceEstimate = noiseVarEst;
    prep.UserParameters = userParams;
    prep.NumOFDMSymbols = nsym;
    prep.NumSpatialStreams = nss;
    prep.NumSegments = L;
    prep.ChannelCoding = channelCoding;
    prep.RUSize = ruSize;
    prep.RUSize80MHzSubblock = ruSize80MHzSubblock;
    prep.LDPCParameters = ldpcParams;
    prep.InputClass = class(rx);
end

function y = ehtSegmentParseSymbols(x,Ncbpss)

    offfsetNcbpss = cumsum([0 Ncbpss]);
    L = numel(Ncbpss); % Number of frequency subblocks
    y = cell(1,L);
    for i=1:L
        y{i} = x(offfsetNcbpss(i)+(1:Ncbpss(i)),:,:);
    end

end

function [userIdx,numUsers,ldpcParams,csi,cfg,isEHTMU] = parseInput(numInputArg,nsd,nss,varargin)

    csiInputFlag = false; % If no CSI input is present

    if isa(varargin{1},'wlanEHTMUConfig') || isa(varargin{1},'wlanEHTTBConfig') || isa(varargin{1},'wlanEHTRecoveryConfig') % wlanEHTDataBitRecover(RX,NOISEVAREST,CFG,...)
        cfg = varargin{1};
        csi = ones(nsd,nss); % If no CSI input is present then assume 1 for processing
    elseif numInputArg>3 && (isa(varargin{2},'wlanEHTMUConfig') || isa(varargin{2},'wlanEHTTBConfig') || isa(varargin{2},'wlanEHTRecoveryConfig')) % wlanEHTDataBitRecover(RX,NOISEVAREST,CSI,CFG,...)
        csi = varargin{1};
        cfg = varargin{2};
        csiInputFlag = true; % CSI input is present
    else
        coder.internal.error('wlan:wlanEHTDataBitRecover:IncorrectDataBitRecoverSyntax');
    end

    isEHTMU = isa(cfg,'wlanEHTMUConfig');

    mode = compressionMode(cfg);
    if any(mode==[0 2]) && isEHTMU % OFDMA or MU-MIMO
        if nargin==3 || (numInputArg==4 && csiInputFlag) % wlanEHTDataBitRecover(RX,NOISEVAREST,CFG)
            coder.internal.error('wlan:shared:ExpectedUserNumber');
        elseif numInputArg>3 && isnumeric(varargin{2}) % wlanEHTDataBitRecover(RX,NOISEVAREST,CFG,USERIDX,NV)
            userIdx = varargin{2};
            numArgPreNV = 4;
            ldpcParams = wlan.internal.parseOptionalInputs(mfilename,varargin{numArgPreNV-1:end});
        elseif numInputArg>3 && isnumeric(varargin{3}) && csiInputFlag
            userIdx = varargin{3}; % wlanEHTDataBitRecover(RX,NOISEVAREST,CSI,CFG,USERIDX,NV)
            numArgPreNV = 5;
            ldpcParams = wlan.internal.parseOptionalInputs(mfilename,varargin{numArgPreNV-1:end});
        else % wlanEHTDataBitRecover(RX,NOISEVAREST,CFG,NV), wlanEHTDataBitRecover(RX,NOISEVAREST,CSI,CFG,NV)
            coder.internal.error('wlan:shared:ExpectedUserNumber');
        end
        numUsers = length(cfg.User);
    else % Single user EHT MU packet or EHT Recovery config
        numUsers = 1;
        userIdx = 1;
        if csiInputFlag % wlanEHTDataBitRecover(RX,NOISEVAREST,CSI,CFG,...)
            if numInputArg>4
                if isnumeric(varargin{3}) % wlanEHTDataBitRecover(RX,NOISEVAREST,CSI,CFG,USERIDX,NV)
                    numArgPreNV = 5;
                    ldpcParams = wlan.internal.parseOptionalInputs(mfilename,varargin{numArgPreNV-1:end});
                else % wlanEHTDataBitRecover(RX,NOISEVAREST,CSI,CFG,NV)
                    numArgPreNV = 4;
                    ldpcParams = wlan.internal.parseOptionalInputs(mfilename,varargin{numArgPreNV-1:end});
                end
            else % wlanEHTDataBitRecover(RX,NOISEVAREST,CSI,CFG)
                ldpcParams = wlan.internal.parseOptionalInputs(mfilename);
            end
        else % wlanEHTDataBitRecover(RX,NOISEVAREST,CFG,...)
            if numInputArg>3
                if isnumeric(varargin{2}) % wlanEHTDataBitRecover(RX,NOISEVAREST,CFG,USERIDX,NV)
                    numArgPreNV = 4;
                    ldpcParams = wlan.internal.parseOptionalInputs(mfilename,varargin{numArgPreNV-1:end});
                else % wlanEHTDataBitRecover(RX,NOISEVAREST,CFG,NV)
                    numArgPreNV = 3;
                    ldpcParams = wlan.internal.parseOptionalInputs(mfilename,varargin{numArgPreNV-1:end});
                end
            else % wlanEHTDataBitRecover(RX,NOISEVAREST,CFG)
                ldpcParams = wlan.internal.parseOptionalInputs(mfilename);
            end
        end
    end
end
