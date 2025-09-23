-------------------------------------------------------------------------------
-- Title      : Acquisition Control Block
-- Project    : EPIX Readout
-------------------------------------------------------------------------------
-- File       : ReadoutControl.vhd
-- Company    : SLAC National Accelerator Laboratory
-------------------------------------------------------------------------------
-- Description:
-- Readout control block
-------------------------------------------------------------------------------
-- This file is part of 'EPIX Development Firmware'.
-- It is subject to the license terms in the LICENSE.txt file found in the 
-- top-level directory of this distribution and at: 
--    https://confluence.slac.stanford.edu/display/ppareg/LICENSE.html. 
-- No part of 'EPIX Development Firmware', including this file, 
-- may be copied, modified, propagated, or distributed except according to 
-- the terms contained in the LICENSE.txt file.
-------------------------------------------------------------------------------

LIBRARY ieee;
use ieee.std_logic_1164.all;
use ieee.std_logic_unsigned.all;
use ieee.std_logic_arith.all;

library surf;
use surf.StdRtlPkg.all;
use surf.AxiStreamPkg.all;
use surf.SsiPkg.all;

use work.EpixPkgGen2.all;

library UNISIM;
use UNISIM.vcomponents.all;

entity ReadoutControl is
   generic (
      TPD_G                      : time := 1 ns;
      ASIC_TYPE_G                : AsicType;
      MASTER_AXI_STREAM_CONFIG_G : AxiStreamConfigType := ssiAxiStreamConfig(4, TKEEP_COMP_C)
   );
   port (

      -- Clocks and reset
      sysClk              : in    sl;
      sysClkRst           : in    sl;

      -- Configuration
      epixConfig          : in    EpixConfigType;
      epixConfigExt       : in    EpixConfigExtType;

      -- Data for headers
      acqCount            : in    slv(31 downto 0);

      -- Frame counter out to register control
      seqCount            : out   slv(31 downto 0);

      -- Opcode to insert into frame
      opCode              : in    slv(7 downto 0);
      
      -- Run control
      acqStart            : in    sl;
      readValid           : in    slv(15 downto 0);
      readDone            : out   sl;
      acqBusy             : in    sl;
      dataSend            : in    sl;

      -- ADC Data
      adcPulse            : in    sl;
      adcValid            : in    slv(19 downto 0);
      adcData             : in    Slv16Array(19 downto 0);
      
      -- monitoring data (ADC board gen2)
      envData             : in    Slv32Array(8 downto 0);

      -- Data out interface
      mAxisMaster         : out AxiStreamMasterType;
      mAxisSlave          : in  AxiStreamSlaveType;

      -- MPS
      mpsOut              : out   sl;

      --TG
      runSync_i : in sl;  
      daqSync_i : in sl;
      
      -- EPIX10KA bank-deserialized digital outputs
      doutOut             : in  Slv2Array(15 downto 0);
      doutRd              : out slv(15 downto 0);
      doutValid           : in  slv(15 downto 0)
   );
end ReadoutControl;

-- Define architecture
architecture ReadoutControl of ReadoutControl is
   
   constant NCOL_C       : integer          := getNumColumns(ASIC_TYPE_G);
   constant WORDS_PER_SUPER_ROW_C  : integer := getWordsPerSuperRow(ASIC_TYPE_G);

   -- Timeout in clock cycles
   constant STUCK_TIMEOUT_C : slv(31 downto 0) := conv_std_logic_vector(1250000,32); --12.5 ms 
   -- Depth of FIFO 
   constant CH_FIFO_ADDR_WIDTH_C : integer := 10;
   -- Hard coded words in the data stream for now
   constant LANE_C     : slv( 1 downto 0) := "00";
   constant VC_C       : slv( 1 downto 0) := "00";
   constant QUAD_C     : slv( 1 downto 0) := "00";
   constant ZEROWORD_C : slv(31 downto 0) := x"00000000";
   -- Register delay for simulation
   constant TPD_C : time := 0.5 ns;

   
   -- State definitions
   type StateType is (IDLE_S,ARMED_S,HEADER_S,READ_FIFO_S,
                      ENV_DATA_S,TPS_DATA_S,FOOTER_S);

   -- Local Signals
   type RegType is record
      readDone          : sl;
      testPattern       : sl;
      seqCountEn        : sl;
      fillCnt           : slv(CH_FIFO_ADDR_WIDTH_C-1 downto 0);
      chCnt             : slv(3 downto 0);
      timeoutCnt        : slv(31 downto 0);
      stuckTimeout      : slv(31 downto 0);
      clearFifos        : sl;
      error             : sl;
      wordCnt           : slv(31 downto 0);
      adcData           : Slv16Array(19 downto 0);
      overSampleSizePwr : slv(6 downto 0);
      mAxisMaster       : AxiStreamMasterType;
      state             : StateType;
   end record;
   constant REG_INIT_C : RegType := (
      '0',
      '0',
      '0',
      (others => '0'),
      (others => '0'),
      (others => '0'),
      (others => '0'),
      '0',
      '0',
      (others => '0'),
      (others => (others => '0')),      
      (others => '0'),
      AXI_STREAM_MASTER_INIT_C,
      IDLE_S
   );
   
   signal r   : RegType := REG_INIT_C;
   signal rin : RegType;
   
   signal memRst         : sl := '0';
   signal dataSendEdge   : sl;
   signal acqStartEdge   : sl;
   signal adcMemWrEn     : slv(15 downto 0);
   signal adcMemRdOrder  : std_logic_vector(15 downto 0);
   signal adcMemRdRdy    : slv(15 downto 0);
   signal adcMemRdValid  : slv(15 downto 0);
   signal adcMemOflow    : slv(15 downto 0);
   signal adcMemOflowAny : std_logic;
   signal adcMemRdData   : Slv16Array(15 downto 0);
   
   signal adcFifoWrEn    : slv(15 downto 0);
   signal adcFifoEmpty   : slv(15 downto 0);
   signal adcFifoOflow   : slv(15 downto 0);
   signal adcFifoRdValid : slv(15 downto 0);
   signal adcFifoRdEn    : slv(15 downto 0);
   signal adcFifoRdData  : Slv32Array(15 downto 0);
   signal adcFifoWrData  : Slv16Array(15 downto 0);
   signal fifoOflowAny   : sl := '0';
   signal fifoEmptyAll   : sl := '0';
   signal intSeqCount    : slv(31 downto 0) := (others => '0');

   signal adcDataToReorder : Slv16Array(19 downto 0);
   signal tpsData          : Slv16Array(3 downto 0);

   type chanMap is array(15 downto 0) of integer range 0 to 15;
   signal channelOrder   : chanMap;
   signal doutOrder      : chanMap;
   signal channelValid   : slv(15 downto 0);
   
   signal adcDataOvs     : Slv16Array(15 downto 0);

   attribute dont_touch : string;
   attribute dont_touch of r : signal is "true";
   
   attribute keep : string;
   attribute keep of fifoEmptyAll : signal is "true";
   attribute keep of r : signal is "true";
   attribute keep of adcFifoRdValid : signal is "true";
   attribute keep of adcFifoRdEn : signal is "true";
   
begin
   
   -- moving average on ADC data for optional oversampling
   G_Ovs : for i in 15 downto 0 generate
      signal avgDataTmp       : Slv21Array(15 downto 0);
      signal avgDataTmpValid  : slv(15 downto 0);
      signal avgDataMux       : Slv14Array(15 downto 0);
   begin
      
      U_MovingAvg : entity surf.BoxcarIntegrator
         generic map (
            TPD_G        => TPD_G,
            DATA_WIDTH_G => 14,
            ADDR_WIDTH_G => 7
         )
         port map (
            clk      => sysClk,
            rst      => sysClkRst,
            -- Configuration, intCount is 0 based, 0 = 1, 1 = 2, 1023 = 1024
            intCount => r.overSampleSizePwr,
            -- Inbound Interface
            ibValid  => adcValid(i),
            ibData   => adcData(i)(13 downto 0),
            -- Outbound Interface
            obValid  => avgDataTmpValid(i),
            obData   => avgDataTmp(i)
         );

      avgDataMux(i) <= 
         avgDataTmp(i)(20 downto 7) when epixConfigExt.oversampleSize = 7 else
         avgDataTmp(i)(19 downto 6) when epixConfigExt.oversampleSize = 6 else
         avgDataTmp(i)(18 downto 5) when epixConfigExt.oversampleSize = 5 else
         avgDataTmp(i)(17 downto 4) when epixConfigExt.oversampleSize = 4 else
         avgDataTmp(i)(16 downto 3) when epixConfigExt.oversampleSize = 3 else
         avgDataTmp(i)(15 downto 2) when epixConfigExt.oversampleSize = 2 else
         avgDataTmp(i)(14 downto 1) when epixConfigExt.oversampleSize = 1 else
         avgDataTmp(i)(13 downto 0);
      
      U_Reg : entity surf.RegisterVector
         generic map (
            TPD_G       => TPD_G,
            WIDTH_G     => 15
         )
         port map (
            clk         => sysClk,
            rst         => sysClkRst,
            sig_i(14)   => avgDataTmpValid(i),
            sig_i(13 downto 0) => avgDataMux(i),
            reg_o(14)   => open,
            reg_o(13 downto 0) => adcDataOvs(i)(13 downto 0)
         );
   
   end generate;
   
   
   -- Counter output to register control
   seqCount <= intSeqCount;
   -- Channel Order for ASIC readout (last downto first)
   -- Readout order based on ePix100 ASIC numbering scheme (0 - forward, 1 - backward)
   -- Indexing for the memory readout order is linked to the raw ADC channel
   -- (i.e., if the channel reads out an ASIC from upper half of carrier,
   --  read it backward, otherwise, read it forward)
   G_EPIX100A_CARRIER_ADC_GEN2 : if (ASIC_TYPE_G = EPIX100A_C or ASIC_TYPE_G = EPIX10KA_C) generate
   
      -- EPIX100A and EPIX10KA Carrier Board View:
      --                             TOP
      --     B3    B2    B1    B0     |     B3    B2    B1    B0     
      --                              |
      --                              |
      --                              |
      --           ASIC2              |           ASIC1 
      --                              |
      --                              |
      --                              |
      --------------------------------------------------------------
      --                              |
      --                              |
      --                              |
      --           ASIC3              |           ASIC0 
      --                              |
      --                              |
      --                              |
      --     B0    B1    B2    B3     |     B0    B1    B2    B3     
      --
      -- ADC Channel Mappping for EPIX100A and EPIX10KA GEN2 ADC Board:
      -- 10:'ASIC0_B0',   2:'ASIC0_B1',   1:'ASIC0_B2',   0:'ASIC0_B3'
      --  8:'ASIC1_B0',   9:'ASIC1_B1',   3:'ASIC1_B2',   4:'ASIC1_B3'
      --  5:'ASIC2_B0',   6:'ASIC2_B1',   7:'ASIC2_B2',  15:'ASIC2_B3'
      -- 14:'ASIC3_B0',  13:'ASIC3_B1',  12:'ASIC3_B2',  11:'ASIC3_B3'
      -- 17:'ASIC0_TPS', 19:'ASIC1_TPS', 18:'ASIC2_TPS', 16:'ASIC3_TPS'
      
      -- ADC channel mapping. See above.
      channelOrder <= (8,9,3,4,5,6,7,15,0,1,2,10,11,12,13,14);
      -- readout order per ADC channel (0 - forward, 1 - backward)
      adcMemRdOrder <= "1000001111111000";
      -- used for ADC Pipeline Delay per ASIC
      -- asicOrder <= (2,3,3,3,3,0,1,1,2,2,2,1,1,0,0,0);
      -- valid only for EPIX10KA_C
      -- see DoutDeserializer for dout mapping
      doutOrder <= (4,5,6,7,8,9,10,11,3,2,1,0,15,14,13,12);
      -- all 16 channels used
      channelValid  <= (others => '1');
      -- TPS ADC channnels mapping. See above.
      tpsData(0) <= r.adcData(17);
      tpsData(1) <= r.adcData(19);
      tpsData(2) <= r.adcData(18);
      tpsData(3) <= r.adcData(16);
   end generate;
   G_EPIX10KP_CARRIER_ADC_GEN2 : if (ASIC_TYPE_G = EPIX10KP_C) generate
      
      -- digital outputs are no longer supported for the EPIX10KP_C
      
      channelOrder <= (1,2,0,10,5,7,15, 6,9,4,3,8,10,11,13,12);
      channelValid  <= (others => '1');
      adcMemRdOrder <= "1100010010100111";
      tpsData(0) <= r.adcData(16);
      tpsData(1) <= r.adcData(17);
      tpsData(2) <= r.adcData(18);
      tpsData(3) <= r.adcData(19);
   end generate;
   G_EPIXS_CARRIER_ADC_GEN2 : if (ASIC_TYPE_G = EPIXS_C) generate
   
      -- EPIXS Carrier Board View:
      --                 TOP
      --        B0        |        B0     
      --                  |
      --                  |
      --        U3        |        U2 
      --                  |
      --                  |
      ------------------------------------------
      --                  |
      --                  |
      --        U4        |        U1 
      --                  |
      --                  |
      --        B0        |        B0        
      --
      -- ADC Channel Mappping for EPIXS GEN2 ADC Board:
      --  8:'U1_B0'
      --  1:'U2_B0'
      --  5:'U3_B0'
      -- 12:'U4_B0'
      -- 16:'U1_TPS', 17:'U2_TPS', 18:'U3_TPS', 19:'U4_TPS'
   
      channelOrder <= (0,2,3,4,6,7,9,10,11,13,14,15,1,5,8,12);
      channelValid  <= "0000000000001111";
      adcMemRdOrder <= "0000000000100010";
      tpsData(0) <= r.adcData(16);
      tpsData(1) <= r.adcData(17);
      tpsData(2) <= r.adcData(18);
      tpsData(3) <= r.adcData(19);
   end generate;

   -- Edge detection for signals that interface with other blocks
   U_DataSendEdge : entity surf.SynchronizerEdge
      port map (
         clk        => sysClk,
         rst        => sysClkRst,
         dataIn     => dataSend,
         risingEdge => dataSendEdge
      );
   U_ReadStartEdge : entity surf.SynchronizerEdge
      port map (
         clk        => sysClk,
         rst        => sysClkRst,
         dataIn     => acqStart,
         risingEdge => acqStartEdge
      );
   
   --------------------------------------------------
   -- Simple state machine to just send ADC values --
   --------------------------------------------------
   comb : process (r,epixConfig,epixConfigExt,acqCount,intSeqCount,adcFifoRdData,adcFifoRdValid,doutValid,
                   channelOrder,doutOrder,fifoEmptyAll,acqBusy,adcMemOflowAny,fifoOflowAny,
                   envData,tpsData,acqStartEdge,dataSendEdge,adcFifoEmpty,
                   sysClkRst,mAxisSlave, adcData, adcDataOvs, channelValid, opCode,
                   doutOut,runSync_i,daqSync_i) 
      variable v : RegType;
   begin
      v := r;
      
      -- Reset pulsed signals
      ssiResetFlags(v.mAxisMaster);
      v.mAxisMaster.tData := (others => '0');
      v.seqCountEn := '0';
      
      adcFifoRdEn <= (others=>'0');
      doutRd      <= (others=>'0');

      -- Always grab latest adc data
      for i in 0 to 15 loop
         if epixConfigExt.oversampleEn = '0' then
            v.adcData(i) := adcData(i);
         else
            v.adcData(i) := adcDataOvs(i);
         end if;
      end loop;
      -- TPS data (no oversampling)
      for i in 16 to 19 loop
         v.adcData(i) := adcData(i);
      end loop;
      
      -- convert oversample size to power
      if epixConfigExt.oversampleSize = 0 then
         v.overSampleSizePwr   := "0000000";
      elsif epixConfigExt.oversampleSize = 1 then
         v.overSampleSizePwr   := "0000001";
      elsif epixConfigExt.oversampleSize = 2 then
         v.overSampleSizePwr   := "0000011";
      elsif epixConfigExt.oversampleSize = 3 then
         v.overSampleSizePwr   := "0000111";
      elsif epixConfigExt.oversampleSize = 4 then
         v.overSampleSizePwr   := "0001111";
      elsif epixConfigExt.oversampleSize = 5 then
         v.overSampleSizePwr   := "0011111";
      elsif epixConfigExt.oversampleSize = 6 then
         v.overSampleSizePwr   := "0111111";
      else
         v.overSampleSizePwr   := "1111111";
      end if;
      
      -- Latch overflows (this is reset in IDLE state)
      if (fifoOflowAny = '1' or adcMemOflowAny = '1') then
         v.error := '1';
      end if;
      
      if STUCK_TIMEOUT_C + epixConfig.asicAcqWidth < x"ffffffff" then
         v.stuckTimeout := STUCK_TIMEOUT_C + epixConfig.asicAcqWidth;
      else
         v.stuckTimeout := x"ffffffff";
      end if;
      
      
      -- State outputs
      if mAxisSlave.tReady = '1' then      
         case (r.state) is
            when IDLE_S =>
               v.wordCnt     := (others => '0');
               v.chCnt       := (others => '0');
               v.fillCnt     := (others => '0');
               v.timeoutCnt  := (others => '0');
               v.clearFifos  := '1';
               v.readDone    := '1';
               v.testPattern := epixConfig.testPattern;
               v.error       := '0';
               if acqStartEdge = '1' then
                  v.state := ARMED_S;
               end if;
            when ARMED_S =>
               v.readDone   := '0';
               v.clearFifos := '0';
               v.timeoutCnt := r.timeoutCnt + 1;
               if dataSendEdge = '1' then
                  v.seqCountEn := '1';
                  v.state      := HEADER_S;
               elsif (r.timeoutCnt >= r.stuckTimeout) then
                  v.state := IDLE_S;
               end if;
            when HEADER_S =>
               v.wordCnt            := r.wordCnt + 1;
               v.mAxisMaster.tValid := '1';

               case conv_integer(r.wordCnt) is
                  when 0 => v.mAxisMaster.tData(31 downto 0) := x"000000" & "00" & LANE_C & "00" & VC_C;
                            ssiSetUserSof(MASTER_AXI_STREAM_CONFIG_G, v.mAxisMaster, '1');
                  when 1 => v.mAxisMaster.tData(31 downto 0) := x"0" & "00" & QUAD_C & opCode & acqCount(15 downto 0);
                  when 2 => v.mAxisMaster.tData(31 downto 0) := intSeqCount;
                  when 3 => v.mAxisMaster.tData(31 downto 0) := (31 downto 1 => '0') & runSync_i ;
                  when 4 => v.mAxisMaster.tData(31 downto 0) := (31 downto 1 => '0') & daqSync_i ;
                  when 5 => v.mAxisMaster.tData(31 downto 0) := ZEROWORD_C;
                  when 6 => v.mAxisMaster.tData(31 downto 0) := ZEROWORD_C;
                  when 7 => v.mAxisMaster.tData(31 downto 0) := ZEROWORD_C;
                  when others => v.mAxisMaster.tData(31 downto 0) := ZEROWORD_C;
               end case;
               if (r.wordCnt = 7) then
                  v.wordCnt := (others => '0');
                  v.state := READ_FIFO_S;
               end if;
            when READ_FIFO_S => 
               -- read from ADC FIFOs and from Dout FIFOs (valid for EPIX10KA_C)
               v.mAxisMaster.tData(31 downto 0) := 
                  '0' & doutOut(doutOrder(conv_integer(r.chCnt)))(1) &
                  adcFifoRdData(channelOrder(conv_integer(r.chCnt)))(29 downto 16) & 
                  '0' & doutOut(doutOrder(conv_integer(r.chCnt)))(0) &
                  adcFifoRdData(channelOrder(conv_integer(r.chCnt)))(13 downto 0);
               
               --if adcFifoRdValid(channelOrder(conv_integer(r.chCnt))) = '1' then
               if adcFifoRdValid(channelOrder(conv_integer(r.chCnt))) = '1' and doutValid(doutOrder(conv_integer(r.chCnt))) = '1' then
                  
                  adcFifoRdEn(channelOrder(conv_integer(r.chCnt))) <= '1';
                  doutRd(doutOrder(conv_integer(r.chCnt))) <= '1';
                  
                  if (channelValid(conv_integer(r.chCnt)) = '1') then
                     v.mAxisMaster.tValid := '1';
                  end if;
                  v.fillCnt         := r.fillCnt + 1;
                  if (r.fillCnt = conv_std_logic_vector(NCOL_C/2-1,r.fillCnt'length)) then
                     v.chCnt   := r.chCnt + 1;
                     v.fillCnt := (others => '0');
                  end if;
               else
                  v.timeoutCnt := r.timeoutCnt + 1;
               end if;
               if acqBusy = '0' and fifoEmptyAll = '1' then
                  v.state     := ENV_DATA_S;
               elsif r.error = '1' or r.timeoutCnt >= r.stuckTimeout then
                  v.state     := FOOTER_S;
               end if;
            when ENV_DATA_S =>
               v.wordCnt         := r.wordCnt + 1;
               v.mAxisMaster.tValid := '1';
               if (r.wordCnt = conv_std_logic_vector(WORDS_PER_SUPER_ROW_C,r.wordCnt'length)) then
                  v.mAxisMaster.tData(31 downto 0) := envData(0);
               elsif (r.wordCnt = conv_std_logic_vector(WORDS_PER_SUPER_ROW_C+1,r.wordCnt'length)) then
                  v.mAxisMaster.tData(31 downto 0) := envData(1);
               elsif (r.wordCnt = conv_std_logic_vector(WORDS_PER_SUPER_ROW_C+2,r.wordCnt'length)) then
                  v.mAxisMaster.tData(31 downto 0) := envData(2);
               elsif (r.wordCnt = conv_std_logic_vector(WORDS_PER_SUPER_ROW_C+3,r.wordCnt'length)) then
                  v.mAxisMaster.tData(31 downto 0) := envData(3);
               elsif (r.wordCnt = conv_std_logic_vector(WORDS_PER_SUPER_ROW_C+4,r.wordCnt'length)) then
                  v.mAxisMaster.tData(31 downto 0) := envData(4);
               elsif (r.wordCnt = conv_std_logic_vector(WORDS_PER_SUPER_ROW_C+5,r.wordCnt'length)) then
                  v.mAxisMaster.tData(31 downto 0) := envData(5);
               elsif (r.wordCnt = conv_std_logic_vector(WORDS_PER_SUPER_ROW_C+6,r.wordCnt'length)) then
                  v.mAxisMaster.tData(31 downto 0) := envData(6);
               elsif (r.wordCnt = conv_std_logic_vector(WORDS_PER_SUPER_ROW_C+7,r.wordCnt'length)) then
                  v.mAxisMaster.tData(31 downto 0) := envData(7);
               elsif (r.wordCnt = conv_std_logic_vector(WORDS_PER_SUPER_ROW_C+8,r.wordCnt'length)) then
                  v.mAxisMaster.tData(31 downto 0) := envData(8);
               else
                  v.mAxisMaster.tData(31 downto 0) := ZEROWORD_C;
               end if;
               if (r.wordCnt = conv_std_logic_vector(WORDS_PER_SUPER_ROW_C*2-1,r.wordCnt'length)) then
                  v.wordCnt := (others => '0');
                  v.state   := TPS_DATA_S;
               end if;
            when TPS_DATA_S =>
               v.wordCnt         := r.wordCnt + 1;
               v.mAxisMaster.tValid := '1';
               if (r.wordCnt = 0) then
                  v.mAxisMaster.tData(31 downto 0) := tpsData(1) & tpsData(0);
               elsif (r.wordCnt = 1) then
                  v.mAxisMaster.tData(31 downto 0) := tpsData(3) & tpsData(2);            
               end if;
               if (r.wordCnt = 1) then
                  v.state := FOOTER_S;
               end if;
            when FOOTER_S =>
               ssiSetUserEofe(MASTER_AXI_STREAM_CONFIG_G,v.mAxisMaster,r.error);
               v.mAxisMaster.tData(31 downto 0) := ZEROWORD_C;
               v.mAxisMaster.tValid             := '1';
               v.mAxisMaster.tLast              := '1';
               v.readDone                       := '1';
               v.state                          := IDLE_S;
         end case;
      end if;
 
      -- Synchronous Reset
      if sysClkRst = '1' then
         v := REG_INIT_C;
      end if;

      -- Register the variable for next clock cycle
      rin <= v;
      
      -- Outputs from block
      readDone    <= r.readDone;
      mAxisMaster <= r.mAxisMaster;
      mpsOut      <= '0';
      
   end process comb;
 
 
   seq : process (sysClk) is
   begin
      if rising_edge(sysClk) then
         r <= rin after TPD_G;
      end if;
   end process seq;
   
   --Sequence/frame counter
   process ( sysClk, sysClkRst ) begin
      if ( sysClkRst = '1' ) then
         intSeqCount <= (others => '0');
      elsif rising_edge(sysClk) then
         if epixConfig.seqCountReset = '1' then
            intSeqCount <= (others => '0') after TPD_G;
         elsif r.seqCountEn = '1' then
            intSeqCount <= intSeqCount + 1 after TPD_G;
          end if;
      end if;
   end process;

   --Simple logic to choose which memory to read from
   process(sysClk) begin
      if rising_edge(sysClk) then
         for i in 0 to 15 loop
            adcFifoWrEn(i)   <= adcMemRdValid(i);
            if r.testPattern = '0' then
               adcFifoWrData(i) <= adcMemRdData(i);
            else
               adcFifoWrData(i) <= "0000" & conv_std_logic_vector(i,4) & adcMemRdData(i)(7 downto 0);
            end if;
         end loop;
      end if;
   end process;

   --Blockrams to reorder data
   --Memory and fifos are reset on system reset or on IDLE state
   memRst <= r.clearFifos or sysClkRst;
   --Generate logic
   G_RowBuffers : for i in 0 to 15 generate
      process(sysClk) begin
         if rising_edge(sysClk) then
            if (adcValid(i) = '1') then
               adcDataToReorder(i) <= "00" & r.adcData(i)(13 downto 0);
            end if;
         end if;
      end process;
 
      --Write when the ADC block says data is good AND when AcqControl agrees
      process(sysClk) begin
         if rising_edge(sysClk) then
            adcMemWrEn(i) <= readValid(i) and adcPulse;
         end if;
      end process;

      --Instantiate memory
      U_RowBuffer : entity work.EpixRowBlockRam
      generic map (
         TPD_G        => TPD_G,
         ASIC_TYPE_G  => ASIC_TYPE_G)
      port map (
         sysClk      => sysClk,
         sysClkRst   => sysClkRst,
         wrReset     => r.clearFifos,
         wrData      => adcDataToReorder(i),
         wrEn        => adcMemWrEn(i),
         rdOrder     => adcMemRdOrder(i),
         rdReady     => adcMemRdRdy(i),
         rdStart     => adcMemRdRdy(i),
         overflow    => adcMemOflow(i),
         rdData      => adcMemRdData(i),
         dataValid   => adcMemRdValid(i),
         testPattern => r.testPattern
      );
      
   end generate;
   --Or of all memory overflow bits
   process(sysClk) 
      variable runningOr : std_logic := '0';
   begin
      if rising_edge(sysClk) then
         runningOr := '0';
         for i in 0 to 15 loop
            runningOr := runningOr or adcMemOflow(i);
         end loop;
         adcMemOflowAny <= runningOr;
      end if;
   end process;
   
   -- Instantiate FIFOs
   G_AdcFifos : for i in 0 to 15 generate
      --Instantiate the FIFOs
      U_AdcFifo : entity surf.FifoMux
         generic map(
            WR_DATA_WIDTH_G => 16,
            RD_DATA_WIDTH_G => 32,
            GEN_SYNC_FIFO_G => true,
            ADDR_WIDTH_G    => CH_FIFO_ADDR_WIDTH_C,
            FWFT_EN_G       => true,
            EMPTY_THRES_G   => 1,
            LITTLE_ENDIAN_G => true
         )
         port map(
            rst           => memRst,
            --Write ports
            wr_clk        => sysClk,
            wr_en         => adcFifoWrEn(i),
            din           => adcFifoWrData(i),
            overflow      => adcFifoOflow(i),
            --Read ports
            rd_clk        => sysClk,
            rd_en         => adcFifoRdEn(i),
            dout          => adcFifoRdData(i),
            valid         => adcFifoRdValid(i),
            empty         => adcFifoEmpty(i)
         );
   end generate;
   --Or of all fifo overflow bits
   --And of all fifo empty bits
   PROC_FIFO_LOGIC : process(sysClk) 
      variable runningOr : std_logic := '0';
      variable runningAnd : std_logic := '0';
   begin
      if rising_edge(sysClk) then
         runningOr := '0';
         runningAnd := '1';
         for i in 0 to 15 loop
            runningOr := runningOr or adcFifoOflow(i);
            runningAnd := runningAnd and adcFifoEmpty(i);
         end loop;
         fifoOflowAny <= runningOr;
         fifoEmptyAll <= runningAnd;
      end if;
   end process;
   
end ReadoutControl;

