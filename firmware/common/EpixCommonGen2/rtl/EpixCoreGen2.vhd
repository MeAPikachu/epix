-------------------------------------------------------------------------------
-- File       : EpixCoreGen2.vhd
-- Company    : SLAC National Accelerator Laboratory
-------------------------------------------------------------------------------
-- Description: 
-------------------------------------------------------------------------------
-- This file is part of 'SLAC Firmware Standard Library'.
-- It is subject to the license terms in the LICENSE.txt file found in the 
-- top-level directory of this distribution and at: 
--    https://confluence.slac.stanford.edu/display/ppareg/LICENSE.html. 
-- No part of 'SLAC Firmware Standard Library', including this file, 
-- may be copied, modified, propagated, or distributed except according to 
-- the terms contained in the LICENSE.txt file.
-------------------------------------------------------------------------------
--

library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;
use IEEE.MATH_REAL.all;

library surf;
use surf.StdRtlPkg.all;
use surf.AxiPkg.all;
use surf.AxiLitePkg.all;
use surf.AxiStreamPkg.all;
use surf.SsiPkg.all;
use surf.SsiCmdMasterPkg.all;
use surf.Ad9249Pkg.all;

use work.EpixPkgGen2.all;

library unisim;
use unisim.vcomponents.all;

entity EpixCoreGen2 is
   generic (
      TPD_G             : time := 1 ns;
      PGP_VER           : string          := "PGP2B";       -- "PGP2B" or "PGP3"
      ASIC_TYPE_G       : AsicType;
      BUILD_INFO_G      : BuildInfoType;
      ADC0_INVERT_CH    : slv(7 downto 0) := "00000000";
      ADC1_INVERT_CH    : slv(7 downto 0) := "00000000";
      ADC2_INVERT_CH    : slv(7 downto 0) := "00000000";
      IODELAY_GROUP_G   : string          := "DEFAULT_GROUP";
      SIMULATION_G      : boolean         := false
   );
   port (
      -- Debugging IOs
      led                 : out slv(3 downto 0);
      -- Power enables
      digitalPowerEn      : out sl;
      analogPowerEn       : out sl;
      fpgaOutputEn        : out sl;
      ledEn               : out sl;
      -- Clocks and reset
      powerGood           : in  sl;
      gtRefClk0P          : in  sl;
      gtRefClk0N          : in  sl;
      -- SFP interfaces
      sfpDisable          : out sl;
      -- SFP TX/RX
      gtDataRxP           : in  sl;
      gtDataRxN           : in  sl;
      gtDataTxP           : out sl;
      gtDataTxN           : out sl;
      -- Guard ring DAC
      vGuardDacSclk       : out sl;
      vGuardDacDin        : out sl;
      vGuardDacCsb        : out sl;
      vGuardDacClrb       : out sl;
      -- External Signals
      runTrigger          : in  sl;
      daqTrigger          : in  sl;
      mpsOut              : in sl;
      triggerOut          : out sl;
      -- Board IDs
      serialIdIo          : inout slv(1 downto 0) := "00";
      -- Slow ADC
      slowAdcRefClk       : out sl;
      slowAdcSclk         : out sl;
      slowAdcDin          : out sl;
      slowAdcCsb          : out sl;
      slowAdcDout         : in  sl;
      slowAdcDrdy         : in  sl;
      -- SACI (allow for point-to-point interfaces)
      saciClk             : out sl;
      saciSelL            : out slv(3 downto 0);
      saciCmd             : out sl;
      saciRsp             : in  sl;
      -- Fast ADC Control
      adcSpiClk           : out sl;
      adcSpiDataIn        : in  sl;
      adcSpiDataOut       : out sl;
      adcSpiDataEn        : out sl;
      adcSpiCsb           : out slv(2 downto 0);
      adcPdwn             : out slv(2 downto 0);
      -- Fast ADC readoutCh
      adcClkP             : out slv( 2 downto 0);
      adcClkN             : out slv( 2 downto 0);
      adcFClkP            : in  slv( 2 downto 0);
      adcFClkN            : in  slv( 2 downto 0);
      adcDClkP            : in  slv( 2 downto 0);
      adcDClkN            : in  slv( 2 downto 0);
      adcChP              : in  slv(19 downto 0);
      adcChN              : in  slv(19 downto 0);
      -- ASIC Control
      asicR0              : out sl;
      asicPpmat           : out sl;
      asicPpbe            : out sl;
      asicGrst            : out sl;
      asicAcq             : out sl;
      asic0Dm2            : in  sl;
      asic0Dm1            : in  sl;
      asicRoClk           : out sl;
      asicSync            : out sl;
      -- ASIC digital data
      asicDout            : in  slv(3 downto 0) := "0000";
      -- Boot Memory Ports
      bootCsL             : out sl;
      bootMosi            : out sl;
      bootMiso            : in  sl
--      -- DDR pins
--      ddr3_dq             : inout slv(31 downto 0);
--      ddr3_dqs_n          : inout slv(3 downto 0);
--      ddr3_dqs_p          : inout slv(3 downto 0);
--      ddr3_addr           : out   slv(14 downto 0);
--      ddr3_ba             : out   slv(2 downto 0);
--      ddr3_ras_n          : out   sl;
--      ddr3_cas_n          : out   sl;
--      ddr3_we_n           : out   sl;
--      ddr3_reset_n        : out   sl;
--      ddr3_ck_p           : out   slv(0 to 0);
--      ddr3_ck_n           : out   slv(0 to 0);
--      ddr3_cke            : out   slv(0 to 0);
--      ddr3_cs_n           : out   slv(0 to 0);
--      ddr3_dm             : out   slv(3 downto 0);
--      ddr3_odt            : out   slv(0 to 0)
   );
end EpixCoreGen2;

architecture top_level of EpixCoreGen2 is
   
   --constant AXI_CLK_FREQ_C : real := 120.3125E+6;
   constant AXI_CLK_FREQ_C : real := 129.6875E+6;
   
   constant NUM_AXI_MASTERS_C    : natural := 9;

   constant AXI_VER_INDEX_C      : natural := 0;
   constant ASIC_INDEX_C         : natural := 1;
   constant SACIREGS_INDEX_C     : natural := 2;
   constant PGPSTAT_INDEX_C      : natural := 3;
   constant MONADC_REG_INDEX_C   : natural := 4;
   constant ADC_INDEX_C          : natural := 5;
   constant BOOTMEM_INDEX_C      : natural := 6;
   constant MEM_LOG_INDEX_C      : natural := 7;
   constant SACI_MULPIX_INDEX_C  : natural := 8;

   constant AXI_CONFIG_C : AxiLiteCrossbarMasterConfigArray(NUM_AXI_MASTERS_C-1 downto 0) := genAxiLiteConfig(NUM_AXI_MASTERS_C, x"00000000", 31, 24);

   signal coreClk     : sl;
   signal coreClkRst  : sl;
   signal pgpClk      : sl;
   signal sysRst      : sl;
   signal axiRst      : sl;
   
   signal iDelayCtrlClk : sl;
   signal iDelayCtrlRst : sl;
   
   -- AXI Signals
   signal axiReadMaster   : AxiReadMasterType;
   signal axiReadSlave    : AxiReadSlaveType;
   signal axiWriteMaster  : AxiWriteMasterType;
   signal axiWriteSlave   : AxiWriteSlaveType;
   
   -- AXI-Lite Signals
   signal sAxiReadMaster  : AxiLiteReadMasterArray(2 downto 0);
   signal sAxiReadSlave   : AxiLiteReadSlaveArray(2 downto 0);
   signal sAxiWriteMaster : AxiLiteWriteMasterArray(2 downto 0);
   signal sAxiWriteSlave  : AxiLiteWriteSlaveArray(2 downto 0);
   -- AXI-Lite Signals
   signal mAxiWriteMasters : AxiLiteWriteMasterArray(NUM_AXI_MASTERS_C-1 downto 0); 
   signal mAxiWriteSlaves  : AxiLiteWriteSlaveArray(NUM_AXI_MASTERS_C-1 downto 0); 
   signal mAxiReadMasters  : AxiLiteReadMasterArray(NUM_AXI_MASTERS_C-1 downto 0); 
   signal mAxiReadSlaves   : AxiLiteReadSlaveArray(NUM_AXI_MASTERS_C-1 downto 0); 

   -- AXI-Stream signals
   signal dataAxisMaster      : AxiStreamMasterType;
   signal dataAxisSlave       : AxiStreamSlaveType;
   signal scopeAxisMaster     : AxiStreamMasterType;
   signal scopeAxisSlave      : AxiStreamSlaveType;
   signal monitorAxisMaster   : AxiStreamMasterType;
   signal monitorAxisSlave    : AxiStreamSlaveType;
   signal monEnAxisMaster     : AxiStreamMasterType;
   
   signal envData             : Slv32Array(8 downto 0);
   
   
   -- Command interface
   signal swRun               : sl;
   signal pgpOpCode           : slv(7 downto 0);
   signal pgpOpCodeEn         : sl;
   
   signal delayCtrlRdy     : sl;
   
   -- Power up reset to SERDES block
   signal adcCardPowerUp     : sl;
   signal requestStartupCal  : sl;
   
   signal fpgaReload : sl;
   signal bootSck    : sl;
   
   signal slowAdcData : Slv24Array(8 downto 0);
   
   signal acqStart   : sl;
   signal ddrClk     : sl;
   signal ddrRst     : sl;
   signal calibComplete : sl;
   
   signal iSaciSelL  : slv(3 downto 0);
   signal iSaciClk   : sl;
   signal iSaciCmd   : sl;
   signal iSaciRsp   : sl;
   
   signal mbRst    : sl;
   signal powerBad : sl;
   
   signal adcStreams          : AxiStreamMasterArray(19 downto 0);
   
   
   constant DDR_AXI_CONFIG_C : AxiConfigType := axiConfig(
      ADDR_WIDTH_C => 30,
      DATA_BYTES_C => 16,
      ID_BITS_C    => 4,
      LEN_BITS_C   => 8);
   
   constant START_ADDR_C : slv(DDR_AXI_CONFIG_C.ADDR_WIDTH_C-1 downto 0) := (others => '0');
   constant STOP_ADDR_C  : slv(DDR_AXI_CONFIG_C.ADDR_WIDTH_C-1 downto 0) := (others => '1');
   
   constant SACI_CLK_PERIOD_C : real := saciClkPeriod(ASIC_TYPE_G);
   
   attribute keep : boolean;
   attribute keep of coreClk : signal is true;
   
begin

   
   -- Fixed state logic signals
   sfpDisable     <= '0';
   
   -- SACI signals
   saciSelL       <= iSaciSelL;
   saciClk        <= iSaciClk;
   saciCmd        <= iSaciCmd;
   iSaciRsp       <= saciRsp;
   
   
   ------------------------------------------
   -- Generate clocks from 156.25 MHz PGP  --
   ------------------------------------------
   -- clkIn     : 156.25 MHz PGP
   -- clkOut(0) : 129.6875 MHz system clock (Modify AXI_CLK_FREQ_C when changed)
   U_CoreClockGen0 : entity surf.ClockManager7
   generic map (
      INPUT_BUFG_G         => false,
      FB_BUFG_G            => true,
      NUM_CLOCKS_G         => 1,
      CLKIN_PERIOD_G       => 6.4,
      DIVCLK_DIVIDE_G      => 10,
      --CLKFBOUT_MULT_F_G    => 38.5, 
      CLKFBOUT_MULT_F_G    => 41.5,    -- base 648.4375MHz
      CLKOUT0_DIVIDE_F_G   => 5.0,
      CLKOUT0_PHASE_G      => 0.0,
      CLKOUT0_DUTY_CYCLE_G => 0.5
   )
   port map (
      clkIn     => pgpClk,
      rstIn     => sysRst,
      clkOut(0) => coreClk,
      rstOut(0) => coreClkRst,
      locked    => open
   );
   
   -- clkIn     : 156.25 MHz PGP
   -- clkOut(0) : 200.0 MHz IDELAYCTRL clock
   U_CoreClockGen1 : entity surf.ClockManager7
   generic map (
      INPUT_BUFG_G         => false,
      FB_BUFG_G            => true,
      NUM_CLOCKS_G         => 1,
      CLKIN_PERIOD_G       => 6.4,
      DIVCLK_DIVIDE_G      => 5,
      CLKFBOUT_MULT_F_G    => 32.0,    -- base 1000.0MHz
      CLKOUT0_DIVIDE_F_G   => 5.0,
      CLKOUT0_PHASE_G      => 0.0,
      CLKOUT0_DUTY_CYCLE_G => 0.5
   )
   port map (
      clkIn     => pgpClk,
      rstIn     => sysRst,
      clkOut(0) => iDelayCtrlClk,
      rstOut(0) => iDelayCtrlRst,
      locked    => open
   );
   
   ---------------------
   -- Diagnostic LEDs --
   ---------------------
   led <= "0000";

   ---------------------
   -- PGP Front end   --
   ---------------------
   
   G_PGP2B : if PGP_VER = "PGP2B" generate
   
      U_PgpFrontEnd : entity work.PgpFrontEnd
      generic map (
         TPD_G          => TPD_G,
         SIMULATION_G   => SIMULATION_G
      )
      port map (
         -- GTX 7 Ports
         gtClkP      => gtRefClk0P,
         gtClkN      => gtRefClk0N,
         gtRxP       => gtDataRxP,
         gtRxN       => gtDataRxN,
         gtTxP       => gtDataTxP,
         gtTxN       => gtDataTxN,
         -- Input power status
         powerBad    => powerBad,
         -- Output reset
         pgpRst      => sysRst,
         -- Output clocking
         pgpClk      => pgpClk,
         -- AXI clocking
         axiClk     => coreClk,
         axiRst     => axiRst,
         -- Axi Master Interface - Registers (axiClk domain)
         mAxiLiteReadMaster  => sAxiReadMaster(0),
         mAxiLiteReadSlave   => sAxiReadSlave(0),
         mAxiLiteWriteMaster => sAxiWriteMaster(0),
         mAxiLiteWriteSlave  => sAxiWriteSlave(0),
         -- Axi Slave Interface - PGP Status Registers (axiClk domain)
         sAxiLiteReadMaster  => mAxiReadMasters(PGPSTAT_INDEX_C),
         sAxiLiteReadSlave   => mAxiReadSlaves(PGPSTAT_INDEX_C),
         sAxiLiteWriteMaster => mAxiWriteMasters(PGPSTAT_INDEX_C),
         sAxiLiteWriteSlave  => mAxiWriteSlaves(PGPSTAT_INDEX_C),
         -- Streaming data Links (axiClk domain)      
         dataAxisMaster    => dataAxisMaster,
         dataAxisSlave     => dataAxisSlave,
         scopeAxisMaster   => scopeAxisMaster,
         scopeAxisSlave    => scopeAxisSlave,
         monitorAxisMaster => monitorAxisMaster,
         monitorAxisSlave  => monitorAxisSlave,
         -- Monitoring enable command incoming stream
         monEnAxisMaster   => monEnAxisMaster,
         -- Command interface
         swRun             => swRun,
         -- To access sideband commands
         pgpOpCode         => pgpOpCode,
         pgpOpCodeEn       => pgpOpCodeEn
      );
      powerBad <= not powerGood;
   
   end generate;
   
   G_PGP3 : if PGP_VER = "PGP3" generate
      
      U_Pgp3FrontEnd : entity work.Pgp3FrontEnd
      generic map (
         TPD_G             => TPD_G,
         SIMULATION_G      => SIMULATION_G,
         AXI_CLK_FREQ_G    => AXI_CLK_FREQ_C,
         AXI_BASE_ADDR_G   => AXI_CONFIG_C(PGPSTAT_INDEX_C).baseAddr
      )
      port map (
         -- GTX 7 Ports
         gtClkP      => gtRefClk0P,
         gtClkN      => gtRefClk0N,
         gtRxP       => gtDataRxP,
         gtRxN       => gtDataRxN,
         gtTxP       => gtDataTxP,
         gtTxN       => gtDataTxN,
         -- Input power status
         powerBad    => powerBad,
         -- Output reset
         pgpRst      => sysRst,
         -- Output clocking
         pgpClk      => pgpClk,
         -- AXI clocking
         axiClk     => coreClk,
         axiRst     => axiRst,
         -- Axi Master Interface - Registers (axiClk domain)
         mAxiLiteReadMaster  => sAxiReadMaster(0),
         mAxiLiteReadSlave   => sAxiReadSlave(0),
         mAxiLiteWriteMaster => sAxiWriteMaster(0),
         mAxiLiteWriteSlave  => sAxiWriteSlave(0),
         -- Axi Slave Interface - PGP Status Registers (axiClk domain)
         sAxiLiteReadMaster  => mAxiReadMasters(PGPSTAT_INDEX_C),
         sAxiLiteReadSlave   => mAxiReadSlaves(PGPSTAT_INDEX_C),
         sAxiLiteWriteMaster => mAxiWriteMasters(PGPSTAT_INDEX_C),
         sAxiLiteWriteSlave  => mAxiWriteSlaves(PGPSTAT_INDEX_C),
         -- Streaming data Links (axiClk domain)      
         dataAxisMaster    => dataAxisMaster,
         dataAxisSlave     => dataAxisSlave,
         scopeAxisMaster   => scopeAxisMaster,
         scopeAxisSlave    => scopeAxisSlave,
         monitorAxisMaster => monitorAxisMaster,
         monitorAxisSlave  => monitorAxisSlave,
         -- Monitoring enable command incoming stream
         monEnAxisMaster   => monEnAxisMaster,
         -- Command interface
         swRun             => swRun,
         -- To access sideband commands
         pgpOpCode         => pgpOpCode,
         pgpOpCodeEn       => pgpOpCodeEn
      );
      powerBad <= not powerGood;
   
   end generate;
   
   ---------------------------------------------
   -- Microblaze based ePix Startup Sequencer --
   ---------------------------------------------
   U_MbRst : entity surf.PwrUpRst
      port map (
         clk      => coreClk,
         arst     => axiRst,
         rstOut   => mbRst
      ); 
   
   U_CPU : entity surf.MicroblazeBasicCoreWrapper
   generic map (
      TPD_G            => TPD_G)
   port map (
      -- Master AXI-Lite Interface: [0x00000000:0x7FFFFFFF]
      mAxilWriteMaster => sAxiWriteMaster(1),
      mAxilWriteSlave  => sAxiWriteSlave(1),
      mAxilReadMaster  => sAxiReadMaster(1),
      mAxilReadSlave   => sAxiReadSlave(1),
      -- Interrupt Interface
      interrupt(7 downto 1)   => "0000000",
      interrupt(0)            => requestStartupCal,
      -- Clock and Reset
      clk              => coreClk,
      rst              => mbRst
   );
   
   --------------------------------------------
   -- AXI Lite Crossbar for register control
   -- Master 0 : PGP front end controller
   -- Master 1 : Microblaze startup controller
   -- Master 2 : SACI multipixel controller
   --------------------------------------------
   U_AxiLiteCrossbar : entity surf.AxiLiteCrossbar
   generic map (
      NUM_SLAVE_SLOTS_G  => 3,
      NUM_MASTER_SLOTS_G => NUM_AXI_MASTERS_C,
      MASTERS_CONFIG_G   => AXI_CONFIG_C)
   port map (
      sAxiWriteMasters    => sAxiWriteMaster,
      sAxiWriteSlaves     => sAxiWriteSlave,
      sAxiReadMasters     => sAxiReadMaster,
      sAxiReadSlaves      => sAxiReadSlave,
      mAxiWriteMasters    => mAxiWriteMasters,
      mAxiWriteSlaves     => mAxiWriteSlaves,
      mAxiReadMasters     => mAxiReadMasters,
      mAxiReadSlaves      => mAxiReadSlaves,
      axiClk              => coreClk,
      axiClkRst           => axiRst
   );
   
   --------------------------------------------
   --     ASIC Core                          --
   --------------------------------------------
   U_AsicCore : entity work.AsicCore
   generic map (
      TPD_G                => TPD_G,
      FPGA_BASE_CLOCK_G    => std_logic_vector(to_unsigned(natural(round(AXI_CLK_FREQ_C)), 32)),
      BUILD_INFO_G         => BUILD_INFO_G,
      AXI_CLK_FREQ_G       => AXI_CLK_FREQ_C,
      ASIC_TYPE_G          => ASIC_TYPE_G,
      AXI_BASE_ADDR_G      => AXI_CONFIG_C(ASIC_INDEX_C).baseAddr
   )
   port map (
      -- Clock and Reset
      sysClk               => coreClk,
      sysRst               => sysRst,
      axiRst               => axiRst,
      -- ADC signals
      adcStreams           => adcStreams,
      -- AXI-Lite Register Interface (sysClk domain)
      mAxilReadMaster      => mAxiReadMasters(ASIC_INDEX_C),
      mAxilReadSlave       => mAxiReadSlaves(ASIC_INDEX_C),
      mAxilWriteMaster     => mAxiWriteMasters(ASIC_INDEX_C),
      mAxilWriteSlave      => mAxiWriteSlaves(ASIC_INDEX_C),
      -- ASIC Control
      asicR0               => asicR0   ,
      asicPpmat            => asicPpmat,
      asicPpbe             => asicPpbe ,
      asicGrst             => asicGrst ,
      asicAcq              => asicAcq  ,
      asic0Dm2             => asic0Dm2 ,
      asic0Dm1             => asic0Dm1 ,
      asicRoClk            => asicRoClk,
      asicSync             => asicSync ,
      -- ASIC digital data
      asicDout             => asicDout,
      -- ADC clock
      adcClkP              => adcClkP,
      adcClkN              => adcClkN,
      -- Guard ring DAC
      vGuardDacSclk        => vGuardDacSclk,
      vGuardDacDin         => vGuardDacDin ,
      vGuardDacCsb         => vGuardDacCsb ,
      vGuardDacClrb        => vGuardDacClrb,
      -- Board IDs
      serialIdIo           => serialIdIo,
      -- External Signals
      runTrigger           => runTrigger,
      daqTrigger           => daqTrigger,
      mpsOut               => mpsOut    ,
      triggerOut           => triggerOut,
      -- SW and fiber trigger
      swRun                => swRun,
      pgpOpCode            => pgpOpCode,
      pgpOpCodeEn          => pgpOpCodeEn,
      -- Power enables
      digitalPowerEn       => digitalPowerEn   ,
      analogPowerEn        => analogPowerEn    ,
      fpgaOutputEn         => fpgaOutputEn     ,
      ledEn                => ledEn            ,
      adcCardPowerUp       => adcCardPowerUp   ,
      delayCtrlRdy         => delayCtrlRdy     ,
      requestStartupCal    => requestStartupCal,
      acqStartOut          => acqStart,
      -- env data
      envData              => envData,
      -- Image Data Stream
      dataAxisMaster       => dataAxisMaster,
      dataAxisSlave        => dataAxisSlave ,
      -- Scope Data Stream
      scopeAxisMaster      => scopeAxisMaster,
      scopeAxisSlave       => scopeAxisSlave
   );
   
   --------------------------------------------
   -- SACI interface controller              --
   -------------------------------------------- 
   U_AxiLiteSaciMaster : entity surf.AxiLiteSaciMaster
   generic map (
      AXIL_CLK_PERIOD_G  => 1.0/AXI_CLK_FREQ_C, -- In units of seconds
      AXIL_TIMEOUT_G     => 1.0E-3,  -- In units of seconds
      SACI_CLK_PERIOD_G  => SACI_CLK_PERIOD_C, -- In units of seconds
      SACI_CLK_FREERUN_G => false,
      SACI_RSP_BUSSED_G  => true,
      SACI_NUM_CHIPS_G   => 4)
   port map (
      -- SACI interface
      saciClk           => iSaciClk,
      saciCmd           => iSaciCmd,
      saciSelL          => iSaciSelL,
      saciRsp(0)        => iSaciRsp,
      -- AXI-Lite Register Interface
      axilClk           => coreClk,
      axilRst           => axiRst,
      axilReadMaster    => mAxiReadMasters(SACIREGS_INDEX_C),
      axilReadSlave     => mAxiReadSlaves(SACIREGS_INDEX_C),
      axilWriteMaster   => mAxiWriteMasters(SACIREGS_INDEX_C),
      axilWriteSlave    => mAxiWriteSlaves(SACIREGS_INDEX_C)
   );
   
   ---------------------------------------------
   -- SACI multipixel configurator Master     --
   -- This is used only by epix100a LCLS software due to initial issuees with ASIC
   -- Should this be removed?
   ---------------------------------------------
   G_MULTIPIX_EPIX10KA : if ASIC_TYPE_G = EPIX100A_C generate
   
      U_SaciMultiPixel : entity surf.SaciMultiPixel
      generic map (
         MASK_REG_ADDR_G   => x"01000034",
         SACI_BASE_ADDR_G  => x"02000000"
      )
      port map (
         axilClk           => coreClk,
         axilRst           => axiRst,
         
         -- AXI lite slave port
         sAxilWriteMaster => mAxiWriteMasters(SACI_MULPIX_INDEX_C),
         sAxilWriteSlave  => mAxiWriteSlaves(SACI_MULPIX_INDEX_C),
         sAxilReadMaster  => mAxiReadMasters(SACI_MULPIX_INDEX_C),
         sAxilReadSlave   => mAxiReadSlaves(SACI_MULPIX_INDEX_C),
         
         -- AXI lite master port
         mAxilWriteMaster  => sAxiWriteMaster(2),
         mAxilWriteSlave   => sAxiWriteSlave(2),
         mAxilReadMaster   => sAxiReadMaster(2),
         mAxilReadSlave    => sAxiReadSlave(2)
      );
   
   end generate;
   G_NO_MULTIPIX_EPIX10KA : if ASIC_TYPE_G /= EPIX100A_C generate
      mAxiWriteSlaves(SACI_MULPIX_INDEX_C) <= AXI_LITE_WRITE_SLAVE_INIT_C;
      mAxiReadSlaves(SACI_MULPIX_INDEX_C)  <= AXI_LITE_READ_SLAVE_INIT_C;
      sAxiWriteMaster(2) <= AXI_LITE_WRITE_MASTER_INIT_C;
      sAxiReadMaster(2) <= AXI_LITE_READ_MASTER_INIT_C;
   end generate;
   
   --------------------------------------------
   --     Fast ADC Readout                   --
   --------------------------------------------
   
   U_AdcPhyTop : entity work.AdcPhyTop
   generic map (
      TPD_G               => TPD_G,
      CLK_PERIOD_G        => 1.0/AXI_CLK_FREQ_C,
      IDELAYCTRL_FREQ_G   => 200.0,
      AXI_BASE_ADDR_G     => AXI_CONFIG_C(ADC_INDEX_C).baseAddr,
      ADC0_INVERT_CH      => ADC0_INVERT_CH,
      ADC1_INVERT_CH      => ADC1_INVERT_CH,
      ADC2_INVERT_CH      => ADC2_INVERT_CH,
      IODELAY_GROUP_G     => IODELAY_GROUP_G
   )
   port map (
      -- Clocks and reset
      coreClk             => coreClk,
      coreRst             => axiRst,
      delayCtrlClk        => iDelayCtrlClk,
      delayCtrlRst        => iDelayCtrlRst,
      delayCtrlRdy        => delayCtrlRdy,
      adcCardPowerUp      => adcCardPowerUp,
      -- AXI Lite Bus
      axilReadMaster      => mAxiReadMasters(ADC_INDEX_C),
      axilReadSlave       => mAxiReadSlaves(ADC_INDEX_C),
      axilWriteMaster     => mAxiWriteMasters(ADC_INDEX_C),
      axilWriteSlave      => mAxiWriteSlaves(ADC_INDEX_C),
      -- Fast ADC Control
      adcSpiClk           => adcSpiClk,
      adcSpiDataIn        => adcSpiDataIn,
      adcSpiDataOut       => adcSpiDataOut,
      adcSpiDataEn        => adcSpiDataEn,
      adcSpiCsb           => adcSpiCsb,
      adcPdwn             => adcPdwn,
      -- Fast ADC readoutCh
      adcFClkP            => adcFClkP,
      adcFClkN            => adcFClkN,
      adcDClkP            => adcDClkP,
      adcDClkN            => adcDClkN,
      adcChP              => adcChP,
      adcChN              => adcChN,
      -- ADC data output
      adcStreams          => adcStreams
   );
   
   
   --------------------------------------------
   --     Slow ADC Readout ADC gen 2         --
   -------------------------------------------- 
   U_AdcCntrl: entity work.SlowAdcCntrlAxi
   generic map (
      SYS_CLK_PERIOD_G  => 1.0/AXI_CLK_FREQ_C,	   -- 100MHz
      ADC_CLK_PERIOD_G  => 200.0E-9,	-- 5MHz
      SPI_SCLK_PERIOD_G => 2.0E-6  	   -- 500kHz
   )
   port map ( 
      -- Master system clock
      sysClk            => coreClk,
      sysClkRst         => axiRst,
      
      -- Trigger Control
      adcStart          => acqStart,
      
      -- Monitoring enable command incoming stream
      monEnAxisMaster   => monEnAxisMaster,
      
      -- Env data outputs
      envData           => envData,
      
      -- AXI lite slave port for register access
      axilClk           => coreClk,
      axilRst           => axiRst,
      sAxilWriteMaster  => mAxiWriteMasters(MONADC_REG_INDEX_C),
      sAxilWriteSlave   => mAxiWriteSlaves(MONADC_REG_INDEX_C),
      sAxilReadMaster   => mAxiReadMasters(MONADC_REG_INDEX_C),
      sAxilReadSlave    => mAxiReadSlaves(MONADC_REG_INDEX_C),
      
      -- AXI stream output
      axisClk           => coreClk,
      axisRst           => axiRst,
      mAxisMaster       => monitorAxisMaster,
      mAxisSlave        => monitorAxisSlave,

      -- ADC Control Signals
      adcRefClk         => slowAdcRefClk,
      adcDrdy           => slowAdcDrdy,
      adcSclk           => slowAdcSclk,
      adcDout           => slowAdcDout,
      adcCsL            => slowAdcCsb,
      adcDin            => slowAdcDin
   );
   
   ---------------------------------------------
   -- Microblaze log memory                   --
   ---------------------------------------------
   U_LogMem : entity surf.AxiDualPortRam
   generic map (
      TPD_G            => TPD_G,
      ADDR_WIDTH_G     => 10,
      DATA_WIDTH_G     => 32
   )
   port map (
      axiClk         => coreClk,
      axiRst         => axiRst,
      axiReadMaster  => mAxiReadMasters(MEM_LOG_INDEX_C),
      axiReadSlave   => mAxiReadSlaves(MEM_LOG_INDEX_C),
      axiWriteMaster => mAxiWriteMasters(MEM_LOG_INDEX_C),
      axiWriteSlave  => mAxiWriteSlaves(MEM_LOG_INDEX_C),
      clk            => '0',
      en             => '1',
      we             => '0',
      weByte         => (others => '0'),
      rst            => '0',
      addr           => (others => '0'),
      din            => (others => '0'),
      dout           => open,
      axiWrValid     => open,
      axiWrStrobe    => open,
      axiWrAddr      => open,
      axiWrData      => open
   );
   
   --------------------------
   -- AXI-Lite Version Module
   --------------------------          
   U_AxiVersion : entity surf.AxiVersion
   generic map (
      TPD_G           => TPD_G,
      BUILD_INFO_G    => BUILD_INFO_G,
      EN_DEVICE_DNA_G => false)   
   port map (
      fpgaReload     => fpgaReload,
      -- AXI-Lite Register Interface
      axiReadMaster  => mAxiReadMasters(AXI_VER_INDEX_C),
      axiReadSlave   => mAxiReadSlaves(AXI_VER_INDEX_C),
      axiWriteMaster => mAxiWriteMasters(AXI_VER_INDEX_C),
      axiWriteSlave  => mAxiWriteSlaves(AXI_VER_INDEX_C),
      -- Clocks and Resets
      axiClk         => coreClk,
      axiRst         => axiRst
   );

   ---------------------
   -- FPGA Reboot Module
   ---------------------
   U_Iprog7Series : entity surf.Iprog7Series
   generic map (
      TPD_G => TPD_G)   
   port map (
      clk         => coreClk,
      rst         => axiRst,
      start       => fpgaReload,
      bootAddress => X"00000000"
   );
   
   -----------------------------------------------------
   -- Using the STARTUPE2 to access the FPGA's CCLK port
   -----------------------------------------------------
   U_STARTUPE2 : STARTUPE2
   port map (
      CFGCLK    => open,             -- 1-bit output: Configuration main clock output
      CFGMCLK   => open,  -- 1-bit output: Configuration internal oscillator clock output
      EOS       => open,  -- 1-bit output: Active high output signal indicating the End Of Startup.
      PREQ      => open,             -- 1-bit output: PROGRAM request to fabric output
      CLK       => '0',              -- 1-bit input: User start-up clock input
      GSR       => '0',  -- 1-bit input: Global Set/Reset input (GSR cannot be used for the port name)
      GTS       => '0',  -- 1-bit input: Global 3-state input (GTS cannot be used for the port name)
      KEYCLEARB => '0',  -- 1-bit input: Clear AES Decrypter Key input from Battery-Backed RAM (BBRAM)
      PACK      => '0',              -- 1-bit input: PROGRAM acknowledge input
      USRCCLKO  => bootSck,          -- 1-bit input: User CCLK input
      USRCCLKTS => '0',              -- 1-bit input: User CCLK 3-state enable input
      USRDONEO  => '1',              -- 1-bit input: User DONE pin output control
      USRDONETS => '1'               -- 1-bit input: User DONE 3-state enable output            
   );
   
   --------------------
   -- Boot Flash Module
   --------------------
   U_AxiMicronN25QCore : entity surf.AxiMicronN25QCore
   generic map (
      TPD_G          => TPD_G,
      AXI_CLK_FREQ_G => AXI_CLK_FREQ_C,   -- units of Hz
      SPI_CLK_FREQ_G => 25.0E+6     -- units of Hz
   )
   port map (
      -- FLASH Memory Ports
      csL            => bootCsL,
      sck            => bootSck,
      mosi           => bootMosi,
      miso           => bootMiso,
      -- AXI-Lite Register Interface
      axiReadMaster  => mAxiReadMasters(BOOTMEM_INDEX_C),
      axiReadSlave   => mAxiReadSlaves(BOOTMEM_INDEX_C),
      axiWriteMaster => mAxiWriteMasters(BOOTMEM_INDEX_C),
      axiWriteSlave  => mAxiWriteSlaves(BOOTMEM_INDEX_C),
      -- Clocks and Resets
      axiClk         => coreClk,
      axiRst         => axiRst
   );
   
   
--   --------------------
--   -- DDR memory controller
--   --------------------
--   U_AxiDdrControllerWrapper : entity work.AxiDdrControllerWrapper
--   port map ( 
--      -- AXI Slave     
--      axiReadMaster     => axiReadMaster,
--      axiReadSlave      => axiReadSlave,
--      axiWriteMaster    => axiWriteMaster,
--      axiWriteSlave     => axiWriteSlave,
--      
--      -- DDR PHY Ref clk
--      sysClk            => refClk,
--      dlyClk            => iDelayCtrlClk,
--      
--      -- DDR clock from the DDR controller core
--      ddrClk            => ddrClk,
--      ddrRst            => ddrRst,
--
--      -- DRR Memory interface ports
--      ddr3_dq           => ddr3_dq,
--      ddr3_dqs_n        => ddr3_dqs_n,
--      ddr3_dqs_p        => ddr3_dqs_p,
--      ddr3_addr         => ddr3_addr,
--      ddr3_ba           => ddr3_ba,
--      ddr3_ras_n        => ddr3_ras_n,
--      ddr3_cas_n        => ddr3_cas_n,
--      ddr3_we_n         => ddr3_we_n,
--      ddr3_reset_n      => ddr3_reset_n,
--      ddr3_ck_p         => ddr3_ck_p,
--      ddr3_ck_n         => ddr3_ck_n,
--      ddr3_cke          => ddr3_cke,
--      ddr3_cs_n         => ddr3_cs_n,
--      ddr3_dm           => ddr3_dm,
--      ddr3_odt          => ddr3_odt,
--      calibComplete     => calibComplete
--   );
--   
--   U_AxiMemTester : entity surf.AxiMemTester
--   generic map (
--      TPD_G            => TPD_G,
--      START_ADDR_G     => START_ADDR_C,
--      STOP_ADDR_G      => STOP_ADDR_C,
--      AXI_CONFIG_G     => DDR_AXI_CONFIG_C
--   )
--   port map (
--      -- AXI-Lite Interface
--      axilClk         => coreClk,
--      axilRst         => axiRst,
--      axilReadMaster  => mAxiReadMasters(TESTMEM_AXI_INDEX_C),
--      axilReadSlave   => mAxiReadSlaves(TESTMEM_AXI_INDEX_C),
--      axilWriteMaster => mAxiWriteMasters(TESTMEM_AXI_INDEX_C),
--      axilWriteSlave  => mAxiWriteSlaves(TESTMEM_AXI_INDEX_C),
--      memReady        => open,
--      memError        => open,
--      -- DDR Memory Interface
--      axiClk          => ddrClk,
--      axiRst          => ddrRst,
--      start           => calibComplete,
--      axiWriteMaster  => axiWriteMaster,
--      axiWriteSlave   => axiWriteSlave,
--      axiReadMaster   => axiReadMaster,
--      axiReadSlave    => axiReadSlave
--   );
   
end top_level;
