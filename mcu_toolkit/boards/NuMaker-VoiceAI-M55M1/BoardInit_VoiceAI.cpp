/**************************************************************************//**
 * @file     BoardInit_VoiceAI.cpp
 * @version  V1.00
 * @brief    Target board initiate function -- NuMaker-VoiceAI-M55M1
 *
 * One file, two uses.  tools/board_support.py installs it over the NuML
 * generic template's BoardInit.cpp after project generation (the toolkit has
 * no VoiceAI template; the project is generated from the GestureAI one), and
 * known_sound/tools/install_device_project.py copies it into the standalone
 * verifier.  It must therefore not depend on anything only one of those
 * projects has -- BoardConfig.h in particular is optional here.
 *
 * Board facts.  From UM_NuMaker-VoiceAI-M55M1_EN_Rev0.01 sheets "M55M1_LQFP64",
 * "MEMS Digital MIC", "UART" (extension connectors) and "Leds and Buttons",
 * and from the board itself read back over SWD (SYS PDID 0xA2000020, Ethos-U
 * ID register 0x10104201, APROM readable to exactly 2 MiB):
 *
 *   MCU        M55M1R2LJC7E per the UM, LQFP64.  Nuvoton's tools file this
 *              part under M5531: Nu-Link's Chip Select must be M5531 for
 *              Download to accept it, and the Keil pack lists M5531R2LJAE in
 *              the M5531 subfamily with no NPU component.  The Ethos-U55 is
 *              nonetheless present -- ID register 0x10104201, the value the
 *              BSP driver expects -- and the Known Sound verifier passes all
 *              14 vectors on it.  It is initialised below exactly as on the
 *              other two boards.
 *   MEMS mic   MP34DT05-M (U4) on the FIRST PDM pin pair:
 *                PA.4 = DMIC0_CLK, PA.5 = DMIC0_DAT   ->  DMIC channel 0.
 *              U10 is unpopulated, so the board is mono.  R79 (0 R) ties the
 *              DMIC0_CLKLP net (PA.3) to DMIC0_CLK: PA.3 must never also be
 *              muxed as CLKLP or two outputs drive one net.  The pin mux is
 *              not done here -- the audio runtimes do it from the board
 *              profile (BOARD_DMIC_SET_CLK()/BOARD_DMIC_SET_DAT() in the Known
 *              Sound pack, the rendered NuML_StartDmic() in the KWS pack) so
 *              the pins and the channel mask come from one table.
 *   Console    UART4, PB.11 TXD / PB.10 RXD, 5-pin header J3, 115200 8N1.
 *              PB.4/PB.5 -- the GestureAI template's UART5 -- are I2S0 to the
 *              NAU88L21 codec on this board and are not brought out.
 *   USB        HSUSB Type-C J1, device only (CC1/CC2 5.1 K to GND, ID open).
 *              printf is tee'd to a CDC virtual COM port by the runtimes.
 *   LEDs       LED_Y PC.4, LED_G PC.5, both active LOW (VDD - 330 R - LED).
 *   Button     BTN1 PA.7, active LOW with a 10 K pull-up.  Input only: the
 *              switch shorts the pin to GND when pressed.
 *   SD card    SD1: nCD PA.6, CLK PB.6, CMD PB.7, DAT0-3 PA.8-PA.11.  Not
 *              brought up.  The generic template's SDCard1_PinConfig() puts
 *              SD1_CMD on PA.5 -- this board's microphone data pin -- and the
 *              audio runtimes keep their model in internal flash anyway.
 *
 * @copyright SPDX-License-Identifier: Apache-2.0
 * @copyright Copyright (C) 2023 Nuvoton Technology Corp. All rights reserved.
 ******************************************************************************/
#include <cstdio>

#include "NuMicro.h"
#include "log_macros.h"

#include "ethosu_npu_init.h"

/* The Known Sound verifier compiles this file with BoardConfig.h present and a
 * board define selected; the NuML generic project has no BoardConfig.h at all.
 * Check the define only when the header is there to define it. */
#if defined(__has_include)
    #if __has_include("BoardConfig.h")
        #include "BoardConfig.h"
        #if !defined(BOARD_NUMAKER_VOICEAI_M55M1)
            #error "BoardInit_VoiceAI.cpp built without -DBOARD_NUMAKER_VOICEAI_M55M1"
        #endif
    #endif
#endif

#define DESIGN_NAME "NuMaker-VoiceAI-M55M1"

/* The console on this board is UART4.  The whole DEBUG_PORT_* family derives
 * from DEBUG_PORT_UART_IDX (DEBUG_PORT, _MODULE, _CLKSEL, _CLKDIV, _RST), and
 * retarget.c's SendChar_ToUART() spins on DEBUG_PORT->FIFOSTS -- so an index
 * that names a UART this firmware never clocks makes printf hang, not lose a
 * character.  Fail at compile time instead.
 *
 * DEBUG_PORT=UART4 (the shape of define the GestureAI template uses) is not
 * enough: it changes DEBUG_PORT alone and leaves the module/clock/reset
 * macros on UART0. */
#if !defined(DEBUG_PORT_UART_IDX) || (DEBUG_PORT_UART_IDX != 4)
    #error "NuMaker-VoiceAI-M55M1 console is UART4 on PB.11/PB.10 (header J3): add DEBUG_PORT_UART_IDX=4 to the target-wide defines"
#endif

/* The BSP's SetDebugUartMFP() (system_M55M1.c) has branches for UART index 0,
 * 5 and 6 only; any other index stops the build with "[Miss] Set UART MFP"
 * from inside a library file.  NVT_DBG_UART_OFF removes that whole block --
 * the three __WEAK functions and their prototypes -- so the UART4 versions
 * are supplied here.  Nothing else in the BSP or the templates calls them. */
#if !defined(NVT_DBG_UART_OFF)
    #error "NVT_DBG_UART_OFF must be defined target-wide: the BSP SetDebugUartMFP() has no UART4 branch"
#endif

/* UART4 lives in UARTSEL0/UARTDIV0; clk.h has no CLK_UARTSEL1_UART4SEL_*. */
#if defined(DEBUG_PORT_UART_GRP_IDX) && (DEBUG_PORT_UART_GRP_IDX != 0)
    #error "DEBUG_PORT_UART_GRP_IDX must stay 0 for UART4"
#endif

static void SetDebugUartMFP(void)
{
    /* Console on header J3: PB.11 = UART4_TXD, PB.10 = UART4_RXD. */
    SET_UART4_RXD_PB10();
    SET_UART4_TXD_PB11();
}

static void SetDebugUartCLK(void)
{
    /* With DEBUG_PORT_UART_IDX=4 these resolve to UART4_MODULE,
     * CLK_UARTSEL0_UART4SEL_HIRC, CLK_UARTDIV0_UART4DIV(1) and SYS_UART4RST. */
    CLK_SetModuleClock(DEBUG_PORT_MODULE, DEBUG_PORT_CLKSEL, DEBUG_PORT_CLKDIV);
    CLK_EnableModuleClock(DEBUG_PORT_MODULE);
    SYS_ResetModule(DEBUG_PORT_RST);
}

static void InitDebugUart(void)
{
    UART_Open(DEBUG_PORT, 115200);
}

static void SYS_Init(void)
{
    /*---------------------------------------------------------------------------------------------------------*/
    /* Init System Clock                                                                                       */
    /*---------------------------------------------------------------------------------------------------------*/

    /* Enable Internal RC 12MHz clock */
    CLK_EnableXtalRC(CLK_SRCCTL_HIRCEN_Msk);

    /* Waiting for Internal RC clock ready */
    CLK_WaitClockReady(CLK_STATUS_HIRCSTB_Msk);

    /* Enable HXT clock (24 MHz crystal X1 on this board) */
    CLK_EnableXtalRC(CLK_SRCCTL_HXTEN_Msk);

    /* Waiting for HXT clock ready */
    CLK_WaitClockReady(CLK_STATUS_HXTSTB_Msk);

    /* Switch SCLK clock source to APLL0 and Enable APLL0 220MHz clock */
    CLK_SetBusClock(CLK_SCLKSEL_SCLKSEL_APLL0, CLK_APLLCTL_APLLSRC_HXT, FREQ_220MHZ);

    /* Update System Core Clock */
    SystemCoreClockUpdate();

    /* Enable GPIO module clock */
    CLK_EnableModuleClock(GPIOA_MODULE);
    CLK_EnableModuleClock(GPIOB_MODULE);
    CLK_EnableModuleClock(GPIOC_MODULE);
    CLK_EnableModuleClock(GPIOD_MODULE);
    CLK_EnableModuleClock(GPIOE_MODULE);
    CLK_EnableModuleClock(GPIOF_MODULE);
    CLK_EnableModuleClock(GPIOG_MODULE);
    CLK_EnableModuleClock(GPIOH_MODULE);
    CLK_EnableModuleClock(GPIOI_MODULE);
    CLK_EnableModuleClock(GPIOJ_MODULE);

    /* Enable FMC0 module clock to keep FMC clock when CPU idle but NPU running */
    CLK_EnableModuleClock(FMC0_MODULE);

    /* Enable NPU module clock.  Nuvoton's UAC sample for this board leaves it
     * off, which is why the Ethos-U ID block reads as zeros until it is set. */
    CLK_EnableModuleClock(NPU0_MODULE);

    /* UART4 clock source HIRC, divider 1 -- our own function above. */
    SetDebugUartCLK();

    /* DMIC_APLL1_FREQ_196608KHZ supports 8000/16000/48000 Hz sample rates with
     * 64/128/256 down-sample.  Independent of SCLK: changing the core clock
     * does not change the PDM bit clock.  APLL1 is free for this here -- the
     * GestureAI template hands it to SDH1, which this board init does not
     * bring up. */
    CLK_EnableAPLL(CLK_APLLCTL_APLLSRC_HIRC, DMIC_APLL1_FREQ_196608KHZ, CLK_APLL1_SELECT);
    /* Select DMIC CLK source from APLL1_DIV2. */
    CLK_SetModuleClock(DMIC0_MODULE, CLK_DMICSEL_DMIC0SEL_APLL1_DIV2, MODULE_NoMsk);
    /* Enable DMIC clock. */
    CLK_EnableModuleClock(DMIC0_MODULE);
    /* DMIC IP reset. */
    SYS_ResetModule(SYS_DMIC0RST);

    /* LPPDMA init. */
    CLK_EnableModuleClock(LPPDMA0_MODULE);
    CLK_EnableModuleClock(LPSRAM0_MODULE);
    SYS_ResetModule(SYS_LPPDMA0RST);

    /* Enable HSOTG module clock */
    CLK_EnableModuleClock(HSOTG0_MODULE);

    /* Select HSOTG PHY Reference clock frequency which is from HXT (24 MHz) */
    HSOTG_SET_PHY_REF_CLK(HSOTG_PHYCTL_FSEL_24_0M);

    /* Set HSUSB role to HSUSBD.  The Type-C connector is wired as a UFP
     * (5.1 K on both CC lines, ID left open), so device is the only valid
     * role on this board. */
    SET_HSUSBDROLE();

    /* Enable HSUSB PHY */
    SYS_Enable_HSUSB_PHY();

    /* Enable HSUSBD peripheral clock */
    CLK_EnableModuleClock(HSUSBD0_MODULE);

    /*---------------------------------------------------------------------------------------------------------*/
    /* Init I/O Multi-function                                                                                 */
    /*---------------------------------------------------------------------------------------------------------*/

    /* Debug UART pins: PB.11 TXD / PB.10 RXD.  Our own function above. */
    SetDebugUartMFP();

    /* No DMIC pin mux here -- see the header comment.  Nothing below may touch
     * PA.4 or PA.5.
     *
     * No SD1 pin config here either: the generic template's
     * SDCard1_PinConfig() would route SD1_CMD to PA.5. */

    /* Status LEDs and the user button as plain GPIO. */
    SET_GPIO_PC4();
    SET_GPIO_PC5();
    SET_GPIO_PA7();
}

/**
  * @brief Initiate the hardware resources of board
  * @return 0: Success, <0: Fail
  * @details Initiate clock, UART, NPU
  * \hideinitializer
  */
int BoardInit(void)
{
    /* Unlock protected registers */
    SYS_UnlockReg();

    SYS_Init();

    /* UART init - will enable valid use of printf (stdout
     * re-directed at this UART (UART4) */
    InitDebugUart();

    SYS_LockReg();                   /* Unlock register lock protect */

    /* Both LEDs are active low: drive high (off) before enabling the output
     * so they do not flash during start-up.
     *
     * The port macros expand through __PC(), whose CMSIS definition declares
     * a `register` variable -- legal in the .c files Nuvoton's samples use,
     * deprecated in the C++17 this file is compiled as.  Suppressed at the
     * lines that trigger it rather than left in the log, where an unexplained
     * warning is how people learn to stop reading them. */
#if defined(__clang__)
    #pragma clang diagnostic push
    #pragma clang diagnostic ignored "-Wdeprecated-register"
#endif
    PC->DOUT |= (BIT4 | BIT5);
    GPIO_SetMode(PC, BIT4 | BIT5, GPIO_MODE_OUTPUT);

    /* BTN1 on PA.7 already has an external 10 K pull-up; input mode is enough,
     * and it MUST stay an input. */
    GPIO_SetMode(PA, BIT7, GPIO_MODE_INPUT);
#if defined(__clang__)
    #pragma clang diagnostic pop
#endif

    info("%s: complete\n", __FUNCTION__);

#if defined(ARM_NPU)

    int state;

    /* If Arm Ethos-U NPU is to be used, we initialise it here */
    if (0 != (state = arm_ethosu_npu_init()))
    {
        return state;
    }

#endif /* ARM_NPU */

    /* Print target design info */
    info("Target system: %s\n", DESIGN_NAME);
    info("Console: UART4, PB.11 TXD / PB.10 RXD (header J3), 115200 8N1\n");

    return 0;
}
