/**************************************************************************//**
 * @file     BoardConfig.h
 * @brief    Everything that differs between the boards, in one place.
 *
 * Select with -DBOARD_NUMAKER_X_M55M1D, -DBOARD_NUGESTUREAI_M55M1 or
 * -DBOARD_NUMAKER_VOICEAI_M55M1.
 *
 * The microphone is the difference that matters and the one that fails
 * silently.  M55M1 has a single DMIC controller with four channels reached
 * through two pin pairs: the DMIC0_* pins carry channels 0/1 and the DMIC1_*
 * pins carry channels 2/3.  Enabling the wrong channel captures nothing at all
 * -- no error, just silence -- so the pins and the channel mask are declared
 * together and must stay consistent.
 *
 * The second difference is where printf comes out, declared here too so that
 * no board's transport is scattered through main.cpp.
 ******************************************************************************/
#ifndef BOARD_CONFIG_H
#define BOARD_CONFIG_H

#if !defined(BOARD_NUMAKER_X_M55M1D) && !defined(BOARD_NUGESTUREAI_M55M1) \
    && !defined(BOARD_NUMAKER_VOICEAI_M55M1)
    #define BOARD_NUMAKER_X_M55M1D
#endif

#if defined(BOARD_NUMAKER_X_M55M1D)

    #define BOARD_NAME              "NuMaker-X-M55M1D"

    /* On-board DMIC on PB4/PB5, channel 0.  Same as every audio sample in the
     * BSP, and what BoardInit.cpp already configures. */
    #define BOARD_DMIC_SET_CLK()    SET_DMIC0_CLK_PB4()
    #define BOARD_DMIC_SET_DAT()    SET_DMIC0_DAT_PB5()
    #define BOARD_DMIC_CHANNEL      0
    #define BOARD_DMIC_CHANNEL_MSK  DMIC_CTL_CHEN0_Msk

    /* printf goes to the debug UART, which BoardInit has already opened.
     * Nothing to start and nothing to pump. */
    #define BOARD_LOG_TRANSPORT     "debug UART"
    #define BOARD_LOG_INIT()        ((void)0)
    #define BOARD_LOG_PUMP()        ((void)0)
    #define BOARD_LOG_WAIT()        ((void)0)
    #define BOARD_LOG_DROPPED()     0u

#elif defined(BOARD_NUGESTUREAI_M55M1)

    #define BOARD_NAME              "NuMaker-GestureAI-M55M1"

    /* MP34DT05-M (U4) on PB2/PB3, channel 2.
     *
     * Not PB4/PB5: those are this board's debug UART5, which the user manual
     * confirms is where the firmware prints.  From UM Rev1.00 section 5.7
     * (MEMS Digital MIC circuit, nets DMIC1_CLK / DMIC1_DAT) and section 5.2
     * (MCU pin table).  L/R is pulled to GND by R22 with R21 unpopulated, so
     * it is the left channel and the LCHEDGE23 reset default applies. */
    #define BOARD_DMIC_SET_CLK()    SET_DMIC1_CLK_PB2()
    #define BOARD_DMIC_SET_DAT()    SET_DMIC1_DAT_PB3()
    #define BOARD_DMIC_CHANNEL      2
    #define BOARD_DMIC_CHANNEL_MSK  DMIC_CTL_CHEN2_Msk

    #define BOARD_LOG_USB_CDC       1
    #define BOARD_LOG_UART_NAME     "UART5"

#elif defined(BOARD_NUMAKER_VOICEAI_M55M1)

    #define BOARD_NAME              "NuMaker-VoiceAI-M55M1"

    /* MP34DT05-M (U4) on PA4/PA5 -- the FIRST PDM pin pair -- channel 0.
     *
     * From UM Rev0.01 sheet "MEMS Digital MIC" (nets DMIC0_CLK / DMIC0_DAT on
     * PA.4 / PA.5; U10 unpopulated, so mono) and sheet "M55M1_LQFP64".  Same
     * channel as the X board, different pins: PB4/PB5 are I2S0 to the codec
     * here.  R79 (0 R) ties DMIC0_CLKLP (PA.3) to the DMIC0_CLK net, so PA.3
     * is never also muxed as CLKLP.
     *
     * The channel was established on hardware, not from the L/R strap (which
     * the UM does not show legibly): Nuvoton's DMIC_UAC sample for this board
     * documents a sweep of all four channel x latch-edge combinations over SWD
     * against the same ambient noise, and channel 0 with LCHEDGE01 at its
     * reset value gave the most audio-band energy and the least
     * sample-to-sample hash.  Its live level meter on this board reads a
     * non-zero ambient peak on channel 0 with the reset latch edge. */
    #define BOARD_DMIC_SET_CLK()    SET_DMIC0_CLK_PA4()
    #define BOARD_DMIC_SET_DAT()    SET_DMIC0_DAT_PA5()
    #define BOARD_DMIC_CHANNEL      0
    #define BOARD_DMIC_CHANNEL_MSK  DMIC_CTL_CHEN0_Msk

    #define BOARD_LOG_USB_CDC       1
    #define BOARD_LOG_UART_NAME     "UART4"

#else
    #error "No board selected"
#endif

#if defined(BOARD_LOG_USB_CDC)

    /* printf is tee'd to a USB CDC virtual COM port so the board enumerates as
     * a COM port on its own, and still to the debug UART for a probe on the
     * header.
     *
     * The ring is filled from stdout_putchar and drained by BOARD_LOG_PUMP()
     * from main context -- the bulk IN endpoint is filled with PIO, so the ISR
     * only reports that the previous packet drained.  A pump that is not called
     * often enough does not block printf; it drops bytes, which is why
     * BOARD_LOG_DROPPED() is reported rather than assumed to be zero.
     *
     * BOARD_LOG_WAIT() blocks until the host opens the port, bounded, because
     * the self-test prints its verdict once and a host takes a second or two to
     * enumerate.  It gives up rather than refusing to boot without a terminal.
     */
    #include "numl_cdc.h"
    #include "numl_usbd.h"

    #define BOARD_LOG_TRANSPORT     "HSUSB CDC + " BOARD_LOG_UART_NAME
    #define BOARD_LOG_INIT()        NuML_USBD_Init()
    #define BOARD_LOG_PUMP()        NuML_CDC_Pump()
    #define BOARD_LOG_WAIT()        NuML_BoardLogWait()
    #define BOARD_LOG_DROPPED()     NuML_CDC_DroppedBytes()

    #ifndef BOARD_LOG_WAIT_MS
        #define BOARD_LOG_WAIT_MS   10000
    #endif

/* 10 ms per step: CLK_SysTickDelay counts a 24-bit SysTick, which at 220 MHz
 * tops out around 76 ms, so a single long delay would silently wrap. */
static inline void NuML_BoardLogWait(void)
{
    int i;

    for (i = 0; i < (BOARD_LOG_WAIT_MS / 10); i++)
    {
        if (NuML_CDC_IsOpen())
        {
            return;
        }

        CLK_SysTickDelay(10000);
    }
}

#endif /* BOARD_LOG_USB_CDC */

#endif /* BOARD_CONFIG_H */
