/**************************************************************************//**
 * @file     numl_dmic.c
 * @brief    16 kHz mono DMIC capture into a sliding ring buffer.
 ******************************************************************************/
#include <string.h>

#include "NuMicro.h"

#include "BoardConfig.h"
#include "numl_dmic.h"

/* LPPDMA channel 2 is what every BSP audio sample uses for DMIC0. */
#define NUML_DMIC_LPPDMA_CH     2

/* 50 ms per half of the ping-pong: short enough that a dropped block is a
 * small hole, long enough that the interrupt is not hot. */
#define NUML_DMIC_BLOCK         800

/* Two seconds, so a one-second window can be read while the next half second
 * is still arriving. */
#define NUML_DMIC_RING          32000

/* The DMA writes these behind the CPU's back, so they must not be cached. */
NVT_NONCACHEABLE __ALIGNED(32) static int16_t s_ai16Block[2][NUML_DMIC_BLOCK];
NVT_NONCACHEABLE __ALIGNED(32) static LPDSCT_T s_sDescriptor[2];

/* 64 KiB, and DTCM is 128 KiB shared with the stack, the heap and every other
 * driver's state.  A project that also links FatFs, the SD host and USB has
 * nothing like that left, so the ring goes to the scatter's SRAM01 region
 * instead.  Only the CPU touches it -- the DMA lands in s_ai16Block above --
 * so ordinary cacheable memory is correct here. */
static int16_t s_ai16Ring[NUML_DMIC_RING] __attribute__((section(".bss.sram.data")));
static volatile uint32_t s_u32Write;
static volatile uint32_t s_u32Read;
static volatile uint32_t s_u32Block;
static volatile uint32_t s_u32Overruns;

static int32_t s_i32Started;

static uint32_t NuML_Dmic_Count(void)
{
    return s_u32Write - s_u32Read;
}

void LPPDMA_IRQHandler(void)
{
    const uint32_t u32Status = LPPDMA_GET_TD_STS(LPPDMA);

    if (u32Status & (1U << NUML_DMIC_LPPDMA_CH))
    {
        LPPDMA_CLR_TD_FLAG(LPPDMA, (1U << NUML_DMIC_LPPDMA_CH));

        const int16_t *pi16Src = s_ai16Block[s_u32Block];
        uint32_t i;

        s_u32Block ^= 1U;

        /* Drop the oldest block rather than tearing the newest: a reader that
         * fell behind gets a reported discontinuity instead of a frame
         * stitched from two different moments. */
        if ((NuML_Dmic_Count() + NUML_DMIC_BLOCK) > NUML_DMIC_RING)
        {
            s_u32Read += NUML_DMIC_BLOCK;
            s_u32Overruns++;
        }

        for (i = 0; i < NUML_DMIC_BLOCK; i++)
        {
            s_ai16Ring[(s_u32Write + i) % NUML_DMIC_RING] = pi16Src[i];
        }

        s_u32Write += NUML_DMIC_BLOCK;
    }

    __DSB();
    __ISB();
}

int32_t NuML_Dmic_Init(uint32_t u32SampleRate)
{
    uint32_t i;

    s_u32Write = 0;
    s_u32Read = 0;
    s_u32Block = 0;
    s_u32Overruns = 0;
    s_i32Started = 0;

    SYS_UnlockReg();

    /* 196.608 MHz divides exactly by the audio rates, so the DMIC clock has no
     * fractional error to accumulate. */
    CLK_EnableAPLL(CLK_APLLCTL_APLLSRC_HIRC, DMIC_APLL1_FREQ_196608KHZ,
                   CLK_APLL1_SELECT);
    CLK_SetModuleClock(DMIC0_MODULE, CLK_DMICSEL_DMIC0SEL_APLL1_DIV2, MODULE_NoMsk);
    CLK_EnableModuleClock(DMIC0_MODULE);
    SYS_ResetModule(SYS_DMIC0RST);

    CLK_EnableModuleClock(LPPDMA0_MODULE);
    CLK_EnableModuleClock(LPSRAM0_MODULE);
    SYS_ResetModule(SYS_LPPDMA0RST);

    BOARD_DMIC_SET_CLK();
    BOARD_DMIC_SET_DAT();

    SYS_LockReg();

    DMIC_Open(DMIC0);
    DMIC_SET_DOWNSAMPLE(DMIC0, DMIC_DOWNSAMPLE_256);

    const uint32_t u32Actual = DMIC_SetSampleRate(DMIC0, u32SampleRate);

    DMIC_SetFIFOWidth(DMIC0, DMIC_FIFOWIDTH_16);
    DMIC_ClearFIFO(DMIC0);

    while (!DMIC_IS_FIFOEMPTY(DMIC0))
    {
    }

    DMIC_ResetDSP(DMIC0);
    DMIC_SetDSPGainVolume(DMIC0, BOARD_DMIC_CHANNEL_MSK, 36); /* +36 dB */

    for (i = 0; i < 2; i++)
    {
        s_sDescriptor[i].CTL =
            ((NUML_DMIC_BLOCK - 1U) << PDMA_DSCT_CTL_TXCNT_Pos) |
            PDMA_WIDTH_16 | PDMA_SAR_FIX | PDMA_DAR_INC |
            PDMA_REQ_SINGLE | PDMA_OP_SCATTER;
        s_sDescriptor[i].SA = (uint32_t)&DMIC0->FIFO;
        s_sDescriptor[i].DA = (uint32_t)s_ai16Block[i];
        s_sDescriptor[i].NEXT = (uint32_t)&s_sDescriptor[i ^ 1U];
    }

    LPPDMA_Open(LPPDMA, (1U << NUML_DMIC_LPPDMA_CH));
    LPPDMA_SetTransferMode(LPPDMA, NUML_DMIC_LPPDMA_CH, LPPDMA_DMIC0_RX,
                           TRUE, (uint32_t)&s_sDescriptor[0]);
    LPPDMA_EnableInt(LPPDMA, NUML_DMIC_LPPDMA_CH, LPPDMA_INT_TRANS_DONE);
    NVIC_SetPriority(LPPDMA_IRQn, 1);
    NVIC_EnableIRQ(LPPDMA_IRQn);

    return (int32_t)u32Actual;
}

int32_t NuML_Dmic_Start(void)
{
    if (s_i32Started)
    {
        return 0;
    }

    DMIC_EnableChMsk(DMIC0, BOARD_DMIC_CHANNEL_MSK);
    CLK_SysTickDelay(10000); /* let the DSP settle before the FIFO matters */
    DMIC_ClearFIFO(DMIC0);

    while (!DMIC_IS_FIFOEMPTY(DMIC0))
    {
    }

    DMIC_ENABLE_LPPDMA(DMIC0);
    s_i32Started = 1;
    return 0;
}

int32_t NuML_Dmic_Available(void)
{
    return (int32_t)NuML_Dmic_Count();
}

int32_t NuML_Dmic_Peek(int16_t *pi16Out, int32_t i32Samples)
{
    const uint32_t u32Primask = __get_PRIMASK();
    int32_t i;

    __disable_irq();

    const uint32_t u32Available = NuML_Dmic_Count();
    const uint32_t u32Read = s_u32Read;

    if (!u32Primask)
    {
        __enable_irq();
    }

    if ((uint32_t)i32Samples > u32Available)
    {
        i32Samples = (int32_t)u32Available;
    }

    for (i = 0; i < i32Samples; i++)
    {
        pi16Out[i] = s_ai16Ring[(u32Read + (uint32_t)i) % NUML_DMIC_RING];
    }

    return i32Samples;
}

void NuML_Dmic_Consume(int32_t i32Samples)
{
    const uint32_t u32Primask = __get_PRIMASK();

    __disable_irq();

    if ((uint32_t)i32Samples > NuML_Dmic_Count())
    {
        i32Samples = (int32_t)NuML_Dmic_Count();
    }

    s_u32Read += (uint32_t)i32Samples;

    if (!u32Primask)
    {
        __enable_irq();
    }
}

uint32_t NuML_Dmic_Overruns(void)
{
    return s_u32Overruns;
}
