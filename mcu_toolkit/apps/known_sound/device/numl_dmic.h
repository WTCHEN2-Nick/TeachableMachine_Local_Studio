/**************************************************************************//**
 * @file     numl_dmic.h
 * @brief    16 kHz mono DMIC capture into a sliding ring buffer.
 *
 * Written rather than reused from the BSP: DMICRecord.c in the sample tree
 * carries a USB Audio Class function through the same interrupt, feeds its
 * ring buffer from inside the USB transmit path, and enables HSUSBD_IRQn from
 * its init.  Separating the capture from the USB streaming turned out to be a
 * bigger change than writing the capture, and this firmware has no USB
 * function to justify pulling in the descriptors and the HSUSBD driver.
 *
 * Configuration values (APLL1, downsample 256, 16-bit FIFO, +36 dB) are the
 * ones the BSP driver uses; the pins and channel come from BoardConfig.h.
 ******************************************************************************/
#ifndef NUML_DMIC_H
#define NUML_DMIC_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Bring up DMIC0 and the LPPDMA ping-pong.
 *
 * Returns the rate the hardware actually achieved, which the caller should
 * report: the DMIC divider cannot hit every rate exactly, and a frontend
 * tuned for 16 kHz fed 15.9 kHz drifts in a way nothing else will flag.
 * Returns a negative value if the peripheral could not be opened.
 */
int32_t NuML_Dmic_Init(uint32_t u32SampleRate);

/* Start capturing.  Returns 0 on success. */
int32_t NuML_Dmic_Start(void);

/* Samples currently held in the ring. */
int32_t NuML_Dmic_Available(void);

/* Copy the oldest i32Samples out without consuming them, so a window can
 * advance by less than its own length.  Returns the count copied. */
int32_t NuML_Dmic_Peek(int16_t *pi16Out, int32_t i32Samples);

/* Drop the oldest i32Samples. */
void NuML_Dmic_Consume(int32_t i32Samples);

/* Blocks dropped because the reader fell behind.  Non-zero means the ring
 * overflowed and the audio is no longer continuous. */
uint32_t NuML_Dmic_Overruns(void);

#ifdef __cplusplus
}
#endif

#endif /* NUML_DMIC_H */
