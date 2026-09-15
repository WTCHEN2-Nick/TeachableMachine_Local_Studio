/**************************************************************************//**
 * @file     numl_yamnet_frontend.c
 * @brief    Known Sound YAMNet log-mel frontend (96 x 64 patch).
 ******************************************************************************/
#include <math.h>

#include "numl_yamnet_frontend.h"
#include "numl_yamnet_tables.h"

/* TensorFlow's _raised_cosine_window computes, in float32:
 *
 *     n       = window_length + periodic*even - 1   = 400 for a periodic
 *                                                     window of even length
 *     cos_arg = float32(2*pi) * float32(k) / n
 *     w[k]    = 0.5 - 0.5 * cos(cos_arg)
 *
 * The divisor is the window length, not length-1.  A symmetric window would
 * divide by 399 and end at exactly 0 instead of 6.16e-5.
 */
#define NUML_TWO_PI_F  6.283185307179586f

/* From yamnet_frontend.json: log_offset.  Inside the logarithm, so silence
 * gives ln(0.001) rather than -inf. */
#define NUML_LOG_OFFSET_F  0.001f

#define NUML_FFT_N  NUML_YAMNET_FFT_LENGTH

void NuML_Yamnet_Window(float *pfOut)
{
    int i;

    for (i = 0; i < NUML_YAMNET_FRAME_LENGTH; i++)
    {
        const float fArg = (NUML_TWO_PI_F * (float)i) / (float)NUML_YAMNET_FRAME_LENGTH;

        pfOut[i] = 0.5f - 0.5f * cosf(fArg);
    }
}

void NuML_Yamnet_MelMatrix(float *pfOut)
{
    int k, b;

    for (k = 0; k < NUML_YAMNET_SPECTRUM_BINS * NUML_YAMNET_MEL_BANDS; k++)
    {
        pfOut[k] = 0.0f;
    }

    for (k = 0; k < NUML_YAMNET_SPECTRUM_BINS; k++)
    {
        b = (int)g_au8MelFirstBand[k];
        pfOut[k * NUML_YAMNET_MEL_BANDS + b] = g_afMelWeight0[k];

        if ((b + 1) < NUML_YAMNET_MEL_BANDS)
        {
            pfOut[k * NUML_YAMNET_MEL_BANDS + b + 1] = g_afMelWeight1[k];
        }
    }
}

/*---------------------------------------------------------------------------
 * Real FFT
 *
 * Radix-2 decimation-in-time over 512 points, kept deliberately plain: the
 * host test and the firmware compile this same translation unit, so the
 * numbers cannot drift between them.  Replacing it with arm_rfft_fast_f32 for
 * speed is a later step, and the tests here will report exactly how much that
 * changes the result.
 *-------------------------------------------------------------------------*/

static void NuML_Fft512(float *pfRe, float *pfIm)
{
    int i, j, k, step, group;

    /* Bit reversal over 9 bits. */
    for (i = 1, j = 0; i < NUML_FFT_N; i++)
    {
        int bit = NUML_FFT_N >> 1;

        for (; j & bit; bit >>= 1)
        {
            j ^= bit;
        }

        j ^= bit;

        if (i < j)
        {
            const float fRe = pfRe[i];
            const float fIm = pfIm[i];

            pfRe[i] = pfRe[j];
            pfIm[i] = pfIm[j];
            pfRe[j] = fRe;
            pfIm[j] = fIm;
        }
    }

    for (step = 1; step < NUML_FFT_N; step <<= 1)
    {
        /* A stage of half-width `step` needs exp(-i*pi*k/step), which is
         * W[k * (256/step)] in the shared table.  Reading a table built in
         * double keeps the angle accurate to float32 instead of deriving it
         * from a float32 pi/step, whose error raises the FFT noise floor
         * enough to swamp the quiet bins of a pure tone. */
        const int iStride = (NUML_FFT_N / 2) / step;

        for (group = 0; group < NUML_FFT_N; group += (step << 1))
        {
            for (k = 0; k < step; k++)
            {
                const float fWr = g_afTwiddleRe[k * iStride];
                const float fWi = g_afTwiddleIm[k * iStride];
                const int   iA  = group + k;
                const int   iB  = iA + step;
                const float fTr = pfRe[iB] * fWr - pfIm[iB] * fWi;
                const float fTi = pfRe[iB] * fWi + pfIm[iB] * fWr;

                pfRe[iB] = pfRe[iA] - fTr;
                pfIm[iB] = pfIm[iA] - fTi;
                pfRe[iA] = pfRe[iA] + fTr;
                pfIm[iA] = pfIm[iA] + fTi;
            }
        }
    }
}

/* |FFT| of a single windowed frame.  pfMag receives NUML_YAMNET_SPECTRUM_BINS
 * values. */
static void NuML_FrameMagnitude(
    const float *pfSrc,
    const float *pfWindow,
    float *pfMag
)
{
    float afRe[NUML_FFT_N];
    float afIm[NUML_FFT_N];
    int i;

    /* Window the 400 live samples; the remaining 112 are zero.  The pad goes
     * at the tail, matching tf.signal.stft -- centring it would rotate the
     * phase, which leaves |X| alone for a steady tone and changes it for
     * every transient. */
    for (i = 0; i < NUML_YAMNET_FRAME_LENGTH; i++)
    {
        afRe[i] = pfSrc[i] * pfWindow[i];
        afIm[i] = 0.0f;
    }

    for (i = NUML_YAMNET_FRAME_LENGTH; i < NUML_FFT_N; i++)
    {
        afRe[i] = 0.0f;
        afIm[i] = 0.0f;
    }

    NuML_Fft512(afRe, afIm);

    /* Magnitude, not power: no squaring. */
    for (i = 0; i < NUML_YAMNET_SPECTRUM_BINS; i++)
    {
        pfMag[i] = sqrtf(afRe[i] * afRe[i] + afIm[i] * afIm[i]);
    }
}

/* ln(mel + offset) for one frame's magnitude spectrum. */
static void NuML_FrameLogMel(const float *pfMag, float *pfDst)
{
    int k, b;

    for (b = 0; b < NUML_YAMNET_MEL_BANDS; b++)
    {
        pfDst[b] = 0.0f;
    }

    /* Scatter each bin into the at most two mel bands it touches.  Bins
     * outside 125..7500 Hz -- including DC -- carry zero weight, so they
     * contribute nothing without needing a special case. */
    for (k = 0; k < NUML_YAMNET_SPECTRUM_BINS; k++)
    {
        const int   iBand = (int)g_au8MelFirstBand[k];
        const float fMag  = pfMag[k];

        pfDst[iBand] += fMag * g_afMelWeight0[k];

        if ((iBand + 1) < NUML_YAMNET_MEL_BANDS)
        {
            pfDst[iBand + 1] += fMag * g_afMelWeight1[k];
        }
    }

    for (b = 0; b < NUML_YAMNET_MEL_BANDS; b++)
    {
        pfDst[b] = logf(pfDst[b] + NUML_LOG_OFFSET_F);
    }
}

/* Shared by LogMel and Patch so both compute identical values, and so neither
 * has to hold a whole clip's magnitude spectrogram (100 KiB) in RAM. */
static void NuML_LogMelFrames(const float *pfPcm, int iFrames, float *pfOut)
{
    float afWindow[NUML_YAMNET_FRAME_LENGTH];
    float afMagnitude[NUML_YAMNET_SPECTRUM_BINS];
    int frame;

    NuML_Yamnet_Window(afWindow);

    for (frame = 0; frame < iFrames; frame++)
    {
        NuML_FrameMagnitude(
            pfPcm + (frame * NUML_YAMNET_FRAME_STEP), afWindow, afMagnitude);
        NuML_FrameLogMel(afMagnitude, pfOut + (frame * NUML_YAMNET_MEL_BANDS));
    }
}

void NuML_Yamnet_Magnitude(const float *pfPcm, float *pfOut)
{
    float afWindow[NUML_YAMNET_FRAME_LENGTH];
    int frame;

    NuML_Yamnet_Window(afWindow);

    for (frame = 0; frame < NUML_YAMNET_CLIP_FRAMES; frame++)
    {
        NuML_FrameMagnitude(
            pfPcm + (frame * NUML_YAMNET_FRAME_STEP),
            afWindow,
            pfOut + (frame * NUML_YAMNET_SPECTRUM_BINS));
    }
}

void NuML_Yamnet_LogMel(const float *pfPcm, float *pfOut)
{
    NuML_LogMelFrames(pfPcm, NUML_YAMNET_CLIP_FRAMES, pfOut);
}

void NuML_Yamnet_Patch(const float *pfPcm, float *pfOut)
{
    /* frame(98, 96, 48) yields one patch starting at frame 0, so the last two
     * frames are never needed -- not padded, not centred, simply not
     * computed. */
    NuML_LogMelFrames(pfPcm, NUML_YAMNET_PATCH_FRAMES, pfOut);
}
