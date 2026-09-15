/**************************************************************************//**
 * @file     numl_yamnet_frontend.h
 * @brief    Known Sound YAMNet log-mel frontend (96 x 64 patch).
 *
 * Reproduces the exported yamnet_frontend_reference.py bit-for-bit closely
 * enough that the quantised model sees the same input on the MCU as it did
 * during training:
 *
 *   |STFT(periodic Hann 400, hop 160, FFT 512)|  ->  magnitude, not power
 *   mel filterbank 64 bands, 125..7500 Hz, HTK, no area normalisation
 *   ln(mel + 0.001)                              ->  natural log, no dB, no
 *                                                    normalisation, no DCT
 *   frame(98, 96, 48)                            ->  one 96 x 64 patch
 ******************************************************************************/
#ifndef NUML_YAMNET_FRONTEND_H
#define NUML_YAMNET_FRONTEND_H

#ifdef _WIN32
    #define NUML_FRONTEND_API __declspec(dllexport)
#else
    #define NUML_FRONTEND_API
#endif

#ifdef __cplusplus
extern "C" {
#endif

#define NUML_YAMNET_SAMPLE_RATE      16000
#define NUML_YAMNET_FRAME_LENGTH     400
#define NUML_YAMNET_FRAME_STEP       160
#define NUML_YAMNET_FFT_LENGTH       512
#define NUML_YAMNET_SPECTRUM_BINS    257
#define NUML_YAMNET_MEL_BANDS        64
#define NUML_YAMNET_CLIP_SAMPLES     16000
#define NUML_YAMNET_CLIP_FRAMES      98
#define NUML_YAMNET_PATCH_FRAMES     96

/* Analysis window: periodic Hann of length 400.  Writes NUML_YAMNET_FRAME_LENGTH
 * floats. */
NUML_FRONTEND_API void NuML_Yamnet_Window(float *pfOut);

/* Expands the compact mel filterbank back to the dense
 * NUML_YAMNET_SPECTRUM_BINS x NUML_YAMNET_MEL_BANDS matrix, row-major.  Only
 * the host test uses this; the runtime path reads the compact table directly.
 */
NUML_FRONTEND_API void NuML_Yamnet_MelMatrix(float *pfOut);

/* |STFT| of one 16000-sample clip.
 *
 * pfPcm  : NUML_YAMNET_CLIP_SAMPLES float samples, nominally in [-1, 1]
 * pfOut  : NUML_YAMNET_CLIP_FRAMES x NUML_YAMNET_SPECTRUM_BINS, row-major
 */
NUML_FRONTEND_API void NuML_Yamnet_Magnitude(const float *pfPcm, float *pfOut);

/* ln(mel + 0.001) of one clip.
 *
 * pfOut : NUML_YAMNET_CLIP_FRAMES x NUML_YAMNET_MEL_BANDS, row-major
 */
NUML_FRONTEND_API void NuML_Yamnet_LogMel(const float *pfPcm, float *pfOut);

/* The model input: the leading NUML_YAMNET_PATCH_FRAMES frames of the log-mel
 * spectrogram.  A 1 s clip produces 98 frames and exactly one patch; the last
 * two frames are discarded, never padded.
 *
 * pfOut : NUML_YAMNET_PATCH_FRAMES x NUML_YAMNET_MEL_BANDS, row-major
 */
NUML_FRONTEND_API void NuML_Yamnet_Patch(const float *pfPcm, float *pfOut);

#ifdef __cplusplus
}
#endif

#endif /* NUML_YAMNET_FRONTEND_H */
