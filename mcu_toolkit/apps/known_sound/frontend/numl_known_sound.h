/**************************************************************************//**
 * @file     numl_known_sound.h
 * @brief    Known Sound output stage: int8 model output -> per-class scores.
 *
 * The model ends in a Dense(n, sigmoid): every class carries its own
 * independent confidence score.  They do not sum to one, two sounds may be
 * detected at the same time, and none firing is a valid answer.  Nothing here
 * takes an argmax or normalises, because doing either would misreport what the
 * model was trained to say.
 *
 * The scale and zero point are arguments, never constants: they are chosen by
 * the representative dataset and change on every re-export.  Read them from
 * TfLiteTensor->params at run time.
 ******************************************************************************/
#ifndef NUML_KNOWN_SOUND_H
#define NUML_KNOWN_SOUND_H

#include <stdint.h>

#ifdef _WIN32
    #define NUML_KNOWN_SOUND_API __declspec(dllexport)
#else
    #define NUML_KNOWN_SOUND_API
#endif

#ifdef __cplusplus
extern "C" {
#endif

/* Dequantise the model output into iClasses independent confidence scores.
 *
 *     score = (code - zero_point) * scale
 */
NUML_KNOWN_SOUND_API void NuML_KnownSound_Scores(
    const int8_t *pi8Raw,
    int iClasses,
    float fScale,
    int iZeroPoint,
    float *pfScores
);

/* The smallest output code whose score reaches fThreshold, so the inference
 * loop can compare integers instead of dequantising every class every hop.
 *
 * Returns a value greater than 127 when no code can reach the threshold, so
 * "code >= result" is simply never true.
 */
NUML_KNOWN_SOUND_API int NuML_KnownSound_ThresholdCode(
    float fThreshold,
    float fScale,
    int iZeroPoint
);

#ifdef __cplusplus
}
#endif

#endif /* NUML_KNOWN_SOUND_H */
