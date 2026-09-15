/**************************************************************************//**
 * @file     numl_known_sound.c
 * @brief    Known Sound output stage: int8 model output -> per-class scores.
 ******************************************************************************/
#include <math.h>

#include "numl_known_sound.h"

void NuML_KnownSound_Scores(
    const int8_t *pi8Raw,
    int iClasses,
    float fScale,
    int iZeroPoint,
    float *pfScores
)
{
    int i;

    for (i = 0; i < iClasses; i++)
    {
        pfScores[i] = ((float)pi8Raw[i] - (float)iZeroPoint) * fScale;
    }
}

int NuML_KnownSound_ThresholdCode(float fThreshold, float fScale, int iZeroPoint)
{
    /* A class fires when its score reaches the threshold:
     *
     *     (code - zero_point) * scale >= threshold
     *     code >= threshold / scale + zero_point
     *
     * Codes are integers, so the smallest one that satisfies this is the
     * ceiling.  Rounding instead fires one step early whenever
     * threshold/scale lands between two integers -- at a threshold of 0.65
     * with a scale of 1/256 that means reporting a detection at a score of
     * 0.6484.
     *
     * The result may exceed 127, which is the correct answer for a threshold
     * no code can reach: "code >= result" is then never true.
     */
    return (int)ceilf(fThreshold / fScale) + iZeroPoint;
}
