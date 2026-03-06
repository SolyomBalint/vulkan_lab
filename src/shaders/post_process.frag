#version 450

/**
 * @license
 * Copyright (c) 2011 NVIDIA Corporation. All rights reserved.
 *
 * TO THE MAXIMUM EXTENT PERMITTED BY APPLICABLE LAW, THIS SOFTWARE IS PROVIDED
 * AS IS AND NVIDIA AND ITS SUPPLIERS DISCLAIM ALL WARRANTIES, EITHER EXPRESS
 * OR IMPLIED, INCLUDING, BUT NOT LIMITED TO, NONINFRINGEMENT, IMPLIED WARRANTIES
 * OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE. IN NO EVENT SHALL
 * NVIDIA OR ITS SUPPLIERS BE LIABLE FOR ANY DIRECT, SPECIAL, INCIDENTAL,
 * INDIRECT, OR CONSEQUENTIAL DAMAGES WHATSOEVER (INCLUDING, WITHOUT LIMITATION,
 * DAMAGES FOR LOSS OF BUSINESS PROFITS, BUSINESS INTERRUPTION, LOSS OF BUSINESS
 * INFORMATION, OR ANY OTHER PECUNIARY LOSS) ARISING OUT OF THE USE OF OR
 * INABILITY TO USE THIS SOFTWARE, EVEN IF NVIDIA HAS BEEN ADVISED OF THE
 * POSSIBILITY OF SUCH DAMAGES.
 */

layout(location = 0) in vec2 uv;
layout(location = 0) out vec4 outColor;

layout(set = 0, binding = 3) uniform sampler2D sceneTexture;

#define FXAA_PRESET 5

#if (FXAA_PRESET == 3)
    #define FXAA_EDGE_THRESHOLD      (1.0 / 8.0)
    #define FXAA_EDGE_THRESHOLD_MIN  (1.0 / 16.0)
    #define FXAA_SEARCH_STEPS        16
    #define FXAA_SEARCH_THRESHOLD    (1.0 / 4.0)
    #define FXAA_SUBPIX_CAP          (3.0 / 4.0)
    #define FXAA_SUBPIX_TRIM         (1.0 / 4.0)
#endif

#if (FXAA_PRESET == 4)
    #define FXAA_EDGE_THRESHOLD      (1.0 / 8.0)
    #define FXAA_EDGE_THRESHOLD_MIN  (1.0 / 24.0)
    #define FXAA_SEARCH_STEPS        24
    #define FXAA_SEARCH_THRESHOLD    (1.0 / 4.0)
    #define FXAA_SUBPIX_CAP          (3.0 / 4.0)
    #define FXAA_SUBPIX_TRIM         (1.0 / 4.0)
#endif

#if (FXAA_PRESET == 5)
    #define FXAA_EDGE_THRESHOLD      (1.0 / 8.0)
    #define FXAA_EDGE_THRESHOLD_MIN  (1.0 / 24.0)
    #define FXAA_SEARCH_STEPS        32
    #define FXAA_SEARCH_THRESHOLD    (1.0 / 4.0)
    #define FXAA_SUBPIX_CAP          (3.0 / 4.0)
    #define FXAA_SUBPIX_TRIM         (1.0 / 4.0)
#endif

#define FXAA_SUBPIX_TRIM_SCALE (1.0 / (1.0 - FXAA_SUBPIX_TRIM))

float fxaaLuma(vec3 rgb)
{
    return rgb.y * (0.587 / 0.299) + rgb.x;
}

vec3 fxaaLerp3(vec3 a, vec3 b, float amountOfA)
{
    return (vec3(-amountOfA) * b) + ((a * vec3(amountOfA)) + b);
}

vec4 fxaaTexOff(sampler2D tex, vec2 pos, ivec2 off, vec2 rcpFrame)
{
    float x = pos.x + float(off.x) * rcpFrame.x;
    float y = pos.y + float(off.y) * rcpFrame.y;
    return texture(tex, vec2(x, y));
}

vec3 fxaaPixelShader(vec2 pos, sampler2D tex, vec2 rcpFrame)
{
    vec3 rgbN = fxaaTexOff(tex, pos, ivec2(0, -1), rcpFrame).xyz;
    vec3 rgbW = fxaaTexOff(tex, pos, ivec2(-1, 0), rcpFrame).xyz;
    vec3 rgbM = fxaaTexOff(tex, pos, ivec2(0, 0), rcpFrame).xyz;
    vec3 rgbE = fxaaTexOff(tex, pos, ivec2(1, 0), rcpFrame).xyz;
    vec3 rgbS = fxaaTexOff(tex, pos, ivec2(0, 1), rcpFrame).xyz;

    float lumaN = fxaaLuma(rgbN);
    float lumaW = fxaaLuma(rgbW);
    float lumaM = fxaaLuma(rgbM);
    float lumaE = fxaaLuma(rgbE);
    float lumaS = fxaaLuma(rgbS);
    float rangeMin = min(lumaM, min(min(lumaN, lumaW), min(lumaS, lumaE)));
    float rangeMax = max(lumaM, max(max(lumaN, lumaW), max(lumaS, lumaE)));

    float range = rangeMax - rangeMin;
    if (range < max(FXAA_EDGE_THRESHOLD_MIN, rangeMax * FXAA_EDGE_THRESHOLD)) {
        return rgbM;
    }

    vec3 rgbL = rgbN + rgbW + rgbM + rgbE + rgbS;

    float lumaL = (lumaN + lumaW + lumaE + lumaS) * 0.25;
    float rangeL = abs(lumaL - lumaM);
    float blendL = max(0.0, (rangeL / range) - FXAA_SUBPIX_TRIM) * FXAA_SUBPIX_TRIM_SCALE;
    blendL = min(FXAA_SUBPIX_CAP, blendL);

    vec3 rgbNW = fxaaTexOff(tex, pos, ivec2(-1, -1), rcpFrame).xyz;
    vec3 rgbNE = fxaaTexOff(tex, pos, ivec2(1, -1), rcpFrame).xyz;
    vec3 rgbSW = fxaaTexOff(tex, pos, ivec2(-1, 1), rcpFrame).xyz;
    vec3 rgbSE = fxaaTexOff(tex, pos, ivec2(1, 1), rcpFrame).xyz;
    rgbL += (rgbNW + rgbNE + rgbSW + rgbSE);
    rgbL *= vec3(1.0 / 9.0);

    float lumaNW = fxaaLuma(rgbNW);
    float lumaNE = fxaaLuma(rgbNE);
    float lumaSW = fxaaLuma(rgbSW);
    float lumaSE = fxaaLuma(rgbSE);

    float edgeVert =
        abs((0.25 * lumaNW) + (-0.5 * lumaN) + (0.25 * lumaNE)) +
        abs((0.50 * lumaW) + (-1.0 * lumaM) + (0.50 * lumaE)) +
        abs((0.25 * lumaSW) + (-0.5 * lumaS) + (0.25 * lumaSE));
    float edgeHorz =
        abs((0.25 * lumaNW) + (-0.5 * lumaW) + (0.25 * lumaSW)) +
        abs((0.50 * lumaN) + (-1.0 * lumaM) + (0.50 * lumaS)) +
        abs((0.25 * lumaNE) + (-0.5 * lumaE) + (0.25 * lumaSE));

    bool horzSpan = edgeHorz >= edgeVert;
    float lengthSign = horzSpan ? -rcpFrame.y : -rcpFrame.x;

    if (!horzSpan) {
        lumaN = lumaW;
        lumaS = lumaE;
    }

    float gradientN = abs(lumaN - lumaM);
    float gradientS = abs(lumaS - lumaM);
    lumaN = (lumaN + lumaM) * 0.5;
    lumaS = (lumaS + lumaM) * 0.5;

    if (gradientN < gradientS) {
        lumaN = lumaS;
        gradientN = gradientS;
        lengthSign *= -1.0;
    }

    vec2 posN;
    posN.x = pos.x + (horzSpan ? 0.0 : lengthSign * 0.5);
    posN.y = pos.y + (horzSpan ? lengthSign * 0.5 : 0.0);

    gradientN *= FXAA_SEARCH_THRESHOLD;

    vec2 posP = posN;
    vec2 offNP = horzSpan ? vec2(rcpFrame.x, 0.0) : vec2(0.0, rcpFrame.y);
    float lumaEndN = lumaN;
    float lumaEndP = lumaN;
    bool doneN = false;
    bool doneP = false;
    posN += offNP * vec2(-1.0, -1.0);
    posP += offNP * vec2(1.0, 1.0);

    for (int i = 0; i < FXAA_SEARCH_STEPS; i++) {
        if (!doneN) {
            lumaEndN = fxaaLuma(texture(tex, posN).xyz);
        }
        if (!doneP) {
            lumaEndP = fxaaLuma(texture(tex, posP).xyz);
        }

        doneN = doneN || (abs(lumaEndN - lumaN) >= gradientN);
        doneP = doneP || (abs(lumaEndP - lumaN) >= gradientN);

        if (doneN && doneP) {
            break;
        }
        if (!doneN) {
            posN -= offNP;
        }
        if (!doneP) {
            posP += offNP;
        }
    }

    float dstN = horzSpan ? pos.x - posN.x : pos.y - posN.y;
    float dstP = horzSpan ? posP.x - pos.x : posP.y - pos.y;
    bool directionN = dstN < dstP;
    lumaEndN = directionN ? lumaEndN : lumaEndP;

    if (((lumaM - lumaN) < 0.0) == ((lumaEndN - lumaN) < 0.0)) {
        lengthSign = 0.0;
    }

    float spanLength = (dstP + dstN);
    dstN = directionN ? dstN : dstP;
    float subPixelOffset = (0.5 + (dstN * (-1.0 / spanLength))) * lengthSign;
    vec3 rgbF = texture(tex, vec2(
        pos.x + (horzSpan ? 0.0 : subPixelOffset),
        pos.y + (horzSpan ? subPixelOffset : 0.0)
    )).xyz;
    return fxaaLerp3(rgbL, rgbF, blendL);
}

void main()
{
    vec2 sceneSize = vec2(textureSize(sceneTexture, 0));
    vec2 rcpFrame = 1.0 / sceneSize;
    outColor = vec4(fxaaPixelShader(uv, sceneTexture, rcpFrame), 1.0);
}
