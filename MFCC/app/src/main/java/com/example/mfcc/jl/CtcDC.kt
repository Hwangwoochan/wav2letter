package com.example.mfcc.jl

import android.util.Log

/**
 * CTC Greedy 디코딩 결과를 담는 데이터 클래스
 */
data class DecodedResult(
    val decodedIndices: Array<LongArray>,
    val decodedValues: LongArray,
    val decodedShape: LongArray,
    val logProbability: FloatArray
)

/**
 * CTC Greedy Decoder (Kotlin)
 *
 * - [mergeRepeated] : 같은 클래스 연속 발생 시 하나로 합치는 옵션
 * - [blankIndex]    : blank 토큰 인덱스
 *
 * [decode] 함수에 3D 배열 inputs와 배치별 sequenceLength를 전달하면
 * 디코딩 결과(SparseTensor + logProbability)를 반환합니다.
 */
class CTCGreedyDecoder(
    private val mergeRepeated: Boolean,
    private val blankIndex: Int
) {

    /**
     * CTC Greedy Decoding 수행
     *
     * @param inputs         3D float array: [maxTime, batchSize, numClasses]
     * @param sequenceLength 1D int array: [batchSize]
     * @return               [DecodedResult] (SparseTensor & logProbability)
     */
    fun decode(
        inputs: Array<Array<FloatArray>>,
        sequenceLength: IntArray
    ): DecodedResult {
        // [0] 간단한 입력 검증
        if (inputs.isEmpty()) {
            throw IllegalArgumentException("inputs is empty")
        }

        val maxTime = inputs.size
        val batchSize = inputs[0].size
        val numClasses = inputs[0][0].size

        if (sequenceLength.size != batchSize) {
            throw IllegalArgumentException("sequenceLength.size != batchSize")
        }

        // [1] 각 배치별 디코딩 결과(토큰 리스트) 저장용
        val sequences = Array(batchSize) { mutableListOf<Int>() }

        // [2] 각 배치별 logProbability(누적) 저장
        val logProbability = FloatArray(batchSize) { 0f }

        // [3] 그리디 디코딩
        for (b in 0 until batchSize) {
            val seqLen = sequenceLength[b]
            var prevIndex = -1

            for (t in 0 until seqLen) {
                // inputs[t][b]에서 가장 큰 값을 갖는 클래스 선택 (argmax)
                val row = inputs[t][b]
                var maxVal = row[0]
                var maxClassIdx = 0
                for (c in 1 until numClasses) {
                    if (row[c] > maxVal) {
                        maxVal = row[c]
                        maxClassIdx = c
                    }
                }

                // [디버깅 추가] 각 타임스텝별 argmax, maxVal 출력
                Log.d("CTCDecoder", "batch=$b, time=$t, argmax=$maxClassIdx, maxVal=$maxVal")
                // row 전체를 보고 싶다면 (조심! 로그가 매우 많아질 수 있음):
                // Log.d("CTCDecoder", "batch=$b, time=$t, row=${row.joinToString()}, argmax=$maxClassIdx, maxVal=$maxVal")

                // logProbability[b]에 -maxVal을 누적
                logProbability[b] += (-maxVal)

                // blank가 아니면(= maxClassIdx != blankIndex),
                // mergeRepeated=true일 때 이전 인덱스와 다르면 시퀀스에 추가
                if (maxClassIdx != blankIndex) {
                    if (!(mergeRepeated && maxClassIdx == prevIndex)) {
                        sequences[b].add(maxClassIdx)
                    }
                }
                prevIndex = maxClassIdx
            }
        }

        // [4] SparseTensor 변환
        var totalDecoded = 0
        var maxDecodedLen = 0
        for (b in 0 until batchSize) {
            val len = sequences[b].size
            totalDecoded += len
            if (len > maxDecodedLen) {
                maxDecodedLen = len
            }
        }

        val decodedIndices = Array(totalDecoded) { LongArray(2) }
        val decodedValues = LongArray(totalDecoded)
        val decodedShape = longArrayOf(batchSize.toLong(), maxDecodedLen.toLong())

        var offset = 0
        for (b in 0 until batchSize) {
            val seq = sequences[b]
            for ((t, cls) in seq.withIndex()) {
                decodedIndices[offset][0] = b.toLong()   // batch
                decodedIndices[offset][1] = t.toLong()   // time
                decodedValues[offset] = cls.toLong()
                offset++
            }
        }

        // 최종 결과를 data class에 담아서 반환
        return DecodedResult(
            decodedIndices = decodedIndices,
            decodedValues = decodedValues,
            decodedShape = decodedShape,
            logProbability = logProbability
        )
    }
}
