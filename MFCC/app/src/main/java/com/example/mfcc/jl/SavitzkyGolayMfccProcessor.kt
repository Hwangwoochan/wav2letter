package com.example.mfcc.jl

import org.apache.commons.math3.linear.MatrixUtils
import org.apache.commons.math3.linear.SingularValueDecomposition
import kotlin.math.abs
import kotlin.math.pow
import kotlin.math.sqrt

object SavitzkyGolayMfccProcessor {

    /**
     * Savitzky–Golay 필터의 컨볼루션 계수를 계산합니다.
     *
     * @param windowLength 필터 윈도우 길이 (홀수, 예: 9)
     * @param polyOrder    다항식 차수 (예: 2 또는 3)
     * @param derivOrder   미분 차수 (예: 1 또는 2)
     * @return             길이 windowLength인 컨볼루션 계수 배열
     */
    fun computeSavitzkyGolayCoefficients(windowLength: Int, polyOrder: Int, derivOrder: Int): DoubleArray {
        require(windowLength % 2 == 1) { "windowLength must be odd." }
        require(derivOrder <= polyOrder) { "derivOrder must be <= polyOrder." }
        val m = (windowLength - 1) / 2

        // Vandermonde 행렬 A: 각 행 i에 대해 A[i][j] = (i - m)^j
        val A = Array(windowLength) { i ->
            DoubleArray(polyOrder + 1) { j ->
                (i - m).toDouble().pow(j)
            }
        }

        // Vandermonde 행렬의 의사역행렬 계산
        val A_matrix = MatrixUtils.createRealMatrix(A)
        val svd = SingularValueDecomposition(A_matrix)
        val A_pinv = svd.solver.inverse // shape: (polyOrder+1) x windowLength

        // 미분 계수를 얻기 위해 factorial(derivOrder)를 계산합니다.
        var factorial = 1.0
        for (i in 2..derivOrder) {
            factorial *= i
        }
        val A_pinvData = A_pinv.data  // 2차원 배열: (polyOrder+1) x windowLength
        return DoubleArray(windowLength) { i ->
            factorial * A_pinvData[derivOrder][i]
        }
    }

    /**
     * 주어진 데이터 행렬(data)의 각 행(예: MFCC 계수 시퀀스)에 대해 Savitzky–Golay 필터를
     * 적용하여 델타(미분) 값을 계산합니다.
     *
     * @param data         입력 데이터 행렬 (shape: [nCoeffs x nFrames])
     * @param windowLength 필터 윈도우 길이 (홀수)
     * @param polyOrder    다항식 차수
     * @param derivOrder   미분 차수 (1: 델타, 2: 델타-델타)
     * @return             미분 결과 행렬 (동일한 shape)
     */
    fun savitzkyGolayDelta(data: Array<DoubleArray>, windowLength: Int, polyOrder: Int, derivOrder: Int): Array<DoubleArray> {
        val nCoeffs = data.size
        val nFrames = data[0].size
        val coeffs = computeSavitzkyGolayCoefficients(windowLength, polyOrder, derivOrder)
        val halfWindow = (windowLength - 1) / 2
        val result = Array(nCoeffs) { DoubleArray(nFrames) }

        // 각 계수 행에 대해 시간 축(열) 방향으로 컨볼루션 적용 (mirror padding)
        for (i in 0 until nCoeffs) {
            for (t in 0 until nFrames) {
                var sum = 0.0
                for (k in -halfWindow..halfWindow) {
                    var idx = t + k
                    // mirror padding: 경계에서 반사 방식 사용
                    if (idx < 0) {
                        idx = -idx
                    } else if (idx >= nFrames) {
                        idx = 2 * nFrames - idx - 2
                    }
                    sum += coeffs[k + halfWindow] * data[i][idx]
                }
                result[i][t] = sum
            }
        }
        return result
    }

    /**
     * 각 행(특징 벡터)별로 평균 0, 표준편차 1이 되도록 정규화합니다.
     *
     * @param features 입력 행렬 (shape: [nFeatures x nFrames])
     * @return         정규화된 행렬
     */
    fun normalize(features: Array<DoubleArray>): Array<DoubleArray> {
        val rows = features.size
        val cols = features[0].size
        val norm = Array(rows) { DoubleArray(cols) }
        for (i in 0 until rows) {
            val row = features[i]
            // 각 행의 평균 계산
            val mean = row.average()
            // 각 행의 분산(평균 제곱 차이) 계산 후 표준편차 구함
            val variance = row.map { (it - mean).pow(2) }.average()
            val std = sqrt(variance)
            // 표준편차가 0이면 0으로 처리
            for (j in 0 until cols) {
                norm[i][j] = if (std != 0.0) (features[i][j] - mean) / std else 0.0
            }
        }
        return norm
    }

    /**
     * 여러 행렬을 행(row) 방향으로 연결합니다.
     * 예: a (13 x T), b (13 x T), c (13 x T)를 연결하면 (39 x T) 행렬이 됩니다.
     *
     * @param arrays 연결할 행렬들 (모두 동일한 열 수를 가져야 함)
     * @return       연결된 행렬
     */
    fun concatenateMatrices(vararg arrays: Array<DoubleArray>): Array<DoubleArray> {
        if (arrays.isEmpty()) return arrayOf()
        val cols = arrays[0][0].size
        val totalRows = arrays.sumBy { it.size }
        val result = Array(totalRows) { DoubleArray(cols) }
        var currentRow = 0
        for (array in arrays) {
            if (array[0].size != cols) {
                throw IllegalArgumentException("All matrices must have the same number of columns.")
            }
            for (row in array) {
                row.copyInto(result[currentRow])
                currentRow++
            }
        }
        return result
    }

    /**
     * 원본 MFCC 행렬에 대해 Savitzky–Golay 필터를 이용한 델타 및 델타-델타를 계산하고,
     * 정규화한 후 행 방향으로 연결하여 최종 MFCC 피처(예: 39차원)를 생성합니다.
     *
     * @param mfcc         원본 MFCC 행렬 (shape: [nCoeffs x nFrames], 예: 13 x T)
     * @param windowLength 필터 윈도우 길이 (예: 9)
     * @param polyOrder    다항식 차수 (예: 2)
     * @return             연결된 MFCC 피처 행렬 (예: 39 x nFrames)
     */
    fun processMfccFeaturesSavGolay(mfcc: Array<DoubleArray>, windowLength: Int, polyOrder: Int): Array<DoubleArray> {
        val mfccDelta = savitzkyGolayDelta(mfcc, windowLength, polyOrder, 1)
        val mfccDelta2 = savitzkyGolayDelta(mfcc, windowLength, polyOrder, 2)

        // 각 행렬을 평균 0, 표준편차 1로 정규화 (행별 정규화)
        val normMfcc = normalize(mfcc)
        val normMfccDelta = normalize(mfccDelta)
        val normMfccDelta2 = normalize(mfccDelta2)

        return concatenateMatrices(normMfcc, normMfccDelta, normMfccDelta2)
    }
}
