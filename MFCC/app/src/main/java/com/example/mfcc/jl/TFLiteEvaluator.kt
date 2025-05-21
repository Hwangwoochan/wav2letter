package com.example.mfcc.jl

import android.content.Context
import android.util.Log
import org.tensorflow.lite.Interpreter
import java.io.FileInputStream
import java.nio.MappedByteBuffer
import java.nio.channels.FileChannel
import kotlin.math.min
import kotlin.math.roundToInt

class TFLiteEvaluator(context: Context, modelFileName: String) {

    private val interpreter: Interpreter
    private val inputShape: IntArray
    private val inputScale: Float
    private val inputZeroPoint: Int
    private val outputScale: Float
    private val outputZeroPoint: Int

    init {
        // Load model
        interpreter = Interpreter(loadModelFile(context, modelFileName))

        // Get input/output details
        val inputDetails = interpreter.getInputTensor(0)
        val outputDetails = interpreter.getOutputTensor(0)

        inputShape = inputDetails.shape() // 예: [1, 296, 39]
        inputScale = inputDetails.quantizationParams().scale
        inputZeroPoint = inputDetails.quantizationParams().zeroPoint
        Log.d("TFLiteEvaluator", "input scale: $inputScale, zeroPoint: $inputZeroPoint")

        outputScale = outputDetails.quantizationParams().scale
        outputZeroPoint = outputDetails.quantizationParams().zeroPoint
        Log.d("TFLiteEvaluator", "output scale: $outputScale, zeroPoint: $outputZeroPoint")
    }

    private fun loadModelFile(context: Context, modelFileName: String): MappedByteBuffer {
        val fileDescriptor = context.assets.openFd(modelFileName)
        val inputStream = FileInputStream(fileDescriptor.fileDescriptor)
        val fileChannel = inputStream.channel
        val startOffset = fileDescriptor.startOffset
        val declaredLength = fileDescriptor.declaredLength
        return fileChannel.map(FileChannel.MapMode.READ_ONLY, startOffset, declaredLength)
    }

    // 음수 인덱스를 정규화하는 헬퍼 함수:
    private fun normalizeIndex(index: Int, size: Int): Int {
        return if (index < 0) size + index else index
    }

    // Python의 음수 인덱싱을 모방하여 슬라이딩 윈도우를 추출하는 함수
    private fun getSlidingWindow(
        inputData: Array<FloatArray>,
        start: Int,
        end: Int,
        featureDim: Int
    ): Array<FloatArray> {
        // start, end가 음수인 경우 정규화합니다.
        val normStart = normalizeIndex(start, inputData.size)
        val normEnd = normalizeIndex(end, inputData.size)

        // 슬라이싱할 범위는 0과 inputData.size 사이로 제한
        val actualStart = maxOf(0, normStart)
        val actualEnd = minOf(normEnd, inputData.size)

        val window = inputData.sliceArray(actualStart until actualEnd)
        // 만약 선택된 window의 길이가 원하는 길이(end - start)보다 작으면 zero padding
        if (window.size < (end - start)) {
            val paddingSize = (end - start) - window.size
            return window + Array(paddingSize) { FloatArray(featureDim) }
        }
        return window
    }

    fun evaluate(inputData: Array<FloatArray>): Array<FloatArray> {
        val inputWindowLength = inputShape[1]
        val featureDim = inputShape[2]
        // contextLength는 모델의 receptive field에 따른 추가 영역
        val contextLength = 24 + 2 * (7 * 3 + 16) // 예: 24 + 2*(21+16) = 24 + 74 = 98
        val outputShape = interpreter.getOutputTensor(0).shape() // 예: [1, 1, 148, 38]

        // 패딩된 입력 데이터 생성
        val paddedInput = padInput(inputData, inputWindowLength, featureDim)
        val outputList = mutableListOf<FloatArray>()

        val size = inputWindowLength
        val inner = size - 2 * contextLength
        val dataEnd = paddedInput.size // 전체 paddedInput의 길이 기준
        var dataPos = 0

        while (dataPos < dataEnd) {
            val (start, end, yStart, yEnd) = when {
                dataPos == 0 -> { // 첫 번째 윈도우
                    val start = dataPos
                    val end = start + size
                    val yStart = 0
                    val yEnd = yStart + (size - contextLength) / 2
                    dataPos = end - contextLength
                    listOf(start, end, yStart, yEnd)
                }
                dataPos + inner + contextLength >= dataEnd -> { // 마지막 윈도우
                    val shift = (dataPos + inner + contextLength) - dataEnd
                    val start = dataPos - contextLength - shift
                    val end = start + size
                    val yStart = (shift + contextLength) / 2
                    val yEnd = size / 2
                    dataPos = dataEnd
                    listOf(start, end, yStart, yEnd)
                }
                else -> { // 중간 윈도우
                    val start = dataPos - contextLength
                    val end = start + size
                    val yStart = contextLength / 2
                    val yEnd = yStart + inner / 2
                    dataPos = end - contextLength
                    listOf(start, end, yStart, yEnd)
                }
            }

            // 입력 데이터에서 슬라이딩 윈도우 영역 추출 (음수 인덱스도 올바르게 처리됨)
            val inputWindow = getSlidingWindow(paddedInput, start, end, featureDim)
            Log.d("TFLiteEvaluator", "Window start: $start, end: $end")

            // 슬라이딩 윈도우 데이터 quantize (모델 입력 shape: [1, windowLength, featureDim])
            val quantizedWindow = quantizeInput(inputWindow)
            val inputTensor = arrayOf(quantizedWindow)

            // 모델 출력 텐서의 shape에 맞게 출력 버퍼 생성
            val outputBuffer = Array(outputShape[0]) {
                Array(outputShape[1]) {
                    Array(outputShape[2]) { ByteArray(outputShape[3]) }
                }
            }

            // 모델 추론 실행
            interpreter.run(inputTensor, outputBuffer)

            // 출력 텐서에서 yStart부터 yEnd까지 선택 후, dequantize
            val output = outputBuffer[0][0].sliceArray(yStart until yEnd).map { row ->
                row.map { value ->
                    (value - outputZeroPoint) * outputScale
                }.toFloatArray()
            }.toTypedArray()

            outputList.addAll(output)
        }

        return outputList.toTypedArray()
    }

    private fun quantizeInput(inputWindow: Array<FloatArray>): Array<ByteArray> {
        return Array(inputWindow.size) { i ->
            ByteArray(inputWindow[i].size) { j ->
                val quantizedValue = ((inputWindow[i][j] / inputScale) + inputZeroPoint).roundToInt()
                quantizedValue.toByte()
            }
        }
    }

    private fun padInput(inputData: Array<FloatArray>, inputWindowLength: Int, featureDim: Int): Array<FloatArray> {
        var data = inputData
        while (data.size < inputWindowLength) {
            // 패딩: 마지막 행을 복사
            data += data.last()
        }
        // 입력 데이터 길이가 홀수이면, featureDim 길이의 0 벡터를 추가
        if (data.size % 2 == 1) {
            data += FloatArray(featureDim)
        }
        return data
    }
}
