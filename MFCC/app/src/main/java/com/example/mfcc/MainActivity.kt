package com.example.mfcc

import android.os.Bundle
import android.util.Log
import androidx.appcompat.app.AppCompatActivity
import com.example.mfcc.jl.AudioProcessor
import com.example.mfcc.jl.TFLiteEvaluator
import com.example.mfcc.jl.TfLiteModelSingleton_Sv
import com.example.mfcc.jl.CTCGreedyDecoder
import com.example.mfcc.jl.AudioResampler
import java.io.BufferedReader
import java.io.InputStreamReader
import android.content.Context



class MainActivity : AppCompatActivity() {

    private val TAG = "MainActivity"

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)


//        // AudioResampler 인스턴스를 생성 후 resampleAsset() 호출
//        val audioResampler = AudioResampler()
//        audioResampler.resampleAsset(
//            context = this,
//            assetFileName = "test2.wav",         // assets 폴더 내 파일 이름
//            desiredSampleRate = 16000,            // 리샘플링 대상 샘플 레이트
//            outputFileName = "resampled_test.wav" // 결과 파일 이름
//        )


        makeMel()


        // Load and evaluate the model
//        val logit = evaluateModel()
//        decodeLogit(logit)



    }

    private fun decodeLogit(logit: Array<FloatArray> ){
        val logits = prepareCTCInput(logit)
        val decoder = CTCGreedyDecoder(mergeRepeated = true, blankIndex = 0)
        val result = decoder.decode(logits, arrayOf(148).toIntArray())
        val TAG = "CTCDecoder"

        Log.d(TAG, "decodedIndices:")
        result.decodedIndices.forEach { row ->
            Log.d(TAG, "  batch=${row[0]}, time=${row[1]}")
        }

        // (2) decodedValues
        Log.d(TAG, "decodedValues: ${result.decodedValues.joinToString()}")

        // (3) decodedShape
        Log.d(TAG, "decodedShape: ${result.decodedShape.joinToString()}")

        // (4) logProbability
        Log.d(TAG, "logProbability: ${result.logProbability.joinToString()}")
    }



    private fun evaluateModel(): Array<FloatArray> {
        return try {
            // 모델 파일 이름
            val modelFileName = "tiny_wav2letter_int8.tflite"

            // TFLiteEvaluator 인스턴스 생성
            val evaluator = TFLiteEvaluator(this, modelFileName)

            // 입력 데이터 로드
            val inputData = loadInputDataFromTxt(this, "input.txt")
            //val inputData = loadnumpydata(this, "numpy.txt")

            //val inputData = loadInputDataFromTxt(this, "makedata.txt")


            // 입력 데이터 로그 출력
            inputData.forEachIndexed { rowIndex, row ->
                Log.d("InputData", "Row $rowIndex: ${row.joinToString(", ")}")
            }

            // 평가 수행
            val outputData = evaluator.evaluate(inputData)

            // 출력 데이터 검증
            if (outputData.isEmpty() || outputData[0].isEmpty()) {
                Log.e(TAG, "Invalid output data: Empty array")
            }

            // 출력 결과 로그 출력
            Log.d(TAG, "Output Data Shape: ${outputData.size} x ${outputData[0].size}")
            outputData.forEachIndexed { index, row ->
                // Log.d(TAG, "Row $index: ${row.joinToString(", ")}")
                Log.d(TAG, "${row.joinToString(", ")}")
            }

            outputData // ✅ 평가 결과 반환
        } catch (e: Exception) {
            Log.e(TAG, "Error occurred during evaluation", e)
            emptyArray()
        }
    }



    fun prepareCTCInput(outputList: Array<FloatArray>): Array<Array<FloatArray>> {
        val timeSteps = outputList.size
        val numClasses = outputList[0].size
        val batchSize = 1  // 대부분의 경우 배치 크기는 1

        val yPredict = Array(timeSteps) { Array(batchSize) { FloatArray(numClasses) } }

        for (t in 0 until timeSteps) {
            for (c in 0 until numClasses) {
                yPredict[t][0][c] = outputList[t][c] // 기존 데이터 변환
            }
        }

        return yPredict // Shape: (time, batch, classes)
    }


    fun loadInputDataFromTxt(context: Context, fileName: String): Array<FloatArray> {
        val inputData = mutableListOf<FloatArray>()

        // assets 디렉토리에서 파일 읽기
        val inputStream = assets.open(fileName)
        val reader = BufferedReader(InputStreamReader(inputStream))

        reader.useLines { lines ->
            lines.forEach { line ->
                // 빈 문자열을 건너뜁니다.
                if (line.isNotBlank() && line.contains("[")) {
                    val cleanedLine = line
                        .replace("[", "") // 대괄호 제거
                        .replace("]", "") // 대괄호 제거
                        .trim()

                    if (cleanedLine.isNotEmpty()) { // 빈 문자열이 아닌 경우 처리
                        val floatArray = cleanedLine.split(",") // 쉼표 기준으로 분리
                            .mapNotNull {
                                it.trim().toFloatOrNull() // 문자열을 Float으로 변환, null은 무시
                            }
                            .toFloatArray()

                        if (floatArray.isNotEmpty()) { // 유효한 데이터만 추가
                            inputData.add(floatArray)
                        }
                    }
                }
            }
        }

        // 데이터 읽기 종료
        reader.close()

        return inputData.toTypedArray() // List<FloatArray> -> Array<FloatArray> 변환
    }

    fun loadnumpydata(context: Context, fileName: String): Array<FloatArray> {
        val inputData = mutableListOf<FloatArray>()

        // assets 디렉토리에서 파일 읽기
        context.assets.open(fileName).bufferedReader().use { reader ->
            reader.forEachLine { line ->
                // 줄의 양쪽 공백을 제거한 후, 하나 이상의 공백을 구분자로 분리하고 Float로 변환
                val row = line.trim()
                    .split(Regex("\\s+"))
                    .map { it.toFloat() }
                    .toFloatArray()
                inputData.add(row)
            }
        }

        // List<FloatArray>를 Array<FloatArray>로 변환하여 반환
        return inputData.toTypedArray()
    }

    fun Qt_DataFromTxt(context: Context, fileName: String): Array<IntArray> {
        val inputData = mutableListOf<IntArray>()

        // assets 디렉토리에서 파일 읽기
        val inputStream = assets.open(fileName)
        val reader = BufferedReader(InputStreamReader(inputStream))

        reader.useLines { lines ->
            lines.forEach { line ->
                // 빈 문자열을 건너뜁니다.
                if (line.isNotBlank() && line.contains("[")) {
                    val cleanedLine = line
                        .replace("[", "") // 대괄호 제거
                        .replace("]", "") // 대괄호 제거
                        .trim()

                    if (cleanedLine.isNotEmpty()) { // 빈 문자열이 아닌 경우 처리
                        val IntArray = cleanedLine.split(",") // 쉼표 기준으로 분리
                            .mapNotNull {
                                it.trim().toIntOrNull() // 문자열을 Float으로 변환, null은 무시
                            }
                            .toIntArray()

                        if (IntArray.isNotEmpty()) { // 유효한 데이터만 추가
                            inputData.add(IntArray)
                        }
                    }
                }
            }
        }

        // 데이터 읽기 종료
        reader.close()

        return inputData.toTypedArray()
    }


    private fun load_model(){
        try {
            // Singleton 인스턴스 가져오기
            val tfliteModel = TfLiteModelSingleton_Sv.getInstance(this)

            // 입력 텐서 정보 가져오기
            val inputTensorInfo = tfliteModel.getInputTensorInfo(this)

            Log.d(TAG, "Input Tensor Info:")
            inputTensorInfo.forEachIndexed { index, info ->
                val (shape, dataType) = info
                Log.d(TAG, "Input Tensor $index - Shape: ${shape.contentToString()}, DataType: $dataType")
            }

            // 출력 텐서 정보 가져오기
            val outputTensorInfo = tfliteModel.getOutputTensorInfo(this)
            Log.d(TAG, "Output Tensor Info:")
            outputTensorInfo.forEachIndexed { index, info ->
                val (shape, dataType) = info
                Log.d(TAG, "Output Tensor $index - Shape: ${shape.contentToString()}, DataType: $dataType")
            }

        } catch (e: Exception) {
            Log.e(TAG, "Error occurred while getting tensor info", e)
        }
    }

    private fun makeMel() {
        // AudioProcessor 인스턴스 생성
        val audioProcessor = AudioProcessor()

        // 처리할 오디오 파일 이름 (assets 디렉토리에서 제공)
        val inputFileName = "test.wav"

        // 저장할 결과 파일 이름WW
        val outputFileName = "mfcc_output.txt"

        try {
            //val inputData = loadInputDataFromTxt(this, "input.txt")
            //audioProcessor.saveMfccToTxt(this, inputData,"tmp.txt" )

            // 오디오 처리 시작
            audioProcessor.processAudioFromAssets(this, inputFileName, outputFileName)

            // 로그로 경로 확인
            Log.d(TAG, "오디오 처리가 완료되었습니다. 결과 파일 이름: $outputFileName")

        } catch (e: Exception) {
            Log.e(TAG, "오디오 처리 중 오류 발생: ", e)
        }
    }
}
