package com.example.mfcc.jl;

import android.content.Context
import org.tensorflow.lite.Interpreter
import java.io.FileInputStream
import java.nio.MappedByteBuffer
import java.nio.channels.FileChannel

class TfLiteModelSingleton_Sv private constructor(context: Context) {

    val interpreter: Interpreter

    init {
        // 모델을 로드합니다.
        interpreter = Interpreter(loadModelFile_Sv(context))
    }

    private fun loadModelFile_Sv(context: Context): MappedByteBuffer {
        // assets 폴더에서 모델 파일을 읽어옵니다.
        val fileDescriptor = context.assets.openFd("tiny_wav2letter_int8.tflite")
        val inputStream = FileInputStream(fileDescriptor.fileDescriptor)
        val fileChannel = inputStream.channel
        val startOffset = fileDescriptor.startOffset
        val declaredLength = fileDescriptor.declaredLength
        return fileChannel.map(FileChannel.MapMode.READ_ONLY, startOffset, declaredLength)
    }

    companion object {
        @Volatile private var instance: TfLiteModelSingleton_Sv? = null

        fun getInstance(context: Context): TfLiteModelSingleton_Sv =
            instance ?: synchronized(this) {
                instance ?: TfLiteModelSingleton_Sv(context).also { instance = it }
            }
    }

    fun getOutputTensorInfo(context: Context): List<Pair<IntArray, Int>> {
        // 모든 출력 텐서의 shape과 데이터 타입을 가져옵니다.
        val interpreter = getInstance(context).interpreter
        val outputTensorCount = interpreter.outputTensorCount
        val outputInfoList = mutableListOf<Pair<IntArray, Int>>()
        for (i in 0 until outputTensorCount) {
            val outputTensor = interpreter.getOutputTensor(i)
            val shape = outputTensor.shape() // 텐서의 shape
            val dataType = outputTensor.dataType().ordinal // 텐서의 데이터 타입
            outputInfoList.add(Pair(shape, dataType))
        }
        return outputInfoList
    }

    fun getInputTensorInfo(context: Context): List<Pair<IntArray, Int>> {
        // 모든 입력 텐서의 shape과 데이터 타입을 가져옵니다.
        val interpreter = getInstance(context).interpreter
        val inputTensorCount = interpreter.inputTensorCount
        val inputInfoList = mutableListOf<Pair<IntArray, Int>>()
        for (i in 0 until inputTensorCount) {
            val inputTensor = interpreter.getInputTensor(i)
            val shape = inputTensor.shape() // 텐서의 shape
            val dataType = inputTensor.dataType().ordinal // 텐서의 데이터 타입
            inputInfoList.add(Pair(shape, dataType))
        }
        return inputInfoList
    }
}

