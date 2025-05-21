package com.example.mfcc.jl

import android.content.Context
import android.content.res.AssetManager
import android.util.Log
import com.arthenica.ffmpegkit.FFmpegKit
import com.arthenica.ffmpegkit.ReturnCode
import java.io.File
import java.io.FileOutputStream
import java.io.IOException
import java.io.InputStream

class AudioResampler {

    private val TAG = "AudioResampler"

    /**
     * assets 폴더의 WAV 파일을 내부 캐시 디렉토리로 복사한 후, FFmpegKit을 사용하여 지정 샘플 레이트로 리샘플링합니다.
     *
     * @param context           애플리케이션 컨텍스트
     * @param assetFileName     assets 내의 파일 이름 (예: "test.wav")
     * @param desiredSampleRate 원하는 리샘플링 대상 샘플 레이트 (예: 16000)
     * @param outputFileName    결과 파일 이름 (예: "resampled_test.wav")
     */
    fun resampleAsset(
        context: Context,
        assetFileName: String,
        desiredSampleRate: Int,
        outputFileName: String
    ) {
        try {
            val inputFile = copyAssetToCache(context, assetFileName)
            Log.d(TAG, "Asset copied to: ${inputFile.absolutePath}")

            // 결과 파일은 내부 캐시 디렉토리에 저장합니다.
            val outputFile = File(context.cacheDir, outputFileName)

            // FFmpeg 명령어 구성
            // -i : 입력 파일, -ar : 리샘플링 대상 샘플 레이트
            val command = "-y -i ${inputFile.absolutePath} -ar $desiredSampleRate ${outputFile.absolutePath}"
            Log.d(TAG, "Executing FFmpeg command: $command")

            // FFmpegKit 비동기 실행
            FFmpegKit.executeAsync(command) { session ->
                val returnCode = session.returnCode
                if (ReturnCode.isSuccess(returnCode)) {
                    Log.d(TAG, "FFmpeg execution successful. Resampled file at: ${outputFile.absolutePath}")
                } else {
                    Log.e(TAG, "FFmpeg execution failed. Return code: $returnCode")
                    Log.e(TAG, "Fail stack trace: ${session.failStackTrace}")
                }
            }
        } catch (e: IOException) {
            Log.e(TAG, "Error copying asset file", e)
        }
    }

    /**
     * assets 폴더의 파일을 내부 캐시 디렉토리로 복사하여 File 객체를 반환합니다.
     *
     * 이미 구현해 놓으신 copyAssetToCache() 메서드입니다.
     *
     * @param context       애플리케이션 컨텍스트
     * @param assetFileName 복사할 assets 파일 이름 (예: "test.wav")
     * @return 복사된 File 객체
     * @throws IOException 파일 복사 중 오류 발생 시 예외 발생
     */
    @Throws(IOException::class)
    fun copyAssetToCache(context: Context, assetFileName: String): File {
        val assetManager: AssetManager = context.assets
        val inputStream: InputStream = assetManager.open(assetFileName)
        val tempFile = File(context.cacheDir, assetFileName)
        val outputStream = FileOutputStream(tempFile)

        val buffer = ByteArray(1024)
        var bytesRead: Int
        while (inputStream.read(buffer).also { bytesRead = it } != -1) {
            outputStream.write(buffer, 0, bytesRead)
        }
        inputStream.close()
        outputStream.close()
        return tempFile
    }
}
