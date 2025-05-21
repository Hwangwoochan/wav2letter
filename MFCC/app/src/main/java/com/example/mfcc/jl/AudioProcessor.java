package com.example.mfcc.jl;

import android.content.Context;
import android.content.res.AssetManager;
import android.util.Log;

import com.jlibrosa.audio.JLibrosa;

import java.io.File;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.io.PrintWriter;
import java.util.Arrays;

import com.example.mfcc.jl.SavitzkyGolayMfccProcessor;

public class AudioProcessor {

    private static final String TAG = "AudioProcessor";

    public void processAudioFromAssets(Context context, String fileName, String outputFileName) {
        JLibrosa jLibrosa = new JLibrosa();

        // 기본 샘플링 레이트
        int sampleRate = 48000;

        try {
            // assets 디렉토리에서 파일을 읽어 임시 파일로 저장
            File tempFile = copyAssetToCache(context, fileName);

            // 오디오 데이터를 jLibrosa로 로드
            float[] audioData = jLibrosa.loadAndRead(tempFile.getAbsolutePath(), sampleRate, -1);

            // MFCC 추출
            int nMfcc = 13;  // MFCC 개수 설정
            int n_fft = 512;
            int n_mel = 128;
            int hop_length = 160;

            float[][] mfccValues = jLibrosa.generateMFCCFeatures(audioData, sampleRate, nMfcc, n_fft, n_mel, hop_length);
            //  public float[][] generateMFCCFeatures(float[] magValues, int mSampleRate, int nMFCC, int n_fft, int n_mels, int hop_length) {
            saveMfccToTxt(context, mfccValues, outputFileName);

            // 1. float[][] -> double[][] 변환
            int nRows = mfccValues.length;
            int nCols = mfccValues[0].length;
            double[][] mfccDouble = new double[nRows][nCols];
            for (int i = 0; i < nRows; i++) {
                for (int j = 0; j < nCols; j++) {
                    mfccDouble[i][j] = mfccValues[i][j];
                }
            }

            // 2. Savitzky–Golay 필터를 이용하여 델타 및 델타-델타 계산 후 원본, 델타, 델타-델타를 연결 (예: 39차원)


            // windowLength와 polyOrder는 필요에 따라 조절 (여기서는 windowLength=9, polyOrder=2 사용)
            double[][] processedMfccDouble = SavitzkyGolayMfccProcessor.INSTANCE.processMfccFeaturesSavGolay(mfccDouble, 9, 2);

            // 3. 필요시 double[][] -> float[][] 변환
            float[][] processedMfcc = new float[processedMfccDouble.length][processedMfccDouble[0].length];
            for (int i = 0; i < processedMfccDouble.length; i++) {
                for (int j = 0; j < processedMfccDouble[0].length; j++) {
                    processedMfcc[i][j] = (float) processedMfccDouble[i][j];
                }
            }

            //processedMfcc= transpose(processedMfcc);
            //saveMfccToUseTxt(context, processedMfcc, "makedmfcc");

            // 결과 출력 및 저장

        } catch (Exception e) {
            Log.e(TAG, "오디오 처리 중 오류 발생: ", e);
        }
    }

    // assets의 파일을 임시 캐시 파일로 복사
    public File copyAssetToCache(Context context, String assetFileName) throws IOException {
        AssetManager assetManager = context.getAssets();
        InputStream inputStream = assetManager.open(assetFileName);
        File tempFile = new File(context.getCacheDir(), assetFileName);
        FileOutputStream outputStream = new FileOutputStream(tempFile);

        byte[] buffer = new byte[1024];
        int bytesRead;
        while ((bytesRead = inputStream.read(buffer)) != -1) {
            outputStream.write(buffer, 0, bytesRead);
        }

        inputStream.close();
        outputStream.close();

        return tempFile;
    }
    public static float[][] transpose(float[][] matrix) {
        int rows = matrix.length;
        int cols = matrix[0].length;
        float[][] transposed = new float[cols][rows];
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                transposed[j][i] = matrix[i][j];
            }
        }
        return transposed;
    }

    // MFCC 값을 .txt 파일로 저장
    public void saveMfccToTxt(Context context, float[][] mfccValues, String outputFileName) {
        File outputFile = new File(context.getCacheDir(), outputFileName);
        try (PrintWriter writer = new PrintWriter(outputFile)) {
            for (int i = 0; i < mfccValues.length; i++) {
                // 각 MFCC 벡터를 쉼표로 구분된 문자열로 변환하여 저장
                String mfccLine = Arrays.toString(mfccValues[i])
                        .replace("[", "")
                        .replace("]", "")
                        .replace(" ", "") // 불필요한 공백 제거
                        .replace(",", " ");
                writer.println(mfccLine);
            }
            Log.d(TAG, "MFCC 데이터가 저장되었습니다: " + outputFile.getAbsolutePath());
        } catch (IOException e) {
            Log.e(TAG, "MFCC 데이터를 저장하는 중 오류 발생: ", e);
        }
    }

    private void saveMfccToUseTxt(Context context, float[][] mfccValues, String outputFileName) {
        File outputFile = new File(context.getCacheDir(), outputFileName);
        try (PrintWriter writer = new PrintWriter(outputFile)) {
            for (int i = 0; i < mfccValues.length; i++) {
                // 각 MFCC 벡터를 쉼표로 구분된 문자열로 변환하여 저장
                String mfccLine = Arrays.toString(mfccValues[i]);
                writer.println(mfccLine);
            }
            Log.d(TAG, "MFCC 데이터가 저장되었습니다: " + outputFile.getAbsolutePath());
        } catch (IOException e) {
            Log.e(TAG, "MFCC 데이터를 저장하는 중 오류 발생: ", e);
        }
    }

}



