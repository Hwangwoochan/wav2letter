plugins {
    alias(libs.plugins.android.application)
    alias(libs.plugins.kotlin.android)
}

android {
    namespace = "com.example.mfcc"
    compileSdk = 35

    defaultConfig {
        applicationId = "com.example.mfcc"
        minSdk = 33
        targetSdk = 34
        versionCode = 1
        versionName = "1.0"

        testInstrumentationRunner = "androidx.test.runner.AndroidJUnitRunner"
    }

    buildTypes {
        release {
            isMinifyEnabled = false
            proguardFiles(
                getDefaultProguardFile("proguard-android-optimize.txt"),
                "proguard-rules.pro"
            )
        }
    }
    compileOptions {
        sourceCompatibility = JavaVersion.VERSION_11
        targetCompatibility = JavaVersion.VERSION_11
    }
    kotlinOptions {
        jvmTarget = "11"
    }
}

dependencies {

    implementation(libs.appcompat)
    implementation(libs.material)
    implementation(libs.core.ktx)
    testImplementation(libs.junit)
    androidTestImplementation(libs.ext.junit)
    androidTestImplementation(libs.espresso.core)

    implementation (files("libs\\jlibrosa-1.1.8-SNAPSHOT-jar-with-dependencies.jar"))

    implementation ("be.tarsos.dsp:core:2.5")
    implementation ("be.tarsos.dsp:jvm:2.5")


    // TensorFlow Lite
    implementation("org.tensorflow:tensorflow-lite:2.13.0")
    implementation("org.tensorflow:tensorflow-lite-support:0.4.3") // TFLite 지원 라이브러리
    implementation ("org.tensorflow:tensorflow-lite-task-vision:0.4.0") // TFLite 비전 태스크 지원

    implementation ("com.arthenica:ffmpeg-kit-full:6.0-2")
}