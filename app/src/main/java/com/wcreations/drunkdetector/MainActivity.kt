package com.wcreations.drunkdetector

import android.app.Activity
import android.content.BroadcastReceiver
import android.content.Context
import android.content.Intent
import androidx.appcompat.app.AppCompatActivity
import android.os.Bundle
import android.util.Log
import android.widget.Toast
import java.lang.Exception
import android.content.IntentFilter
import android.content.res.AssetFileDescriptor
import org.tensorflow.lite.Interpreter
import java.io.FileInputStream
import java.nio.MappedByteBuffer
import java.nio.channels.FileChannel


class MainActivity : AppCompatActivity() {
    lateinit var tflite : Interpreter;

    private fun loadModelFile() : MappedByteBuffer {
        val fileDescriptor = assets.openFd("file_saved4.tflite");
        val inputStream = FileInputStream(fileDescriptor.getFileDescriptor());
        val fileChannel = inputStream.channel;
        val startOffset = fileDescriptor.startOffset;
        val declaredLength = fileDescriptor.declaredLength;
        return fileChannel.map(FileChannel.MapMode.READ_ONLY, startOffset, declaredLength);
    }

    companion object {
        val RECEIVER = "com.wcreations.drunkdetector.MyBroadCastReceiver"
    }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)

        tflite = Interpreter(loadModelFile());

        val intentFilter = IntentFilter()
        intentFilter.addAction(RECEIVER)
        registerReceiver(MyBroadCastReceiver(this), intentFilter)    }

    class MyBroadCastReceiver(val activity: MainActivity) : BroadcastReceiver() {
        override fun onReceive(context: Context?, intent: Intent?) {

            try {
                Toast.makeText(context,"received",Toast.LENGTH_SHORT).show()
                activity.tflite.run(input,output)
                activity.startActivity(Intent(activity,MainActivity::class.java))
            }
            catch (ex: Exception) {
                ex.printStackTrace();
            }
        }
    }
}
