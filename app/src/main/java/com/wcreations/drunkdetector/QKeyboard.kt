package com.wcreations.drunkdetector

import android.accessibilityservice.AccessibilityService
import android.accessibilityservice.AccessibilityServiceInfo
import android.content.Intent
import android.content.res.AssetFileDescriptor
import android.view.KeyEvent
import android.view.WindowManager
import android.view.accessibility.AccessibilityEvent
import java.io.FileInputStream
import java.nio.ByteBuffer
import java.nio.channels.FileChannel

class QKeyboard : AccessibilityService() {
    var lastTime = 0L
    var lastText = ""
   // var data = ByteBuffer.allocate(10)
   //var tflite = Interpreter(data)

    override fun onServiceConnected() {
        lastTime = System.currentTimeMillis()
        toast("Start monitoring")
    }

    override fun onAccessibilityEvent(event: AccessibilityEvent) {

        when (event.eventType) {
            AccessibilityEvent.TYPE_VIEW_TEXT_CHANGED -> {
                var text = event.text.toString()
                val delta = (System.currentTimeMillis() - lastTime)*1.0/1000

                if(lastText.length > text.length) {
                    lastText = text
                    text = "del"
                    val intent = Intent()
                    intent.action = MainActivity.RECEIVER
                    sendBroadcast(intent)
                }
                else{
                    lastText = text
                    text = text[text.length-2].toString()
                }

                lastTime = System.currentTimeMillis()
                toast("${text} $delta")
            }
        }

    }


    override fun onInterrupt() {

    }
}
