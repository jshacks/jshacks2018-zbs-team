package com.wcreations.drunkdetector

import android.accessibilityservice.AccessibilityService
import android.view.accessibility.AccessibilityEvent
import java.nio.ByteBuffer

class QKeyboard : AccessibilityService() {
    var lastTime = 0L
    var data = ByteBuffer.allocate(10)
   // var tflite = Interpreter(data)

    override fun onServiceConnected() {
        lastTime = System.currentTimeMillis()
        toast("Start monitoring")
    }

    override fun onAccessibilityEvent(event: AccessibilityEvent) {

        when (event.eventType) {
            AccessibilityEvent.TYPE_VIEW_TEXT_CHANGED -> {
                val data = event.text.toString()
                val delta = (System.currentTimeMillis() - lastTime)*1.0/1000
                toast("${data.last()} $delta")
                lastTime = System.currentTimeMillis()
            }
        }

    }

    override fun onInterrupt() {

    }
}
