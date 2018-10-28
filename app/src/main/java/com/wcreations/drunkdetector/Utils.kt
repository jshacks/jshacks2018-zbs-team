package com.wcreations.drunkdetector

import android.widget.Toast

fun QKeyboard.toast(str: String){
    Toast.makeText(baseContext,str,Toast.LENGTH_SHORT).show()
}