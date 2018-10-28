package com.wcreations.drunkdetector

import android.os.Bundle
import android.view.LayoutInflater
import android.view.ViewGroup
import androidx.appcompat.app.AppCompatDialogFragment
import androidx.fragment.app.FragmentManager

abstract class QDialog(val layoutId: Int) : AppCompatDialogFragment() {
    override fun onCreate(savedInstanceState: Bundle?) {
        setHasOptionsMenu(true)
        super.onCreate(savedInstanceState)
    }


    override fun onCreateView(inflater: LayoutInflater, container: ViewGroup?, savedInstanceState: Bundle?)
            = inflater.inflate(layoutId,container,false)

    fun show(manager:FragmentManager){show(manager,"dialog")}

}