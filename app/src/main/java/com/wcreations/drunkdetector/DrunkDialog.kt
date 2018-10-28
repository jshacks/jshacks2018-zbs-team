package com.wcreations.drunkdetector

import android.os.Bundle
import android.view.View
import kotlinx.android.synthetic.main.drunk_dialog.*

class DrunkDialog : QDialog(R.layout.drunk_dialog) {
    override fun onViewCreated(view: View, savedInstanceState: Bundle?) {
        super.onViewCreated(view, savedInstanceState)
        button.setOnClickListener {
            dismiss()
        }
    }
}