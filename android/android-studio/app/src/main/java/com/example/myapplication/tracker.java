package com.example.myapplication;

import org.opencv.core.Mat;
import android.graphics.Rect;

public interface tracker {

    public void initializeTracker(Mat frame);

    public boolean trackerUpdate(Mat frame);
}
