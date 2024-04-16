package com.example.myapplication;

import android.graphics.Bitmap;
import android.graphics.Canvas;
import android.graphics.Color;
import android.graphics.Matrix;
import android.graphics.Paint;
import android.view.MotionEvent;
import android.view.SurfaceView;
import android.view.View;

import org.opencv.core.Point;

import java.util.Random;

public class RoadLine {
    private Point point1, point2, point3, point4;
    private View circle1, circle2, circle3, circle4;

    private float globalCoeff = 1f;

//    public  void setParameters(Point point1, Point point2, Point point3, Point point4, float globalCoeff) {
//        this.point1 = point1;
//        this.point2 = point2;
//        this.point3 = point3;
//        this.point4 = point4;
//        this.globalCoeff = globalCoeff;
//
//    }

    VideoProcess videoProcess;

    public void setParameters(View circle1, View circle2, View circle3, View circle4, VideoProcess videoProcess) {
        linePaint.setColor(Color.RED);
        linePaint.setStrokeWidth(5);
        this.circle1 = circle1;
        this.circle2 = circle2;
        this.circle3 = circle3;
        this.circle4 = circle4;
        this.videoProcess = videoProcess;
        drawLines(videoProcess.getFrameBitmap());
    }

    public void movePoint(View circle, MotionEvent event, SurfaceView surfaceView) {
        if (event.getRawX() < surfaceView.getX()
                || event.getRawY() < surfaceView.getY()
                || event.getRawX() > surfaceView.getX() + surfaceView.getWidth()
                || event.getRawY() > surfaceView.getY() + surfaceView.getHeight()
        )
            return;
        // TODO: 12.04.24
        circle.setX((int) event.getRawX());
        circle.setY((int) event.getRawY());

        drawLines(videoProcess.getFrameBitmap());

    }

    private float calculateLocalCoefficient(Point point) {
        // TODO: 12.04.24
        return 1f;
    }

    public float calculateSpeed(Point p1, Point p2, int frames) {
        Random random = new Random();
        return random.nextFloat()*10 + 45;
        // TODO: 12.04.24
//        return 0;
    }

    private final Paint linePaint = new Paint();

    public void drawLines(Bitmap bitmap) {
//        int offsetX = (int) surfaceView.getX() - circle1.getWidth() / 2;
//        int offsetY = (int) surfaceView.getY() - circle1.getHeight() / 2;
//        point1 = new Point(circle1.getX() - offsetX, circle1.getY() - offsetY);
//        point2 = new Point(circle2.getX() - offsetX, circle2.getY() - offsetY);
//        point3 = new Point(circle3.getX() - offsetX, circle3.getY() - offsetY);
//        point4 = new Point(circle4.getX() - offsetX, circle4.getY() - offsetY);
//
//        Canvas canvas = new Canvas(bitmap);
//        canvas.drawLine((int) point3.x, (int) point3.y, (int) point1.x, (int) point1.y ,linePaint);

        // TODO: 12.04.24
    }
}
