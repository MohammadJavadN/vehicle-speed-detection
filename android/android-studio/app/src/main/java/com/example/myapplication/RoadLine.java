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

import java.util.Arrays;
import java.util.Random;

public class RoadLine {
    private Point point1, point2, point3, point4;
    private Point P1, P2;
    private double W1, W2, LineLength;
    private View circle1, circle2, circle3, circle4;

    private float globalCoeff = 0.72f;

//    public  void setParameters(Point point1, Point point2, Point point3, Point point4, float globalCoeff) {
//        this.point1 = point1;
//        this.point2 = point2;
//        this.point3 = point3;
//        this.point4 = point4;
//        this.globalCoeff = globalCoeff;
//
//    }

    public void initializeCircles(View circle1, View circle2, View circle3, View circle4, VideoProcess videoProcess){
        linePaint.setColor(Color.RED);
        linePaint.setStrokeWidth(5);
        this.videoProcess = videoProcess;

        this.circle3 = circle1;
        this.circle1 = circle2;
        this.circle4 = circle3;
        this.circle2 = circle4;

        updateParameters();

        setVisible(View.VISIBLE);
    }

    VideoProcess videoProcess;

    public void updateParameters() {
        setPoints();
//        setVisible(View.INVISIBLE);
//        drawLines(videoProcess.getFrameBitmap());
    }
    private Matrix normToViewTransform;

    void setVisible(int visible){
        circle1.setVisibility(visible);
        circle2.setVisibility(visible);
        circle3.setVisibility(visible);
        circle4.setVisibility(visible);
    }

    private void setPoints(){
        // TODO: 16.04.24
        float w = videoProcess.viewW;
        float h = videoProcess.viewH;
        normToViewTransform = ImageUtils.getTransformationMatrix(
                1,
                1,
                (int) w,
                (int) h,
                90,
                false
        );

        int offsetX = (int) videoProcess.getSurfaceView().getX() - circle1.getWidth() / 2;
        int offsetY = (int) videoProcess.getSurfaceView().getY() - circle1.getHeight() / 2;
        float x1 = circle1.getX() - offsetX;
        float x2 = circle2.getX() - offsetX;
        float x3 = circle3.getX() - offsetX;
        float x4 = circle4.getX() - offsetX;

        float y1 = circle1.getY() - offsetY;
        float y2 = circle2.getY() - offsetY;
        float y3 = circle3.getY() - offsetY;
        float y4 = circle4.getY() - offsetY;

        float dx1 = x1 - x3;
        float dx2 = x2 - x4;
        float dy1 = y3 - y1;
        float dy2 = y4 - y2;

        float Y1 = dy1 > 0 ? h - y1 : y1;
        float Y2 = dy2 > 0 ? h - y2 : y2;
        float Y3 = dy1 > 0 ? y3 : h - y3;
        float Y4 = dy2 > 0 ? y4 : h - y4;
        float X1 = dx1 > 0 ? x1 : w - x1;
        float X2 = dx2 > 0 ? x2 : w - x2;
        float X3 = dx1 > 0 ? w - x3 : x3;
        float X4 = dx2 > 0 ? w - x4 : x4;

        point1 = new Point(
                Math.max(Math.min(x3 + dx1 * Y3 / Math.abs(dy1), w - 1), 0),
                Math.min(Math.max(y3 - dy1 * (X3) / Math.abs(dx1), 1), h)
        );
        point2 = new Point(
                Math.max(Math.min(x4 + dx2 * Y4 / Math.abs(dy2), w - 1), 0),
                Math.min(Math.max(y4 - (dy2 * (X4) / Math.abs(dx2)), 1), h)
        );
        point3 = new Point(
                Math.min(Math.max(x1 - dx1 * (Y1) / Math.abs(dy1), 1), w),
                Math.max(Math.min(y1 + dy1 * X1 / Math.abs(dx1), h - 1), 0)
        );
        point4 = new Point(
                Math.min(Math.max(x2 - dx2 * (Y2) / Math.abs(dy2), 1), w),
                Math.max(Math.min(y2 + dy2 * X2 / Math.abs(dx2), h - 1), 0)
        );
        P1 = new Point((point1.x + point2.x)/2, (point1.y + point2.y)/2);
        P2 = new Point((point3.x + point4.x)/2, (point3.y + point4.y)/2);
        W1 = Math.pow(d(point1, point2), 0.01);
        W2 = Math.pow(d(point3, point4), 0.01);
        LineLength = d(P1, P2);
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

//        drawLines(videoProcess.getFrameBitmap());

    }

    private double calculateLocalCoefficient(Point point) {
        // TODO: 12.04.24
        return (W1 * d(point, P2) + W2 * d(point, P1))/LineLength/Math.max(W1, W2);
    }

    public float calculateSpeed(Point pN1, Point pN2, int frames) {
        float[] pts = {(float) pN1.x, (float) pN1.y, (float) pN2.x, (float) pN2.y};
        System.out.println(P1 + ", " + P2);
        System.out.println(Arrays.toString(pts));
        normToViewTransform.mapPoints(pts);
        System.out.println(Arrays.toString(pts));
        Point p1 = new Point(pts[0], pts[1]);
        Point p2 = new Point(pts[2], pts[3]);
        double coef = calculateLocalCoefficient(p1);
        return (float) (globalCoeff * d(p1, p2) /frames);
//        Random random = new Random();
//        return random.nextFloat()*10 + 45;
        // TODO: 12.04.24
//        return 0;
    }

    private double d(Point p1, Point p2){
        return Math.sqrt(Math.pow(p1.x - p2.x, 2) + Math.pow(p1.y - p2.y, 2));
    }

    private final Paint linePaint = new Paint();

    public void drawLines(Bitmap bitmap) {
        System.out.println("in drawLines, W1=" + W1 + ", W2=" + W2 + ", LineLength=" + LineLength);
        System.out.println("vid.w=" + videoProcess.viewW + ", vid.h=" + videoProcess.viewH);
        System.out.println(" " + point1 + ", " + point2 + ", " + point3 + ", " + point4);
        updateParameters();
        Canvas canvas = new Canvas(bitmap);
        canvas.drawLine((int) point3.x, (int) point3.y, (int) point1.x, (int) point1.y ,linePaint);
        canvas.drawLine((int) point2.x, (int) point2.y, (int) point4.x, (int) point4.y ,linePaint);

        // TODO: 12.04.24
    }
}
