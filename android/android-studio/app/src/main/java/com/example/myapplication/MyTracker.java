package com.example.myapplication;

import android.graphics.Matrix;
import android.graphics.RectF;

import org.opencv.core.Mat;
import org.opencv.core.MatOfByte;
import org.opencv.core.MatOfFloat;
import org.opencv.core.MatOfPoint2f;
import org.opencv.core.Point;
import org.opencv.core.Size;
import org.opencv.core.TermCriteria;
import org.opencv.imgproc.Imgproc;
import org.opencv.video.Video;

import java.util.ArrayList;
import java.util.List;
import java.util.Random;

public class MyTracker {
    private final static int POINT_CNT = 200;
    private final static float BORDER = 0.1f;
    Random random = new Random();
    private Mat prevGray, frameGray;
//    private MatOfPoint2f prevPtsMat, nextPts;
    private Size winSize;
    private TermCriteria criteria;
    private List<Point> prevPts;
    public static MyTracker create(){
        return new MyTracker();
    }

    public MyTracker() {
        prevGray = new Mat();
        frameGray = new Mat();
        winSize = new Size(10, 10);
        criteria = new TermCriteria(TermCriteria.COUNT + TermCriteria.EPS,
                200,
                0.001);

    }
    Matrix normToFrameTransform;
    Matrix frameToNormTransform;
    public void init(Mat frame, RectF rectF){
        normToFrameTransform = ImageUtils.getTransformationMatrix(
                1,
                1,
                frame.width(),
                frame.height(),
                0,
                false
        );
        frameToNormTransform = new Matrix();
        normToFrameTransform.invert(frameToNormTransform);

        RectF boundingBox = new RectF();

        normToFrameTransform.mapRect(boundingBox, rectF);

        Imgproc.cvtColor(frame, prevGray, Imgproc.COLOR_BGR2GRAY);
        prevPts = new ArrayList<>();
        fillPrevPts(boundingBox);
    }
    private void fillPrevPts(RectF boundingBox){
        float w = boundingBox.width() * (1 - 2 * BORDER);
        float h = boundingBox.height() * (1 - 2 * BORDER);
        float x = boundingBox.left + w * BORDER;
        float y = boundingBox.top + h * BORDER;

//        System.out.println("&&& coordinate: " + boundingBox);


        for (int i = 0; i < POINT_CNT; i++) {
            prevPts.add(new Point((int) (random.nextFloat() * w + x), (int) (random.nextFloat() * h + y)));
//            System.out.println(prevPts.get(i));
        }
    }
    public boolean update(Mat frame, RectF boundingBox) {
        System.out.println("enter in update function");
        try {
            Imgproc.cvtColor(frame, frameGray, Imgproc.COLOR_BGR2GRAY);
            // Convert prevPts to float32
            MatOfPoint2f prevPtsMat = new MatOfPoint2f();
            MatOfPoint2f nextPts = new MatOfPoint2f();
            prevPtsMat.fromList(prevPts);

            // Calculate optical flow using Lucas-Kanade method
            MatOfByte status = new MatOfByte();
            MatOfFloat err = new MatOfFloat();
            Video.calcOpticalFlowPyrLK(
                    prevGray, frameGray, prevPtsMat,
                    nextPts, status, err, winSize,
                    0, criteria);

            byte[] statusArr = status.toArray();
            float[] errArr = err.toArray();
            Point[] p0Arr = prevPtsMat.toArray();
            Point[] p1Arr = nextPts.toArray();

            double a = 0, b = 0, c = 0, d = 0;
            double meanXOld = 0, meanXNew = 0, meanYOld = 0, meanYNew = 0;
            double varXOld = 0, varXNew = 0, varYOld = 0, varYNew = 0;
            Point[] goodP0Arr = new Point[statusArr.length];
            Point[] goodP1Arr = new Point[statusArr.length];
//            double[] goodDx = new double[statusArr.length];
//            double[] goodDy = new double[statusArr.length];
            int cnt = 0;

            prevPts = new ArrayList<>();
            for (int i = 0; i < statusArr.length; i++) {
                if (statusArr[i] == 1) {
                    if (errArr[i] > 0.8)
                        continue;
                    System.out.println("error: " + errArr[i]);
                    Point newPt = p1Arr[i];
                    Point oldPt = p0Arr[i];

                    goodP0Arr[cnt] = p0Arr[i];
                    goodP1Arr[cnt] = p1Arr[i];

                    prevPts.add(newPt);

//                    goodDx[cnt] = newPt.x - oldPt.x;
//                    goodDy[cnt] = newPt.y - oldPt.y;

                    meanXNew += newPt.x;
                    meanYNew += newPt.y;

                    meanXOld += oldPt.x;
                    meanYOld += oldPt.y;

                    cnt++;
                }
            }


            if (cnt <= 0)
                return false;
            meanXNew /= cnt;
            meanYNew /= cnt;
            meanXOld /= cnt;
            meanYOld /= cnt;

            for (int i = 0; i < cnt; i++) {
                varXOld += Math.pow(goodP0Arr[i].x - meanXOld, 2);
                varYOld += Math.pow(goodP0Arr[i].y - meanYOld, 2);
                varXNew += Math.pow(goodP1Arr[i].x - meanXNew, 2);
                varYNew += Math.pow(goodP1Arr[i].y - meanYNew, 2);
            }
            varXOld = Math.sqrt(varXOld)/cnt;
            varYOld = Math.sqrt(varYOld)/cnt;
            varXNew = Math.sqrt(varXNew)/cnt;
            varYNew = Math.sqrt(varYNew)/cnt;

            normToFrameTransform.mapRect(boundingBox);

//            if (cnt < (int) (0.3 * POINT_CNT))
//                fillPrevPts(boundingBox);

            double dx = meanXNew - meanXOld;
            double dy = meanYNew - meanYOld;

//            for (int i = 0; i < cnt; i++) {
//                dx += goodDx[i];
//                dy += goodDy[i];
//            }
//            dx /= cnt;
//            dy /= cnt;

//            double wVar = 0;
//            for (int i = 0; i < cnt; i++) {
//                wVar += Math.pow(goodDx[i] - dx, 2);
//            }


            double w = boundingBox.width() * varXNew / varXOld;
            double h = boundingBox.height() * varYNew / varYOld;
            double x = (int) (boundingBox.left + dx);
            double y = (int) (boundingBox.top + dy);
//            boundingBox = new RectF((float) x, (float) y, (float) (x + w), (float) (y + h));
            boundingBox.left = (int) (x);
            boundingBox.top = (int) (y);
            boundingBox.right = (int) (x + w);
            boundingBox.bottom = (int) (y + h);

            frameToNormTransform.mapRect(boundingBox);

            prevGray = frameGray.clone();

            return true;
        }catch (Exception e) {
            System.out.println("MyTracker -> public boolean update(Mat frame, Rect boundingBox):" + e.toString());
        }
        return false;
    }
}
