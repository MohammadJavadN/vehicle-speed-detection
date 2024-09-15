package com.example.myapplication;

import android.graphics.RectF;

import org.opencv.core.Mat;
import org.opencv.core.Point;
import org.opencv.core.Rect;

import java.util.ArrayList;
import java.util.Stack;

public class Car extends Recognition implements tracker {
//    private RectF location;
    private final int ID;
    private int frameNum;
    private final int SPEED_CNT = 10;
    private float[] speeds = new float[SPEED_CNT];
    private int speedCnt = 0;
    private float speed;
    private static int newID = 0;
    private static ArrayList<Car> cars = new ArrayList<>();
    private Rect boundingBox;
    private Point prevCenter;
    private Point currentCenter;
    private static final float FADE_RATE = 0.6f;
    private static final float SCORE_TH = 0.5f;
    private float score; // max score = 1
    public boolean isShow = true;

    public Car(Mat frame, int frameNum, Recognition recognition) {
        super(
            recognition.getLabel(),
            recognition.getLocation(),
            recognition.getProb(),
            recognition.getImgHeight(),
            recognition.getImgWidth()
        );

        this.frameNum = frameNum;
        this.ID = newID++;
        this.speed = 0;
        this.speedCnt = 0;
        this.score = 1;
        initializeTracker(frame);
        cars.add(this);
    }

    public static ArrayList<Car> getCars() {
        if (cars == null)
            cars = new ArrayList<>();
        return cars;
    }

    public int getID() {
        return ID;
    }

    public float getSpeed() {
        return speed;
    }

    public void updateLocation(RectF newLocation, int frameNum, RoadLine roadLine) {
        float tmpSpeed = roadLine.calculateSpeed(
                getCenter(location),
                getCenter(newLocation),
                frameNum - this.frameNum
        );
        speeds[speedCnt % SPEED_CNT] = tmpSpeed;
        speedCnt++;
        float lastSpeed = speed;
        speed = 0;
        int cnt = Math.min(speedCnt, SPEED_CNT);
        for (int i = 0; i < cnt; i++) {
            if ((speeds[i] < 0.8 * lastSpeed || speeds[i] > 1.2 * lastSpeed) && speed > 0)
                speed += lastSpeed/cnt;
            else
                this.speed += speeds[i]/cnt;
        }

        if (speed < 5)
            isShow = false;
        location = newLocation;
        this.frameNum = frameNum;
//        this.score *= FADE_RATE;

    }

    private Point getCenter(RectF rect) {
        return new Point(rect.centerX(), rect.centerY());
    }

    public static void setIDAndSpeed(Recognition recognition, Mat frame, int frameNum, RoadLine roadLine) {
        RectF location2 = recognition.getLocation();
        for (Car car : cars) {
//            if (!car.isShow)
//                continue;
            RectF location1 = car.getLocation();
            float ror1 = ImageUtils.calculateIOR1(location1, location2);
            if (ror1 > 0.2){
//                car.updateLocation(location2, frameNum, roadLine);
                car.score = 1;
                return;
            }
        }
        new Car(frame, frameNum, recognition);
    }

    public int getFrameNum() {
        return frameNum;
    }

    public static void removeOutCars(int frameNum){
        cars.removeIf(car -> car.frameNum < frameNum - 1 && car.isShow);
    }
    private MyTracker tracker;

    @Override
    public void initializeTracker(Mat frame) {
//        tracker = TrackerVit.create();
        tracker = MyTracker.create();
//        this.boundingBox = new Rect(
//                locationInt.left,
//                locationInt.top,
//                locationInt.width(),
//                locationInt.height()
//                );
        // Initialize the tracker with the first frame and bounding boxes of detected objects
        tracker.init(frame, location);
    }

    @Override
    public boolean trackerUpdate(Mat frame) {
        System.out.println("enter in trackerUpdate function");
        prevCenter = getCenter(location);

        boolean located =  tracker.update(frame, location);
        setLocation(location);

        System.out.println("in trackerUpdate, tracker.update(frame, boundingBox) done");
        System.out.println("LLL located: " + located + ", score: " + score);

//        locationInt.left = boundingBox.x;
//        locationInt.top = boundingBox.y;
//        locationInt.right = boundingBox.x + boundingBox.width;
//        locationInt.bottom = boundingBox.y + boundingBox.height;
//        setLocationInt(locationInt);
        currentCenter = getCenter(location);

        this.score *= FADE_RATE;

        return located;
    }

    public static void updateAllTrackers(Mat frame){
        System.out.println("enter in updateAllTrackers function");
        cars.removeIf(car -> !car.trackerUpdate(frame) && car.isShow);
    }

    public static void removeUnderScoreCars(){
        cars.removeIf(car -> car.score < SCORE_TH && car.isShow);
    }

    private void updateSpeed(RoadLine roadLine){
        this.speed = roadLine.calculateSpeed(
                prevCenter,
                currentCenter,
                1
        );
    }

    public static void updateAllSpeeds(RoadLine roadLine){
        for (Car car: cars) {
            car.updateSpeed(roadLine);
        }
    }
}
