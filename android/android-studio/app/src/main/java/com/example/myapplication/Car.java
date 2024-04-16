package com.example.myapplication;

import android.graphics.RectF;

import org.opencv.core.Point;

import java.util.ArrayList;

public class Car {
    private RectF location;
    private int ID;
    private int frameNum;
    private float speed;
    private static int newID = 0;
    private static ArrayList<Car> cars = new ArrayList<>();

    public Car(RectF location, int frameNum) {
        this.location = location;
        this.frameNum = frameNum;
        this.ID = newID++;
        this.speed = 0;
        cars.add(this);
    }

    public static ArrayList<Car> getCars() {
        if (cars == null)
            cars = new ArrayList<>();
        return cars;
    }

    public RectF getLocation() {
        return location;
    }

    public int getID() {
        return ID;
    }

    public float getSpeed() {
        return speed;
    }

    public void updateLocation(RectF newLocation, int frameNum, RoadLine roadLine) {
        this.speed = roadLine.calculateSpeed(
                getCenter(location),
                getCenter(newLocation),
                frameNum - this.frameNum
        );
        location = newLocation;
        this.frameNum = frameNum;
    }

    private Point getCenter(RectF rect) {
        return new Point(rect.centerX(), rect.centerY());
    }

    public static void setIDAndSpeed(RectF location2, int frameNum, RoadLine roadLine) {
        for (Car car : cars) {
            if (frameNum - car.frameNum < 3) {
                RectF location1 = car.getLocation();
                if (ImageUtils.isNear(location1, location2)){
                    car.updateLocation(location2, frameNum, roadLine);
                    return;
                }
            }
        }
        new Car(location2, frameNum);
    }

    public int getFrameNum() {
        return frameNum;
    }

    public static void removeOutCars(int frameNum){
        cars.removeIf(car -> car.frameNum < frameNum - 3);
    }
}
