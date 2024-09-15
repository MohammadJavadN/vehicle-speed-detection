package com.example.myapplication;

import android.graphics.Bitmap;
import android.graphics.Canvas;
import android.graphics.Color;
import android.graphics.Matrix;
import android.graphics.Paint;
import android.graphics.RectF;
import android.view.SurfaceView;

import org.opencv.core.Mat;
import org.opencv.core.Size;
import org.opencv.videoio.VideoCapture;
import org.opencv.videoio.VideoWriter;
import org.opencv.videoio.Videoio;

import java.util.ArrayList;
import java.util.List;

public class VideoProcess {

    // SCALE_SIZE must be equal as the required input tensor size of the model
    // SCALE_SIZE1 is for the first detect model and SCALE_SIZE2 is for the second
    private static final int SCALE_SIZE1 = 300, SCALE_SIZE2 = 300;
    private static final int TEXT_SIZE = 40;
    private static int frameNum;
    private static int maxFrames = 500;
    public RoadLine roadLine;
    VideoWriter videoWriter;
    VideoCapture cap;
    private Bitmap frameBitmap;
    private Mat frame;
    private int previewHeight, previewWidth;
    public int viewH, viewW;
    private Bitmap scaledBitmap;
    private Matrix frameToCropTransform;
    private Matrix cropToFrameTransform;
    private Matrix normToCropTransform;
    Matrix normToFrameTransform;
    private Matrix cropToScaledCropTransform;
    private Detector detector;

    public SurfaceView getSurfaceView() {
        return surfaceView;
    }

    private SurfaceView surfaceView;
    private boolean ifWriteVideo = false;
    private Paint borderPaint, textPaint, textBack;
    public VideoProcess(String inVideoPath, SurfaceView surfaceView, Detector detector, String outVideoPath) {
        // Release resources
        if (cap != null && cap.isOpened())
            cap.release();
        if (videoWriter != null && videoWriter.isOpened())
            videoWriter.release();

        this.cap = new VideoCapture();
        cap.open(inVideoPath);
        int fourcc = VideoWriter.fourcc('m', 'p', '4', 'v');
        double fps = cap.get(Videoio.CAP_PROP_FPS);
        int width = (int) cap.get(Videoio.CAP_PROP_FRAME_WIDTH);
        int height = (int) cap.get(Videoio.CAP_PROP_FRAME_HEIGHT);
        this.videoWriter = new VideoWriter(outVideoPath, fourcc, fps, new Size(width, height));

        frameNum = 0;
        if (readNextFrameAsBitmap()) {
            this.previewHeight = frame.rows();
            this.previewWidth = frame.cols();
        }

        this.surfaceView = surfaceView;
        this.detector = detector;

        frameToCropTransform = ImageUtils.getTransformationMatrix(
                previewWidth,
                previewHeight,
                SCALE_SIZE1,
                SCALE_SIZE1,
                0,
                false
        );

        normToCropTransform = ImageUtils.getTransformationMatrix(
                1,
                1,
                SCALE_SIZE1,
                SCALE_SIZE1,
                0,
                false
        );
        normToFrameTransform = ImageUtils.getTransformationMatrix(
                1,
                1,
                previewWidth,
                previewHeight,
                0,
                false
        );
//
        cropToFrameTransform = new Matrix();
        frameToCropTransform.invert(cropToFrameTransform);

        scaledBitmap = Bitmap.createBitmap(SCALE_SIZE1, SCALE_SIZE1, Bitmap.Config.ARGB_8888);
        borderPaint = new Paint();
        borderPaint.setColor(Color.RED);
        borderPaint.setStyle(Paint.Style.STROKE);
        borderPaint.setStrokeWidth(5);
        textPaint = new Paint();
        textPaint.setColor(Color.RED);
        textPaint.setTextSize(TEXT_SIZE);
        textPaint.setStrokeWidth(5);
        textBack = new Paint();
        textBack.setColor(Color.YELLOW);
        textBack.setStyle(Paint.Style.FILL_AND_STROKE);
        roadLine = new RoadLine();

    }

    public Bitmap getFrameBitmap() {
        return frameBitmap;
    }

    private void ScaleImg(Bitmap srcBitmap, Bitmap dstBitmap, Matrix transformMatrix) {
        Canvas canvas = new Canvas(dstBitmap);
        canvas.drawBitmap(srcBitmap, transformMatrix, null);
    }

    protected boolean doProcessForNextFrame() {
        if (readNextFrameAsBitmap()) {

            System.out.println("in doProcessForNextFrame, readNextFrameAsBitmap() done");

            List<Recognition> recognitionList = extractCarsFromFrame();

            System.out.println("in doProcessForNextFrame, extractCarsFromFrame() done");

            ArrayList<Car> cars = assignIDAndSpeed(recognitionList);

            System.out.println("in doProcessForNextFrame, assignIDAndSpeed() done");

            drawCarsSpeed(frameBitmap, cars);

            System.out.println("in doProcessForNextFrame, drawCarsSpeed() done");

            show(true);

            // TODO: 12.04.24
//            saveOutput();

            return true;
        }
        return false;

    }

    protected List<Recognition> extractCarsFromFrame(){

        // scale the image to the required input size of model
        ScaleImg(frameBitmap, scaledBitmap, frameToCropTransform);
        // detect objects using detector
        List<Recognition> recognitionList = detector.detect(scaledBitmap, frameBitmap);

        // remove unnecessary detected recognitions
        String[] classes = {"car"};
        ImageUtils.filteroutRecognitions(classes, recognitionList);

        // merge the overlapped and adjacent recognitions
        ImageUtils.mergeRecognitions(this.previewHeight, this.previewWidth, recognitionList);

        return recognitionList;
    }

    private void saveOutput() {
        if (ifWriteVideo) // TODO
            try {
                videoWriter.write(frame);
            } catch (Exception e) {
                System.out.println(e.toString());
            }
    }

    public boolean show(boolean isShowLine) {
        Canvas canvas = surfaceView.getHolder().lockCanvas();
        if (canvas != null) {
            // Render the frame onto the canvas
            // Ensure you have a valid 'frame' variable holding the current frame
            viewW = canvas.getWidth();
            viewH = canvas.getHeight();
            Matrix matrix = new Matrix();
            matrix.postRotate(90);
            Bitmap scaledBitmap = Bitmap.createScaledBitmap(frameBitmap, viewH, viewW, false);
            Bitmap rotatedBitmap = Bitmap.createBitmap(scaledBitmap, 0, 0, viewH, viewW, matrix, true);
            if (isShowLine)
                roadLine.drawLines(rotatedBitmap);
            canvas.drawBitmap(rotatedBitmap, 0, 0, null);
            surfaceView.getHolder().unlockCanvasAndPost(canvas);
            return true;
        }
        return false;
    }

    private ArrayList<Car> assignIDAndSpeed(List<Recognition> recognitionList) {
        Car.updateAllTrackers(frame);

        System.out.println("in assignIDAndSpeed, Car.updateAllTrackers(frame) done");

        Car.updateAllSpeeds(roadLine);

        System.out.println("in assignIDAndSpeed, Car.updateAllSpeeds(roadLine) done");

//        for (Recognition recognition : recognitionList) {
//            for (Car car: Car.getCars()) {
//                if (!car.isShow)
//                    if (ImageUtils.calculateIOR1(recognition.getLocation(), car.location) > 0.5)
//                        recognition.setLocation(new RectF(0,0,0,0));
//            }
//        }

        for (Recognition recognition : recognitionList) {
            Car.setIDAndSpeed(recognition, frame, frameNum, roadLine);
        }
        System.out.println("in assignIDAndSpeed, Car.setIDAndSpeed(recognition, frame, frameNum, context) done");

        Car.removeUnderScoreCars(); // todo
//        Car.removeOutCars(frameNum); // todo
//        ArrayList<Car> cars = new ArrayList<>();
//        for (Car car: Car.getCars()) {
//            if (car.getFrameNum() == frameNum)
//                cars.add(car);
//        }
        return Car.getCars();
    }

    private void drawCarsSpeed(Bitmap bitmap, ArrayList<Car> cars) {
        Canvas canvas = new Canvas(bitmap);
        Matrix frameToCanvasMatrix = getFrameToCanvasMatrix(canvas, bitmap);
        for (Car car : cars) {
            if (!car.isShow)
                continue;
            RectF rect = new RectF();
            normToFrameTransform.mapRect(rect, car.getLocation());


            frameToCanvasMatrix.mapRect(rect);
            float cornerSize = Math.min(rect.width(), rect.height()) / 8.0f;
            canvas.drawRect(rect, borderPaint);

            //Draw Label Text
            String labelString = "ID: " + car.getID() + ", speed: " + ((float) ((int) car.getSpeed() * 100) / 100);
            canvas.drawRect(rect.left, rect.top - TEXT_SIZE, Math.max(rect.right, rect.left + labelString.length()), rect.top, textBack);
            canvas.drawText(
                    labelString,
                    rect.left,
                    rect.top,
                    textPaint
            );
        }
    }

    private Matrix getFrameToCanvasMatrix(Canvas canvas, Bitmap bitmap) {
        int frameWidth = bitmap.getWidth();
        int frameHeight = bitmap.getHeight();

        float multiplierW = canvas.getWidth() / (float) frameWidth;
        float multiplierH = canvas.getHeight() / (float) frameHeight;

        return ImageUtils.getTransformationMatrix(
                frameWidth,
                frameHeight,
                (int) (multiplierW * frameWidth),
                (int) (multiplierH * frameHeight),
                0,
                false
        );
    }

    private boolean readNextFrameAsBitmap() {
        frame = new Mat();
        if (frameNum < maxFrames) {
            boolean ret = cap.read(frame);
            if (!ret) {
                return false;
            }
            frameNum++;
            this.frameBitmap = ImageUtils.matToBitmap(frame);
            return true;
        }
        return false;
    }


}
