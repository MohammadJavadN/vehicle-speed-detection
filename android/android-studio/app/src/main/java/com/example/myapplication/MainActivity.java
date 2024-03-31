package com.example.myapplication;

import android.Manifest;
import android.content.Context;
import android.content.pm.PackageManager;
import android.database.Cursor;
import android.graphics.Bitmap;
import android.graphics.Canvas;
import android.graphics.Matrix;
import android.net.Uri;
import android.os.Build;
import android.os.Bundle;
import android.provider.MediaStore;
import android.view.SurfaceView;
import android.widget.Switch;
import android.widget.Toast;

import androidx.activity.EdgeToEdge;
import androidx.activity.result.ActivityResultLauncher;
import androidx.activity.result.contract.ActivityResultContracts;
import androidx.appcompat.app.AppCompatActivity;
import androidx.core.app.ActivityCompat;
import androidx.core.content.ContextCompat;

import com.example.myapplication.ml.LicensePlateDetectorFloat32;
import com.example.myapplication.ml.SpeedPredictionModel;
import com.example.myapplication.ml.SpeedPredictionModelSideView;

import org.opencv.android.Utils;
import org.opencv.core.MatOfByte;
import org.opencv.core.MatOfFloat;
import org.opencv.core.MatOfPoint;
import org.opencv.core.MatOfPoint2f;
import org.opencv.core.TermCriteria;
import org.opencv.imgproc.Imgproc;
import org.opencv.video.Video;
import org.opencv.videoio.VideoWriter;
import org.tensorflow.lite.DataType;
import org.tensorflow.lite.support.tensorbuffer.TensorBuffer;

import java.io.File;
import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.concurrent.Executors;
import java.util.concurrent.ScheduledExecutorService;
import java.util.concurrent.TimeUnit;

import org.opencv.core.Core;
import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.videoio.VideoCapture;
import org.opencv.videoio.Videoio;


import org.opencv.core.Point;
import org.opencv.core.Scalar;
import org.opencv.core.Size;


public class MainActivity extends AppCompatActivity {

    private static String inVideoPath = "/sdcard/Download/Video.mp4";
    private static String outVideoPath = "/sdcard/Download/ou_.mp4";
    private static int maxFrames = 500;

    private static VideoCapture cap;
    private static VideoWriter out;
    private static Scalar speedColor = new Scalar(255,0,0);

    private static LicensePlateDetectorFloat32 plateDetectorModel;
    private static SpeedPredictionModel speedPredictionModel;
    private static SpeedPredictionModelSideView speedPredictionModelSideView;
    private static TensorBuffer plateInputFeature;
    private static TensorBuffer speedInputFeature;
    private static TensorBuffer sideSpeedInputFeature;
    private static int height, width;
    private static Mat prevGray;
    private static List<Point> prevPts = new ArrayList<>();

    private static int frameNum;
    private static Size winSize;
    private static TermCriteria criteria;
    private static int imW;
    private static int imH;

    ActivityResultLauncher<String> filechoser;
    SurfaceView surfaceView;
    private ScheduledExecutorService scheduledExecutorService;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        EdgeToEdge.enable(this);
        setContentView(R.layout.activity_main);
        getPermission();
        init();

        surfaceView = findViewById(R.id.surfaceView);

        filechoser = registerForActivityResult(
                new ActivityResultContracts.GetContent(),
                o -> {
                    String path = getRealPathFromURI(MainActivity.this, o); // TODO: uncomment
                    if (path == null)
                        path = MainActivity.inVideoPath;
                    if(new File(path).exists())
                        MainActivity.inVideoPath = path;
                    String[] paths = MainActivity.inVideoPath.split("/");
                    paths[paths.length-1] = "output.mp4";
                    MainActivity.outVideoPath = String.join("/", paths);
                    Toast.makeText(getApplicationContext(),
                            "out_path: " + outVideoPath,
                            Toast.LENGTH_LONG).show();
//                    MainActivity.inVideoPath = getRealPathFromURI(MainActivity.this, o); // TODO: uncomment
                    System.out.println(MainActivity.inVideoPath);

                    // Start updating frames periodically
                    startUpdatingFrames();
                }
        );

    }

    private static final int PERMISSION_REQUEST_CODE = 100;

    void getPermission(){
        if (ContextCompat.checkSelfPermission(MainActivity.this,
                Manifest.permission.WRITE_EXTERNAL_STORAGE)
                != PackageManager.PERMISSION_GRANTED) {
            ActivityCompat.requestPermissions(MainActivity.this,
                    new String[]{Manifest.permission.WRITE_EXTERNAL_STORAGE},
                    PERMISSION_REQUEST_CODE);
        }
        if (ContextCompat.checkSelfPermission(MainActivity.this,
                Manifest.permission.READ_EXTERNAL_STORAGE)
                != PackageManager.PERMISSION_GRANTED) {
            ActivityCompat.requestPermissions(MainActivity.this,
                    new String[]{Manifest.permission.READ_EXTERNAL_STORAGE},
                    PERMISSION_REQUEST_CODE);
        }

        else if (ContextCompat.checkSelfPermission(MainActivity.this,
                Manifest.permission.READ_MEDIA_VIDEO)
                != PackageManager.PERMISSION_GRANTED) {
            if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.TIRAMISU) {
                ActivityCompat.requestPermissions(MainActivity.this,
                        new String[]{Manifest.permission.READ_MEDIA_VIDEO},
                        PERMISSION_REQUEST_CODE);
            }
        }
    }

    public String getRealPathFromURI(Context context, Uri contentUri) {
        String filePath = null;
        Cursor cursor = null;
        try {
            String[] projection = {MediaStore.Video.Media.DATA};
            cursor = context.getContentResolver().query(contentUri, projection, null, null, null);
            if (cursor != null && cursor.moveToFirst()) {
                int columnIndex = cursor.getColumnIndexOrThrow(MediaStore.Video.Media.DATA);
                filePath = cursor.getString(columnIndex);
            }
        } catch (Exception e) {
            e.printStackTrace();
        } finally {
            if (cursor != null) {
                cursor.close();
            }
        }
        return filePath;
    }

    private void startUpdatingFrames() {
        MainActivity.cap = new VideoCapture();
        MainActivity.cap.open(MainActivity.inVideoPath);
        int fourcc = VideoWriter.fourcc('m', 'p', '4', 'v');
        double fps = MainActivity.cap.get(Videoio.CAP_PROP_FPS);
        int width = (int) MainActivity.cap.get(Videoio.CAP_PROP_FRAME_WIDTH);
        int height = (int) MainActivity.cap.get(Videoio.CAP_PROP_FRAME_HEIGHT);
        MainActivity.out = new VideoWriter(MainActivity.outVideoPath, fourcc, fps, new Size(width, height));

        scheduledExecutorService = Executors.newSingleThreadScheduledExecutor();
        // Render the frame onto the canvas
        Runnable updateFrameTask = () -> {
            Mat frame;
            if (((Switch)(findViewById(R.id.plateSw))).isChecked())
                frame = predictAndVisualize(); // TODO
            else
                frame = predictWithoutPlate(((Switch)(findViewById(R.id.sideViewSw))).isChecked());
            System.out.println("activated:" + ((Switch)(findViewById(R.id.plateSw))).isChecked());
            if (frame != null) {
                if (((Switch)(findViewById(R.id.saveSw))).isChecked())
                    try {
                        MainActivity.out.write(frame);
                    } catch (Exception e) {
                        System.out.println(e.toString());
                    }
                Canvas canvas = surfaceView.getHolder().lockCanvas();
                if (canvas != null) {
                    // Render the frame onto the canvas
                    // Ensure you have a valid 'frame' variable holding the current frame
                    int w = canvas.getWidth();
                    int h = canvas.getHeight();
                    Matrix matrix = new Matrix();
                    matrix.postRotate(90);
                    Bitmap scaledBitmap = Bitmap.createScaledBitmap(matToBitmap(frame), h, w, false);
                    Bitmap rotatedBitmap = Bitmap.createBitmap(scaledBitmap, 0, 0, h, w, matrix, true);
                    canvas.drawBitmap(rotatedBitmap, 0, 0, null);
                    surfaceView.getHolder().unlockCanvasAndPost(canvas);
                }
            } else
                onDestroy();
        };

        // Schedule the task to run every 33 milliseconds (30 frames per second)
        scheduledExecutorService.scheduleAtFixedRate(
                updateFrameTask,
                0, // Initial delay
                1, // Period (milliseconds)
                TimeUnit.MILLISECONDS);
    }

    @Override
    protected void onDestroy() {
        super.onDestroy();
        // Release resources
        if (MainActivity.cap != null && MainActivity.cap.isOpened())
            MainActivity.cap.release();
        if (MainActivity.out != null && MainActivity.out.isOpened())
            MainActivity.out.release();

        if (scheduledExecutorService != null) {
            scheduledExecutorService.shutdown();
        }
    }

    public static Bitmap matToBitmap(Mat inputMat) {
        System.out.println("%%%" + inputMat.cols() + inputMat.rows());
        Bitmap outputBitmap = Bitmap.createBitmap(inputMat.cols(), inputMat.rows(), Bitmap.Config.ARGB_8888);
        Utils.matToBitmap(inputMat, outputBitmap);
        return outputBitmap;
    }

    public void browseVideo(android.view.View view) {
        filechoser.launch("video/*");
    }

    public static int getId(float x, float y, int imW) {
        int gs = (int) (imW/4.8);  // grid_width
        return (int) ((y / gs) * (imW / gs) + (x / gs));
    }

    public static int getLane(int x, int y) {
        if (x < 0.307*imW) {
            return 1;
        }
        if (x < 0.49 * imW + 0.37 * y) {
            return 2;
        }
        if (x < 0.6875 * imW + 0.8 * y) {
            return 3;
        }
        return 4;
    }

    public void init() {
        System.loadLibrary("opencv_java4");
        if (MainActivity.maxFrames == 0) {
            MainActivity.maxFrames = (int) MainActivity.cap.get(Videoio.CAP_PROP_FRAME_COUNT);
        }

        try {
            plateDetectorModel = LicensePlateDetectorFloat32.newInstance(MainActivity.this);
            plateInputFeature = TensorBuffer.createFixedSize(new int[]{1, 640, 640, 3},
                    DataType.FLOAT32);
            MainActivity.height = 640;
            MainActivity.width = 640;

            speedPredictionModel = SpeedPredictionModel.newInstance(MainActivity.this);
            // Creates inputs for reference.
            speedInputFeature = TensorBuffer.createFixedSize(new int[]{1, 3}, DataType.FLOAT32);

            speedPredictionModelSideView = SpeedPredictionModelSideView.newInstance(MainActivity.this);
            sideSpeedInputFeature = TensorBuffer.createFixedSize(new int[]{1, 3}, DataType.FLOAT32);

        } catch (IOException e) {
            throw new RuntimeException(e);
        }

        MainActivity.winSize = new Size(10, 10);
        MainActivity.criteria = new TermCriteria(TermCriteria.COUNT + TermCriteria.EPS,
                200,
                0.001);
        MainActivity.prevGray = null;
        MainActivity.prevPts = null;
        MainActivity.frameNum = 0;
    }

    public static Mat predictWithoutPlate(boolean sideView) {
        Mat frame = new Mat();
        if (MainActivity.frameNum < MainActivity.maxFrames) {
            boolean ret = cap.read(frame);
            if (!ret) {
                return null;
            }

            MainActivity.frameNum++;
            imH = frame.rows();
            imW = frame.cols();

            Mat imageRgb = new Mat();
            Imgproc.cvtColor(frame, imageRgb, Imgproc.COLOR_BGR2RGB);

            Mat frameGray = new Mat();
            Imgproc.cvtColor(frame, frameGray, Imgproc.COLOR_BGR2GRAY);


            try {
                if (MainActivity.frameNum > 2) {
                    float[] laneSpeeds = {0, 0, 0, 0};
                    int[] cntOfLaneSpeeds = {0, 0, 0, 0};

                    MatOfPoint prevPtsMat_ = new MatOfPoint();

                    Imgproc.goodFeaturesToTrack(MainActivity.prevGray, prevPtsMat_,150,0.3,5, new Mat(),7,false,0.04);

                    MatOfPoint2f prevPtsMat = new MatOfPoint2f(prevPtsMat_.toArray());

                    MatOfPoint2f nextPts = new MatOfPoint2f();
                    // Calculate optical flow using Lucas-Kanade method
                    MatOfByte status = new MatOfByte();
                    MatOfFloat err = new MatOfFloat();
                    Video.calcOpticalFlowPyrLK(
                            MainActivity.prevGray, frameGray, prevPtsMat,
                            nextPts, status, err, MainActivity.winSize,
                            0, MainActivity.criteria);

                    byte[] StatusArr = status.toArray();
                    Point[] p0Arr = prevPtsMat.toArray();
                    Point[] p1Arr = nextPts.toArray();
                    Mat mask = Mat.zeros(frame.size(), CvType.CV_8UC3);

                    for (int i = 0; i<StatusArr.length ; i++ ) {
                        if (StatusArr[i] == 1) {
                            Point newPt = p0Arr[i];

                            Point oldPt = p1Arr[i];

                            double a = newPt.x;
                            double b = newPt.y;
                            double c = oldPt.x;
                            double d = oldPt.y;

                            if (b < 0.33 * imH) {
                                continue;
                            }

                            int lane = getLane((int) newPt.x, (int) newPt.y);

                            double pixelSpeed = Math.sqrt(Math.pow(a - c, 2) + Math.pow(b - d, 2));

                            double[] input_data = {a, b, pixelSpeed};

                            // Get output tensor (predicted_speed_function)
                            float predictedSpeed = predictSpeed(input_data, sideView);
                            if (pixelSpeed > 15) {
                                if (!sideView){
                                    predictedSpeed = 1.3f * predictedSpeed;
                                    laneSpeeds[lane - 1] += predictedSpeed;
                                    cntOfLaneSpeeds[lane - 1]++;
                                } else {
                                    laneSpeeds[0] += predictedSpeed;
                                    cntOfLaneSpeeds[0]++;
                                }
                                Imgproc.line(mask, newPt, oldPt, new Scalar(255, 0, 0), 4 * imW / 1920);
                                Imgproc.circle(mask, newPt, (5 * imW) / 1920, new Scalar(255, 0, 0), -1);
                            }
                        }
                    }
                    if (sideView){
                        // Draw lines and circles
                        int speed = (int) (laneSpeeds[0]/cntOfLaneSpeeds[0]*10);

                        if (speed < 20)
                            speed = 0;
                        Imgproc.putText(frame,
                                Float.toString((float) speed /10),
                                new Point(0.45 * imW, 55),
                                Imgproc.FONT_HERSHEY_PLAIN,
                                5.5*imW/1920,
                                speedColor,
                                6*imW/1920);
                    } else
                        for (int lane = 0; lane < 4; lane++) {
                            // Draw lines and circles
                            int speed = (int) (laneSpeeds[lane]/cntOfLaneSpeeds[lane]*10);

                            if (speed < 5)
                                speed = 0;
                            Imgproc.putText(frame,
                                    Float.toString((float) speed /10),
                                    new Point((lane+1) * 0.208 * imW, 55),
                                    Imgproc.FONT_HERSHEY_PLAIN,
                                    4.5*imW/1920,
                                    speedColor,
                                    6*imW/1920);

                        }
                    // Add mask to frame
                    Core.add(frame, mask, frame);
                }
            }catch (Exception e){
                System.out.println(e.toString());
            }

            MainActivity.prevGray = frameGray.clone();

            // Return the processed frame
            return frame;

        }
        return null;

    }

    public static Mat predictAndVisualize() {
        Mat frame = new Mat();
        if (MainActivity.frameNum < MainActivity.maxFrames) {
            boolean ret = cap.read(frame);
            if (!ret) {
                return null;
            }

            MainActivity.frameNum++;
            imH = frame.rows();
            imW = frame.cols();

            Mat imageRgb = new Mat();
            Imgproc.cvtColor(frame, imageRgb, Imgproc.COLOR_BGR2RGB);


            Mat imageResized = new Mat();
            Imgproc.resize(imageRgb, imageResized, new Size(MainActivity.width, MainActivity.height));

            System.out.println(imageResized.get(0,0)[0]);
            Core.normalize(imageResized, //todo: imageResized to input
                    imageResized, 0.0, 1.0, Core.NORM_MINMAX, CvType.CV_32FC3);
            System.out.println(imageResized.get(0,0)[0]);


            Map<Integer, List<Float>> plates = predictPlates(imageResized);

            Mat frameGray = new Mat();
            Imgproc.cvtColor(frame, frameGray, Imgproc.COLOR_BGR2GRAY);


            try {
                if (MainActivity.frameNum > 2 && !MainActivity.prevPts.isEmpty()) {

                    // Convert prevPts to float32
                    MatOfPoint2f prevPtsMat = new MatOfPoint2f();
                    prevPtsMat.fromList(prevPts);

                    MatOfPoint2f nextPts = new MatOfPoint2f();
                    // Calculate optical flow using Lucas-Kanade method
                    Video.calcOpticalFlowPyrLK(
                            MainActivity.prevGray, frameGray, prevPtsMat,
                            nextPts, new MatOfByte(), new MatOfFloat(), MainActivity.winSize,
                            5, MainActivity.criteria);
                    Point[] p0Arr = prevPtsMat.toArray();
                    Point[] p1Arr = nextPts.toArray();
                    // Draw the tracks
                    //    private MatOfPoint2f nextPts = new MatOfPoint2f();
                    Mat mask = Mat.zeros(frame.size(), CvType.CV_8UC3);

                    for (int i = 0; i < prevPts.size(); i++) {
                        Point newPt = p0Arr[i];
                        Point oldPt = p1Arr[i];

                        double a = newPt.x;
                        double b = newPt.y;
                        double c = oldPt.x;
                        double d = oldPt.y;

                        if (b < 0.18 * imH) {
                            continue;
                        }

                        double pixelSpeed = Math.sqrt(Math.pow(a - c, 2) + Math.pow(b - d, 2));

                        double[] input_data = {a, b, pixelSpeed};

                        // Get output tensor (predicted_speed_function)
                        float predictedSpeed = predictSpeed(input_data, false)*10;
                        int speed = (int) predictedSpeed;
                        // Draw lines and circles
                        Imgproc.line(mask, newPt, oldPt, new Scalar(255, 0, 0), 2);
                        Imgproc.circle(mask, newPt, 5, new Scalar(255, 0, 0), -1);
                        // Draw label text
                        Imgproc.putText(frame,
                                Float.toString((float) speed /10),
                                newPt,
                                Imgproc.FONT_HERSHEY_PLAIN,
                                4.5*imW/1920,
                                speedColor,
                                6*imW/1920);

                        int lane = getLane((int) newPt.x, (int) newPt.y);
                        Imgproc.putText(frame,
                                Float.toString((float) speed /10),
                                new Point(lane * 0.208 * imW, 55),
                                Imgproc.FONT_HERSHEY_PLAIN,
                                4.5*imW/1920,
                                speedColor,
                                6*imW/1920);

                    }
                    // Add mask to frame
                    Core.add(frame, mask, frame);
                }
            }catch (Exception e){
                System.out.println(e.toString());
            }

            MainActivity.prevPts = new ArrayList<>();
            for (List<Float> sumBoxCnt : plates.values()) {
                float sumX = sumBoxCnt.get(0);
                float sumY = sumBoxCnt.get(1);
                float sumW = sumBoxCnt.get(2);
                float sumH = sumBoxCnt.get(3);
                float cnt = sumBoxCnt.get(4);

                float xCenter = sumX / cnt;
                float yCenter = sumY / cnt;
                float w = sumW / cnt;
                float h = sumH / cnt;

                MainActivity.prevPts.add(new Point((int) (xCenter * imW), (int) (yCenter * imH)));

                int x1 = (int) ((xCenter - w / 2) * imW);
                int y1 = (int) ((yCenter - h / 2) * imH);
                int x2 = (int) ((xCenter + w / 2) * imW);
                int y2 = (int) ((yCenter + h / 2) * imH);

                Imgproc.rectangle(frame, new Point(x1, y1), new Point(x2, y2), speedColor, 2);
            }

            MainActivity.prevGray = frameGray.clone();

            // Return the processed frame
            return frame;

        }
        return null;

    }

    public static ByteBuffer doubleToByteBuffer(double[] data){
        int bufferSize = data.length * 4;
        ByteBuffer byteBuffer = ByteBuffer.allocateDirect(bufferSize);
        byteBuffer.order(ByteOrder.nativeOrder()); // Set the byte order to native

        // Check if the buffer has enough capacity to hold all the data
        if (byteBuffer.remaining() < bufferSize) {
            throw new RuntimeException("ByteBuffer does not have enough capacity to hold all the data");
        }

        // Put the data into the ByteBuffer
        for (double value : data) {
            byteBuffer.putFloat((float) value);
        }

        byteBuffer.rewind(); // Rewind the buffer to the beginning

        return byteBuffer;
    }
    private static float predictSpeed(double[] inputData, boolean sideView) {
        if (sideView){
            double MEAN_A = 936.88328756;
            double MEAN_B = 256.01818182;
            double MEAN_P = 36.73230519;
            double VAR_A = 255.618194466;
            double VAR_B = 111.727442858;
            double VAR_P = 634.69753378;
            double[] normalized;
            normalized = new double[]{(inputData[0] - MEAN_A) / VAR_A,
                    (inputData[1] - MEAN_B) / VAR_B,
                    (inputData[2] - MEAN_P) / VAR_P};
            sideSpeedInputFeature.loadBuffer(doubleToByteBuffer(normalized));

            // Runs model inference and gets result.
            SpeedPredictionModelSideView.Outputs outputs = speedPredictionModelSideView.process(sideSpeedInputFeature);
            TensorBuffer outputFeature0 = outputs.getOutputFeature0AsTensorBuffer();
            return outputFeature0.getFloatValue(0);
        }
        else {
            double MEAN_A = 936.88328756;
            double MEAN_B = 617.87426442;
            double MEAN_P = 42.33951691;
            double VAR_A = 550.843393004;
            double VAR_B = 322.851840306;
            double VAR_P = 26.414411479;
            double[] normalized;
            normalized = new double[]{(inputData[0] - MEAN_A) / VAR_A,
                    (inputData[1] - MEAN_B) / VAR_B,
                    (inputData[2] - MEAN_P) / VAR_P};
            speedInputFeature.loadBuffer(doubleToByteBuffer(normalized));

            // Runs model inference and gets result.
            SpeedPredictionModel.Outputs outputs = speedPredictionModel.process(speedInputFeature);
            TensorBuffer outputFeature0 = outputs.getOutputFeature0AsTensorBuffer();
            return outputFeature0.getFloatValue(0);
        }
    }

    public static ByteBuffer matToByteBuffer(Mat mat) {
        // Ensure the Mat is of type CV_32FC3
        if (mat.type() != CvType.CV_32FC3) {
            throw new IllegalArgumentException("Input Mat must be of type CV_32FC3");
        }

        // Get the number of bytes needed for the Mat data
        int numChannels = mat.channels();
        int bufferSize = (int) (mat.total() * numChannels * 4);
        System.out.println(mat.total() + ", " + numChannels + ", " + 4);
        // Create a ByteBuffer with the appropriate size
        ByteBuffer byteBuffer = ByteBuffer.allocateDirect(bufferSize);
        byteBuffer.order(ByteOrder.nativeOrder()); // Set the byte order to native

        // Get the data elements from the Mat and put them into the ByteBuffer
        float[] data = new float[(int) mat.total() * numChannels];
        mat.get(0, 0, data);

        // Check if the buffer has enough capacity to hold all the data
        if (byteBuffer.remaining() < bufferSize) {
            throw new RuntimeException("ByteBuffer does not have enough capacity to hold all the data");
        }

        // Put the data into the ByteBuffer
        for (float value : data) {
            byteBuffer.putFloat(value);
        }

        byteBuffer.rewind(); // Rewind the buffer to the beginning

        return byteBuffer;
    }

    public static Map<Integer, List<Float>> predictPlates(Mat frame) {
        try {
            plateInputFeature.loadBuffer(matToByteBuffer(frame));
        } catch (Exception e) {
            System.out.println(e.toString());
        }

        // Runs model inference and gets result.
        LicensePlateDetectorFloat32.Outputs outputs = plateDetectorModel.process(plateInputFeature);
        TensorBuffer outputFeature0 = outputs.getOutputFeature0AsTensorBuffer();

        return computeBoxesAndScores(outputFeature0);
    }

    private static Map<Integer, List<Float>> computeBoxesAndScores(TensorBuffer outFeature) {
        System.out.println(Arrays.toString(outFeature.getShape()));

        float[] floats = outFeature.getFloatArray();
        final int N = outFeature.getShape()[2];
        final float threshold = 0.09f;
        Map<Integer, List<Float>> plates = new HashMap<>();
        for (int i = 0; i < N; i++) {
            if(floats[4*N+i] > threshold)
            {
                System.out.println("it is bigger than threshold!");
                float xCenter = floats[i];
                float yCenter = floats[i + N];
                float w = floats[i + 2*N];
                float h = floats[i + 3*N];

                if ((yCenter > 0.104) && (w < 0.093) && (h < 0.052)) {
                    int carId = getId(xCenter * imW, yCenter * imH, imW);

                    List<Float> sumBoxCnt = plates.getOrDefault(carId, Arrays.asList(0f, 0f, 0f, 0f, 0f));
                    assert sumBoxCnt != null;
                    float sumX = sumBoxCnt.get(0);
                    float sumY = sumBoxCnt.get(1);
                    float sumW = sumBoxCnt.get(2);
                    float sumH = sumBoxCnt.get(3);
                    float cnt = sumBoxCnt.get(4);

                    sumBoxCnt.set(0, sumX + xCenter);
                    sumBoxCnt.set(1, sumY + yCenter);
                    sumBoxCnt.set(2, sumW + w);
                    sumBoxCnt.set(3, sumH + h);
                    sumBoxCnt.set(4, cnt + 1);

                    plates.put(carId, sumBoxCnt);
                }
                System.out.println("go for next...");

            }
        }
        System.out.println("all plates of this frame are detected!");
        return plates;

    }
}


