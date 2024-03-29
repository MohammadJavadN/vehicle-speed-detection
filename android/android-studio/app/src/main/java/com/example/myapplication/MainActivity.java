package com.example.myapplication;

import android.Manifest;
import android.content.Context;
import android.content.pm.PackageManager;
import android.database.Cursor;
import android.graphics.Bitmap;
import android.graphics.Canvas;
import android.net.Uri;
import android.os.Bundle;
import android.os.Handler;
import android.os.Looper;
import android.provider.MediaStore;
import android.view.SurfaceView;
import android.widget.ImageView;

import androidx.activity.EdgeToEdge;
import androidx.activity.result.ActivityResultLauncher;
import androidx.activity.result.contract.ActivityResultContracts;
import androidx.appcompat.app.AppCompatActivity;
import androidx.core.app.ActivityCompat;
import androidx.core.content.ContextCompat;

import com.example.myapplication.ml.LicensePlateDetectorFloat32;
import com.example.myapplication.ml.SpeedPredictionModel;

import org.opencv.android.Utils;
import org.opencv.core.MatOfByte;
import org.opencv.core.MatOfFloat;
import org.opencv.core.MatOfPoint2f;
import org.opencv.core.TermCriteria;
import org.opencv.imgproc.Imgproc;
import org.opencv.video.Video;
import org.tensorflow.lite.DataType;
import org.tensorflow.lite.support.tensorbuffer.TensorBuffer;

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

//    private String xmlPath = "../data/vehicles.xml";
//    private String outVideoPath = "../data/out.mp4";
    private static String inVideoPath = "/sdcard/Download/out.mp4";
//    private static String inVideoPath = "/sdcard/Download/video.h264";
//    private String licensePlateDetectorModelPath = "../models/license_plate_detector_float32.tflite";
    private static int maxFrames = 500;

//    private Map<Integer, Object[]> vehicles;
    private static VideoCapture cap;
//    private VideoWriter out;

    private static LicensePlateDetectorFloat32 plateDetectorModel;
    private static SpeedPredictionModel speedPredictionModel;
    private static TensorBuffer plateInputFeature;
    private static TensorBuffer speedInputFeature;
    //    private Interpreter pi, si;
//    private Interpreter.Options options;
    private static int height, width;
    private final static float inputMean = 127.5f;
    private final static float inputStd = 127.5f;
    private final static double MEAN_A = 936.88328756;
    private final static double MEAN_B = 617.87426442;
    private final static double MEAN_P = 42.33951691;
    private final static double VAR_A = 303428.44361634;
    private final static double VAR_B = 104233.31078923;
    private final static double VAR_P = 697.72113379;
    private final static int MAX_BOXES = 10;
//    private Map<Integer, Float> laneSpeeds;
    private static Mat prevGray;
    private static List<Point> prevPts = new ArrayList<>();

    private static int frameNum;
    private static Size winSize;
    private static TermCriteria criteria;
    private static int imW;
    private static int imH;

    ActivityResultLauncher<String> filechoser;
    private ImageView imageView;
    SurfaceView surfaceView;
    private Thread videoThread;
    private boolean isRunning = false;
    private Handler handler;
    private Runnable updateFrameTask;
    private ScheduledExecutorService scheduledExecutorService;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        EdgeToEdge.enable(this);
        setContentView(R.layout.activity_main);
//        ViewCompat.setOnApplyWindowInsetsListener(findViewById(R.id.main), (v, insets) -> {
//            Insets systemBars = insets.getInsets(WindowInsetsCompat.Type.systemBars());
//            v.setPadding(systemBars.left, systemBars.top, systemBars.right, systemBars.bottom);
//            return insets;
//        });
        getPermission();
        init();


        // Initialize views
//        imageView = findViewById(R.id.imageView);
        surfaceView = findViewById(R.id.surfaceView);



        // Initialize handler
        handler = new Handler(Looper.getMainLooper());


        filechoser = registerForActivityResult(
                new ActivityResultContracts.GetContent(),
                o -> {
//                    MainActivity.inVideoPath = o.getPath();
//                    MainActivity.inVideoPath = getRealPathFromURI(MainActivity.this, o); // TODO: uncomment
                    System.out.println(MainActivity.inVideoPath);

//                    init();
                    /*
                    SurfaceHolder surfaceHolder = surfaceView.getHolder();
                    surfaceHolder.addCallback(new SurfaceHolder.Callback() {
                        @Override
                        public void surfaceCreated(SurfaceHolder holder) {
                            startVideoThread();
                        }

                        @Override
                        public void surfaceChanged(SurfaceHolder holder, int format, int width, int height) {}

                        @Override
                        public void surfaceDestroyed(SurfaceHolder holder) {
                            stopVideoThread();
                        }
                    });
                    startVideoThread();
                    */
                    // Start updating frames periodically
                    startUpdatingFrames(); //TODO: uncomment

                    /*
                    Thread thread = new Thread(new MyRunnable());
                    thread.start();

                    // Show a toast message to indicate that the thread has started
//                    Toast.makeText(this, "Thread started", Toast.LENGTH_SHORT).show();
                    */
                }
        );

    }


    private void startVideoThread() {
        MainActivity.cap = new VideoCapture();
        MainActivity.cap.open(MainActivity.inVideoPath);
        isRunning = true;
        videoThread = new Thread(() -> {
            while (isRunning) {
                Mat frame = predictAndVisualize();
                if (frame != null) {
                    Canvas canvas = surfaceView.getHolder().lockCanvas();
                    if (canvas != null) {
                        // Render the frame onto the canvas
                        // Ensure you have a valid 'frame' variable holding the current frame
                        canvas.drawBitmap(matToBitmap(frame), 0, 0, null);
                        surfaceView.getHolder().unlockCanvasAndPost(canvas);
                    }
                }else
                    stopVideoThread();
            }
//            cap.release();
        });
        videoThread.start();
    }


    private void stopVideoThread() {
        isRunning = false;
        try {
            videoThread.join();
        } catch (InterruptedException e) {
            e.printStackTrace();
        }
    }

    private static final int PERMISSION_REQUEST_CODE = 100;

    void getPermission(){
        if (ContextCompat.checkSelfPermission(MainActivity.this,
                Manifest.permission.READ_EXTERNAL_STORAGE)
                != PackageManager.PERMISSION_GRANTED) {
            ActivityCompat.requestPermissions(MainActivity.this,
                    new String[]{Manifest.permission.READ_EXTERNAL_STORAGE},
                    PERMISSION_REQUEST_CODE);
        }
//        else if (ContextCompat.checkSelfPermission(MainActivity.this,
//                Manifest.permission.INTERNET)
//                != PackageManager.PERMISSION_GRANTED) {
//            ActivityCompat.requestPermissions(MainActivity.this,
//                    new String[]{Manifest.permission.INTERNET},
//                    PERMISSION_REQUEST_CODE);
//        } else if (ContextCompat.checkSelfPermission(MainActivity.this,
//                Manifest.permission.READ_MEDIA_VIDEO)
//                != PackageManager.PERMISSION_GRANTED) {
//            if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.TIRAMISU) {
//                ActivityCompat.requestPermissions(MainActivity.this,
//                        new String[]{Manifest.permission.READ_MEDIA_VIDEO},
//                        PERMISSION_REQUEST_CODE);
//            }
//            else{
//                Toast.makeText(this, "init=" + Build.VERSION.SDK_INT+"tiramisu=" + Build.VERSION_CODES.TIRAMISU, Toast.LENGTH_LONG).show();
//            }
//        }
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
        scheduledExecutorService = Executors.newSingleThreadScheduledExecutor();
        updateFrameTask = () -> {
            Mat frame = predictAndVisualize();
            if (frame != null) {
                Canvas canvas = surfaceView.getHolder().lockCanvas();
                if (canvas != null) {
                    // Render the frame onto the canvas
                    // Ensure you have a valid 'frame' variable holding the current frame
                    canvas.drawBitmap(matToBitmap(frame), 0, 0, null);
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
        if (MainActivity.cap != null && MainActivity.cap.isOpened()) {
            MainActivity.cap.release();
        }
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
    /*
    private static class MyRunnable implements Runnable {
        @Override
        public void run() {
            Mat outFrame = MainActivity.predictAndVisualize();
            // Code to be executed in the background thread
            // This method will be executed when the thread starts
        }
    }
    */
/*
    public Map<Integer, Object[]> parseXml(String xmlPath) {
        Map<Integer, Object[]> vehicles = new HashMap<>();

        try {
            File file = new File(xmlPath);
            DocumentBuilderFactory dbFactory = DocumentBuilderFactory.newInstance();
            DocumentBuilder dBuilder = dbFactory.newDocumentBuilder();
            Document doc = dBuilder.parse(file);

            // doc.getDocumentElement().normalize();

            NodeList vehicleList = doc.getElementsByTagName("vehicle");

            for (int temp = 0; temp < vehicleList.getLength(); temp++) {
                Node node = vehicleList.item(temp);

                if (node.getNodeType() == Node.ELEMENT_NODE) {
                    Element element = (Element) node;

                    if (element.getAttribute("radar").equals("True")) {
                        int iframe = Integer.parseInt(element.getAttribute("iframe"));
                        Element region = (Element) element.getElementsByTagName("region").item(0);
                        int x = Integer.parseInt(region.getAttribute("x"));
                        int y = Integer.parseInt(region.getAttribute("y"));
                        int w = Integer.parseInt(region.getAttribute("w"));
                        int h = Integer.parseInt(region.getAttribute("h"));

                        Element radar = (Element) element.getElementsByTagName("radar").item(0);
                        float speed = Float.parseFloat(radar.getAttribute("speed"));
                        int lane = Integer.parseInt(element.getAttribute("lane"));

                        vehicles.put(iframe + 1, new Object[]{new int[]{x, y, w, h}, speed, lane});
                    }
                }
            }
        } catch (Exception e) {
            e.printStackTrace();
        }

        return vehicles;
    }
*/
    public static int getId(float x, float y, int imW) {
        int gs = 400;  // grid_width
        return (int) ((y / gs) * (imW / gs) + (x / gs));
    }

    public static int getLane(int x, int y) {
        if (x < 590) {
            return 1;
        }
        if (x < 940 + 0.37 * y) {
            return 2;
        }
        if (x < 1320 + 0.8 * y) {
            return 3;
        }
        return 4;
    }

    public void init() {
//        MainActivity.vehicles = parseXml(xmlPath);
//        System.loadLibrary(Core.NATIVE_LIBRARY_NAME);
        System.loadLibrary("opencv_java4");
//        MainActivity.cap = new VideoCapture();
//        MainActivity.cap.open(MainActivity.inVideoPath);
        if (MainActivity.maxFrames == 0) {
            MainActivity.maxFrames = (int) MainActivity.cap.get(Videoio.CAP_PROP_FRAME_COUNT);
        }
        /*
        int fourcc = VideoWriter.fourcc('m', 'p', '4', 'v');
        double fps = MainActivity.cap.get(Videoio.CAP_PROP_FPS);
        int width = (int) MainActivity.cap.get(Videoio.CAP_PROP_FRAME_WIDTH);
        int height = (int) MainActivity.cap.get(Videoio.CAP_PROP_FRAME_HEIGHT);
        MainActivity.out = new VideoWriter(outVideoPath, fourcc, fps, new org.opencv.core.Size(width, height));
        */
        try {
            plateDetectorModel = LicensePlateDetectorFloat32.newInstance(MainActivity.this);
            plateInputFeature = TensorBuffer.createFixedSize(new int[]{1, 640, 640, 3},
                    DataType.FLOAT32);
            MainActivity.height = 640;
            MainActivity.width = 640;

            speedPredictionModel = SpeedPredictionModel.newInstance(MainActivity.this);

            // Creates inputs for reference.
            speedInputFeature = TensorBuffer.createFixedSize(new int[]{1, 3}, DataType.FLOAT32);
        } catch (IOException e) {
            throw new RuntimeException(e);
        }

//        MainActivity.pi = loadTfliteModel(licensePlateDetectorModelPath);
//        MainActivity.options = new Interpreter.Options();
//        MainActivity.options.setNumThreads(4);
//        MainActivity.si = new Interpreter(loadTfliteModel(speedPredictionModelPath), MainActivity.options);
//        MainActivity.sc = loadStandardScaler("../models/std_scaler.bin");
//        MainActivity.height = MainActivity.pid[0][0][0][1];
//        MainActivity.width = MainActivity.pid[0][0][0][2];
/*
        MainActivity.laneSpeeds = new HashMap<>();
        for (int i = 1; i <= 3; i++)
            MainActivity.laneSpeeds.put(i, 0f);
*/
        MainActivity.winSize = new Size(100, 40);
        MainActivity.criteria = new TermCriteria(TermCriteria.COUNT + TermCriteria.EPS,
                10,
                0.03);
        MainActivity.prevGray = null;
        MainActivity.prevPts = null;
        MainActivity.frameNum = 0;
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
                    try {
//                        for (int i = 0; i < prevPts.size(); i++) {
//                            Point pt = prevPts.get(i);
//                            System.out.println(pt);
//                            float[] floats = new float[]{(float) pt.x, (float) pt.y};
                            prevPtsMat.fromList(prevPts);
//                    }
                    } catch (Exception e){
                        System.out.println(e.toString() + "line 463");
                    }

                    MatOfPoint2f nextPts = new MatOfPoint2f();
                    try {
                        System.out.println("npoints= " + prevPtsMat.size() + ", prevPts.size()= " + prevPts.size());
                        // Calculate optical flow using Lucas-Kanade method
                        Video.calcOpticalFlowPyrLK(
                                MainActivity.prevGray, frameGray, prevPtsMat,
                                nextPts, new MatOfByte(), new MatOfFloat(), MainActivity.winSize,
                                5, MainActivity.criteria);
                    }catch (Exception e){
                        System.out.println(e.toString() + "line 476");
                    }
                    Point[] p0Arr = prevPtsMat.toArray();
                    Point[] p1Arr = nextPts.toArray();
                    // Draw the tracks
                    //    private MatOfPoint2f nextPts = new MatOfPoint2f();
                    Mat mask = Mat.zeros(frame.size(), CvType.CV_8UC3);

                    try {
                        for (int i = 0; i < prevPts.size(); i++) {
                            Point newPt = p0Arr[i];
                            Point oldPt = p1Arr[i];

                            double a = newPt.x;
                            double b = newPt.y;
                            double c = oldPt.x;
                            double d = oldPt.y;

                            if (b < 200) {
                                continue;
                            }

                            double pixelSpeed = Math.sqrt(Math.pow(a - c, 2) + Math.pow(b - d, 2));

                            // Prepare input data (example)
                            Mat inputMat = Mat.zeros(1, 3, CvType.CV_32F);
                            inputMat.put(0, 0, (float) a, (float) b, (float) pixelSpeed);

                            // Transform input data using standardScaler
                            double[] normalized;
                            normalized = new double[]{(a - MEAN_A) / VAR_A, (b - MEAN_B) / VAR_B, (pixelSpeed - MEAN_P) / VAR_P};
                            // Get output tensor (predicted_speed_function)
                            float predictedSpeed = predictSpeed(normalized);

                            // Draw lines and circles
                            Imgproc.line(mask, newPt, oldPt, new Scalar(255, 0, 0), 2);
                            Imgproc.circle(frameGray, newPt, 5, new Scalar(255, 0, 0), -1);
                            // Draw label text
                            Imgproc.putText(frame,
                                    String.format("%.2f", predictedSpeed),
                                    newPt,
                                    Imgproc.FONT_HERSHEY_SIMPLEX,
                                    1.5,
                                    new Scalar(0, 0, 255),
                                    4);

                            int lane = getLane((int) newPt.x, (int) newPt.y);
                            Imgproc.putText(frame,
                                    String.format("%.2f", predictedSpeed),
                                    new Point(lane * 400, 35),
                                    Imgproc.FONT_HERSHEY_SIMPLEX,
                                    1.5,
                                    new Scalar(0, 0, 255),
                                    4);

                        }
                    } catch (Exception e){
                        System.out.println(e.toString() + "line 533");
                    }

                    // Add mask to frame
                    Core.add(frame, mask, frame);
                }
            }catch (Exception e){
                System.out.println(e.toString() + "line 540");
            }
/*
            Mat frameGray = new Mat();
            Imgproc.cvtColor(frame, frameGray, Imgproc.COLOR_BGR2GRAY);

            if (MainActivity.frameNum > 2 && MainActivity.prevPts != null) {
                MatOfPoint2f prevPtsMat = new MatOfPoint2f();
                prevPtsMat.fromList(MainActivity.prevPts);
                MatOfPoint2f nextPts = new MatOfPoint2f();
                MatOfByte status = new MatOfByte();
                MatOfFloat err = new MatOfFloat();
                Video.calcOpticalFlowPyrLK(
                        MainActivity.prevGray, frameGray, prevPtsMat,
                        nextPts, status, err, MainActivity.winSize, 5, MainActivity.criteria
                );

                byte[] statusArr = status.toArray();
                float[] errArr = err.toArray();
                List<Point> nextPtsList = nextPts.toList();
                Mat mask = Mat.zeros(frame.size(), CvType.CV_8UC3);
                for (int i = 0; i < nextPtsList.size(); i++) {
                    if (statusArr[i] == 1 && nextPtsList.get(i).y < 200) {
                        Point newPt = nextPtsList.get(i);
                        Point oldPt = prevPts.get(i);
                        double pixelSpeed = Math.sqrt(Math.pow(newPt.x - oldPt.x, 2) + Math.pow(newPt.y - oldPt.y, 2));

                        Mat inputMat = new Mat(1, 3, CvType.CV_32FC1);
                        inputMat.put(0, 0, (float) newPt.x, (float) newPt.y, (float) pixelSpeed);
                        Mat outputMat = new Mat();
                        si.run(inputMat, outputMat);

                        float predictedSpeed = (float) outputMat.get(0, 0)[0];

                        Imgproc.line(mask, newPt, oldPt, new Scalar(255, 0, 0), 2);
                        Imgproc.circle(frame, newPt, 5, new Scalar(255, 0, 0), -1);
                        Imgproc.putText(frame, String.format("%.2f", predictedSpeed), newPt, Imgproc.FONT_HERSHEY_SIMPLEX, 1.5, new Scalar(0, 0, 255), 4);

                        int lane = getLane((int) newPt.x, (int) newPt.y);
                        Imgproc.putText(frame, String.format("%.2f", predictedSpeed), new Point(getLane((int) newPt.x, (int) newPt.y) * 400, 35), Imgproc.FONT_HERSHEY_SIMPLEX, 1.5, new Scalar(0, 0, 255), 4);

                        Core.add(frame, mask, frame);

                        if (vehicles.containsKey(frameNum)) {
                            Object[] vehicleData = vehicles.get(frameNum);
                            int[] region = (int[]) vehicleData[0];
                            float realSpeed = (float) vehicleData[1];
                            int lane = (int) vehicleData[2];
                            int carId = getId(region[0], region[1]);
                            if (carId == getId((int) newPt.x, (int) newPt.y)) {
                                if (verbose != 0) {
                                    System.out.print("frame_num= " + frameNum + ", ");
                                    System.out.println("speeds= " + realSpeed + ", " + pixelSpeed);
                                }
                            }
                        } else if (pixelSpeed > minPixelSpeed && getLane((int) newPt.x, (int) newPt.y) < 4) {
                            if (verbose != 0) {
                                System.out.println("frame_num= " + frameNum + ", lane= " + getLane((int) newPt.x, (int) newPt.y) + ", lane_speeds= " + laneSpeeds.get(getLane((int) newPt.x, (int) newPt.y)) + ", pixel_speed= " + pixelSpeed);
                            }
                        }
                    }
                }
            }

*/

            MainActivity.prevPts = new ArrayList<>();
            for (List<Float> sumBoxCnt : plates.values()) {
                System.out.println("Calculating mean of boxes1");
                float sumX = sumBoxCnt.get(0);
                float sumY = sumBoxCnt.get(1);
                float sumW = sumBoxCnt.get(2);
                float sumH = sumBoxCnt.get(3);
                float cnt = sumBoxCnt.get(4);
                System.out.println("Calculating mean of boxes2");

                float xCenter = sumX / cnt;
                float yCenter = sumY / cnt;
                float w = sumW / cnt;
                float h = sumH / cnt;
                System.out.println("Calculating mean of boxes3");

                MainActivity.prevPts.add(new Point((int) (xCenter * imW), (int) (yCenter * imH)));
                System.out.println("Calculating mean of boxes4");

                int x1 = (int) ((xCenter - w / 2) * imW);
                int y1 = (int) ((yCenter - h / 2) * imH);
                int x2 = (int) ((xCenter + w / 2) * imW);
                int y2 = (int) ((yCenter + h / 2) * imH);
                System.out.println("Calculating mean of boxes5");

                Imgproc.rectangle(frame, new Point(x1, y1), new Point(x2, y2), new Scalar(10, 255, 0), 2);
                System.out.println("Calculating mean of boxes6");
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
    private static float predictSpeed(double[] inputData) {
//        try {
//            SpeedPredictionModel model = SpeedPredictionModel.newInstance(MainActivity.this);
//
//            // Creates inputs for reference.
//            TensorBuffer inputFeature0 = TensorBuffer.createFixedSize(new int[]{1, 3}, DataType.FLOAT32);
        speedInputFeature.loadBuffer(doubleToByteBuffer(inputData));

        // Runs model inference and gets result.
        SpeedPredictionModel.Outputs outputs = speedPredictionModel.process(speedInputFeature);
        TensorBuffer outputFeature0 = outputs.getOutputFeature0AsTensorBuffer();

//            // Releases model resources if no longer used.
//            speedPredictionModel.close();
//        } catch (IOException e) {
//            // TODO Handle the exception
//        }
        System.out.println("out.shape = " + Arrays.toString(outputFeature0.getShape()));
        System.out.println(outputFeature0.getFloatValue(0));
        return outputFeature0.getFloatValue(0); // TODO
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


//            // Calculate the number of bytes needed to store the Mat data
//            int dataSize = mat.rows() * mat.cols() * mat.channels() * 4;
//            System.out.println("dataSize=" + dataSize);
//
//            // Get the Mat data as a byte array
//            byte[] byteArray = new byte[dataSize];
//            mat.get(0, 0, byteArray);
//            System.out.println("got");
//
//            return ByteBuffer.wrap(byteArray);



//        float[] floatArray = mat.toArray();
//
//        // Create a ByteBuffer with the appropriate size
//        int dataSize = floatArray.length * Float.BYTES;
//        ByteBuffer byteBuffer = ByteBuffer.allocate(dataSize);
//        byteBuffer.order(ByteOrder.nativeOrder()); // Set byte order to native order (little-endian or big-endian)
//
//        // Put each floating-point value into the ByteBuffer as bytes
//        for (float floatValue : floatArray) {
//            byteBuffer.putFloat(floatValue);
//        }
//
//        // Reset position to beginning of the ByteBuffer
//        byteBuffer.rewind();
//
//        return byteBuffer;


//        // Convert Mat to MatOfByte
//        MatOfByte matOfByte = new MatOfByte();
//        org.opencv.imgcodecs.Imgcodecs.imencode(".jpg", mat, matOfByte);
//
//        // Convert MatOfByte to ByteBuffer
//        byte[] byteArray = matOfByte.toArray();
//        ByteBuffer byteBuffer = ByteBuffer.wrap(byteArray);
//        System.out.println("size of byteArray = " + byteArray.length);
//        return byteBuffer;
        }

        public static Map<Integer, List<Float>> predictPlates(Mat frame) {
//        try {
//            LicensePlateDetectorFloat32 model = LicensePlateDetectorFloat32.newInstance(MainActivity.this);
//
//            // Creates inputs for reference.
//            TensorBuffer inputFeature0 = TensorBuffer.createFixedSize(new int[]{1, 640, 640, 3}, DataType.FLOAT32);
            try {
                plateInputFeature.loadBuffer(matToByteBuffer(frame));
            } catch (Exception e) {
                System.out.println(e.toString());
            }

            // Runs model inference and gets result.
            LicensePlateDetectorFloat32.Outputs outputs = plateDetectorModel.process(plateInputFeature);
            TensorBuffer outputFeature0 = outputs.getOutputFeature0AsTensorBuffer();

            return computeBoxesAndScores(outputFeature0);

//            // Releases model resources if no longer used.
//            plateDetectorModel.close(); // TODO: change it's place
//        } catch (IOException e) {
//            // TODO Handle the exception
//        }
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

                if ((yCenter > 250f / imW) && (w < 100f / imH) && (h < 100f / imW)) {
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
//                System.out.println("###########" +floats[i] + ", " + floats[i+N] + ", " + floats[i+2*N] + ", " + floats[i+3*N] + ", " + floats[i+4*N]);
        }
        System.out.println("all plates of this frame are detected!");
        return plates;

    }


}


