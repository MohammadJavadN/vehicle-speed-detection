/** Copyright 2019 The TensorFlow Authors. All Rights Reserved.

 Licensed under the Apache License, Version 2.0 (the "License");
 you may not use this file except in compliance with the License.
 You may obtain a copy of the License at

 http://www.apache.org/licenses/LICENSE-2.0

 Unless required by applicable law or agreed to in writing, software
 distributed under the License is distributed on an "AS IS" BASIS,
 WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 See the License for the specific language governing permissions and
 limitations under the License.
 ==============================================================================*/

package com.example.myapplication;

import android.graphics.Bitmap;
import android.graphics.Matrix;
import android.graphics.Rect;
import android.graphics.RectF;
import android.util.Size;

import org.opencv.android.Utils;
import org.opencv.core.Mat;

import java.util.Arrays;
import java.util.HashSet;
import java.util.List;
import java.util.Set;

public class ImageUtils {

    static final int kMaxChannelValue = 262143;

    public static Bitmap matToBitmap(Mat inputMat) {
        System.out.println("%%%" + inputMat.cols() + inputMat.rows());
        Bitmap outputBitmap = Bitmap.createBitmap(inputMat.cols(), inputMat.rows(), Bitmap.Config.ARGB_8888);
        Utils.matToBitmap(inputMat, outputBitmap);
        return outputBitmap;
    }
    public static int getYUVByteSize(Size size) {
        final int ySize = size.getWidth() * size.getHeight();
        final int uvSize = ((size.getWidth() + 1) / 2) * ((size.getHeight() + 1) / 2) * 2;

        return ySize + uvSize;
    }

    public static Matrix getTransformationMatrix(
            final int srcWidth,
            final int srcHeight,
            final int dstWidth,
            final int dstHeight,
            final int applyRotation,
            final boolean maintainAspectRatio) {
        final Matrix matrix = new Matrix();

        if (applyRotation != 0) {

            // Translate so center of image is at origin.
            matrix.postTranslate(-srcWidth / 2.0f, -srcHeight / 2.0f);

            // Rotate around origin.
            matrix.postRotate(applyRotation);
        }

        // Account for the already applied rotation, if any, and then determine how
        // much scaling is needed for each axis.
        final boolean transpose = (Math.abs(applyRotation) + 90) % 180 == 0;

        final int inWidth = transpose ? srcHeight : srcWidth;
        final int inHeight = transpose ? srcWidth : srcHeight;

        // Apply scaling if necessary.
        if (inWidth != dstWidth || inHeight != dstHeight) {
            final float scaleFactorX = dstWidth / (float) inWidth;
            final float scaleFactorY = dstHeight / (float) inHeight;

            if (maintainAspectRatio) {
                // Scale by minimum factor so that dst is filled completely while
                // maintaining the aspect ratio. Some image may fall off the edge.
                final float scaleFactor = Math.max(scaleFactorX, scaleFactorY);
                matrix.postScale(scaleFactor, scaleFactor);
            } else {
                // Scale exactly to fill dst from src.
                matrix.postScale(scaleFactorX, scaleFactorY);
            }
        }

        if (applyRotation != 0) {
            // Translate back from origin centered reference to destination frame.
            matrix.postTranslate(dstWidth / 2.0f, dstHeight / 2.0f);
        }

        return matrix;
    }

    public static void convertYUV420SPToARGB8888(byte[] input, int width, int height, int[] output) {
        final int frameSize = width * height;
        for (int j = 0, yp = 0; j < height; j++) {
            int uvp = frameSize + (j >> 1) * width;
            int u = 0;
            int v = 0;

            for (int i = 0; i < width; i++, yp++) {
                int y = 0xff & input[yp];
                if ((i & 1) == 0) {
                    v = 0xff & input[uvp++];
                    u = 0xff & input[uvp++];
                }

                output[yp] = YUV2RGB(y, u, v);
            }
        }
    }

    private static int YUV2RGB(int y, int u, int v) {
        // Adjust and check YUV values
        y = (y - 16) < 0 ? 0 : (y - 16);
        u -= 128;
        v -= 128;

        // This is the floating point equivalent. We do the conversion in integer
        // because some Android devices do not have floating point in hardware.
        // nR = (int)(1.164 * nY + 2.018 * nU);
        // nG = (int)(1.164 * nY - 0.813 * nV - 0.391 * nU);
        // nB = (int)(1.164 * nY + 1.596 * nV);
        int y1192 = 1192 * y;
        int r = (y1192 + 1634 * v);
        int g = (y1192 - 833 * v - 400 * u);
        int b = (y1192 + 2066 * u);

        // Clipping RGB values to be inside boundaries [ 0 , kMaxChannelValue ]
        r = r > kMaxChannelValue ? kMaxChannelValue : (r < 0 ? 0 : r);
        g = g > kMaxChannelValue ? kMaxChannelValue : (g < 0 ? 0 : g);
        b = b > kMaxChannelValue ? kMaxChannelValue : (b < 0 ? 0 : b);

        return 0xff000000 | ((r << 6) & 0xff0000) | ((g >> 2) & 0xff00) | ((b >> 10) & 0xff);
    }

    /**
     *
     * @param bitmap original image
     * @param recognition detected box
     * @return a new Bitmap of the detected area
     */
    protected static Bitmap cropImg(Bitmap bitmap, Recognition recognition) {
        return Bitmap.createBitmap(bitmap, recognition.getLocationInt().left,
                recognition.getLocationInt().top,
                recognition.getLocationInt().right - recognition.getLocationInt().left,
                recognition.getLocationInt().bottom - recognition.getLocationInt().top);
    }

    /**
     *
     * @param classes an array of the class labels you need
     * @param recognitionList list of all detected objects
     * @return the useless detected objects are removed from the recognition list
     */
    protected static List<Recognition> filteroutRecognitions(String[] classes, List<Recognition> recognitionList){
        Set<String> classesSet = new HashSet<String>(Arrays.asList(classes));
        for (int i = recognitionList.size()-1; i >0 ; i--) {
            if(!classesSet.contains(recognitionList.get(i).getLabel())) recognitionList.remove(i);
        }
        return recognitionList;
    }

    /**
     * @param imgHeight       height of the input image
     * @param imgWidth        width of the input image
     * @param recognitionList list of detected objects
     */
    protected static void mergeRecognitions(int imgHeight, int imgWidth, List<Recognition> recognitionList) {
        boolean flag = false;
        boolean ifModify;
        Rect rect1, rect2;
        RectF rectF1, rectF2;
        System.out.println("recognitionList.size(): " + recognitionList.size());
        while (!flag) {
            ifModify = false;
            for (int i = recognitionList.size() - 1; i > 0; i--) {
                rect1 = recognitionList.get(i).getLocationInt();
                rectF1 = recognitionList.get(i).getLocation();
                for (int j = i - 1; j > 0; j--) {
                    rect2 = recognitionList.get(j).getLocationInt();
                    rectF2 = recognitionList.get(j).getLocation();

                    // if IOU > 0.3, merge the two rects
                    if (calculateIOU2(rect1, rect2) > 0.3) {
                        recognitionList.get(i).setLocationInt(mergeTwoRects(rect1, rect2));
                        recognitionList.remove(j);
                        ifModify = true;
                        break;
                    }

                    // if the rects are close to each other, merge them
//                    if (isNear(imgHeight, imgWidth, rect1, rect2)) {
                    if (isNear(rectF1, rectF2)) {
                        recognitionList.get(i).setLocationInt(mergeTwoRects(rect1, rect2));
                        recognitionList.remove(j);
                        ifModify = true;
                        break;
                    }
//                    if (rectF2.centerX() < rectF1.right
//                            && rectF2.centerX() > rectF1.left
//                            && rectF2.centerY() < rectF1.bottom
//                            && rectF2.centerY() > rectF1.top
//                            && rectF2.width() < rectF1.width()) {
////                        recognitionList.get(i).setLocationInt(mergeTwoRects(rect1, rect2));
//                        recognitionList.remove(j);
//                        ifModify = true;
//                        break;
//                    }
                    if (rectF2.width() > 0.5 || rectF2.height() > 0.4 || rectF2.width() < 0.1 || rectF2.height() < 0.1) {
//                        recognitionList.get(i).setLocationInt(mergeTwoRects(rect1, rect2));
                        recognitionList.remove(j);
                        ifModify = true;
                        break;
                    }
                }
                if (ifModify) break;
            }
            if (!ifModify) flag = true;
        }
        System.out.println("recognitionList.size(): " + recognitionList.size());

        for (Recognition recognition : recognitionList
        ) {
            recognition.setLabel("merged");
            recognition.setProb(1.0f);
        }

    }

    private static Rect mergeTwoRects(Rect rect1, Rect rect2) {
        Rect mergedRect = new Rect();
        mergedRect.left = Math.min(rect1.left, rect2.left);
        mergedRect.right = Math.max(rect1.right, rect2.right);
        mergedRect.top = Math.min(rect1.top, rect2.top);
        mergedRect.bottom = Math.max(rect1.bottom, rect2.bottom);
        return mergedRect;

    }

    /**
     * Intersection over Union = area of overlap / area of union
     */
    private static float calculateIOU(Rect rect1, Rect rect2) {
        float IOU;
        int width1 = rect1.right - rect1.left, width2 = rect2.right - rect2.left;
        int height1 = rect1.bottom - rect1.top, height2 = rect2.bottom - rect2.top;
        int overlapWidth, overlapHeight;
        int overlapArea, totalArea;

        if (rect1.left > rect2.right || rect2.left > rect1.right || rect1.top > rect2.bottom || rect2.top > rect1.bottom)
            return 0;
        else {
            overlapWidth = Math.abs(Math.min(rect1.right, rect2.right) - Math.max(rect1.left, rect2.left));
            overlapHeight = Math.abs(Math.min(rect1.bottom, rect2.bottom) - Math.max(rect1.top, rect2.top));
            overlapArea = overlapHeight * overlapWidth;
            totalArea = width1 * height1 + width2 * height2;
            IOU = overlapArea / (float) (totalArea - overlapArea);
            return IOU;
        }
    }

    private static float calculateIOU2(Rect rect1, Rect rect2) {
        float IOU;
        int width1 = rect1.width(), width2 = rect2.width();
        int height1 = rect1.height(), height2 = rect2.height();
        int overlapWidth, overlapHeight;
        int overlapArea, minArea;

        if (rect1.left > rect2.right || rect2.left > rect1.right || rect1.top > rect2.bottom || rect2.top > rect1.bottom)
            return 0;
        else {
            overlapWidth = Math.abs(Math.min(rect1.right, rect2.right) - Math.max(rect1.left, rect2.left));
            overlapHeight = Math.abs(Math.min(rect1.bottom, rect2.bottom) - Math.max(rect1.top, rect2.top));
            overlapArea = overlapHeight * overlapWidth;
            minArea = Math.min(width1 * height1, width2 * height2);
            IOU = overlapArea / (float) (minArea);
            return IOU;
        }
    }
    /**
     * determine whether the distance between two rect is too close
     */
    static boolean isNear(int imgHeight, int imgWidth, Rect rect1, Rect rect2) {
        boolean xNear, yNear;
        final float THRESHOLD = 0.2f;
        // Only if there is a small one among the two rects, we need to determine if they are close
        if (isSmall(imgHeight, imgWidth, rect1) || isSmall(imgHeight, imgWidth, rect2)) {
            // xNear = (the distance in the horizontal direction < set distance threshold)
            xNear = Math.abs((rect1.right + rect1.left) / 2.0 - (rect2.right + rect2.left) / 2.0)
                    < THRESHOLD * imgWidth + (rect1.right - rect1.left) / 2.0 + (rect2.right - rect2.left) / 2.0;
            // yNear = (the distance in the vertical direction < set distance threshold)
            yNear = Math.abs((rect1.bottom + rect1.top) / 2.0 - (rect2.bottom + rect2.top) / 2.0)
                    < THRESHOLD * imgHeight + (rect1.bottom - rect1.top) / 2.0 + (rect2.bottom - rect2.top) / 2.0;
            return xNear && yNear;
        }
        // If they are all relative big rects, we don't want them merge, so just return false
        return false;
    }

    static boolean isNear(RectF rect1, RectF rect2) {
        boolean xNear, yNear;
        final float THRESHOLD = 0.1f;
        // Only if there is a small one among the two rects, we need to determine if they are close
        if (isSmall(rect1) || isSmall(rect2)) {
//            // xNear = (the distance in the horizontal direction < set distance threshold)
//            xNear = Math.abs((rect1.right + rect1.left) / 2.0 - (rect2.right + rect2.left) / 2.0)
//                    < THRESHOLD + (rect1.right - rect1.left) / 2.0 + (rect2.right - rect2.left) / 2.0;
//            // yNear = (the distance in the vertical direction < set distance threshold)
//            yNear = Math.abs((rect1.bottom + rect1.top) / 2.0 - (rect2.bottom + rect2.top) / 2.0)
//                    < THRESHOLD + (rect1.bottom - rect1.top) / 2.0 + (rect2.bottom - rect2.top) / 2.0;
//            return xNear && yNear;
            xNear = Math.abs(rect1.centerX() - rect2.centerX()) < THRESHOLD + rect1.width()/3 + rect2.width()/3;
            yNear = Math.abs(rect1.centerY() - rect2.centerY()) < THRESHOLD + rect1.height()/3 + rect2.height()/3;
            return xNear && yNear;
        } else {
//        double r1 = rect1.width()/rect2.width();
//        double r2 = rect1.height()/rect2.height();
//        boolean sizeNear = r1 > 0.8 && r1 < 1.2 && r2 > 0.8 && r2 < 1.2;
            return Math.abs(rect1.centerX() - rect2.centerX()) / Math.min(rect1.width(), rect2.width()) + Math.abs(rect1.centerY() - rect2.centerY()) / Math.min(rect1.height(), rect2.height()) < 0.5;
        }
        // If they are all relative big rects, we don't want them merge, so just return false
//        return false;
    }
    /**
     * determine whether a rect is small compared with the original image
     */
    private static boolean isSmall(RectF rect) {
        float th = 0.2f;
        return ((rect.bottom - rect.top) < th
                && (rect.right - rect.left) < th);
    }
    private static boolean isSmall(int imgHeight, int imgWidth, Rect rect) {
        float th = 0.2f;
        return ((rect.bottom - rect.top) < th * imgHeight
                && (rect.right - rect.left) < th * imgWidth);
    }

    /**
     * do extension to all rects in a recognitionList
     *
     * @param EXTENSION how many pixels you want to expand outward for each edge
     */
    protected static List<Recognition> extendRecognitions(int imgHeight, int imgWidth, List<Recognition> recognitionList, int EXTENSION) {
        for (Recognition recognition : recognitionList
        ) {
            recognition.setLocationInt(extendBoundary(imgHeight, imgWidth, recognition.getLocationInt(), EXTENSION));
        }
        return recognitionList;
    }


    private static Rect extendBoundary(int imgHeight, int imgWidth, Rect rect, int EXTENSION) {
        rect.left = Math.max(0, rect.left - EXTENSION);
        rect.top = Math.max(0, rect.top - EXTENSION);
        rect.right = Math.min(imgWidth, rect.right + EXTENSION);
        rect.bottom = Math.min(imgHeight, rect.bottom + EXTENSION);
        return rect;
    }

    /**
     * map the bounding box to the original image
     */
    protected static Recognition convertRecognitiontoOriginalImg(Recognition recognition, Recognition recognitionSmall) {
        Rect newLocationInt = new Rect(recognition.getLocationInt().left + recognitionSmall.getLocationInt().left,
                recognition.getLocationInt().top + recognitionSmall.getLocationInt().top,
                recognition.getLocationInt().left + recognitionSmall.getLocationInt().right,
                recognition.getLocationInt().top + recognitionSmall.getLocationInt().bottom);

        recognitionSmall.setImgHeight(recognition.getImgHeight());
        recognitionSmall.setImgWidth(recognition.getImgWidth());
        recognitionSmall.setLocationInt(newLocationInt);
        return recognitionSmall;
    }
}
