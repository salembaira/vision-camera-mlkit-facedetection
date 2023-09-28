// FaceDetectionModule.java

package com.rnmlkit.facedetection;

import android.graphics.PointF;
import android.graphics.Rect;
import android.net.Uri;
import android.graphics.BitmapFactory;
import android.graphics.Bitmap;
import android.media.Image;

import java.util.concurrent.CountDownLatch;

import androidx.annotation.NonNull;
import androidx.annotation.Nullable;

import java.util.Map;

import com.facebook.react.bridge.Arguments;
import com.facebook.react.bridge.ReactApplicationContext;
import com.facebook.react.bridge.ReactContextBaseJavaModule;
import com.facebook.react.bridge.ReactMethod;
import com.facebook.react.bridge.Promise;
import com.facebook.react.bridge.ReadableMap;
import com.facebook.react.bridge.WritableArray;
import com.facebook.react.bridge.WritableMap;
import com.google.android.gms.tasks.OnFailureListener;
import com.google.android.gms.tasks.OnSuccessListener;
import com.google.mlkit.vision.common.InputImage;
import com.google.mlkit.vision.face.Face;
import com.google.mlkit.vision.face.FaceContour;
import com.google.mlkit.vision.face.FaceDetection;
import com.google.mlkit.vision.face.FaceDetector;
import com.google.mlkit.vision.face.FaceDetectorOptions;
import com.google.mlkit.vision.face.FaceLandmark;

import com.mrousavy.camera.frameprocessor.Frame;
import com.mrousavy.camera.frameprocessor.FrameProcessorPlugin;

import java.io.IOException;
import java.util.List;
import java.net.URL;

public class FaceDetectionModule extends FrameProcessorPlugin {

    private WritableArray detectionResult;  // Store the detection result
    private Exception detectionError;  // Store the detection error if any

    @Nullable
    @Override
    public Object callback(@NonNull Frame frame, @Nullable Map<String, Object> arguments) {
        Image mediaImage = frame.getImage();
        InputImage image = InputImage.fromMediaImage(mediaImage, frame.getImageInfo().getRotationDegrees());
        try {
            return detect(image, arguments);
        } catch (Exception e) {
            // Handle error and possibly return some error indication
            return "error detecting faces";
        }
    }

    private FaceDetectorOptions getOptions(ReadableMap map) {
        FaceDetectorOptions.Builder builder = new FaceDetectorOptions.Builder();
        String performanceMode = map.getString("performanceMode");
        if (performanceMode != null && performanceMode.equals("accurate")) {
            builder.setPerformanceMode(FaceDetectorOptions.PERFORMANCE_MODE_ACCURATE);
        }
        String landmarkMode = map.getString("landmarkMode");
        if (landmarkMode != null && landmarkMode.equals("all")) {
            builder.setLandmarkMode(FaceDetectorOptions.LANDMARK_MODE_ALL);
        }
        String contourMode = map.getString("contourMode");
        if (contourMode != null && contourMode.equals("all")) {
            builder.setContourMode(FaceDetectorOptions.CONTOUR_MODE_ALL);
        }
        String classificationMode = map.getString("classificationMode");
        if (classificationMode != null && classificationMode.equals("all")) {
            builder.setClassificationMode(FaceDetectorOptions.CLASSIFICATION_MODE_ALL);
        }
        double minFaceSize = map.hasKey("minFaceSize") ? map.getDouble("minFaceSize") : -1;
        if (minFaceSize > 0 && minFaceSize <= 1) {
            builder.setMinFaceSize((float) minFaceSize);
        }
        boolean trackingEnabled = map.hasKey("trackingEnabled") && map.getBoolean("trackingEnabled");
        if (trackingEnabled) {
            builder.enableTracking();
        }
        return builder.build();
    }

    private ReadableMap rectToMap(Rect rect) {
        WritableMap map = Arguments.createMap();
        map.putInt("width", rect.width());
        map.putInt("height", rect.height());
        map.putInt("top", rect.top);
        map.putInt("left", rect.left);
        return map;
    }

    private ReadableMap pointToMap(PointF point) {
        WritableMap map = Arguments.createMap();
        map.putDouble("x", point.x);
        map.putDouble("y", point.y);
        return map;
    }

    private ReadableMap contourToMap(FaceContour contour) {
        WritableArray points = Arguments.createArray();
        for (PointF point : contour.getPoints()) {
            points.pushMap(pointToMap(point));
        }
        WritableMap map = Arguments.createMap();
        map.putArray("points", points);
        return map;
    }

    private ReadableMap landmarkToMap(FaceLandmark landmark) {
        WritableMap map = Arguments.createMap();
        map.putMap("position", pointToMap(landmark.getPosition()));
        return map;
    }

    private ReadableMap faceToMap(Face face, ReadableMap options) {
        WritableMap map = Arguments.createMap();
        map.putMap("frame", rectToMap(face.getBoundingBox()));
        map.putDouble("rotationX", face.getHeadEulerAngleX());
        map.putDouble("rotationY", face.getHeadEulerAngleY());
        map.putDouble("rotationZ", face.getHeadEulerAngleZ());

        // If landmark detection was enabled (mouth, ears, eyes, cheeks, and nose):
        String landmarkMode = options.getString("landmarkMode");
        if (landmarkMode != null && landmarkMode.equals("all")) {
            WritableMap landmarks = Arguments.createMap();

            FaceLandmark leftEar = face.getLandmark(FaceLandmark.LEFT_EAR);
            if (leftEar != null) {
                landmarks.putMap("leftEar", landmarkToMap(leftEar));
            }

            FaceLandmark rightEar = face.getLandmark(FaceLandmark.RIGHT_EAR);
            if (rightEar != null) {
                landmarks.putMap("rightEar", landmarkToMap(rightEar));
            }

            FaceLandmark leftEye = face.getLandmark(FaceLandmark.LEFT_EYE);
            if (leftEye != null) {
                landmarks.putMap("leftEye", landmarkToMap(leftEye));
            }

            FaceLandmark rightEye = face.getLandmark(FaceLandmark.RIGHT_EYE);
            if (rightEye != null) {
                landmarks.putMap("rightEye", landmarkToMap(rightEye));
            }

            FaceLandmark noseBase = face.getLandmark(FaceLandmark.NOSE_BASE);
            if (noseBase != null) {
                landmarks.putMap("noseBase", landmarkToMap(noseBase));
            }

            FaceLandmark leftCheek = face.getLandmark(FaceLandmark.LEFT_CHEEK);
            if (leftCheek != null) {
                landmarks.putMap("leftCheek", landmarkToMap(leftCheek));
            }

            FaceLandmark rightCheek = face.getLandmark(FaceLandmark.RIGHT_CHEEK);
            if (rightCheek != null) {
                landmarks.putMap("rightCheek", landmarkToMap(rightCheek));
            }

            FaceLandmark mouthLeft = face.getLandmark(FaceLandmark.MOUTH_LEFT);
            if (mouthLeft != null) {
                landmarks.putMap("mouthLeft", landmarkToMap(mouthLeft));
            }

            FaceLandmark mouthRight = face.getLandmark(FaceLandmark.MOUTH_RIGHT);
            if (mouthRight != null) {
                landmarks.putMap("mouthRight", landmarkToMap(mouthRight));
            }

            FaceLandmark mouthBottom = face.getLandmark(FaceLandmark.MOUTH_BOTTOM);
            if (mouthBottom != null) {
                landmarks.putMap("mouthBottom", landmarkToMap(mouthBottom));
            }

            map.putMap("landmarks", landmarks);
        }

        // If contour detection was enabled:
        String contourMode = options.getString("contourMode");
        if (contourMode != null && contourMode.equals("all")) {
            WritableMap contours = Arguments.createMap();
            FaceContour faceContour = face.getContour(FaceContour.FACE);
            if (faceContour != null) {
                contours.putMap("face", contourToMap(faceContour));
            }

            FaceContour leftEye = face.getContour(FaceContour.LEFT_EYE);
            if (leftEye != null) {
                contours.putMap("leftEye", contourToMap(leftEye));
            }

            FaceContour rightEye = face.getContour(FaceContour.RIGHT_EYE);
            if (rightEye != null) {
                contours.putMap("rightEye", contourToMap(rightEye));
            }

            FaceContour leftCheek = face.getContour(FaceContour.LEFT_CHEEK);
            if (leftCheek != null) {
                contours.putMap("leftCheek", contourToMap(leftCheek));
            }

            FaceContour rightCheek = face.getContour(FaceContour.RIGHT_CHEEK);
            if (rightCheek != null) {
                contours.putMap("rightCheek", contourToMap(rightCheek));
            }

            FaceContour noseBottom = face.getContour(FaceContour.NOSE_BOTTOM);
            if (noseBottom != null) {
                contours.putMap("noseBottom", contourToMap(noseBottom));
            }

            FaceContour noseBridge = face.getContour(FaceContour.NOSE_BRIDGE);
            if (noseBridge != null) {
                contours.putMap("noseBridge", contourToMap(noseBridge));
            }

            FaceContour leftEyebrowTop = face.getContour(FaceContour.LEFT_EYEBROW_TOP);
            if (leftEyebrowTop != null) {
                contours.putMap("leftEyebrowTop", contourToMap(leftEyebrowTop));
            }

            FaceContour rightEyebrowTop = face.getContour(FaceContour.RIGHT_EYEBROW_TOP);
            if (rightEyebrowTop != null) {
                contours.putMap("rightEyebrowTop", contourToMap(rightEyebrowTop));
            }

            FaceContour leftEyebrowBottom = face.getContour(FaceContour.LEFT_EYEBROW_BOTTOM);
            if (leftEyebrowBottom != null) {
                contours.putMap("leftEyebrowBottom", contourToMap(leftEyebrowBottom));
            }

            FaceContour rightEyebrowBottom = face.getContour(FaceContour.RIGHT_EYEBROW_BOTTOM);
            if (rightEyebrowBottom != null) {
                contours.putMap("rightEyebrowBottom", contourToMap(rightEyebrowBottom));
            }

            FaceContour upperLipTop = face.getContour(FaceContour.UPPER_LIP_TOP);
            if (upperLipTop != null) {
                contours.putMap("upperLipTop", contourToMap(upperLipTop));
            }

            FaceContour lowerLipTop = face.getContour(FaceContour.LOWER_LIP_TOP);
            if (lowerLipTop != null) {
                contours.putMap("lowerLipTop", contourToMap(lowerLipTop));
            }

            FaceContour upperLipBottom = face.getContour(FaceContour.UPPER_LIP_BOTTOM);
            if (upperLipBottom != null) {
                contours.putMap("upperLipBottom", contourToMap(upperLipBottom));
            }

            FaceContour lowerLipBottom = face.getContour(FaceContour.LOWER_LIP_BOTTOM);
            if (lowerLipBottom != null) {
                contours.putMap("lowerLipBottom", contourToMap(lowerLipBottom));
            }

            map.putMap("contours", contours);
        }

        // If classification was enabled:
        if (face.getSmilingProbability() != null) {
            map.putDouble("smilingProbability", face.getSmilingProbability());
        }

        if (face.getLeftEyeOpenProbability() != null) {
            map.putDouble("leftEyeOpenProbability", face.getLeftEyeOpenProbability());
        }

        if (face.getRightEyeOpenProbability() != null) {
            map.putDouble("rightEyeOpenProbability", face.getRightEyeOpenProbability());
        }

        // If face tracking was enabled:
        if (face.getTrackingId() != null) {
            map.putInt("trackingID", face.getTrackingId());
        }
        return map;
    }

    private ReadableArray detect(InputImage image, ReadableMap optionsMap) throws Exception {
        final CountDownLatch latch = new CountDownLatch(1);  // Used to wait for the async operation to complete
        detectionResult = null;  // Reset the detection result
        detectionError = null;  // Reset the detection error

        FaceDetectorOptions options = getOptions(optionsMap);
        FaceDetector detector = FaceDetection.getClient(options);

        detector.process(image)
                .addOnSuccessListener(faces -> {
                    detectionResult = Arguments.createArray();
                    for (Face face : faces) {
                        detectionResult.pushMap(faceToMap(face, optionsMap));
                    }
                    latch.countDown();  // Release the latch
                })
                .addOnFailureListener(e -> {
                    detectionError = e;  // Store the error
                    latch.countDown();  // Release the latch
                });

        latch.await();  // Wait for the async operation to complete

        if (detectionError != null) {
            throw detectionError;  // Throw the error if any
        }

        return detectionResult;  // Return the result
    }
}
