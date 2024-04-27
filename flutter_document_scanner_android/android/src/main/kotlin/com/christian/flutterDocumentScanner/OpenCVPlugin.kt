package com.christian.flutterDocumentScanner

import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.graphics.Color
import android.util.Log
import io.flutter.plugin.common.MethodChannel
import org.opencv.core.*
import org.opencv.imgcodecs.Imgcodecs
import org.opencv.imgproc.Imgproc
import java.io.ByteArrayOutputStream


class OpenCVPlugin {
    companion object {
        fun findContourPhoto(
            result: MethodChannel.Result, byteData: ByteArray, minContourArea: Double
        ) {
            try {
                val src = Imgcodecs.imdecode(MatOfByte(*byteData), Imgcodecs.IMREAD_UNCHANGED)

                val documentContour = findBiggestContour(src, minContourArea)

                // TODO: Use for when to use real time transmission
                // Scalar -> RGB(235, 228, 44)
                // Imgproc.drawContours(src, listOf(documentContour), -1, Scalar(44.0, 228.0, 235.0), 10)

                // Instantiating an empty MatOfByte class
                val matOfByte = MatOfByte()

                // Converting the Mat object to MatOfByte
                Imgcodecs.imencode(".jpg", src, matOfByte)
                val byteArray: ByteArray = matOfByte.toArray()


                val points = mutableListOf<Map<String, Any>>()

                if (documentContour != null) {
                    points.add(
                        mapOf(
                            "x" to documentContour.toList()[0].x,
                            "y" to documentContour.toList()[0].y
                        )
                    )
                    points.add(
                        mapOf(
                            "x" to documentContour.toList()[3].x,
                            "y" to documentContour.toList()[3].y
                        )
                    )
                    points.add(
                        mapOf(
                            "x" to documentContour.toList()[2].x,
                            "y" to documentContour.toList()[2].y
                        )
                    )
                    points.add(
                        mapOf(
                            "x" to documentContour.toList()[1].x,
                            "y" to documentContour.toList()[1].y
                        )
                    )
                }

                val resultEnd = mapOf(
                    "height" to src.height(),
                    "width" to src.width(),
                    "points" to points,
                    "image" to byteArray
                )

                result.success(resultEnd)

            } catch (e: java.lang.Exception) {
                result.error("FlutterDocumentScanner-Error", "Android: " + e.message, e)
            }
        }

        private fun findBiggestContour(src: Mat, minContourArea: Double): MatOfPoint? {
            // Converting to RGB from BGR
            val dstColor = Mat()
            Imgproc.cvtColor(src, dstColor, Imgproc.COLOR_BGR2RGB)

            // Converting to gray
            Imgproc.cvtColor(dstColor, dstColor, Imgproc.COLOR_BGR2GRAY)

            // Applying blur and threshold
            val dstBilateral = Mat()
            Imgproc.bilateralFilter(dstColor, dstBilateral, 9, 75.0, 75.0, Core.BORDER_DEFAULT)
            Imgproc.adaptiveThreshold(
                dstBilateral,
                dstBilateral,
                255.0,
                Imgproc.ADAPTIVE_THRESH_GAUSSIAN_C,
                Imgproc.THRESH_BINARY,
                115,
                4.0
            )

            // Median blur replace center pixel by median of pixels under kernel
            val dstBlur = Mat()
            Imgproc.GaussianBlur(dstBilateral, dstBlur, Size(5.0, 5.0), 0.0)
            Imgproc.medianBlur(dstBlur, dstBlur, 11)


            val dstBorder = Mat()
            Core.copyMakeBorder(dstBlur, dstBorder, 5, 5, 5, 5, Core.BORDER_CONSTANT)


            val dstCanny = Mat()
            Imgproc.Canny(dstBorder, dstCanny, 75.0, 200.0)

            // Close gaps between edges (double page clouse => rectangle kernel)
            val dstEnd = Mat()
            Imgproc.morphologyEx(
                dstCanny, dstEnd, Imgproc.MORPH_CLOSE, Mat.ones(intArrayOf(5, 11), CvType.CV_32F)
            )


            // Getting contours
            val contours = mutableListOf<MatOfPoint>()
            val hierarchy = Mat()
            Imgproc.findContours(
                dstEnd, contours, hierarchy, Imgproc.RETR_TREE, Imgproc.CHAIN_APPROX_SIMPLE
            )
            hierarchy.release()

            // Finding the biggest rectangle otherwise return original corners
            val height = dstEnd.height()
            val width = dstEnd.width()
            val maxContourArea = (width - 10) * (height - 10)

            var maxArea = 0.0
            var documentContour = MatOfPoint()

            for (contour in contours) {
                val contour2f = MatOfPoint2f()
                contour.convertTo(contour2f, CvType.CV_32FC2)
                val perimeter = Imgproc.arcLength(contour2f, true)

                val approx2f = MatOfPoint2f()
                Imgproc.approxPolyDP(contour2f, approx2f, 0.03 * perimeter, true)

                // Page has 4 corners and it is convex
                val approx = MatOfPoint()
                approx2f.convertTo(approx, CvType.CV_32S)
                val isContour = Imgproc.isContourConvex(approx)
                val isLessCurrentArea = Imgproc.contourArea(approx) > maxArea
                val isLessMaxArea = maxArea < maxContourArea

                if (approx.total()
                        .toInt() == 4 && isContour && isLessCurrentArea && isLessMaxArea
                ) {
                    maxArea = Imgproc.contourArea(approx)
                    documentContour = approx
                }
            }

            if (Imgproc.contourArea(documentContour) < minContourArea) {
                return null
            }

            return documentContour
        }


        fun adjustingPerspective(
            byteData: ByteArray, points: List<Map<String, Any>>, result: MethodChannel.Result
        ) {
            try {
                val src = Imgcodecs.imdecode(MatOfByte(*byteData), Imgcodecs.IMREAD_UNCHANGED)
                val documentContour = MatOfPoint(
                    Point("${points[0]["x"]}".toDouble(), "${points[0]["y"]}".toDouble()),
                    Point("${points[1]["x"]}".toDouble(), "${points[1]["y"]}".toDouble()),
                    Point("${points[2]["x"]}".toDouble(), "${points[2]["y"]}".toDouble()),
                    Point("${points[3]["x"]}".toDouble(), "${points[3]["y"]}".toDouble())
                )

                val imgWithPerspective = warpPerspective(src, documentContour)

                // Instantiating an empty MatOfByte class
                val matOfByte = MatOfByte()

                // Converting the Mat object to MatOfByte
                Imgcodecs.imencode(".jpg", imgWithPerspective, matOfByte)
                val byteArray: ByteArray = matOfByte.toArray()

                result.success(byteArray)

            } catch (e: java.lang.Exception) {
                result.error("FlutterDocumentScanner-Error", "Android: " + e.message, e)
            }
        }

        private fun warpPerspective(src: Mat, documentContour: MatOfPoint): Mat {
            val srcContour = MatOfPoint2f(
                Point(0.0, 0.0),
                Point((src.width() - 1).toDouble(), 0.0),
                Point((src.width() - 1).toDouble(), (src.height() - 1).toDouble()),
                Point(0.0, (src.height() - 1).toDouble())
            )

            val dstContour = MatOfPoint2f(
                documentContour.toList()[0],
                documentContour.toList()[1],
                documentContour.toList()[2],
                documentContour.toList()[3]
            )
            val warpMat = Imgproc.getPerspectiveTransform(dstContour, srcContour)

            val dstWarPerspective = Mat()
            Imgproc.warpPerspective(src, dstWarPerspective, warpMat, src.size())

            return dstWarPerspective
        }


        fun applyFilter(result: MethodChannel.Result, byteData: ByteArray, filter: Int) {
            try {
                val filterType: FilterType = when (filter) {
                    1 -> FilterType.Natural
                    2 -> FilterType.Gray
                    3 -> FilterType.Eco

                    else -> FilterType.Natural
                }

                val bitmap = BitmapFactory.decodeByteArray(byteData, 0, byteData.size)
                // Calculate average brightness
                var totalBrightness = 0
                for (x in 0 until bitmap.width) {
                    for (y in 0 until bitmap.height) {
                        val pixel = bitmap.getPixel(x, y)
                        val brightness = Color.red(pixel) + Color.green(pixel) + Color.blue(pixel)
                        totalBrightness += brightness
                    }
                }

                val averageBrightness = totalBrightness / (bitmap.width * bitmap.height)

                Log.d("brightness", "$totalBrightness, avg brightness: $averageBrightness")

                val src: Mat = if (averageBrightness in 490..492) {
                    Imgcodecs.imdecode(MatOfByte(*byteData), Imgcodecs.IMREAD_UNCHANGED)
                } else {
                    val adjustedBrightness = adjustBrightness(byteData)
                    Imgcodecs.imdecode(
                        MatOfByte(*adjustedBrightness), Imgcodecs.IMREAD_UNCHANGED
                    )
                }

                var dstEnd = Mat()

                when (filterType) {
                    FilterType.Natural -> dstEnd = src

                    FilterType.Gray -> Imgproc.cvtColor(src, dstEnd, Imgproc.COLOR_BGR2GRAY)

                    FilterType.Eco -> {
                        // convert to grey scale
                        val dstColor = Mat()
                        Imgproc.cvtColor(src, dstColor, Imgproc.COLOR_BGR2GRAY)

                        // blur the image
                        val dstGaussian = Mat()
                        Imgproc.GaussianBlur(dstColor, dstGaussian, Size(5.0, 5.0), 0.0)

                        // remove the minor dotted part to make image clearer
                        val dstCanny = Mat()
                        Imgproc.Canny(dstGaussian, dstCanny, 200.0, 100.0)

                        // make border cleaner and sharpner
                        val dstDilate = Mat()
                        val kernel =
                            Imgproc.getStructuringElement(Imgproc.MORPH_RECT, Size(3.0, 3.0))
                        Imgproc.dilate(dstCanny, dstDilate, kernel, Point(-1.0, -1.0), 1)

                        // set border width
                        val dstErode = Mat()
                        val kernel1 =
                            Imgproc.getStructuringElement(Imgproc.MORPH_RECT, Size(1.0, 1.0))
                        Imgproc.erode(dstDilate, dstErode, kernel1, Point(-1.0, -1.0), 1)

                        val dstThreshold = Mat()
                        Imgproc.adaptiveThreshold(
                            dstErode,
                            dstThreshold,
                            255.0,
                            Imgproc.ADAPTIVE_THRESH_GAUSSIAN_C,
                            Imgproc.THRESH_BINARY,
                            7,
                            2.0
                        )

//                        Imgproc.medianBlur(dstThreshold, dstEnd, 3)
                        val dstMedianBlur = Mat()
                        Imgproc.medianBlur(dstThreshold, dstMedianBlur, 3)


                        val contours: List<MatOfPoint> = ArrayList()
                        val hierarchy = Mat()
                        Imgproc.findContours(
                            dstMedianBlur,
                            contours,
                            hierarchy,
                            Imgproc.RETR_LIST,
                            Imgproc.CHAIN_APPROX_SIMPLE
                        )
                        var biggest = MatOfPoint()
                        val imgCols: Int = src.cols()
                        val imgRows: Int = src.rows()

                        var big: List<MatOfPoint>? =
                            findBiggestSecondBiggestAndThirdBiggestContours(
                                contours,
                                imgCols,
                                imgRows
                            )

                        try {
                            if (big == null) {
                                return
                            }
                        } catch (e: CvException) {
                            Log.e(
                                "OMR",
                                "OpenCV Exception: " + e.message
                            )
                            return
                        }

                        if (big.size == 1) {
                            biggest = big[0]
                        } else if (big.size == 2) {
                            if (Imgproc.contourArea(big[0]) - Imgproc.contourArea(big[1]) >= 2000) {
                                biggest = big[1]
                            } else if (Imgproc.contourArea(big[1]) - Imgproc.contourArea(big[0]) >= 2000) {
                                biggest = big[0]
                            }
                        } else if (big.size == 3) {
                            val firstBig: Double
                            val secondBig: Double
                            val thirdBig: Double
                            val firstIndex: Int
                            val secondIndex: Int
                            val thirdIndex: Int
                            if (Imgproc.contourArea(big[0]) > Imgproc.contourArea(big[1]) && Imgproc.contourArea(
                                    big[0]
                                ) > Imgproc.contourArea(
                                    big[2]
                                )
                            ) {
                                firstBig = Imgproc.contourArea(big[0])
                                firstIndex = 0
                                if (Imgproc.contourArea(big[1]) > Imgproc.contourArea(big[2])) {
                                    secondIndex = 1
                                    thirdIndex = 2
                                    secondBig = Imgproc.contourArea(big[1])
                                    thirdBig = Imgproc.contourArea(big[2])
                                } else {
                                    secondIndex = 2
                                    thirdIndex = 1
                                    secondBig = Imgproc.contourArea(big[2])
                                    thirdBig = Imgproc.contourArea(big[1])
                                }
                            } else if (Imgproc.contourArea(big[1]) > Imgproc.contourArea(big[0]) && Imgproc.contourArea(
                                    big[1]
                                ) > Imgproc.contourArea(big[2])
                            ) {
                                firstBig = Imgproc.contourArea(big[1])
                                firstIndex = 1
                                if (Imgproc.contourArea(big[0]) > Imgproc.contourArea(big[2])) {
                                    secondIndex = 0
                                    thirdIndex = 2
                                    secondBig = Imgproc.contourArea(big[0])
                                    thirdBig = Imgproc.contourArea(big[2])
                                } else {
                                    secondIndex = 2
                                    thirdIndex = 0
                                    secondBig = Imgproc.contourArea(big[2])
                                    thirdBig = Imgproc.contourArea(big[0])
                                }
                            } else {
                                firstBig = Imgproc.contourArea(big[2])
                                firstIndex = 2
                                if (Imgproc.contourArea(big[0]) > Imgproc.contourArea(big[1])) {
                                    secondIndex = 0
                                    thirdIndex = 1
                                    secondBig = Imgproc.contourArea(big[0])
                                    thirdBig = Imgproc.contourArea(big[1])
                                } else {
                                    secondIndex = 1
                                    thirdIndex = 0
                                    secondBig = Imgproc.contourArea(big[1])
                                    thirdBig = Imgproc.contourArea(big[0])
                                }
                            }

                            biggest = if (firstBig - secondBig >= 3000) {
                                big[secondIndex]
                            } else {
                                big[thirdIndex]
                            }
                        }

                        val reorderedPoints: Array<Point?> = reorderPoints(biggest.toArray())
                        val reorderedContour = MatOfPoint()
                        reorderedContour.fromArray(*reorderedPoints)
                        biggest = reorderedContour

                        if (!biggest.empty()) {
                            dstEnd = warpPerspective(
                                src,
                                biggest
                            )
                        }
                    }
                }


                // Instantiating an empty MatOfByte class
                val matOfByte = MatOfByte()

                // Converting the Mat object to MatOfByte
                Imgcodecs.imencode(".jpg", dstEnd, matOfByte)
                val byteArray: ByteArray = matOfByte.toArray()

                result.success(byteArray)

            } catch (e: java.lang.Exception) {
                result.error("FlutterDocumentScanner-Error", "Android: " + e.message, e)
            }
        }

        private fun adjustBrightness(imageData: ByteArray): ByteArray {
            // Decode the byte array into a Bitmap
            val bitmap = BitmapFactory.decodeByteArray(imageData, 0, imageData.size)

            // Calculate average brightness
            var totalBrightness = 0
            for (x in 0 until bitmap.width) {
                for (y in 0 until bitmap.height) {
                    val pixel = bitmap.getPixel(x, y)
                    val brightness = Color.red(pixel) + Color.green(pixel) + Color.blue(pixel)
                    totalBrightness += brightness
                }
            }
            val averageBrightness = totalBrightness / (bitmap.width * bitmap.height)

            // Define brightness thresholds
            val lowThreshold = 490
            val highThreshold = 492

            // Adjust brightness based on thresholds
            val targetBrightness = (lowThreshold + highThreshold) / 2
            val brightnessFactor = targetBrightness.toFloat() / averageBrightness

            // Create a mutable bitmap for adjusting brightness
            val adjustedBitmap = bitmap.copy(Bitmap.Config.ARGB_8888, true)

            // Adjust brightness of each pixel
            for (x in 0 until adjustedBitmap.width) {
                for (y in 0 until adjustedBitmap.height) {
                    val pixel = adjustedBitmap.getPixel(x, y)
                    val newPixel = Color.argb(
                        Color.alpha(pixel),
                        (Color.red(pixel) * brightnessFactor).toInt().coerceIn(0, 255),
                        (Color.green(pixel) * brightnessFactor).toInt().coerceIn(0, 255),
                        (Color.blue(pixel) * brightnessFactor).toInt().coerceIn(0, 255)
                    )
                    adjustedBitmap.setPixel(x, y, newPixel)
                }
            }

            // Convert adjusted bitmap to byte array
            val byteArrayOutputStream = ByteArrayOutputStream()
            adjustedBitmap.compress(Bitmap.CompressFormat.PNG, 100, byteArrayOutputStream)
            return byteArrayOutputStream.toByteArray()
        }

        private fun findBiggestSecondBiggestAndThirdBiggestContours(
            contours: List<MatOfPoint>,
            imgCols: Int,
            imgRows: Int
        ): List<MatOfPoint>? {
            val resultContours: MutableList<MatOfPoint> = ArrayList()
            var maxArea = 0.0
            var secondMaxArea = 0.0
            var thirdMaxArea = 0.0
            var biggestContour = MatOfPoint()
            var secondBiggestContour = MatOfPoint()
            var thirdBiggestContour = MatOfPoint()
            for (contour in contours) {
                val area = Imgproc.contourArea(contour)
                if (area > 5000) {
                    val peri = Imgproc.arcLength(MatOfPoint2f(*contour.toArray()), true)
                    val approx = MatOfPoint2f()
                    Imgproc.approxPolyDP(
                        MatOfPoint2f(*contour.toArray()),
                        approx,
                        0.02 * peri,
                        true
                    )

                    // Check if contour has a reasonable area and is not too large
                    if (area < imgCols * imgRows - 10000) {
                        if (area > maxArea && approx.total() == 4L) {
                            thirdBiggestContour = secondBiggestContour
                            thirdMaxArea = secondMaxArea
                            secondBiggestContour = biggestContour
                            secondMaxArea = maxArea
                            biggestContour = MatOfPoint(*approx.toArray())
                            maxArea = area
                        } else if (area > secondMaxArea && approx.total() == 4L) {
                            thirdBiggestContour = secondBiggestContour
                            thirdMaxArea = secondMaxArea
                            secondBiggestContour = MatOfPoint(*approx.toArray())
                            secondMaxArea = area
                        } else if (area > thirdMaxArea && approx.total() == 4L) {
                            thirdBiggestContour = MatOfPoint(*approx.toArray())
                            thirdMaxArea = area
                        }
                    }
                }
            }
            if (biggestContour.toArray().isNotEmpty()) {
                resultContours.add(biggestContour)
            }
            if (secondBiggestContour.toArray().isNotEmpty()) {
                resultContours.add(secondBiggestContour)
            }
            if (thirdBiggestContour.toArray().isNotEmpty()) {
                resultContours.add(thirdBiggestContour)
            }
            return resultContours.ifEmpty { null }
        }

        private fun reorderPoints(myPoints: Array<Point>): Array<Point?> {
            val myPointsNew = arrayOfNulls<Point>(4)
            val matOfPoint2f = MatOfPoint2f()
            matOfPoint2f.fromArray(*myPoints)
            val sums = DoubleArray(4)
            val diffs = DoubleArray(4)
            for (i in 0..3) {
                sums[i] = myPoints[i].x + myPoints[i].y
                diffs[i] = myPoints[i].y - myPoints[i].x
            }
            val minSumIndex = findMinIndex(sums)
            val maxSumIndex = findMaxIndex(sums)
            val minDiffIndex = findMinIndex(diffs)
            val maxDiffIndex = findMaxIndex(diffs)
            myPointsNew[0] = myPoints[minSumIndex]
            myPointsNew[1] = myPoints[minDiffIndex]
            myPointsNew[2] = myPoints[maxSumIndex]
            myPointsNew[3] = myPoints[maxDiffIndex]
            return myPointsNew
        }

        private fun findMinIndex(arr: DoubleArray): Int {
            var minVal = arr[0]
            var minIndex = 0
            for (i in 1 until arr.size) {
                if (arr[i] < minVal) {
                    minVal = arr[i]
                    minIndex = i
                }
            }
            return minIndex
        }

        private fun findMaxIndex(arr: DoubleArray): Int {
            var maxVal = arr[0]
            var maxIndex = 0
            for (i in 1 until arr.size) {
                if (arr[i] > maxVal) {
                    maxVal = arr[i]
                    maxIndex = i
                }
            }
            return maxIndex
        }

    }

    private enum class FilterType {
        Natural, Gray, Eco,
    }
}
