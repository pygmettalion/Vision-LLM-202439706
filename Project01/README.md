# Jawline Destiny Theory 

A real-time facial similarity project inspired by the playful idea that people with similar jawlines are destined for each other.

The project uses MediaPipe Face Landmarker and OpenCV to detect two faces simultaneously and calculate a custom **Destiny Percentage**.

## Features

- Real-time webcam input and two-person face detection
- Lower-face landmark and jawline contour visualization
- Geometric feature extraction and normalization
- Weighted jawline similarity calculation
- Real-time percentage display with EMA smoothing

## Technologies

- Python
- OpenCV
- MediaPipe Face Landmarker
- NumPy

## Similarity Algorithm

The similarity calculation is a custom rule-based algorithm consisting of six stages.

### 1. Facial Landmark Detection

MediaPipe Face Landmarker detects up to two faces in each camera frame using `num_faces=2`.

The detected landmarks provide normalized facial coordinates, which are converted into pixel coordinates for geometric measurements.

### 2. Feature Extraction

The `extract_features()` function extracts four lower-face features.

| Feature | Calculation | Weight |
|---|---|---|
| Mouth Width Ratio | Mouth width / Jaw width | 25% |
| Lower-Face Height Ratio | Nose-region-to-chin distance / Jaw width | 20% |
| Jaw Taper Ratio | Lower-jaw width / Jaw width | 20% |
| Jawline Contour Similarity | Comparison of 13 normalized jawline landmarks | 35% |

The landmark indices used are:

- Mouth corners: `61`, `291`
- Jaw width reference: `172`, `397`
- Lower jaw: `148`, `377`
- Nose-region reference: `2`
- Chin: `152`

### 3. Distance Calculation and Normalization

Euclidean distance is used to measure the distance between facial landmarks.

`distance = sqrt((x2 - x1)² + (y2 - y1)²)`

Raw distances are divided by the jaw width to reduce differences caused by face size and camera distance.

The jawline contour is also normalized:

`normalized_point = (point - jaw_center) / jaw_width`

The algorithm uses 13 jawline landmarks:

`[172, 136, 150, 149, 176, 148, 152, 377, 400, 378, 379, 365, 397]`

This allows the two jawline shapes to be compared using corresponding landmark positions.

### 4. Similarity Calculation

For the three facial ratios, the `ratio_similarity()` function measures relative differences:

`S = max(0, 1 - |A - B| / max(|A|, |B|, 0.000001))`

Identical ratios receive a similarity score of 1.0.

For jawline contours, the algorithm calculates the Euclidean distance between corresponding normalized landmarks and obtains the mean distance.

`contour_error = mean(EuclideanDistance(A[i], B[i]))`

`contour_similarity = max(0, 1 - contour_error / 0.30)`

The value `0.30` is an experimentally adjustable heuristic scale, not a scientifically validated threshold.

### 5. Weighted Destiny Score

The final similarity score is calculated using a weighted combination of four features:

`Destiny (%) = 100 × (0.25 × Mouth + 0.20 × Height + 0.20 × Taper + 0.35 × Contour)`

The result is restricted to a range of 0–100%.

The weights are manually defined design parameters rather than values learned from training data.

### 6. Exponential Moving Average (EMA)

An Exponential Moving Average reduces fluctuations in the displayed similarity score:

`Smoothed Score = 0.75 × Previous Score + 0.25 × Current Score`

EMA makes the real-time percentage display more stable when detected landmark positions change slightly between frames.

## How to Run

1. Install dependencies using `pip install -r requirements-windows.txt`.
2. Place the MediaPipe model at `models/face_landmarker.task`.
3. Connect a webcam.
4. Run `python main.py`.
5. Position two faces in front of the camera to display the Destiny Percentage.

Press **Q** or **ESC** to exit.

## Limitations

The score is based on selected 2D facial landmarks. Head angle, facial expression, camera perspective, and lighting can affect the result.

The weights and contour threshold have not been validated using a facial similarity dataset.

The program uses a pre-trained MediaPipe model for facial landmark detection, but the similarity calculation itself is a custom geometric algorithm without additional model training.

Compatibility with Jetson/ARM Linux requires separate environment testing.

## Note

The Destiny Percentage is a custom jawline similarity score developed for educational and entertainment purposes. It does not represent a scientifically validated measure of romantic compatibility or destiny.
