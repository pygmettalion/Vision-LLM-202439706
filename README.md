# Vision-LLM-202439706
202439706 조예연 LLM

## 02. Computer Vision

### Overview
Practiced image processing techniques using OpenCV, focusing on color space conversion, image enhancement, and object segmentation.

### Implementation

**1. Color Space Conversion**
- Converted images between RGB, HSV, and LAB color spaces.
- Separated and visualized individual color channels.
- Applied color-based thresholding to generate binary masks.

**2. Image Enhancement (CLAHE)**
- Applied CLAHE to the L channel in LAB color space.
- Improved local contrast in images affected by uneven lighting.
- Generated binary masks using the enhanced basketball image.

**3. Morphological Operations**
- Applied Opening to remove small noise from binary masks.
- Applied Closing to fill small gaps in segmented regions.

**4. Contour Detection**
- Detected object contours using `cv2.findContours()`.
- Selected the largest contour to identify the basketball region.
- Visualized the selected region using `cv2.drawContours()`.

### Result
Applied CLAHE and morphological operations to improve basketball segmentation under uneven lighting conditions. Extracted the basketball region using color thresholding and contour detection.

### File
- `02_Computer-Vision.ipynb`