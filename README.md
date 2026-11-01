# Workshop 1 -- Adding Sunglasses to a Passport-Size Photo Using OpenCV

## Aim

To add sunglasses to a passport-size face photograph using OpenCV by
applying image resizing, alpha-channel masking, and arithmetic
operations.

## Procedure

1.  Import the required libraries: OpenCV (`cv2`), NumPy, and
    Matplotlib.
2.  Load the face photograph (`my.jpeg`) using `cv2.imread()` and
    display the original image.
3.  Load the sunglasses PNG image (`Image-Source-PlusPNG.com.png`) with
    its alpha channel using `cv2.imread()` with the `-1` flag.
4.  Resize the sunglasses image to 350 × 135 pixels so that it fits over
    the eye region.
5.  Separate the sunglasses image into its BGR color channels and alpha
    channel.
6.  Display the sunglasses color channels and alpha mask.
7.  Create a copy of the face image and place the resized sunglasses
    image over the eye region using array slicing.
8.  Merge the alpha channel into a three-channel mask and convert its
    values to 0 and 1.
9.  Extract the eye region of the face image.
10. Apply the mask to the eye region and sunglasses image using
    `cv2.multiply()`.
11. Combine the masked eye region and masked sunglasses using
    `cv2.add()`.
12. Replace the original eye region with the combined result and display
    the original and augmented images side by side.

## Program

``` python
# Import libraries
import cv2
import numpy as np
import matplotlib.pyplot as plt

# Load the Face Image
faceImage = cv2.imread('my.jpeg')
plt.imshow(faceImage[:, :, ::-1])
plt.title("Face")

# Load the Sunglass image with Alpha channel
glassPNG = cv2.imread('Image-Source-PlusPNG.com.png', -1)
plt.imshow(glassPNG[:, :, ::-1])
plt.title("glassPNG")

# Resize the image to fit over the eye region
glassPNG = cv2.resize(glassPNG, (350, 135))
print("image Dimension ={}".format(glassPNG.shape))

# Separate the Color and alpha channels
glassBGR = glassPNG[:, :, 0:3]
glassMask1 = glassPNG[:, :, 3]

# Display the images for clarity
plt.figure(figsize=[15, 15])
plt.subplot(121)
plt.imshow(glassBGR[:, :, ::-1])
plt.title('Sunglass Color channels')

plt.subplot(122)
plt.imshow(glassMask1, cmap='gray')
plt.title('Sunglass Alpha channel')

# Make a copy
faceWithGlassesNaive = faceImage.copy()

# Replace the eye region with the sunglass image
faceWithGlassesNaive[300:435, 260:610] = glassBGR
plt.imshow(faceWithGlassesNaive[..., ::-1])

# Make the dimensions of the mask same as the input image
glassMask = cv2.merge((glassMask1, glassMask1, glassMask1))

# Make the values [0,1] for arithmetic operations
glassMask = np.uint8(glassMask / 255)

# Make a copy
faceWithGlassesArithmetic = faceImage.copy()

# Get the eye region from the face image
eyeROI = faceWithGlassesArithmetic[300:435, 260:610]

# Use the mask to create the masked eye region
maskedEye = cv2.multiply(eyeROI, (1 - glassMask))

# Use the mask to create the masked sunglass region
maskedGlass = cv2.multiply(glassBGR, glassMask)

# Combine the sunglasses in the eye region
eyeRoiFinal = cv2.add(maskedEye, maskedGlass)

# Display the intermediate results
plt.figure(figsize=[20, 20])
plt.subplot(131)
plt.imshow(maskedEye[..., ::-1])
plt.title("Masked Eye Region")

plt.subplot(132)
plt.imshow(maskedGlass[..., ::-1])
plt.title("Masked Sunglass Region")

plt.subplot(133)
plt.imshow(eyeRoiFinal[..., ::-1])
plt.title("Augmented Eye and Sunglass")

# Replace the eye ROI with the output from the previous section
faceWithGlassesArithmetic[300:435, 260:610] = eyeRoiFinal

# Display the final result
plt.figure(figsize=[20, 20])
plt.subplot(121)
plt.imshow(faceImage[:, :, ::-1])
plt.title("Original Image")

plt.subplot(122)
plt.imshow(faceWithGlassesArithmetic[:, :, ::-1])
plt.title("With Sunglasses")

plt.show()
```

## Output

1.  **Original Face Image:** The input passport-size photograph is
    loaded and displayed.
2.  **Sunglasses Image:** The PNG sunglasses image is loaded with its
    alpha channel and resized to 350 × 135 pixels. The displayed
    dimension is `(135, 350, 4)`.
3.  **Sunglasses Color and Alpha Channels:** The BGR color channels and
    grayscale alpha mask are displayed separately.
4.  **Direct Overlay:** A copy of the face image is displayed with the
    sunglasses placed over the eye region.
5.  **Masked Eye Region:** The eye region is processed using the inverse
    sunglasses mask.
6.  **Masked Sunglasses Region:** The sunglasses are processed using the
    alpha mask.
7.  **Augmented Eye Region:** The masked eye and sunglasses regions are
    combined using `cv2.add()`.
8.  **Final Output:** The original photograph and the photograph with
    sunglasses are displayed side by side.

*Note: The output depends on the input face and sunglasses images. Keep
`my.jpeg` and `Image-Source-PlusPNG.com.png` in the working directory
before executing the notebook. The eye-region coordinates are fixed as
`[300:435, 260:610]` in this program.*

## Result

Thus, sunglasses were successfully added to the passport-size photograph
using OpenCV. The alpha-channel mask and arithmetic operations were used
to blend the sunglasses with the eye region and display the final
augmented image.
