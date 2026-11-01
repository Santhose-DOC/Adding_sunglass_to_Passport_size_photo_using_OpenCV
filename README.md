# Workshop 1 -- Adding Sunglasses to a Passport-Size Photo Using OpenCV

### Name Developed by: SANTHOSE AROCKIRAJ J
### RegisterNumber:  21222430248

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

**Original Face Image:**

<img width="280" height="347" alt="image" src="https://github.com/user-attachments/assets/78f7ee3d-c9d8-47b4-a6f2-00d30f0e95cc" />


**Sunglasses Image:** 

<img width="448" height="229" alt="image" src="https://github.com/user-attachments/assets/99f1ce34-fe61-48ca-acb2-964022146c34" />


**Sunglasses Color and Alpha Channels:** 

<img width="871" height="202" alt="image" src="https://github.com/user-attachments/assets/2469b6e7-825b-4336-ba24-e5ec91c57c92" />


**Direct Overlay:** 

<img width="293" height="338" alt="image" src="https://github.com/user-attachments/assets/f4867a96-5b82-4d6b-b863-9bdfee038db9" />


**Masked Eye Region:** 

<img width="703" height="317" alt="image" src="https://github.com/user-attachments/assets/8587f8ee-aa23-4e4f-ab38-9bd77dcf3759" />



**Masked Sunglasses Region:**

<img width="755" height="343" alt="image" src="https://github.com/user-attachments/assets/5f94d295-f60e-4f0b-a4d3-6b35a17499ec" />


**Augmented Eye Region:**

<img width="700" height="342" alt="image" src="https://github.com/user-attachments/assets/6b90aeab-af14-4d3f-9c76-15c7d29ee0fc" />


**Final Output:** 

<img width="870" height="556" alt="image" src="https://github.com/user-attachments/assets/05030860-acc2-4af1-8da5-e6444a1f2505" />


## Result

Thus, sunglasses were successfully added to the passport-size photograph
using OpenCV. The alpha-channel mask and arithmetic operations were used
to blend the sunglasses with the eye region and display the final
augmented image.
