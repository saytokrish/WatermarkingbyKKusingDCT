import cv2
import numpy as np

def embed_watermark(image_path, watermark_path, output_path, alpha=0.5):
    # Load the input image and watermark
    image = cv2.imread(image_path)  # Load as a color image
    watermark = cv2.imread(watermark_path, cv2.IMREAD_UNCHANGED)
    
    # Ensure watermark has the same dimensions as the image
    watermark = cv2.resize(watermark, (image.shape[1], image.shape[0]))

    # Convert watermark to BGR if it is grayscale
    if len(watermark.shape) == 2 or watermark.shape[2] == 1:  # Grayscale image
        watermark = cv2.cvtColor(watermark, cv2.COLOR_GRAY2BGR)

    # Embed the watermark using alpha blending
    watermarked_image = cv2.addWeighted(image, 1, watermark, alpha, 0)

    # Save the watermarked image
    cv2.imwrite(output_path, watermarked_image)
    print(f"Watermarked image saved to {output_path}")


def extract_watermark(watermarked_path, original_path, output_path, alpha=0.5):
    """
    Extract the watermark from a watermarked image.
    
    Parameters:
        watermarked_path (str): Path to the watermarked image.
        original_path (str): Path to the original image.
        output_path (str): Path to save the extracted watermark.
        alpha (float): Strength of the watermark used during embedding.
    """
    # Load the watermarked image and the original image
    watermarked_image = cv2.imread(watermarked_path)
    original_image = cv2.imread(original_path)
    
    # Ensure both images are the same size
    if watermarked_image.shape != original_image.shape:
        raise ValueError("Watermarked and original images must have the same dimensions.")
    
    # Extract the watermark
    watermark = (watermarked_image - original_image * (1 - alpha)) / alpha
    watermark = np.clip(watermark, 0, 255).astype(np.uint8)  # Ensure valid pixel values

    # Save the extracted watermark
    cv2.imwrite(output_path, watermark)
    print(f"Extracted watermark saved to {output_path}")


# File paths
    """
     Parameters:
    - image_path: Path to the input image.
    - watermark_path: Path to the watermark image.
    - output_path: Path to save the watermarked image.
    - alpha: Strength of the watermark embedding.
    """

image_path = "kkimg.jpg"
watermark_path = "kkwm.jpg"
watermarked_output_path = "kk1.jpg"
extracted_watermark_output_path = "kk2.jpg"

# Embedding the watermark
embed_watermark(image_path, watermark_path, watermarked_output_path, alpha=0.01)

# Extracting the watermark
extract_watermark(watermarked_output_path, image_path, extracted_watermark_output_path, alpha=0.01)