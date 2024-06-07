import cv2
import numpy as np
from skimage.metrics import structural_similarity as ssim
from PIL import Image
import imagehash

def bilateral_filtering(
        img: np.uint8,
        spatial_variance: float,
        intensity_variance: float,
        kernel_size: int,
) -> np.uint8:
    if img is None:
        raise ValueError("Image not loaded.")

    img = img / 255.0  # Normalize to [0, 1] range
    img_filtered = np.zeros_like(img)  # Placeholder of the filtered image

    sizeX, sizeY, channels = img.shape

    kernel_half_size = kernel_size // 2

    for i in range(sizeX):
        for j in range(sizeY):
            for c in range(channels):
                filtered_pixel_value = 0.0
                normalization_factor = 0.0
                for k in range(-kernel_half_size, kernel_half_size + 1):
                    for l in range(-kernel_half_size, kernel_half_size + 1):
                        x = i + k
                        y = j + l
                        if 0 <= x < sizeX and 0 <= y < sizeY:
                            spatial_weight = np.exp(-(k ** 2 + l ** 2) / (2 * spatial_variance ** 2))
                            intensity_weight = np.exp(-(img[i, j, c] - img[x, y, c]) ** 2 / (2 * intensity_variance ** 2))
                            bilateral_weight = spatial_weight * intensity_weight
                            filtered_pixel_value += img[x, y, c] * bilateral_weight
                            normalization_factor += bilateral_weight
                img_filtered[i, j, c] = filtered_pixel_value / normalization_factor

    img_filtered = (img_filtered * 255).astype(np.uint8)
    return img_filtered

def calculate_ssim(img1, img2):
    ssim_index, _ = ssim(img1, img2, full=True, channel_axis=-1)
    return ssim_index

def calculate_mse(img1, img2):
    mse = np.sum((img1 - img2) ** 2) / float(img1.size)
    return mse

def calculate_cross_correlation(img1, img2):
    cross_correlation = cv2.matchTemplate(img1, img2, cv2.TM_CCOEFF_NORMED)[0, 0]
    return cross_correlation

def calculate_perceptual_hash_similarity(img1, img2):
    hash1 = imagehash.dhash(Image.fromarray(img1))
    hash2 = imagehash.dhash(Image.fromarray(img2))
    hash_similarity = 1 - (hash1 - hash2) / len(hash1.hash) ** 2
    return hash_similarity

def calculate_histogram_similarity(img1, img2):
    hist1 = cv2.calcHist([img1], [0, 1, 2], None, [256, 256, 256], [0, 256, 0, 256, 0, 256])
    hist2 = cv2.calcHist([img2], [0, 1, 2], None, [256, 256, 256], [0, 256, 0, 256, 0, 256])
    similarity = cv2.compareHist(hist1, hist2, cv2.HISTCMP_CORREL)
    return similarity

def calculate_overall_similarity_score(ssim_index, mse, cross_correlation, hash_similarity, hist_similarity):
    # Normalize each metric to the range [0, 1]
    ssim_normalized = ssim_index
    mse_normalized = 1.0 - (mse / (mse + 1.0))  # Normalize MSE to [0, 1], minimizing the effect of large MSE
    cross_correlation_normalized = (cross_correlation + 1.0) / 2.0  # Normalize Cross-Correlation to [0, 1]
    hash_similarity_normalized = (hash_similarity + 1.0) / 2.0  # Normalize Hash Similarity to [0, 1]
    hist_similarity_normalized = (hist_similarity + 1.0) / 2.0  # Normalize Histogram Similarity to [0, 1]

    # Assign equal weight to each metric
    weight_ssim = 1.0
    weight_mse = 1.0
    weight_cross_correlation = 1.0
    weight_hash_similarity = 1.0
    weight_hist_similarity = 1.0

    # Calculate the overall similarity score
    overall_similarity_score = (
        weight_ssim * ssim_normalized +
        weight_mse * mse_normalized +
        weight_cross_correlation * cross_correlation_normalized +
        weight_hash_similarity * hash_similarity_normalized +
        weight_hist_similarity * hist_similarity_normalized
    ) / (weight_ssim + weight_mse + weight_cross_correlation + weight_hash_similarity + weight_hist_similarity)

    # Convert to a percentage
    overall_similarity_score_percent = overall_similarity_score * 100.0

    return overall_similarity_score_percent

if __name__ == '__main__':
    # Example usage
    image_path1 = '/Users/JOSH/Desktop/CS 4391                     (Vision)/Project/pics/1700/6.jpg'
    image_path2 = '/Users/JOSH/Desktop/CS 4391                     (Vision)/Project/pics/1700/7.jpg'

    # Load the images
    img1 = cv2.imread(image_path1)
    img2 = cv2.imread(image_path2)

    if img1 is None:
        raise ValueError("Image 1 not loaded.")
    if img2 is None:
        raise ValueError("Image 2 not loaded.")

    # Resize images to a common size
    common_height, common_width = 256, 256  # Adjust dimensions as needed
    img1 = cv2.resize(img1, (common_width, common_height))
    img2 = cv2.resize(img2, (common_width, common_height))

    # Apply bilateral filtering
    spatial_variance = 50
    intensity_variance = 50
    kernel_size = 5

    try:
        filtered_img1 = bilateral_filtering(img1, spatial_variance, intensity_variance, kernel_size)
        filtered_img2 = bilateral_filtering(img2, spatial_variance, intensity_variance, kernel_size)
    except ValueError as e:
        print(f"Error during bilateral filtering: {e}")
        exit()

    # Calculate SSIM
    ssim_index = calculate_ssim(filtered_img1, filtered_img2)
    print(f"Structural Similarity Index (SSIM): {ssim_index:.4f} \n(Range: [0, 1]) [Optimal: 1]\n")

    # Calculate Mean Squared Error (MSE)
    mse = calculate_mse(filtered_img1, filtered_img2)
    print(f"Mean Squared Error (MSE): {mse:.4f} \n(Range: [0, ∞)) [Optimal: 0]\n")

    # Calculate Cross-Correlation
    cross_correlation = calculate_cross_correlation(filtered_img1, filtered_img2)
    print(f"Cross-Correlation: {cross_correlation:.4f} \n(Range: [-1, 1]) [Optimal: 1]\n")

    # Calculate Perceptual Hashing Similarity
    hash_similarity = calculate_perceptual_hash_similarity(filtered_img1, filtered_img2)
    print(f"Perceptual Hashing Similarity: {hash_similarity:.4f} \n(Range: [0, 1]) [Optimal: 1]\n")

    # Calculate Histogram Similarity
    hist_similarity = calculate_histogram_similarity(filtered_img1, filtered_img2)
    print(f"Histogram Similarity: {hist_similarity:.4f} \n(Range: [-1, 1]) [Optimal: 1]\n")

    overall_similarity_score = calculate_overall_similarity_score(ssim_index, mse, cross_correlation, hash_similarity, hist_similarity)
    print(f"Overall Similarity Score: {overall_similarity_score:.2f} out of 100")
