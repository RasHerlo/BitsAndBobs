import numpy as np
import matplotlib.pyplot as plt
import argparse
import os

def normalize_image(img):
    """Normalize image to 0-1 range and then scale to 0-255"""
    img_norm = (img - img.min()) / (img.max() - img.min())
    return (img_norm * 255).astype(np.uint8)

def calculate_ncc(img1, img2):
    """Calculate normalized cross-correlation between two images"""
    img1 = img1.astype(float)
    img2 = img2.astype(float)
    
    # Subtract means
    img1 = img1 - np.mean(img1)
    img2 = img2 - np.mean(img2)
    
    # Calculate NCC
    numerator = np.sum(img1 * img2)
    denominator = np.sqrt(np.sum(img1**2) * np.sum(img2**2))
    
    return numerator / denominator if denominator != 0 else 0

def calculate_local_ncc(img1, img2, window_size=21):
    """Calculate local NCC map using sliding window"""
    h, w = img1.shape
    ncc_map = np.zeros((h, w))
    
    # Pad images
    pad = window_size // 2
    img1_pad = np.pad(img1, pad, mode='constant')
    img2_pad = np.pad(img2, pad, mode='constant')
    
    # Calculate NCC for each window position
    for i in range(h):
        for j in range(w):
            patch1 = img1_pad[i:i+window_size, j:j+window_size]
            patch2 = img2_pad[i:i+window_size, j:j+window_size]
            ncc_map[i, j] = calculate_ncc(patch1, patch2)
    
    return ncc_map

def create_image_figure(image1_path, image2_path):
    # Read the images
    img1 = plt.imread(image1_path)
    img2 = plt.imread(image2_path)
    
    # Get center coordinates
    h1, w1 = img1.shape
    h2, w2 = img2.shape
    center1 = (h1//2, w1//2)
    center2 = (h2//2, w2//2)
    
    # Extract central 412x412 regions
    size = 412
    img1_center = img1[center1[0]-size//2:center1[0]+size//2, 
                      center1[1]-size//2:center1[1]+size//2]
    img2_center = img2[center2[0]-size//2:center2[0]+size//2, 
                      center2[1]-size//2:center2[1]+size//2]
    
    # Normalize images to 0-255 range
    img1_norm = normalize_image(img1)
    img2_norm = normalize_image(img2)
    
    # Get directory names
    dir1 = os.path.dirname(image1_path)
    dir2 = os.path.dirname(image2_path)
    
    # Create figure with four subplots
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
    
    # Create cyan image (green + blue channels)
    cyan_img = np.zeros((*img1_norm.shape, 3), dtype=np.uint8)
    cyan_img[..., 1] = img1_norm  # Green channel
    cyan_img[..., 2] = img1_norm  # Blue channel
    
    # Create magenta image (red + blue channels)
    magenta_img = np.zeros((*img2_norm.shape, 3), dtype=np.uint8)
    magenta_img[..., 0] = img2_norm  # Red channel
    magenta_img[..., 2] = img2_norm  # Blue channel
    
    # Create overlay
    overlay = np.zeros((*img1_norm.shape, 3), dtype=np.uint8)
    overlay[..., 1] = img1_norm  # Green channel from first image
    overlay[..., 2] = img1_norm  # Blue channel from first image
    overlay[..., 0] = img2_norm  # Red channel from second image
    
    # Calculate NCC
    global_ncc = calculate_ncc(img1_center, img2_center)
    local_ncc = calculate_local_ncc(img1_center, img2_center)
    
    # Plot images
    ax1.imshow(cyan_img)
    ax1.set_title(f'Image 1 (Cyan)\nFrom: {dir1}')
    
    ax2.imshow(magenta_img)
    ax2.set_title(f'Image 2 (Magenta)\nFrom: {dir2}')
    
    # Plot overlay with red rectangle
    ax3.imshow(overlay)
    rect = plt.Rectangle((center1[1]-size//2, center1[0]-size//2), 
                        size, size, fill=False, edgecolor='red', 
                        linestyle='--', linewidth=2)
    ax3.add_patch(rect)
    ax3.set_title('Cyan-Magenta Overlay')
    
    # Plot NCC map
    ncc_plot = ax4.imshow(local_ncc, cmap='viridis')
    plt.colorbar(ncc_plot, ax=ax4)
    ax4.set_title(f'Local NCC Map\nGlobal NCC: {global_ncc:.3f}')
    
    plt.tight_layout()
    plt.show()

def main():
    parser = argparse.ArgumentParser(description='Create a figure with two images and their overlay')
    parser.add_argument('image1', help='Path to the first image')
    parser.add_argument('image2', help='Path to the second image')
    
    args = parser.parse_args()
    create_image_figure(args.image1, args.image2)

if __name__ == '__main__':
    main() 