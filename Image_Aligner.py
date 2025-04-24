import numpy as np
import matplotlib.pyplot as plt
import matplotlib.widgets as widgets
from matplotlib.patches import Rectangle
from matplotlib.lines import Line2D
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

class ImageAligner:
    def __init__(self, image1_path, image2_path):
        # Read and normalize images
        self.img1 = plt.imread(image1_path)
        self.img2 = plt.imread(image2_path)
        self.img1_norm = normalize_image(self.img1)
        self.img2_norm = normalize_image(self.img2)
        
        # Get center coordinates
        h1, w1 = self.img1.shape
        h2, w2 = self.img2.shape
        self.center1 = (h1//2, w1//2)
        self.center2 = (h2//2, w2//2)
        
        # Initialize displacement
        self.dx = 0
        self.dy = 0
        self.increment = 3
        
        # Create figure and subplots
        self.fig = plt.figure(figsize=(15, 10))
        gs = self.fig.add_gridspec(2, 3, width_ratios=[1, 1, 0.2])
        
        # Create subplots
        self.ax1 = self.fig.add_subplot(gs[0, 0])
        self.ax2 = self.fig.add_subplot(gs[0, 1])
        self.ax3 = self.fig.add_subplot(gs[1, 0])
        self.ax4 = self.fig.add_subplot(gs[1, 1])
        self.ax_controls = self.fig.add_subplot(gs[:, 2])
        
        # Initialize plots
        self.update_plots()
        
        # Create controls
        self.create_controls()
        
        # Connect event handlers
        self.fig.canvas.mpl_connect('key_press_event', self.on_key)
        
    def create_controls(self):
        # Create slider for increment
        ax_slider = plt.axes([0.8, 0.7, 0.15, 0.03])
        self.slider = widgets.Slider(ax_slider, 'Step', 1, 10, valinit=3, valstep=1)
        self.slider.on_changed(self.update_increment)
        
        # Create arrow buttons
        button_width = 0.05
        button_height = 0.05
        button_spacing = 0.01
        
        # Up button
        ax_up = plt.axes([0.8, 0.5, button_width, button_height])
        self.btn_up = widgets.Button(ax_up, '↑')
        self.btn_up.on_clicked(lambda x: self.move_image(0, 1))
        
        # Down button
        ax_down = plt.axes([0.8, 0.4, button_width, button_height])
        self.btn_down = widgets.Button(ax_down, '↓')
        self.btn_down.on_clicked(lambda x: self.move_image(0, -1))
        
        # Left button
        ax_left = plt.axes([0.75, 0.45, button_width, button_height])
        self.btn_left = widgets.Button(ax_left, '←')
        self.btn_left.on_clicked(lambda x: self.move_image(-1, 0))
        
        # Right button
        ax_right = plt.axes([0.85, 0.45, button_width, button_height])
        self.btn_right = widgets.Button(ax_right, '→')
        self.btn_right.on_clicked(lambda x: self.move_image(1, 0))
        
        # Reset button
        ax_reset = plt.axes([0.8, 0.3, button_width, button_height])
        self.btn_reset = widgets.Button(ax_reset, 'Reset')
        self.btn_reset.on_clicked(self.reset_position)
        
    def update_increment(self, val):
        self.increment = int(val)
        
    def move_image(self, dx, dy):
        # Calculate new position
        new_dx = self.dx + dx * self.increment
        new_dy = self.dy + dy * self.increment
        
        # Check bounds
        if abs(new_dx) <= 100 and abs(new_dy) <= 100:
            self.dx = new_dx
            self.dy = new_dy
            self.update_plots()
        
    def reset_position(self, event):
        self.dx = 0
        self.dy = 0
        self.update_plots()
        
    def update_plots(self):
        # Clear all plots
        self.ax1.clear()
        self.ax2.clear()
        self.ax3.clear()
        self.ax4.clear()
        
        # Create cyan image
        cyan_img = np.zeros((*self.img1_norm.shape, 3), dtype=np.uint8)
        cyan_img[..., 1] = self.img1_norm
        cyan_img[..., 2] = self.img1_norm
        
        # Create magenta image
        magenta_img = np.zeros((*self.img2_norm.shape, 3), dtype=np.uint8)
        magenta_img[..., 0] = self.img2_norm
        magenta_img[..., 2] = self.img2_norm
        
        # Create overlay with displacement
        overlay = np.zeros((*self.img1_norm.shape, 3), dtype=np.uint8)
        overlay[..., 1] = self.img1_norm
        overlay[..., 2] = self.img1_norm
        
        # Apply displacement to magenta image
        h, w = self.img2_norm.shape
        y_start = max(0, self.dy)
        y_end = min(h, h + self.dy)
        x_start = max(0, self.dx)
        x_end = min(w, w + self.dx)
        
        overlay[y_start:y_end, x_start:x_end, 0] = self.img2_norm[max(0, -self.dy):min(h, h-self.dy), 
                                                                 max(0, -self.dx):min(w, w-self.dx)]
        
        # Plot images
        self.ax1.imshow(cyan_img)
        self.ax1.set_title('Image 1 (Cyan)')
        
        self.ax2.imshow(magenta_img)
        self.ax2.set_title('Image 2 (Magenta)')
        
        # Plot overlay with analysis region and crosshair
        self.ax3.imshow(overlay)
        size = 412
        rect = Rectangle((self.center1[1]-size//2, self.center1[0]-size//2), 
                        size, size, fill=False, edgecolor='red', 
                        linestyle='--', linewidth=2)
        self.ax3.add_patch(rect)
        
        # Add crosshair at current position
        x = self.center1[1] + self.dx
        y = self.center1[0] + self.dy
        crosshair_h = Line2D([x-10, x+10], [y, y], color='yellow', linestyle='--', linewidth=1)
        crosshair_v = Line2D([x, x], [y-10, y+10], color='yellow', linestyle='--', linewidth=1)
        self.ax3.add_line(crosshair_h)
        self.ax3.add_line(crosshair_v)
        
        # Add displacement text
        self.ax3.set_title(f'Overlay\nDisplacement: ({self.dx}, {self.dy})')
        
        # Calculate and plot NCC
        img1_center = self.img1[self.center1[0]-size//2:self.center1[0]+size//2, 
                              self.center1[1]-size//2:self.center1[1]+size//2]
        img2_center = self.img2[self.center2[0]-size//2+self.dy:self.center2[0]+size//2+self.dy, 
                              self.center2[1]-size//2+self.dx:self.center2[1]+size//2+self.dx]
        
        global_ncc = calculate_ncc(img1_center, img2_center)
        local_ncc = calculate_local_ncc(img1_center, img2_center)
        
        ncc_plot = self.ax4.imshow(local_ncc, cmap='viridis')
        self.ax4.set_title(f'Local NCC Map\nGlobal NCC: {global_ncc:.3f}')
        
        plt.draw()
        
    def on_key(self, event):
        if event.key == 'up':
            self.move_image(0, 1)
        elif event.key == 'down':
            self.move_image(0, -1)
        elif event.key == 'left':
            self.move_image(-1, 0)
        elif event.key == 'right':
            self.move_image(1, 0)

def main():
    parser = argparse.ArgumentParser(description='Create a figure with two images and their overlay')
    parser.add_argument('image1', help='Path to the first image')
    parser.add_argument('image2', help='Path to the second image')
    
    args = parser.parse_args()
    aligner = ImageAligner(args.image1, args.image2)
    plt.show()

if __name__ == '__main__':
    main() 