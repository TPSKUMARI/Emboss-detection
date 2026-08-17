#!/usr/bin/env python3
"""
Combined Script: Grid Completion with ROI Masking, Cell Numbering, Text Halo Detection
Integrates provided grid3.py code for edge detection, ROI removal, and grid line completion.
Adds cell numbering based on detected grid lines.
Integrates color2.py for text halo detection on the final numbered overlay.
Updated: No resizing for outputs to maintain high quality and original size.
New: Resizes display image for interactive point selection to max 800px side to make clicking easier.
New: All outputs saved to 'outputs' folder.
Verified: Full workflow - edge detection, ROI selection & masking, grid completion from surrounding lines, cell numbering.
"""
import cv2
import numpy as np
from scipy.signal import find_peaks
from scipy import ndimage
from scipy.ndimage import gaussian_filter, zoom
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import sys
import os

# Global variables for mouse callback
drawing = False
points = []
img_copy = None
display_scale = 1.0  # To be set later

def mouse_callback(event, x, y, flags, param):
    global drawing, points, img_copy, display_scale
    
    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
        # Store points in original image coordinates
        orig_x = int(x / display_scale)
        orig_y = int(y / display_scale)
        points.append((orig_x, orig_y))
        # Draw on display image
        cv2.circle(img_copy, (x, y), 5, (0, 0, 255), -1)  # Red dot for selected point
        cv2.imshow('Select 4 Points', img_copy)
        
        if len(points) == 4:
            drawing = False
            # Draw lines between points to visualize quadrilateral (on display)
            pts_display = np.array([(int(px * display_scale), int(py * display_scale)) for px, py in points], np.int32)
            pts_display = pts_display.reshape((-1, 1, 2))
            cv2.polylines(img_copy, [pts_display], True, (255, 0, 0), 2)  # Blue lines
            cv2.imshow('Select 4 Points', img_copy)
            cv2.waitKey(1000)  # Pause to show selection

def detect_edges(image_path, method='canny', threshold1=50, threshold2=100):
    """
    Detect edges in an image using various methods.
   
    Args:
        image_path: Path to input image
        method: 'canny', 'sobel', 'laplacian', or 'scharr'
        threshold1: First threshold (for Canny)
        threshold2: Second threshold (for Canny)
   
    Returns:
        Edge map as numpy array
    """
    # Load image
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Could not load image from {image_path}")
   
    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
   
    # Apply Gaussian blur
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
   
    # Apply edge detection
    if method.lower() == 'canny':
        edges = cv2.Canny(blurred, threshold1, threshold2)
   
    elif method.lower() == 'sobel':
        sobelx = cv2.Sobel(blurred, cv2.CV_64F, 1, 0, ksize=3)
        sobely = cv2.Sobel(blurred, cv2.CV_64F, 0, 1, ksize=3)
        edges = np.sqrt(sobelx**2 + sobely**2)
        edges = np.uint8(edges * 255 / edges.max())
   
    elif method.lower() == 'laplacian':
        edges = cv2.Laplacian(blurred, cv2.CV_64F)
        edges = np.uint8(np.absolute(edges))
   
    elif method.lower() == 'scharr':
        scharrx = cv2.Scharr(blurred, cv2.CV_64F, 1, 0)
        scharry = cv2.Scharr(blurred, cv2.CV_64F, 0, 1)
        edges = np.sqrt(scharrx**2 + scharry**2)
        edges = np.uint8(edges * 255 / edges.max())
   
    else:
        raise ValueError(f"Unknown method: {method}")
   
    return edges, img  # Return both edges and original for display

def remove_roi(image, points, target_image_type='edges'):
    """
    Remove (mask out) the ROI defined by 4 points in the target image.
   
    Args:
        image: Original image or edges
        points: List of 4 (x, y) coordinates
        target_image_type: 'original' or 'edges' to apply mask to
   
    Returns:
        Masked image
    """
    if len(points) != 4:
        raise ValueError("Exactly 4 points must be provided for the quadrilateral ROI.")
    
    # Create mask for the quadrilateral
    mask = np.zeros(image.shape[:2], np.uint8)
    pts = np.array(points, np.int32)
    pts = pts.reshape((-1, 1, 2))
    cv2.fillPoly(mask, [pts], 255)
    
    # If applying to original color image, expand mask to 3 channels
    if len(image.shape) == 3:
        mask = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
    
    # Set ROI to black (remove)
    masked_image = image.copy()
    masked_image[mask == 255] = 0  # Black out the ROI
    
    return masked_image

def get_bounding_box(points):
    """
    Get the axis-aligned bounding box from points.
    """
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    return min(xs), max(xs), min(ys), max(ys)

def group_close_lines(lines, tolerance=3):
    """
    Group close lines into single average position to avoid duplicates.
    """
    if not lines:
        return []
    lines = sorted(lines)
    grouped = []
    current = [lines[0]]
    for l in lines[1:]:
        if l - current[-1] < tolerance:
            current.append(l)
        else:
            grouped.append(np.mean(current))
            current = [l]
    grouped.append(np.mean(current))
    return sorted(grouped)

def detect_horizontal_ys(edges_crop, tolerance=5, proj_threshold=0.015, min_peak_distance=8):
    """
    Detect horizontal line positions (y) using projection profiles in the crop.
    Returns grouped average y positions. Robust for small lines.
    Adjusted for better detection of faint/missing horizontals.
    """
    if edges_crop.shape[1] == 0:  # Empty crop
        return []
    
    # Dilate horizontally to connect small gaps in lines
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 1))
    dilated = cv2.dilate(edges_crop, kernel, iterations=1)
    
    # Horizontal projection: sum along columns for each row
    proj = np.sum(dilated.astype(np.float32), axis=1)
    
    # Normalize projection
    proj_max = np.max(proj)
    if proj_max == 0:
        return []
    
    # Threshold for peaks
    height = proj_max * proj_threshold
    
    # Find peaks
    peaks, _ = find_peaks(proj, height=height, distance=min_peak_distance)
    
    # Refine peaks with sub-pixel accuracy (optional, using quadratic fit)
    refined_peaks = []
    for peak in peaks:
        if peak > 0 and peak < len(proj) - 1:
            # Quadratic interpolation for sub-pixel, with zero-denominator check
            y0, y1, y2 = proj[peak-1:peak+2]
            denom = 2 * (y1 - y0 - y2 + y1)
            if abs(denom) < 1e-6:  # Avoid division by zero or near-zero
                refined = peak
            else:
                refined = peak + (y1 - y0) / denom
            refined_peaks.append(refined)
        else:
            refined_peaks.append(peak)
    
    if not refined_peaks:
        return []
    
    refined_peaks = sorted(refined_peaks)
    # Group close peaks
    grouped = group_close_lines(refined_peaks, tolerance)
    
    return grouped

def detect_vertical_xs(edges_crop, tolerance=5, proj_threshold=0.01, min_peak_distance=12):
    """
    Detect vertical line positions (x) using projection profiles in the crop.
    Returns grouped average x positions. Robust for small lines.
    """
    if edges_crop.shape[0] == 0:  # Empty crop
        return []
    
    # Dilate vertically to connect small gaps in lines
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 5))
    dilated = cv2.dilate(edges_crop, kernel, iterations=1)
    
    # Vertical projection: sum along rows for each column
    proj = np.sum(dilated.astype(np.float32), axis=0)
    
    # Normalize projection
    proj_max = np.max(proj)
    if proj_max == 0:
        return []
    
    # Threshold for peaks
    height = proj_max * proj_threshold
    
    # Find peaks
    peaks, _ = find_peaks(proj, height=height, distance=min_peak_distance)
    
    # Refine peaks with sub-pixel accuracy
    refined_peaks = []
    for peak in peaks:
        if peak > 0 and peak < len(proj) - 1:
            x0, x1, x2 = proj[peak-1:peak+2]
            denom = 2 * (x1 - x0 - x2 + x1)
            if abs(denom) < 1e-6:  # Avoid division by zero or near-zero
                refined = peak
            else:
                refined = peak + (x1 - x0) / denom
            refined_peaks.append(refined)
        else:
            refined_peaks.append(peak)
    
    if not refined_peaks:
        return []
    
    refined_peaks = sorted(refined_peaks)
    # Group close peaks
    grouped = group_close_lines(refined_peaks, tolerance)
    
    return grouped

def match_lines(lines1, lines2, max_dist=12):
    """
    Match lines between two sets by greedy closest assignment within distance.
    Returns list of average positions for matched pairs.
    """
    lines1 = sorted(lines1)
    lines2 = sorted(lines2)
    matched = []
    used2 = set()
    for l1 in lines1:
        closest_dist = float('inf')
        closest_l2 = None
        closest_idx = -1
        for idx, l2 in enumerate(lines2):
            if idx in used2:
                continue
            d = abs(l1 - l2)
            if d < closest_dist:
                closest_dist = d
                closest_l2 = l2
                closest_idx = idx
        if closest_dist < max_dist:
            matched.append((l1 + closest_l2) / 2)
            used2.add(closest_idx)
    return sorted(matched)

def complete_grid(masked_edges, points, h, w):
    """
    Detect and match horizontal and vertical lines across the masked region using projections.
    Draws completing line segments only in the ROI bounding box gap.
    Additionally draws top, bottom, left, and right boundary lines of the ROI bbox.
   
    Args:
        masked_edges: Binary edge image with ROI masked
        points: The 4 points defining the ROI
        h, w: Height and width of the image
   
    Returns:
        Completed grid edges image
    """
    min_x, max_x, min_y, max_y = get_bounding_box(points)
    
    # Expand crops slightly to capture lines near the boundary
    margin = 80
    left_start = max(0, min_x - margin)
    left_end = min_x + margin
    right_start = max(max_x - margin, 0)
    right_end = min(w, max_x + margin)
    top_start = max(0, min_y - margin)
    top_end = min_y + margin
    bottom_start = max(max_y - margin, 0)
    bottom_end = min(h, max_y + margin)
    
    # Crops
    left_crop = masked_edges[:, left_start:left_end]
    right_crop = masked_edges[:, right_start:right_end]
    top_crop = masked_edges[top_start:top_end, :]
    bottom_crop = masked_edges[bottom_start:bottom_end, :]
    
    # Detect horizontal lines: left and right of mask (to match y positions)
    left_horz_ys = detect_horizontal_ys(left_crop)
    right_horz_ys = detect_horizontal_ys(right_crop)
    matched_horz_ys = match_lines(left_horz_ys, right_horz_ys)
    matched_horz_ys = group_close_lines(matched_horz_ys)
    
    # Detect vertical lines: top and bottom of mask (to match x positions)
    top_vert_xs = detect_vertical_xs(top_crop)
    bottom_vert_xs = detect_vertical_xs(bottom_crop)
    matched_vert_xs = match_lines(top_vert_xs, bottom_vert_xs)
    matched_vert_xs = group_close_lines(matched_vert_xs)
    
    # Draw matched line segments on copy (only in the gap)
    output = masked_edges.copy()
    
    # Horizontal completions: draw segments across x-gap at matched y
    for y in matched_horz_ys:
        if min_y <= y <= max_y:
            cv2.line(output, (min_x, int(y)), (max_x, int(y)), 255, 1)
    
    # Vertical completions: draw segments across y-gap at matched x
    for x in matched_vert_xs:
        if min_x <= x <= max_x:
            cv2.line(output, (int(x), min_y), (int(x), max_y), 255, 1)
    
    # Draw the 4 boundary lines of the ROI bounding box
    # Top line
    cv2.line(output, (min_x, min_y), (max_x, min_y), 255, 1)
    # Bottom line
    cv2.line(output, (min_x, max_y), (max_x, max_y), 255, 1)
    # Left line
    cv2.line(output, (min_x, min_y), (min_x, max_y), 255, 1)
    # Right line
    cv2.line(output, (max_x, min_y), (max_x, max_y), 255, 1)
    
    print(f"✓ Detected left/right horz ys: {len(left_horz_ys)} / {len(right_horz_ys)} -> matched {len(matched_horz_ys)}")
    print(f"✓ Detected top/bottom vert xs: {len(top_vert_xs)} / {len(bottom_vert_xs)} -> matched {len(matched_vert_xs)}")
    print("✓ Added top, bottom, left, and right boundary lines.")
    
    return output, matched_horz_ys, matched_vert_xs  # Return lines for numbering

def add_numbers_to_grid(overlay_img, horz_ys, vert_xs, points):
    """
    Add sequential numbers to each grid cell center.
    Assumes row-major ordering starting from 1.
    """
    numbered_img = overlay_img.copy()
    min_x, max_x, min_y, max_y = get_bounding_box(points)
    
    # Sort lines (should already be sorted)
    horz_ys = sorted(horz_ys)
    vert_xs = sorted(vert_xs)
    
    if len(horz_ys) < 2 or len(vert_xs) < 2:
        print("Warning: Insufficient lines detected for gridding. Skipping numbering.")
        return numbered_img
    
    num_rows = len(horz_ys) - 1
    num_cols = len(vert_xs) - 1
    print(f"✓ Grid dimensions: {num_rows} rows x {num_cols} cols")
    
    for row in range(num_rows):
        y1 = horz_ys[row]
        y2 = horz_ys[row + 1]
        cy = int((y1 + y2) / 2)
        
        for col in range(num_cols):
            x1 = vert_xs[col]
            x2 = vert_xs[col + 1]
            cx = int((x1 + x2) / 2)
            
            # Clip to image bounds and ROI for safety
            cx = max(min_x + 10, min(cx, max_x - 10))  # Padding
            cy = max(min_y + 10, min(cy, max_y - 10))
            
            number = row * num_cols + col + 1
            cv2.putText(numbered_img, str(number), (cx, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
    
    return numbered_img

def overlay_edges(image_path, edges, alpha=0.3, color=(0, 255, 0)):
    """
    Overlay edges on the original image.
   
    Args:
        image_path: Path to original image
        edges: Edge map (binary image)
        alpha: Transparency of overlay (0-1)
        color: RGB color for edges
   
    Returns:
        Image with edges overlaid
    """
    img = cv2.imread(image_path)
   
    # Create colored edge map
    edges_colored = np.zeros_like(img)
    edges_colored[edges > 0] = color
   
    # Blend with original
    overlay = cv2.addWeighted(img, 1-alpha, edges_colored, alpha, 0)
   
    return overlay

def detect_halos(img_path, output_dir):
    """
    Detect text halos from color2.py logic.
    Saves halo visualization images to output_dir.
    Updated: No resizing for input to maintain high quality.
    """
    # Load the image without resizing to maintain quality
    img = mpimg.imread(img_path)
    print("Loaded full-size image for halo detection to maintain quality.")

    gray = np.dot(img[..., :3], [0.2989, 0.5870, 0.1140])

    # Adaptive thresholding for text mask (local Gaussian blur)
    blurred = gaussian_filter(gray, sigma=15)
    text_threshold_local = blurred - 20  # Darker than local mean; tune -20 for sensitivity
    text_mask = gray < text_threshold_local
    text_mask = text_mask.astype(bool)  # Ensure bool

    # Halo candidates: brighter areas
    halo_threshold = 110  # Tune: lower for fainter halos
    halo_candidate = gray > halo_threshold

    # Morphological gradient for text edges: dilate - erode
    struct = np.ones((3, 3), dtype=bool)
    dilated = ndimage.binary_dilation(text_mask, structure=struct)
    eroded = ndimage.binary_erosion(text_mask, structure=struct)
    morph_gradient = dilated ^ eroded  # XOR for boundary pixels

    # Robust halo: edge pixels that are bright (halo) or adjacent bright
    edge_halo = morph_gradient & halo_candidate
    dilated_edge = ndimage.binary_dilation(morph_gradient, structure=struct)
    adjacent_halo = dilated_edge & halo_candidate & ~text_mask & ~morph_gradient

    halo_robust = edge_halo | adjacent_halo

    # Clean: Remove small noise
    labels, num_features = ndimage.label(halo_robust)
    sizes = ndimage.sum(halo_robust, labels, range(num_features + 1))
    mask_clean = sizes[labels] > 5  # Keep components >5 pixels
    halo_clean = halo_robust & mask_clean

    # Stats
    total_halo_pixels = np.sum(halo_clean)
    num_regions = np.sum(sizes[1:] > 5)
    print(f'Significant halo regions (>5 pixels): {num_regions}')
    print(f'Total halo pixels: {total_halo_pixels}')
    if num_regions > 0:
        top_sizes = sorted(sizes[1:][sizes[1:] > 5], reverse=True)[:5]
        print(f'Top 5 halo sizes: {top_sizes}')

    # Efficient saving: Use direct array saving to avoid matplotlib figure memory overhead
    # 1. Grayscale Original
    plt.imsave(os.path.join(output_dir, 'original_gray_halo.png'), gray, cmap='gray', dpi=300)

    # 2. Halo White Overlay (grayscale with halos set to 255)
    overlay_white = np.copy(gray.astype(np.uint8))
    overlay_white[halo_clean] = 255
    plt.imsave(os.path.join(output_dir, 'halo_white_overlay.png'), overlay_white, cmap='gray', dpi=300)

    # 3. Red-Tinted Overlay on Color Image (modify RGB directly)
    colored_overlay = np.copy(img)
    if len(colored_overlay.shape) == 3 and colored_overlay.shape[2] == 3:
        # Set halo pixels to red [1,0,0] (assuming normalized 0-1; adjust if uint8)
        if colored_overlay.dtype == np.float32 or colored_overlay.dtype == np.float64:
            colored_overlay[halo_clean] = [1, 0, 0]
        else:  # uint8, scale 255
            colored_overlay[halo_clean] = [255, 0, 0]
    plt.imsave(os.path.join(output_dir, 'halo_red_overlay.png'), colored_overlay, dpi=300)

    # 4. Text Edges (Morph Gradient for debugging)
    plt.imsave(os.path.join(output_dir, 'text_edges_halo.png'), morph_gradient.astype(np.uint8), cmap='gray', dpi=300)

    print(f"\nHalo detection images saved to {output_dir}:")
    print("- original_gray_halo.png")
    print("- halo_white_overlay.png")
    print("- halo_red_overlay.png")
    print("- text_edges_halo.png (shows raw text boundaries)")

if __name__ == "__main__":
    # Create outputs folder
    output_dir = 'outputs'
    os.makedirs(output_dir, exist_ok=True)
    print(f"Output directory: {output_dir}")
   
    if len(sys.argv) < 2:
        print("Combined Grid Completion, Numbering, Halo Detection Tool (High Quality Mode)")
        print("=" * 70)
        print("\nUsage:")
        print("  python combined_script.py <input_image> [output] [method] [threshold1] [threshold2]")
        print("\nMethods:")
        print("  canny     - Canny edge detection (default)")
        print("  sobel     - Sobel gradient")
        print("  laplacian - Laplacian edge detection")
        print("  scharr    - Scharr gradient")
        print("\nWorkflow:")
        print(" 1. Detects edges and displays original image for point selection.")
        print(" 2. Click 4 points to define quadrilateral ROI (red dots, blue lines).")
        print(" 3. Press any key to proceed after selection.")
        print(" 4. Masks ROI in edges.")
        print(" 5. Detects/matches grid lines using projections and draws completing segments in ROI gap.")
        print(" 6. Draws top, bottom, left, and right boundary lines of ROI bbox.")
        print(" 7. Overlays grid on original.")
        print(" 8. Adds sequential numbers to grid cells.")
        print(" 9. Runs text halo detection on the numbered overlay.")
        print(f"All outputs saved to '{output_dir}' folder in full size/high quality (no resizing).")
        print("Saves: masked_edges.png, completed_grid.png, numbered_overlay.png, and halo images.")
        print("\nExample:")
        print("  python combined_script.py m1.jpg masked_edges.png canny 50 100")
        sys.exit(0)
   
    input_path = sys.argv[1]
    output_path = os.path.join(output_dir, sys.argv[2] if len(sys.argv) > 2 else 'masked_edges.png')
    method = sys.argv[3] if len(sys.argv) > 3 else 'canny'
    threshold1 = int(sys.argv[4]) if len(sys.argv) > 4 else 50
    threshold2 = int(sys.argv[5]) if len(sys.argv) > 5 else 100
   

    print(f"Detecting edges using {method} method...")
    edges, original = detect_edges(input_path, method, threshold1, threshold2)
    
    # Resize for interactive point selection (max 800px side for easier clicking)
    height, width = original.shape[:2]
    display_max = 800
    display_scale = min(display_max / height, display_max / width)
    display_width = int(width * display_scale)
    display_height = int(height * display_scale)
    display_img = cv2.resize(original, (display_width, display_height))
    img_copy = display_img.copy()
    print(f"Display resized to {display_width}x{display_height} (scale: {display_scale:.3f}) for easier point selection.")
    
    # Interactive point selection on resized display image
    cv2.namedWindow('Select 4 Points')
    cv2.setMouseCallback('Select 4 Points', mouse_callback)
    
    print("Instructions:")
    print(" - Click 4 points on the image to define the ROI quadrilateral.")
    print(" - After 4th point, the selection will be highlighted.")
    print(" - Press any key to apply mask and proceed.")
    print(" - Close window with ESC or 'q' if needed.")
    
    while len(points) < 4:
        cv2.imshow('Select 4 Points', img_copy)
        key = cv2.waitKey(1) & 0xFF
        if key == 27 or key == ord('q'):  # ESC or q to quit
            print("Selection cancelled.")
            cv2.destroyAllWindows()
            sys.exit(0)
    
    cv2.destroyAllWindows()
    
    print("Applying ROI mask to edges...")
    masked_edges = remove_roi(edges, points, target_image_type='edges')
    
    cv2.imwrite(output_path, masked_edges)
    print(f"✓ Masked edges (ROI removed) saved to {output_path}")
    
    # Calculate statistics (original edges vs masked)
    original_edge_pixels = np.count_nonzero(edges)
    masked_edge_pixels = np.count_nonzero(masked_edges)
    total_pixels = edges.shape[0] * edges.shape[1]
    original_percentage = (original_edge_pixels / total_pixels) * 100
    masked_percentage = (masked_edge_pixels / total_pixels) * 100
    print(f"✓ Original edge pixels: {original_edge_pixels:,} ({original_percentage:.2f}%)")
    print(f"✓ Masked edge pixels: {masked_edge_pixels:,} ({masked_percentage:.2f}%)")
    print(f"✓ Removed edge pixels: {original_edge_pixels - masked_edge_pixels:,}")
    
    # Complete the grid
    print("\nCompleting grid lines...")
    h, w = edges.shape
    completed_edges, horz_ys, vert_xs = complete_grid(masked_edges, points, h, w)
    completed_path = os.path.join(output_dir, 'completed_grid.png')
    cv2.imwrite(completed_path, completed_edges)
    print(f"✓ Completed grid edges saved to {completed_path}")
    
    # Overlay the completed grid on original image
    overlay_path = os.path.join(output_dir, 'overlay.png')
    overlay = overlay_edges(input_path, completed_edges)
    cv2.imwrite(overlay_path, overlay)
    
    # Add numbers to the overlay
    print("\nAdding numbers to grid cells...")
    numbered_overlay = add_numbers_to_grid(overlay, horz_ys, vert_xs, points)
    numbered_path = os.path.join(output_dir, 'numbered_overlay.png')
    cv2.imwrite(numbered_path, numbered_overlay)
    print(f"✓ Numbered grid overlaid on original saved to {numbered_path}")
    
    # Run text halo detection on the numbered overlay (full size, high quality)
    print("\nRunning text halo detection...")
    detect_halos(numbered_path, output_dir)
    
    print(f"✓ All processing complete! Full-size high-quality images saved to '{output_dir}'.")