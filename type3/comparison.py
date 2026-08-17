import matplotlib.image as mpimg
import numpy as np
from scipy import ndimage
from scipy.ndimage import gaussian_filter, zoom
import matplotlib.pyplot as plt
from pathlib import Path
import json

class WhiteEmbossingDetector:
    """
    Specialized detector for white embossing shade around black text on gray background.
    Optimized for Jordan t-shirt labels.
    """
    
    def __init__(self, max_size=2000):
        self.max_size = max_size
    
    def load_and_preprocess(self, image_path, target_shape=None):
        """Load and optionally resize image."""
        img = mpimg.imread(image_path)
        
        if img.dtype == np.uint8:
            img = img.astype(float) / 255.0
        
        if target_shape is not None:
            target_height, target_width = target_shape
            current_height, current_width = img.shape[:2]
            
            if (current_height != target_height) or (current_width != target_width):
                scale_h = target_height / current_height
                scale_w = target_width / current_width
                img = zoom(img, (scale_h, scale_w, 1), order=1)
                print(f"  Resized to {target_height}x{target_width}")
            return img
        
        if img.shape[0] > self.max_size or img.shape[1] > self.max_size:
            height, width = img.shape[:2]
            scale = min(self.max_size / height, self.max_size / width)
            new_height, new_width = int(height * scale), int(width * scale)
            img = zoom(img, (scale, scale, 1), order=1)
            print(f"  Resized to {new_height}x{new_width}")
        
        return img
    
    def detect_embossing_shade(self, img):
        """
        Detect white embossing shade around ALL text elements: letters, symbols, icons, logos.
        Includes detailed thickness analysis.
        
        Strategy:
        1. Identify ALL dark elements (text, symbols, icons, logos)
        2. Identify gray background (middle range)
        3. Detect white embossing at multiple distances from elements
        4. Analyze thickness distribution in detail
        """
        # Convert to grayscale (0-255)
        gray = np.dot(img[..., :3], [0.2989, 0.5870, 0.1140])
        gray_255 = (gray * 255).astype(np.uint8)
        
        print(f"\n  📊 Image Analysis:")
        print(f"    Brightness range: {gray_255.min()} - {gray_255.max()}")
        print(f"    Mean: {gray_255.mean():.1f}, Median: {np.median(gray_255):.1f}")
        
        # Calculate brightness percentiles
        p05 = np.percentile(gray_255, 5)
        p10 = np.percentile(gray_255, 10)
        p20 = np.percentile(gray_255, 20)
        p50 = np.percentile(gray_255, 50)
        p80 = np.percentile(gray_255, 80)
        p90 = np.percentile(gray_255, 90)
        p95 = np.percentile(gray_255, 95)
        print(f"    Percentiles - 5th: {p05:.0f}, 10th: {p10:.0f}, 20th: {p20:.0f}")
        print(f"                  50th: {p50:.0f}, 80th: {p80:.0f}, 90th: {p90:.0f}, 95th: {p95:.0f}")
        
        # Step 1: Detect ALL DARK ELEMENTS (text, symbols, icons, logos)
        # Use multiple thresholds to capture everything
        
        # Very dark elements (main text, solid symbols)
        very_dark_threshold = p10 + 10
        very_dark_elements = gray_255 < very_dark_threshold
        
        # Medium dark elements (lighter text, thin symbols, logo edges)
        medium_dark_threshold = p20 + 15
        medium_dark_elements = gray_255 < medium_dark_threshold
        
        # Combine all dark elements
        all_dark_elements = very_dark_elements | medium_dark_elements
        
        # Clean the mask to remove tiny noise but keep small symbols
        all_dark_elements = ndimage.binary_opening(all_dark_elements, structure=np.ones((2, 2)))
        all_dark_elements = ndimage.binary_closing(all_dark_elements, structure=np.ones((4, 4)))
        
        # Remove very small isolated pixels (likely noise, not symbols)
        labels_dark, num_dark = ndimage.label(all_dark_elements)
        sizes_dark = ndimage.sum(all_dark_elements, labels_dark, range(num_dark + 1))
        # Keep regions with at least 3 pixels (captures even small symbols)
        all_dark_elements = all_dark_elements & (sizes_dark[labels_dark] > 3)
        
        total_dark_pixels = np.sum(all_dark_elements)
        print(f"\n  🔤 Element Detection:")
        print(f"    All dark elements (text/symbols/icons): {total_dark_pixels:,} pixels")
        
        # Step 2: Detect WHITE EMBOSSING at multiple thickness levels
        # We'll search at different distances from elements to capture full thickness
        
        print(f"\n  ⚪ White Embossing Detection:")
        
        # Define white threshold
        white_threshold = p90 - 3  # Very bright pixels
        white_pixels = gray_255 >= white_threshold
        print(f"    White threshold: {white_threshold:.0f}")
        print(f"    Total white pixels in image: {np.sum(white_pixels):,}")
        
        # Create search zones at different distances
        struct = np.ones((3, 3))
        
        # Zone 1: Very close to elements (1-3 pixels) - thin embossing
        zone1 = all_dark_elements.copy()
        for _ in range(3):
            zone1 = ndimage.binary_dilation(zone1, structure=struct)
        zone1 = zone1 & ~all_dark_elements
        
        # Zone 2: Medium distance (4-6 pixels) - medium embossing
        zone2 = all_dark_elements.copy()
        for _ in range(6):
            zone2 = ndimage.binary_dilation(zone2, structure=struct)
        zone2 = zone2 & ~all_dark_elements
        zone2_only = zone2 & ~zone1
        
        # Zone 3: Far distance (7-10 pixels) - thick embossing
        zone3 = all_dark_elements.copy()
        for _ in range(10):
            zone3 = ndimage.binary_dilation(zone3, structure=struct)
        zone3 = zone3 & ~all_dark_elements
        zone3_only = zone3 & ~zone1 & ~zone2
        
        # Zone 4: Very far (11-15 pixels) - extra thick embossing (rare)
        zone4 = all_dark_elements.copy()
        for _ in range(15):
            zone4 = ndimage.binary_dilation(zone4, structure=struct)
        zone4 = zone4 & ~all_dark_elements
        zone4_only = zone4 & ~zone1 & ~zone2 & ~zone3
        
        total_search_zone = zone1 | zone2 | zone3 | zone4
        
        print(f"    Search zone pixels: {np.sum(total_search_zone):,}")
        
        # Find white pixels in each zone
        emboss_zone1 = white_pixels & zone1
        emboss_zone2 = white_pixels & zone2_only
        emboss_zone3 = white_pixels & zone3_only
        emboss_zone4 = white_pixels & zone4_only
        
        print(f"    Zone 1 (thin, 1-3px): {np.sum(emboss_zone1):,} white pixels")
        print(f"    Zone 2 (medium, 4-6px): {np.sum(emboss_zone2):,} white pixels")
        print(f"    Zone 3 (thick, 7-10px): {np.sum(emboss_zone3):,} white pixels")
        print(f"    Zone 4 (extra thick, 11-15px): {np.sum(emboss_zone4):,} white pixels")
        
        # Combine all embossing
        embossing_mask = emboss_zone1 | emboss_zone2 | emboss_zone3 | emboss_zone4
        
        print(f"    Total embossing before refinement: {np.sum(embossing_mask):,}")
        
        # Step 3: Refine using local contrast
        # True embossing should be brighter than immediate surroundings
        local_mean = gaussian_filter(gray_255.astype(float), sigma=4)
        brighter_than_local = gray_255 > (local_mean + 5)
        
        # Apply local contrast filter
        embossing_mask_refined = embossing_mask & brighter_than_local
        
        print(f"    After local contrast filter: {np.sum(embossing_mask_refined):,}")
        
        # Step 4: Clean up tiny isolated noise
        labels, num = ndimage.label(embossing_mask_refined)
        sizes = ndimage.sum(embossing_mask_refined, labels, range(num + 1))
        # Keep regions with at least 3 pixels (small but valid embossing)
        embossing_mask_clean = embossing_mask_refined & (sizes[labels] > 3)
        
        final_embossing_pixels = np.sum(embossing_mask_clean)
        print(f"    Final embossing pixels: {final_embossing_pixels:,}")
        
        # Step 5: DETAILED THICKNESS ANALYSIS
        print(f"\n  📏 Thickness Analysis:")
        
        # Calculate exact distance from each embossing pixel to nearest dark element
        dist_from_elements = ndimage.distance_transform_edt(~all_dark_elements)
        
        if final_embossing_pixels > 0:
            # Get distances for all embossing pixels
            embossing_distances = dist_from_elements[embossing_mask_clean]
            
            # Thickness statistics
            avg_thickness = np.mean(embossing_distances)
            median_thickness = np.median(embossing_distances)
            min_thickness = np.min(embossing_distances)
            max_thickness = np.max(embossing_distances)
            std_thickness = np.std(embossing_distances)
            
            print(f"    Average thickness: {avg_thickness:.2f} pixels")
            print(f"    Median thickness: {median_thickness:.2f} pixels")
            print(f"    Min thickness: {min_thickness:.2f} pixels")
            print(f"    Max thickness: {max_thickness:.2f} pixels")
            print(f"    Std deviation: {std_thickness:.2f} pixels")
            
            # Thickness distribution (detailed)
            very_thin = np.sum((embossing_distances > 0) & (embossing_distances <= 2))
            thin = np.sum((embossing_distances > 2) & (embossing_distances <= 4))
            medium = np.sum((embossing_distances > 4) & (embossing_distances <= 6))
            thick = np.sum((embossing_distances > 6) & (embossing_distances <= 8))
            very_thick = np.sum((embossing_distances > 8) & (embossing_distances <= 10))
            extra_thick = np.sum(embossing_distances > 10)
            
            print(f"\n    Thickness Distribution:")
            print(f"      Very thin (0-2px): {very_thin:,} pixels ({very_thin/final_embossing_pixels*100:.1f}%)")
            print(f"      Thin (2-4px): {thin:,} pixels ({thin/final_embossing_pixels*100:.1f}%)")
            print(f"      Medium (4-6px): {medium:,} pixels ({medium/final_embossing_pixels*100:.1f}%)")
            print(f"      Thick (6-8px): {thick:,} pixels ({thick/final_embossing_pixels*100:.1f}%)")
            print(f"      Very thick (8-10px): {very_thick:,} pixels ({very_thick/final_embossing_pixels*100:.1f}%)")
            print(f"      Extra thick (>10px): {extra_thick:,} pixels ({extra_thick/final_embossing_pixels*100:.1f}%)")
            
            # Calculate "dominant thickness" - where most embossing is
            thickness_categories = {
                'very_thin': very_thin,
                'thin': thin,
                'medium': medium,
                'thick': thick,
                'very_thick': very_thick,
                'extra_thick': extra_thick
            }
            dominant_thickness_category = max(thickness_categories, key=thickness_categories.get)
            
            # Intensity analysis
            embossing_intensities = gray_255[embossing_mask_clean]
            avg_intensity = np.mean(embossing_intensities)
            min_intensity = np.min(embossing_intensities)
            max_intensity = np.max(embossing_intensities)
            
            # Count regions
            emb_labels, num_regions = ndimage.label(embossing_mask_clean)
            region_sizes = ndimage.sum(embossing_mask_clean, emb_labels, range(1, num_regions + 1))
            largest_region = np.max(region_sizes) if len(region_sizes) > 0 else 0
            avg_region_size = np.mean(region_sizes) if len(region_sizes) > 0 else 0
            
        else:
            avg_thickness = 0
            median_thickness = 0
            min_thickness = 0
            max_thickness = 0
            std_thickness = 0
            very_thin = thin = medium = thick = very_thick = extra_thick = 0
            dominant_thickness_category = 'none'
            avg_intensity = 0
            min_intensity = 0
            max_intensity = 0
            num_regions = 0
            largest_region = 0
            avg_region_size = 0
        
        # Calculate final statistics
        total_pixels = gray_255.size
        embossing_percentage = (final_embossing_pixels / total_pixels) * 100
        embossing_to_element_ratio = (final_embossing_pixels / total_dark_pixels) if total_dark_pixels > 0 else 0
        
        stats = {
            'total_pixels': int(total_pixels),
            'element_pixels': int(total_dark_pixels),
            'embossing_pixels': int(final_embossing_pixels),
            'embossing_percentage': float(embossing_percentage),
            'embossing_to_element_ratio': float(embossing_to_element_ratio),
            
            # Thickness statistics
            'avg_thickness': float(avg_thickness),
            'median_thickness': float(median_thickness),
            'min_thickness': float(min_thickness),
            'max_thickness': float(max_thickness),
            'std_thickness': float(std_thickness),
            'dominant_thickness_category': dominant_thickness_category,
            
            # Thickness distribution
            'very_thin_pixels': int(very_thin),
            'thin_pixels': int(thin),
            'medium_pixels': int(medium),
            'thick_pixels': int(thick),
            'very_thick_pixels': int(very_thick),
            'extra_thick_pixels': int(extra_thick),
            
            # Intensity
            'avg_intensity': float(avg_intensity),
            'min_intensity': int(min_intensity),
            'max_intensity': int(max_intensity),
            
            # Regions
            'num_regions': int(num_regions),
            'largest_region': int(largest_region),
            'avg_region_size': float(avg_region_size),
            
            # Thresholds used
            'white_threshold_used': float(white_threshold),
            'very_dark_threshold': float(very_dark_threshold),
            'medium_dark_threshold': float(medium_dark_threshold),
            
            'image_brightness': {
                'min': int(gray_255.min()),
                'max': int(gray_255.max()),
                'mean': float(gray_255.mean()),
                'p05': float(p05),
                'p10': float(p10),
                'p20': float(p20),
                'p50': float(p50),
                'p80': float(p80),
                'p90': float(p90),
                'p95': float(p95)
            }
        }
        
        # Create thickness map for visualization
        thickness_map = np.zeros_like(gray_255, dtype=float)
        thickness_map[embossing_mask_clean] = dist_from_elements[embossing_mask_clean]
        
        return {
            'embossing_mask': embossing_mask_clean,
            'element_mask': all_dark_elements,
            'white_pixels': white_pixels,
            'search_zone': total_search_zone,
            'thickness_map': thickness_map,
            'gray': gray_255,
            'original': img,
            'stats': stats
        }
    
    def compare_embossing(self, ref_result, test_result):
        """Compare embossing between reference and test image with detailed thickness analysis."""
        ref_stats = ref_result['stats']
        test_stats = test_result['stats']
        
        # Percentage comparison
        percentage_diff = test_stats['embossing_percentage'] - ref_stats['embossing_percentage']
        
        if ref_stats['embossing_percentage'] > 0:
            percentage_ratio = (test_stats['embossing_percentage'] / ref_stats['embossing_percentage']) * 100
        else:
            percentage_ratio = 0
        
        # Ratio comparison
        ratio_diff = test_stats['embossing_to_element_ratio'] - ref_stats['embossing_to_element_ratio']
        
        # DETAILED THICKNESS COMPARISON
        thickness_diff = test_stats['avg_thickness'] - ref_stats['avg_thickness']
        median_thickness_diff = test_stats['median_thickness'] - ref_stats['median_thickness']
        max_thickness_diff = test_stats['max_thickness'] - ref_stats['max_thickness']
        
        # Thickness category distribution comparison
        ref_thin_pct = (ref_stats['thin_pixels'] + ref_stats['very_thin_pixels']) / ref_stats['embossing_pixels'] * 100 if ref_stats['embossing_pixels'] > 0 else 0
        test_thin_pct = (test_stats['thin_pixels'] + test_stats['very_thin_pixels']) / test_stats['embossing_pixels'] * 100 if test_stats['embossing_pixels'] > 0 else 0
        
        ref_thick_pct = (ref_stats['thick_pixels'] + ref_stats['very_thick_pixels'] + ref_stats['extra_thick_pixels']) / ref_stats['embossing_pixels'] * 100 if ref_stats['embossing_pixels'] > 0 else 0
        test_thick_pct = (test_stats['thick_pixels'] + test_stats['very_thick_pixels'] + test_stats['extra_thick_pixels']) / test_stats['embossing_pixels'] * 100 if test_stats['embossing_pixels'] > 0 else 0
        
        thin_distribution_diff = test_thin_pct - ref_thin_pct
        thick_distribution_diff = test_thick_pct - ref_thick_pct
        
        # Verdict based on coverage AND thickness
        # IMPORTANT: Less embossing or thinner = GOOD/PASS
        #            More embossing or thicker = BAD/FAIL
        
        coverage_excess = percentage_diff > 0.2  # More than 0.2% extra
        coverage_less = percentage_diff < -0.2   # Less than -0.2% (this is OK)
        
        thickness_excess = thickness_diff > 1.0  # More than 1px thicker
        thickness_less = thickness_diff < -1.0   # More than 1px thinner (this is OK)
        
        # Classification logic
        if coverage_excess and thickness_excess:
            # TOO MUCH and TOO THICK - BAD
            verdict = "❌ FAIL - TOO MUCH embossing AND TOO THICK"
            quality = "FAIL - EXCESS"
        elif coverage_excess and not thickness_excess:
            # TOO MUCH coverage but thickness OK
            if percentage_diff > 0.5:
                verdict = "❌ FAIL - TOO MUCH embossing coverage"
                quality = "FAIL - EXCESS COVERAGE"
            else:
                verdict = "⚠️ WARNING - Slightly more embossing"
                quality = "MARGINAL - MORE COVERAGE"
        elif thickness_excess and not coverage_excess:
            # TOO THICK but coverage OK
            if thickness_diff > 2.0:
                verdict = "❌ FAIL - Embossing TOO THICK"
                quality = "FAIL - TOO THICK"
            else:
                verdict = "⚠️ WARNING - Slightly thicker embossing"
                quality = "MARGINAL - THICKER"
        elif coverage_less or thickness_less:
            # LESS or THINNER - This is GOOD (better quality)
            if coverage_less and thickness_less:
                verdict = "✅ PASS - Less embossing and thinner (Good quality)"
                quality = "PASS - MINIMAL"
            elif coverage_less:
                verdict = "✅ PASS - Less embossing coverage (Good)"
                quality = "PASS - LESS COVERAGE"
            else:
                verdict = "✅ PASS - Thinner embossing (Good)"
                quality = "PASS - THINNER"
        else:
            # Similar to reference
            verdict = "✅ PASS - Similar to reference"
            quality = "PASS - SIMILAR"
        
        return {
            'reference_percentage': float(ref_stats['embossing_percentage']),
            'test_percentage': float(test_stats['embossing_percentage']),
            'percentage_difference': float(percentage_diff),
            'percentage_ratio': float(percentage_ratio),
            'reference_pixels': int(ref_stats['embossing_pixels']),
            'test_pixels': int(test_stats['embossing_pixels']),
            'pixel_difference': int(test_stats['embossing_pixels'] - ref_stats['embossing_pixels']),
            
            'reference_ratio': float(ref_stats['embossing_to_element_ratio']),
            'test_ratio': float(test_stats['embossing_to_element_ratio']),
            'ratio_difference': float(ratio_diff),
            
            # Detailed thickness comparison
            'reference_avg_thickness': float(ref_stats['avg_thickness']),
            'test_avg_thickness': float(test_stats['avg_thickness']),
            'avg_thickness_difference': float(thickness_diff),
            
            'reference_median_thickness': float(ref_stats['median_thickness']),
            'test_median_thickness': float(test_stats['median_thickness']),
            'median_thickness_difference': float(median_thickness_diff),
            
            'reference_max_thickness': float(ref_stats['max_thickness']),
            'test_max_thickness': float(test_stats['max_thickness']),
            'max_thickness_difference': float(max_thickness_diff),
            
            'reference_dominant_thickness': ref_stats['dominant_thickness_category'],
            'test_dominant_thickness': test_stats['dominant_thickness_category'],
            
            'reference_thin_percentage': float(ref_thin_pct),
            'test_thin_percentage': float(test_thin_pct),
            'thin_distribution_difference': float(thin_distribution_diff),
            
            'reference_thick_percentage': float(ref_thick_pct),
            'test_thick_percentage': float(test_thick_pct),
            'thick_distribution_difference': float(thick_distribution_diff),
            
            'verdict': verdict,
            'quality': quality
        }
    
    def save_visualization(self, result, output_path, title="White Embossing Detection"):
        """Save detailed visualization with thickness analysis."""
        fig, axes = plt.subplots(3, 3, figsize=(18, 15))
        
        img = result['original']
        gray = result['gray']
        element_mask = result['element_mask']
        embossing_mask = result['embossing_mask']
        thickness_map = result['thickness_map']
        white_pixels = result['white_pixels']
        search_zone = result['search_zone']
        stats = result['stats']
        
        # Row 1
        axes[0, 0].imshow(img)
        axes[0, 0].set_title('Original Image', fontsize=12, fontweight='bold')
        axes[0, 0].axis('off')
        
        axes[0, 1].imshow(gray, cmap='gray', vmin=0, vmax=255)
        axes[0, 1].set_title(f'Grayscale\nRange: {stats["image_brightness"]["min"]}-{stats["image_brightness"]["max"]}', 
                            fontsize=11)
        axes[0, 1].axis('off')
        
        axes[0, 2].imshow(element_mask, cmap='gray')
        axes[0, 2].set_title(f'ALL Elements Detected\n(Text, Symbols, Icons, Logos)\n{stats["element_pixels"]:,} pixels', 
                            fontsize=10, fontweight='bold')
        axes[0, 2].axis('off')
        
        # Row 2
        axes[1, 0].imshow(white_pixels, cmap='gray')
        axes[1, 0].set_title(f'White Pixels\nThreshold: {stats["white_threshold_used"]:.0f}', 
                            fontsize=11)
        axes[1, 0].axis('off')
        
        axes[1, 1].imshow(embossing_mask, cmap='gray')
        axes[1, 1].set_title(f'White Embossing Detected\n{stats["embossing_pixels"]:,} pixels', 
                            fontsize=11, fontweight='bold', color='red')
        axes[1, 1].axis('off')
        
        # Thickness map with colorbar
        thickness_plot = axes[1, 2].imshow(thickness_map, cmap='hot', vmin=0, vmax=15)
        axes[1, 2].set_title(f'Thickness Map\nAvg: {stats["avg_thickness"]:.2f}px, Max: {stats["max_thickness"]:.1f}px', 
                            fontsize=10, fontweight='bold')
        axes[1, 2].axis('off')
        plt.colorbar(thickness_plot, ax=axes[1, 2], label='Distance (pixels)', fraction=0.046)
        
        # Row 3
        # Original with RED marking
        overlay_red = np.copy(img)
        overlay_red[embossing_mask] = [1.0, 0.0, 0.0]
        axes[2, 0].imshow(overlay_red)
        axes[2, 0].set_title(f'Embossing in RED\n{stats["embossing_percentage"]:.3f}% coverage', 
                            fontsize=11, fontweight='bold', color='red')
        axes[2, 0].axis('off')
        
        # Original with CYAN marking
        overlay_cyan = np.copy(img)
        overlay_cyan[embossing_mask] = [0.0, 1.0, 1.0]
        axes[2, 1].imshow(overlay_cyan)
        axes[2, 1].set_title(f'Embossing in CYAN\n{stats["embossing_percentage"]:.3f}% coverage', 
                            fontsize=11, fontweight='bold', color='cyan')
        axes[2, 1].axis('off')
        
        # Statistics
        axes[2, 2].axis('off')
        stats_text = f"""EMBOSSING STATISTICS
{'='*40}

COVERAGE:
  Percentage: {stats['embossing_percentage']:.3f}%
  Pixels: {stats['embossing_pixels']:,}
  Element Pixels: {stats['element_pixels']:,}
  Emboss/Element Ratio: {stats['embossing_to_element_ratio']:.3f}

THICKNESS ANALYSIS:
  Average: {stats['avg_thickness']:.2f} px
  Median: {stats['median_thickness']:.2f} px
  Max: {stats['max_thickness']:.1f} px
  Std Dev: {stats['std_thickness']:.2f} px
  Dominant: {stats['dominant_thickness_category']}

THICKNESS DISTRIBUTION:
  Very thin (0-2px): {stats['very_thin_pixels']:,}
  Thin (2-4px): {stats['thin_pixels']:,}
  Medium (4-6px): {stats['medium_pixels']:,}
  Thick (6-8px): {stats['thick_pixels']:,}
  Very thick (8-10px): {stats['very_thick_pixels']:,}
  Extra thick (>10px): {stats['extra_thick_pixels']:,}

INTENSITY:
  Avg: {stats['avg_intensity']:.1f}/255
  Range: {stats['min_intensity']}-{stats['max_intensity']}

REGIONS:
  Count: {stats['num_regions']}
"""
        axes[2, 2].text(0.05, 0.95, stats_text, fontsize=8, verticalalignment='top',
                       family='monospace', transform=axes[2, 2].transAxes)
        
        plt.suptitle(title, fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()


def process_jordan_labels(image_list, output_dir='claudeoutput'):
    """
    Process Jordan t-shirt label images to detect white embossing shade.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)
    
    detector = WhiteEmbossingDetector()
    
    reference_path = image_list[0]
    test_paths = image_list[1:]
    
    print("\n" + "="*90)
    print("WHITE EMBOSSING SHADE DETECTOR - Jordan T-Shirt Labels")
    print("="*90)
    print("Detects: White shade/outline around black text on gray background")
    print("="*90)
    print(f"Reference: {reference_path}")
    print(f"Test Images: {', '.join([Path(p).name for p in test_paths])}")
    print("="*90)
    
    # Process reference
    print(f"\n{'='*90}")
    print(f"📸 REFERENCE: {reference_path}")
    print(f"{'='*90}")
    ref_img = detector.load_and_preprocess(reference_path)
    ref_result = detector.detect_embossing_shade(ref_img)
    ref_shape = ref_result['gray'].shape
    
    print(f"\n✅ REFERENCE SUMMARY:")
    print(f"  White Embossing: {ref_result['stats']['embossing_percentage']:.3f}%")
    print(f"  Embossing Pixels: {ref_result['stats']['embossing_pixels']:,}")
    print(f"  Emboss/Element Ratio: {ref_result['stats']['embossing_to_element_ratio']:.3f}")
    print(f"  Avg Thickness: {ref_result['stats']['avg_thickness']:.2f} px")
    print(f"  Median Thickness: {ref_result['stats']['median_thickness']:.2f} px")
    print(f"  Dominant Thickness: {ref_result['stats']['dominant_thickness_category']}")
    
    detector.save_visualization(
        ref_result,
        output_dir / 'reference_embossing.png',
        title=f"Reference: {Path(reference_path).name}"
    )
    print(f"  ✓ Saved: reference_embossing.png")
    
    # Process test images
    results = []
    
    for i, test_path in enumerate(test_paths, 1):
        print(f"\n{'='*90}")
        print(f"📸 TEST {i}/{len(test_paths)}: {test_path}")
        print(f"{'='*90}")
        
        try:
            test_img = detector.load_and_preprocess(test_path, target_shape=ref_shape)
            test_result = detector.detect_embossing_shade(test_img)
            
            print(f"\n✅ TEST SUMMARY:")
            print(f"  White Embossing: {test_result['stats']['embossing_percentage']:.3f}%")
            print(f"  Embossing Pixels: {test_result['stats']['embossing_pixels']:,}")
            print(f"  Emboss/Element Ratio: {test_result['stats']['embossing_to_element_ratio']:.3f}")
            print(f"  Avg Thickness: {test_result['stats']['avg_thickness']:.2f} px")
            print(f"  Median Thickness: {test_result['stats']['median_thickness']:.2f} px")
            print(f"  Dominant Thickness: {test_result['stats']['dominant_thickness_category']}")
            
            # Compare
            comparison = detector.compare_embossing(ref_result, test_result)
            
            print(f"\n📊 COMPARISON WITH REFERENCE:")
            print(f"  Coverage Difference: {comparison['percentage_difference']:+.3f}%")
            print(f"  Pixel Difference: {comparison['pixel_difference']:+,}")
            print(f"  Ratio Difference: {comparison['ratio_difference']:+.3f}")
            print(f"  Avg Thickness Difference: {comparison['avg_thickness_difference']:+.2f} px")
            print(f"  Median Thickness Difference: {comparison['median_thickness_difference']:+.2f} px")
            print(f"  Max Thickness Difference: {comparison['max_thickness_difference']:+.2f} px")
            print(f"  Thin Distribution Diff: {comparison['thin_distribution_difference']:+.1f}%")
            print(f"  Thick Distribution Diff: {comparison['thick_distribution_difference']:+.1f}%")
            print(f"  {comparison['verdict']}")
            print(f"  Quality: {comparison['quality']}")
            
            test_name = Path(test_path).stem
            detector.save_visualization(
                test_result,
                output_dir / f'{test_name}_embossing.png',
                title=f"Test: {Path(test_path).name}"
            )
            print(f"  ✓ Saved: {test_name}_embossing.png")
            
            results.append({
                'image_name': test_name,
                'image_path': str(test_path),
                'stats': test_result['stats'],
                'comparison': comparison
            })
            
        except Exception as e:
            print(f"❌ Error: {e}")
            import traceback
            traceback.print_exc()
    
    # Ranking
    print(f"\n{'='*90}")
    print("🏆 RANKING BY WHITE EMBOSSING COVERAGE")
    print("="*90)
    
    all_images = [{
        'name': Path(reference_path).stem + " [REFERENCE]",
        'percentage': ref_result['stats']['embossing_percentage'],
        'pixels': ref_result['stats']['embossing_pixels'],
        'ratio': ref_result['stats']['embossing_to_element_ratio'],
        'avg_thickness': ref_result['stats']['avg_thickness'],
        'dominant_thickness': ref_result['stats']['dominant_thickness_category'],
        'is_ref': True
    }]
    
    for r in results:
        all_images.append({
            'name': r['image_name'],
            'percentage': r['stats']['embossing_percentage'],
            'pixels': r['stats']['embossing_pixels'],
            'ratio': r['stats']['embossing_to_element_ratio'],
            'avg_thickness': r['stats']['avg_thickness'],
            'dominant_thickness': r['stats']['dominant_thickness_category'],
            'quality': r['comparison']['quality'],
            'is_ref': False
        })
    
    sorted_images = sorted(all_images, key=lambda x: x['percentage'], reverse=True)
    
    print()
    for i, img in enumerate(sorted_images, 1):
        symbol = "👑" if i == 1 else f"{i}."
        quality_tag = f" [{img['quality']}]" if not img['is_ref'] else ""
        print(f"{symbol} {img['name']}{quality_tag}")
        print(f"   Coverage: {img['percentage']:.3f}%")
        print(f"   Pixels: {img['pixels']:,}")
        print(f"   Ratio: {img['ratio']:.3f}")
        print(f"   Avg Thickness: {img['avg_thickness']:.2f} px")
        print(f"   Dominant: {img['dominant_thickness']}")
        print()
    
    # Chart
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    names = [img['name'] for img in sorted_images]
    percentages = [img['percentage'] for img in sorted_images]
    ratios = [img['ratio'] for img in sorted_images]
    colors = ['#FF6B6B' if img['is_ref'] else '#4ECDC4' for img in sorted_images]
    
    # Coverage chart
    bars1 = ax1.barh(names, percentages, color=colors, edgecolor='black', linewidth=2)
    ax1.set_xlabel('White Embossing Coverage (%)', fontsize=13, fontweight='bold')
    ax1.set_title('White Embossing Coverage Comparison', fontsize=14, fontweight='bold')
    ax1.grid(axis='x', alpha=0.3)
    
    for bar, pct in zip(bars1, percentages):
        ax1.text(bar.get_width(), bar.get_y() + bar.get_height()/2,
                f'  {pct:.3f}%', va='center', fontweight='bold')
    
    # Ratio chart
    bars2 = ax2.barh(names, ratios, color=colors, edgecolor='black', linewidth=2)
    ax2.set_xlabel('Embossing-to-Text Ratio', fontsize=13, fontweight='bold')
    ax2.set_title('Embossing-to-Text Ratio Comparison', fontsize=14, fontweight='bold')
    ax2.grid(axis='x', alpha=0.3)
    
    for bar, ratio in zip(bars2, ratios):
        ax2.text(bar.get_width(), bar.get_y() + bar.get_height()/2,
                f'  {ratio:.3f}', va='center', fontweight='bold')
    
    from matplotlib.patches import Patch
    legend = [Patch(facecolor='#FF6B6B', edgecolor='black', label='Reference'),
              Patch(facecolor='#4ECDC4', edgecolor='black', label='Test Images')]
    fig.legend(handles=legend, loc='upper center', ncol=2, bbox_to_anchor=(0.5, 0.98))
    
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(output_dir / 'comparison_chart.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    # Save report
    report = {
        'reference': {'path': str(reference_path), 'stats': ref_result['stats']},
        'test_results': results,
        'ranking': sorted_images
    }
    
    with open(output_dir / 'embossing_report.json', 'w') as f:
        json.dump(report, f, indent=2)
    
    print("="*90)
    print("✅ PROCESSING COMPLETE")
    print("="*90)
    print(f"\n📁 Results saved to: {output_dir}/")
    print(f"  - Individual visualizations with RED marking")
    print(f"  - comparison_chart.png (ranking charts)")
    print(f"  - embossing_report.json (detailed data)")
    print("\n💡 Check the RED markings - they show the detected white embossing!")
    
    return report


if __name__ == "__main__":
    # Your images (put them in same folder as this script)
    image_list = [
        'ref.jpg',   # Reference image
        'b1.jpg',     # Test image 1
        'b2.jpg',
        'b3.jpg',
        'b4.jpg',
        'b5.jpg',
        'b26.jpg',
        'b27.jpg',
        'b28.jpg',
        'b29.jpg',
        'b30.jpg',
    # Test image 3
    ]
    
    print("\n" + "="*90)
    print("🎯 WHITE EMBOSSING SHADE DETECTOR")
    print("="*90)
    print("\nSpecifically designed for Jordan t-shirt labels:")
    print("  ✓ Gray background")
    print("  ✓ Black text/letters")
    print("  ✓ White embossing shade around text")
    print("="*90)
    
    report = process_jordan_labels(image_list, output_dir='type3')
    
    print("\n✅ All done! Check the 'claudeoutput' folder for results!")