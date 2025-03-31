# src/utils/visualization.py
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import cv2
from pathlib import Path
from typing import Optional, Tuple, List, Dict, Any
import logging
from skimage import measure
import io

logger = logging.getLogger(__name__)

class VisualQA:
    """
    Visualization utilities for quality assurance during preprocessing.
    """
    
    # Color maps for different visualization tasks
    COLORMAP_SEGMENTATION = LinearSegmentedColormap.from_list(
        'segmentation', [(0, 0, 0, 0), (1, 0, 0, 0.7)]
    )
    
    COLORMAP_CLASSES = {
        'normal': (0, 1, 0, 0.7),             # Green
        'partially_sclerotic': (1, 1, 0, 0.7), # Yellow
        'sclerotic': (1, 0, 0, 0.7),          # Red
        'uncertain': (1, 0, 1, 0.7)           # Magenta
    }
    
    @staticmethod
    def visualize_patch(
        image: np.ndarray, 
        mask: Optional[np.ndarray] = None,
        output_path: Optional[str] = None,
        dpi: int = 100
    ) -> None:
        """
        Visualize a patch with optional overlay of segmentation mask.
        
        Args:
            image: RGB image array
            mask: Binary segmentation mask
            output_path: Path to save visualization
            dpi: DPI for saved figure
        """
        plt.figure(figsize=(10, 10))
        
        if mask is not None:
            plt.subplot(1, 3, 1)
            plt.title("Original Image")
            plt.imshow(image)
            plt.axis('off')
            
            plt.subplot(1, 3, 2)
            plt.title("Segmentation Mask")
            plt.imshow(mask, cmap='gray')
            plt.axis('off')
            
            plt.subplot(1, 3, 3)
            plt.title("Overlay")
            plt.imshow(image)
            plt.imshow(mask, cmap=VisualQA.COLORMAP_SEGMENTATION)
            plt.axis('off')
        else:
            plt.imshow(image)
            plt.axis('off')
        
        plt.tight_layout()
        
        if output_path:
            plt.savefig(output_path, dpi=dpi, bbox_inches='tight')
            plt.close()
        else:
            plt.show()
    
    @staticmethod
    def visualize_class_distribution(
        class_counts: Dict[str, int],
        output_path: Optional[str] = None,
        title: str = "Class Distribution"
    ) -> None:
        """
        Visualize the distribution of classes in the dataset.
        
        Args:
            class_counts: Dictionary mapping class names to counts
            output_path: Path to save visualization
            title: Plot title
        """
        plt.figure(figsize=(10, 6))
        
        # Create bar chart
        classes = list(class_counts.keys())
        counts = [class_counts[cls] for cls in classes]
        
        bars = plt.bar(classes, counts, color=['green', 'yellow', 'red', 'magenta'])
        
        # Add counts above bars
        for bar, count in zip(bars, counts):
            plt.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 5,
                str(count),
                ha='center',
                fontweight='bold'
            )
        
        plt.title(title)
        plt.ylabel('Count')
        plt.xlabel('Class')
        
        # Add percentage labels
        total = sum(counts)
        for i, (bar, count) in enumerate(zip(bars, counts)):
            percentage = 100 * count / total
            plt.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() / 2,
                f"{percentage:.1f}%",
                ha='center',
                color='white',
                fontweight='bold'
            )
        
        plt.tight_layout()
        
        if output_path:
            plt.savefig(output_path, dpi=100, bbox_inches='tight')
            plt.close()
        else:
            plt.show()
    
    @staticmethod
    def create_montage(
        images: List[np.ndarray],
        masks: Optional[List[np.ndarray]] = None,
        rows: int = 4,
        cols: int = 6,
        output_path: Optional[str] = None,
        title: str = "Patch Montage"
    ) -> np.ndarray:
        """
        Create a montage of multiple patches for quick review.
        
        Args:
            images: List of image arrays
            masks: List of mask arrays (optional)
            rows: Number of rows in montage
            cols: Number of columns in montage
            output_path: Path to save visualization
            title: Plot title
            
        Returns:
            Montage array
        """
        if len(images) > rows * cols:
            logger.warning(f"Too many images ({len(images)}) for montage with {rows}x{cols} grid. "
                         f"Truncating to first {rows * cols} images.")
            images = images[:rows * cols]
            if masks is not None:
                masks = masks[:rows * cols]
        
        # Determine the size of each thumbnail
        if len(images) > 0:
            h, w, c = images[0].shape
        else:
            return np.zeros((100, 100, 3), dtype=np.uint8)
        
        # Create empty montage
        montage = np.zeros((rows * h, cols * w, c), dtype=np.uint8)
        
        # Fill montage with images
        idx = 0
        for i in range(rows):
            if idx >= len(images):
                break
                
            for j in range(cols):
                if idx >= len(images):
                    break
                    
                # Get current image
                img = images[idx]
                
                # If mask is provided, overlay it
                if masks is not None and idx < len(masks):
                    mask = masks[idx]
                    # Convert mask to RGB with alpha
                    mask_rgb = np.zeros((h, w, 4), dtype=np.float32)
                    mask_rgb[..., 0] = 1.0  # Red
                    mask_rgb[..., 3] = mask * 0.5  # Alpha
                    
                    # Convert image to RGBA
                    img_rgba = np.zeros((h, w, 4), dtype=np.float32)
                    img_rgba[..., :3] = img / 255.0
                    img_rgba[..., 3] = 1.0
                    
                    # Blend
                    blended = img_rgba * (1 - mask_rgb[..., 3:]) + mask_rgb * mask_rgb[..., 3:]
                    img = (blended[..., :3] * 255).astype(np.uint8)
                
                # Place in montage
                montage[i*h:(i+1)*h, j*w:(j+1)*w] = img
                idx += 1
        
        if output_path:
            cv2.imwrite(output_path, cv2.cvtColor(montage, cv2.COLOR_RGB2BGR))
        
        return montage
    
    @staticmethod
    def analyze_mask_quality(
        mask: np.ndarray,
        min_area: int = 100,
        max_area: int = 10000,
        circularity_threshold: float = 0.7
    ) -> Dict[str, Any]:
        """
        Analyze the quality of a segmentation mask.
        
        Args:
            mask: Binary segmentation mask
            min_area: Minimum acceptable area for a connected component
            max_area: Maximum acceptable area for a connected component
            circularity_threshold: Minimum circularity for a "good" glomerulus
            
        Returns:
            Dictionary with quality metrics
        """
        # Find connected components
        labels = measure.label(mask)
        props = measure.regionprops(labels)
        
        # Analyze each component
        results = {
            'num_components': len(props),
            'components': [],
            'has_large_component': False,
            'has_small_component': False,
            'has_non_circular': False
        }
        
        for prop in props:
            # Calculate circularity: 4*pi*area / perimeter^2
            area = prop.area
            perimeter = prop.perimeter
            circularity = 4 * np.pi * area / (perimeter * perimeter) if perimeter > 0 else 0
            
            component = {
                'area': area,
                'perimeter': perimeter,
                'circularity': circularity,
                'centroid': prop.centroid,
                'bbox': prop.bbox
            }
            
            # Check quality
            if area < min_area:
                component['issue'] = 'too_small'
                results['has_small_component'] = True
            elif area > max_area:
                component['issue'] = 'too_large'
                results['has_large_component'] = True
            elif circularity < circularity_threshold:
                component['issue'] = 'non_circular'
                results['has_non_circular'] = True
            else:
                component['issue'] = None
                
            results['components'].append(component)
            
        return results
    
    @staticmethod
    def create_report(
        slide_name: str,
        class_counts: Dict[str, int],
        sample_images: Dict[str, List[np.ndarray]],
        output_dir: str
    ) -> str:
        """
        Create a visual report for quality assurance.
        
        Args:
            slide_name: Name of the slide
            class_counts: Dictionary mapping class names to counts
            sample_images: Dictionary mapping class names to sample images
            output_dir: Directory to save report
            
        Returns:
            Path to the report file
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Generate components
        
        # 1. Class distribution
        dist_path = output_dir / f"{slide_name}_class_distribution.png"
        VisualQA.visualize_class_distribution(
            class_counts, 
            str(dist_path),
            title=f"Class Distribution - {slide_name}"
        )
        
        # 2. Sample montages for each class
        montage_paths = {}
        for class_name, images in sample_images.items():
            if images:
                montage_path = output_dir / f"{slide_name}_{class_name}_montage.png"
                VisualQA.create_montage(
                    images,
                    output_path=str(montage_path),
                    title=f"{class_name.capitalize()} Samples - {slide_name}"
                )
                montage_paths[class_name] = montage_path
        
        # 3. Create HTML report
        html_path = output_dir / f"{slide_name}_report.html"
        
        with open(html_path, 'w') as f:
            f.write(f"""
            <!DOCTYPE html>
            <html>
            <head>
                <title>QA Report - {slide_name}</title>
                <style>
                    body {{ font-family: Arial, sans-serif; margin: 20px; }}
                    h1, h2, h3 {{ color: #333; }}
                    .container {{ margin-bottom: 30px; }}
                    .image-container {{ margin: 10px 0; }}
                    img {{ max-width: 100%; }}
                    table {{ border-collapse: collapse; width: 100%; }}
                    th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
                    th {{ background-color: #f2f2f2; }}
                    tr:nth-child(even) {{ background-color: #f9f9f9; }}
                </style>
            </head>
            <body>
                <h1>Quality Assurance Report - {slide_name}</h1>
                <div class="container">
                    <h2>Class Distribution</h2>
                    <div class="image-container">
                        <img src="{dist_path.name}" alt="Class Distribution">
                    </div>
                    
                    <h2>Dataset Summary</h2>
                    <table>
                        <tr>
                            <th>Class</th>
                            <th>Count</th>
                            <th>Percentage</th>
                        </tr>
            """)
            
            total = sum(class_counts.values())
            for class_name, count in class_counts.items():
                percentage = 100 * count / total if total > 0 else 0
                f.write(f"""
                        <tr>
                            <td>{class_name.capitalize()}</td>
                            <td>{count}</td>
                            <td>{percentage:.1f}%</td>
                        </tr>
                """)
            
            f.write(f"""
                    </table>
                </div>
            """)
            
            # Add sample montages
            for class_name, montage_path in montage_paths.items():
                f.write(f"""
                <div class="container">
                    <h2>{class_name.capitalize()} Samples</h2>
                    <div class="image-container">
                        <img src="{montage_path.name}" alt="{class_name.capitalize()} Samples">
                    </div>
                </div>
                """)
            
            # Close HTML
            f.write("""
            </body>
            </html>
            """)
        
        logger.info(f"Generated QA report: {html_path}")
        return str(html_path)
