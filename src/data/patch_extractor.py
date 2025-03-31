# src/data/patch_extractor.py
import numpy as np
import logging
import os
from pathlib import Path
from typing import List, Tuple, Dict, Optional, Any, Union
import cv2
from concurrent.futures import ThreadPoolExecutor

from ..utils.slide_utils import SlideReader
from ..utils.qupath_utils import QuPathAnnotationParser

logger = logging.getLogger(__name__)

class GlomeruliPatchExtractor:
    """
    Extracts glomeruli patches from whole slide images based on QuPath annotations.
    """
    
    def __init__(self, 
                 output_dir: str, 
                 patch_size: int = 256, 
                 level: Optional[int] = None,
                 magnification: Optional[float] = 20.0,
                 context_factor: float = 1.2,
                 n_workers: int = 4):
        """
        Initialize the patch extractor.
        
        Args:
            output_dir: Directory to save extracted patches
            patch_size: Size of the output patches (square)
            level: Pyramid level to extract from (overrides magnification)
            magnification: Target magnification for extraction
            context_factor: Factor to enlarge bounding box for context
            n_workers: Number of threads for parallel processing
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.patch_size = patch_size
        self.level = level
        self.magnification = magnification
        self.context_factor = context_factor
        self.n_workers = n_workers
        
        # Create output subdirectories for each class
        for class_name in QuPathAnnotationParser.CLASS_MAPPING.keys():
            class_dir = self.output_dir / class_name
            class_dir.mkdir(exist_ok=True)
            
        logger.info(f"Initialized GlomeruliPatchExtractor with output to {output_dir}")
        
    def process_slide(self, 
                     slide_path: str, 
                     annotation_path: str,
                     save_thumbnails: bool = True) -> Dict[str, int]:
        """
        Process a single slide and extract glomeruli patches.
        
        Args:
            slide_path: Path to the .svs slide file
            annotation_path: Path to the QuPath GeoJSON annotation file
            save_thumbnails: Whether to save visualization thumbnails
            
        Returns:
            Dictionary with counts of extracted patches per class
        """
        slide_name = Path(slide_path).stem
        logger.info(f"Processing slide: {slide_name}")
        
        # Load slide and annotations
        reader = SlideReader(slide_path)
        parser = QuPathAnnotationParser(annotation_path)
        
        # Determine extraction level
        level_to_use = self._determine_extraction_level(reader)
        downsample_factor = reader.level_downsamples[level_to_use]
        logger.info(f"Using level {level_to_use} with downsample factor {downsample_factor:.2f}")
        
        # Get glomeruli annotations
        glomeruli = parser.get_glomeruli_annotations()
        if not glomeruli:
            logger.warning(f"No glomeruli annotations found in {annotation_path}")
            reader.close()
            return {}
            
        # Save slide thumbnail with annotations if requested
        if save_thumbnails:
            self._save_annotated_thumbnail(reader, glomeruli, slide_name)
            
        # Extract patches
        counts = self._extract_patches(reader, glomeruli, level_to_use, downsample_factor, slide_name)
        
        reader.close()
        return counts
        
    def _determine_extraction_level(self, reader: SlideReader) -> int:
        """
        Determine the appropriate level for patch extraction.
        
        Args:
            reader: Slide reader instance
            
        Returns:
            Level index to use for extraction
        """
        if self.level is not None:
            if self.level >= reader.level_count:
                logger.warning(f"Requested level {self.level} exceeds available levels. "
                             f"Using maximum available level {reader.level_count - 1}")
                return reader.level_count - 1
            return self.level
            
        # Determine level based on target magnification
        orig_mag = SlideReader.get_magnification(reader.properties)
        if orig_mag is None:
            logger.warning("Could not determine slide magnification. Using level 0.")
            return 0
            
        downsample_factor = orig_mag / self.magnification
        return reader.get_best_level_for_downsample(downsample_factor)
        
    def _save_annotated_thumbnail(self, 
                                reader: SlideReader, 
                                glomeruli: List[Dict[str, Any]], 
                                slide_name: str):
        """
        Save a thumbnail of the slide with annotations overlaid.
        
        Args:
            reader: Slide reader instance
            glomeruli: List of glomeruli annotations
            slide_name: Name of the slide for output filename
        """
        try:
            thumbnail_size = (1024, 1024)
            thumbnail = reader.get_thumbnail(thumbnail_size)
            
            # Scale factor from level 0 to thumbnail
            scale_x = thumbnail.shape[1] / reader.dimensions[0]
            scale_y = thumbnail.shape[0] / reader.dimensions[1]
            
            # Create RGB image (copy to avoid modifying the original)
            vis_img = thumbnail.copy()
            
            # Color mapping for classes
            colors = {
                0: (0, 255, 0),    # normal: green
                1: (255, 255, 0),  # partially_sclerotic: yellow
                2: (0, 0, 255),    # sclerotic: red
                3: (255, 0, 255)   # uncertain: magenta
            }
            
            # Draw annotations
            for glom in glomeruli:
                coords = glom['coordinates']
                class_id = glom['class_id']
                
                # Scale coordinates to thumbnail size
                scaled_coords = coords.copy()
                scaled_coords[:, 0] *= scale_x
                scaled_coords[:, 1] *= scale_y
                
                # Convert to int32 for OpenCV
                points = scaled_coords.astype(np.int32)
                
                # Draw polygon
                cv2.polylines(vis_img, [points], True, colors[class_id], 2)
                
            # Save visualization
            vis_path = self.output_dir / f"{slide_name}_annotated.png"
            cv2.imwrite(str(vis_path), cv2.cvtColor(vis_img, cv2.COLOR_RGB2BGR))
            logger.info(f"Saved annotated thumbnail to {vis_path}")
            
        except Exception as e:
            logger.error(f"Failed to save annotated thumbnail: {str(e)}")
    
    def _extract_patches(self, 
                        reader: SlideReader, 
                        glomeruli: List[Dict[str, Any]], 
                        level: int, 
                        downsample_factor: float, 
                        slide_name: str) -> Dict[str, int]:
        """
        Extract patches for all glomeruli.
        
        Args:
            reader: Slide reader instance
            glomeruli: List of glomeruli annotations
            level: Pyramid level to extract from
            downsample_factor: Downsample factor for the level
            slide_name: Name of the slide
            
        Returns:
            Dictionary with counts of extracted patches per class
        """
        # Prepare batch of tasks for parallel processing
        tasks = []
        for i, glom in enumerate(glomeruli):
            class_name = glom['class_name'].lower()
            class_id = glom['class_id']
            
            # Get bounding box with context
            bbox = self._get_bounding_box_with_context(glom['coordinates'], 
                                                      reader.dimensions,
                                                      self.context_factor)
            
            # Scale bbox to extraction level
            level_bbox = (
                int(bbox[0] / downsample_factor),
                int(bbox[1] / downsample_factor),
                int(bbox[2] / downsample_factor),
                int(bbox[3] / downsample_factor)
            )
            
            # Calculate size and location
            width = level_bbox[2] - level_bbox[0]
            height = level_bbox[3] - level_bbox[1]
            location = (bbox[0], bbox[1])
            
            patch_filename = f"{slide_name}_glom{i:03d}_{class_name}.png"
            output_path = self.output_dir / class_name / patch_filename
            
            tasks.append((
                reader, 
                location, 
                level, 
                (width, height), 
                output_path, 
                self.patch_size,
                class_id
            ))
        
        # Process in parallel
        counts = {class_name: 0 for class_name in QuPathAnnotationParser.CLASS_MAPPING.keys()}
        
        with ThreadPoolExecutor(max_workers=self.n_workers) as executor:
            for class_name in executor.map(self._extract_single_patch, tasks):
                if class_name:
                    counts[class_name] += 1
        
        logger.info(f"Extracted patches: {counts}")
        return counts
    
    @staticmethod
    def _extract_single_patch(task_args) -> Optional[str]:
        """
        Extract a single glomeruli patch (for parallel processing).
        
        Args:
            task_args: Tuple containing extraction parameters
            
        Returns:
            Class name if successful, None otherwise
        """
        reader, location, level, size, output_path, patch_size, class_id = task_args
        
        try:
            # Read region from slide
            region = reader.read_region(location, level, size)
            
            # Resize to target patch size
            if region.shape[0] != patch_size or region.shape[1] != patch_size:
                region = cv2.resize(region, (patch_size, patch_size))
                
            # Save patch
            cv2.imwrite(str(output_path), cv2.cvtColor(region, cv2.COLOR_RGB2BGR))
            
            # Return class name for counting
            return list(QuPathAnnotationParser.CLASS_MAPPING.keys())[class_id]
        except Exception as e:
            logger.error(f"Failed to extract patch at {location}: {str(e)}")
            return None
    
    @staticmethod
    def _get_bounding_box_with_context(
        coordinates: np.ndarray, 
        slide_dimensions: Tuple[int, int],
        context_factor: float = 1.2
    ) -> Tuple[int, int, int, int]:
        """
        Calculate bounding box with added context around a glomerulus.
        
        Args:
            coordinates: Array of polygon coordinates
            slide_dimensions: (width, height) of the slide
            context_factor: Factor to enlarge the bounding box
            
        Returns:
            Tuple (x_min, y_min, x_max, y_max)
        """
        x_min, y_min = np.min(coordinates, axis=0)
        x_max, y_max = np.max(coordinates, axis=0)
        
        # Calculate center
        center_x = (x_min + x_max) / 2
        center_y = (y_min + y_max) / 2
        
        # Calculate half-width and half-height with context
        half_width = (x_max - x_min) * context_factor / 2
        half_height = (y_max - y_min) * context_factor / 2
        
        # Make it square by taking max of width and height
        half_size = max(half_width, half_height)
        
        # Calculate new bounds
        x_min_new = int(max(0, center_x - half_size))
        y_min_new = int(max(0, center_y - half_size))
        x_max_new = int(min(slide_dimensions[0], center_x + half_size))
        y_max_new = int(min(slide_dimensions[1], center_y + half_size))
        
        return x_min_new, y_min_new, x_max_new, y_max_new
        
    def process_batch(self, 
                     slide_annotation_pairs: List[Tuple[str, str]],
                     save_thumbnails: bool = True) -> Dict[str, Dict[str, int]]:
        """
        Process a batch of slides.
        
        Args:
            slide_annotation_pairs: List of (slide_path, annotation_path) tuples
            save_thumbnails: Whether to save visualization thumbnails
            
        Returns:
            Nested dictionary with extracted patch counts per class for each slide
        """
        results = {}
        
        for slide_path, annotation_path in slide_annotation_pairs:
            slide_name = Path(slide_path).stem
            try:
                counts = self.process_slide(slide_path, annotation_path, save_thumbnails)
                results[slide_name] = counts
            except Exception as e:
                logger.error(f"Failed to process slide {slide_name}: {str(e)}")
                results[slide_name] = {"error": str(e)}
                
        return results
