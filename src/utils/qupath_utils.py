# src/utils/qupath_utils.py
import json
import numpy as np
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional

logger = logging.getLogger(__name__)

class QuPathAnnotationParser:
    """
    Parser for custom annotation JSON format.
    Extracts regions of interest and associated metadata.
    """
    
    # Class mapping for glomeruli types
    CLASS_MAPPING = {
        "Normal": 0,
        "Partially Sclerotic": 1,
        "Sclerotic": 2,
        "Uncertain": 3
    }
    
    def __init__(self, geojson_path: str):
        """
        Initialize the parser with a JSON file path.
        
        Args:
            geojson_path: Path to annotation JSON file
        """
        self.geojson_path = Path(geojson_path)
        if not self.geojson_path.exists():
            raise FileNotFoundError(f"Annotation file not found: {geojson_path}")
            
        try:
            with open(self.geojson_path, 'r') as f:
                self.annotations = json.load(f)
            logger.info(f"Successfully loaded annotations from: {self.geojson_path.name}")
        except json.JSONDecodeError as e:
            logger.error(f"Invalid JSON in annotation file {geojson_path}: {str(e)}")
            raise
        except Exception as e:
            logger.error(f"Failed to load annotations from {geojson_path}: {str(e)}")
            raise
    
    def get_glomeruli_annotations(self) -> List[Dict[str, Any]]:
        """
        Extract glomeruli annotations with class information.
        
        Returns:
            List of dictionaries with annotation data
        """
        glomeruli = []
        
        # Get the slide key (first key in the JSON)
        slide_keys = list(self.annotations.keys())
        if not slide_keys:
            logger.warning(f"No slide data found in {self.geojson_path}")
            return []
            
        slide_key = slide_keys[0]
        
        # Get annotations for this slide
        slide_data = self.annotations[slide_key]
        annotation_list = slide_data.get('annotations', [])
        
        for annotation in annotation_list:
            # Get classification label
            category = annotation.get('category', '')
            class_name = self._normalize_class_name(category)
            
            # Get bounding box
            bbox = annotation.get('bbox', [])
            if len(bbox) != 4:
                logger.warning(f"Invalid bbox format: {bbox}")
                continue
                
            # Get segmentation polygon
            segmentation = annotation.get('segmentation', [])
            if not segmentation or not isinstance(segmentation[0], list):
                logger.warning(f"Invalid segmentation format: {segmentation}")
                continue
                
            # Convert segmentation format to coordinates
            # The segmentation format appears to be [x1, y1, x2, y2, ...] flattened
            # We need to reshape it to [[x1, y1], [x2, y2], ...]
            coords = []
            polygon = segmentation[0]
            
            # Check if the format is already [x1, y1, x2, y2, ...] or [[x1, y1], [x2, y2], ...]
            if isinstance(polygon[0], list):
                coords = np.array(polygon)
            else:
                # Reshape the flattened array into pairs
                for i in range(0, len(polygon), 2):
                    if i+1 < len(polygon):
                        coords.append([polygon[i], polygon[i+1]])
                coords = np.array(coords)
            
            # Create annotation entry
            glomeruli.append({
                'class_name': class_name,
                'class_id': self.CLASS_MAPPING.get(class_name, 3),  # Default to uncertain
                'coordinates': coords,
                'bbox': bbox,
                'properties': annotation
            })
            
        logger.info(f"Extracted {len(glomeruli)} glomeruli annotations")
        return glomeruli
    
    def _normalize_class_name(self, category: str) -> str:
        """
        Normalize class names to match CLASS_MAPPING.
        
        Args:
            category: Original category string
            
        Returns:
            Normalized class name
        """
        # Try direct match
        if category in self.CLASS_MAPPING:
            return category
            
        # Try case-insensitive match
        category_lower = category.lower()
        for class_name in self.CLASS_MAPPING.keys():
            if class_name.lower() == category_lower:
                return class_name
                
        # Try partial match
        for class_name in self.CLASS_MAPPING.keys():
            if class_name.lower() in category_lower or category_lower in class_name.lower():
                return class_name
                
        logger.warning(f"Unknown category: {category}, defaulting to 'Uncertain'")
        return "Uncertain"
    
    def create_annotation_masks(self, width: int, height: int, 
                               downsample_factor: float = 1.0) -> Dict[str, np.ndarray]:
        """
        Create binary masks for each glomeruli class.
        
        Args:
            width: Width of the output mask
            height: Height of the output mask
            downsample_factor: Factor to scale coordinates
            
        Returns:
            Dictionary mapping class names to binary masks
        """
        from skimage import draw
        
        # Initialize masks for each class
        masks = {class_name: np.zeros((height, width), dtype=np.uint8) 
                for class_name in self.CLASS_MAPPING.keys()}
        
        # Get all glomeruli annotations
        glomeruli = self.get_glomeruli_annotations()
        
        for glom in glomeruli:
            class_name = glom['class_name']
            if class_name not in masks:
                continue
                
            # Scale coordinates by downsample factor
            coords = glom['coordinates'] / downsample_factor
            
            # Create polygon mask
            try:
                rr, cc = draw.polygon(coords[:, 1], coords[:, 0], (height, width))
                masks[class_name][rr, cc] = 1
            except Exception as e:
                logger.error(f"Failed to create mask for {class_name}: {str(e)}")
                
        return masks