  # src/utils/slide_utils.py
import openslide
import numpy as np
import logging
from pathlib import Path
from typing import Tuple, Optional, Dict, List, Any

logger = logging.getLogger(__name__)

class SlideReader:
    """
    Utility class for reading and processing whole slide images (WSI).
    Optimized for .svs files with efficient memory handling.
    """
    
    def __init__(self, slide_path: str, cache_size: int = 10):
        """
        Initialize the slide reader.
        
        Args:
            slide_path: Path to the .svs file
            cache_size: Size of the tile cache (higher = more memory usage but faster processing)
        """
        self.slide_path = Path(slide_path)
        if not self.slide_path.exists():
            raise FileNotFoundError(f"Slide file not found: {slide_path}")
            
        try:
            self.slide = openslide.OpenSlide(str(slide_path))
            logger.info(f"Successfully loaded slide: {self.slide_path.name}")
        except Exception as e:
            logger.error(f"Failed to load slide {slide_path}: {str(e)}")
            raise
            
        # Set cache size for tile reading
        self.slide.set_cache(openslide.OpenSlideCache(cache_size))
        
        # Get slide properties
        self.dimensions = self.slide.dimensions
        self.level_count = self.slide.level_count
        self.level_dimensions = self.slide.level_dimensions
        self.level_downsamples = self.slide.level_downsamples
        self.properties = dict(self.slide.properties)
        
        logger.debug(f"Slide dimensions: {self.dimensions}")
        logger.debug(f"Slide levels: {self.level_count}")
        
    def get_thumbnail(self, size: Tuple[int, int] = (1024, 1024)) -> np.ndarray:
        """
        Get a thumbnail of the slide for visualization.
        
        Args:
            size: Requested size of the thumbnail
            
        Returns:
            Thumbnail as RGB numpy array
        """
        thumbnail = self.slide.get_thumbnail(size)
        return np.array(thumbnail)
        
    def read_region(self, 
                   location: Tuple[int, int], 
                   level: int, 
                   size: Tuple[int, int]) -> np.ndarray:
        """
        Read a region from the slide at specified location and level.
        
        Args:
            location: (x, y) coordinates in level 0 reference frame
            level: Pyramid level to read from
            size: Size of region to read
            
        Returns:
            Region as RGB numpy array
        """
        if level >= self.level_count:
            raise ValueError(f"Level {level} out of range. Slide has {self.level_count} levels.")
            
        region = self.slide.read_region(location, level, size)
        # Convert RGBA to RGB
        return np.array(region)[:, :, :3]
        
    def get_best_level_for_downsample(self, downsample_factor: float) -> int:
        """
        Get the best level for a given downsample factor.
        
        Args:
            downsample_factor: Desired downsample factor
            
        Returns:
            Level index
        """
        return self.slide.get_best_level_for_downsample(downsample_factor)
        
    def close(self):
        """Close the slide and release resources."""
        self.slide.close()
        
    def __enter__(self):
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
        
    @staticmethod
    def get_magnification(slide_properties: Dict[str, Any]) -> Optional[float]:
        """
        Extract magnification from slide properties.
        
        Args:
            slide_properties: Dictionary of slide properties
            
        Returns:
            Magnification value or None if not found
        """
        # Try different property keys that might contain magnification info
        for key in ['openslide.objective-power', 'aperio.AppMag']:
            if key in slide_properties:
                try:
                    return float(slide_properties[key])
                except (ValueError, TypeError):
                    pass
        
        logger.warning("Could not determine slide magnification from properties")
        return None
