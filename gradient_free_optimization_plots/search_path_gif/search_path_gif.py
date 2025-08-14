# Author: Simon Blanke
# Email: simon.blanke@yahoo.com
# License: MIT License

from typing import Dict, Any, Optional, List, Union, Callable
import glob
import os
import subprocess
import warnings
from pathlib import Path

from .plot_search_paths import plot_search_paths

warnings.filterwarnings('ignore')


class SearchPathGif:
    """Class for generating animated GIFs of optimization search paths."""
    
    DEFAULT_PATH = "./gifs"
    DEFAULT_INITIALIZE = {"random": 4}
    
    def __init__(self, path: Optional[str] = None) -> None:
        """Initialize SearchPathGif with output path.
        
        Args:
            path: Directory path for output files. Defaults to './gifs'.
        """
        if path is None:
            path = self.DEFAULT_PATH
        
        self.path = Path(os.getcwd()) / path
        self.path.mkdir(parents=True, exist_ok=True)
        
        # Initialize required attributes
        self.optimizer: Optional[Any] = None
        self.opt_para: Dict[str, Any] = {}
        self.initialize: Dict[str, Any] = self.DEFAULT_INITIALIZE.copy()
        self.n_iter: int = 0
        self.random_state: Optional[int] = None
        self.objective_function: Optional[Callable] = None
        self.search_space: Dict[str, Any] = {}
        self.constraints: List = []
        self.name: str = ""
        self.title: Union[str, bool] = True

    def add_optimizer(
        self, 
        optimizer: Any, 
        n_iter: int, 
        opt_para: Optional[Dict[str, Any]] = None, 
        initialize: Optional[Dict[str, Any]] = None, 
        random_state: Optional[int] = None
    ) -> None:
        """Add optimizer configuration.
        
        Args:
            optimizer: The optimizer class to use
            n_iter: Number of iterations to run
            opt_para: Optimizer parameters
            initialize: Initialization parameters
            random_state: Random state for reproducibility
        """
        self.optimizer = optimizer
        self.opt_para = opt_para or {}
        self.initialize = initialize or self.DEFAULT_INITIALIZE.copy()
        self.n_iter = n_iter
        self.random_state = random_state

    def add_test_function(
        self, 
        objective_function: Callable, 
        search_space: Dict[str, Any], 
        constraints: Optional[List] = None
    ) -> None:
        """Add objective function and search space.
        
        Args:
            objective_function: The function to optimize
            search_space: Dictionary defining the search space
            constraints: List of constraints (optional)
        """
        self.objective_function = objective_function
        self.search_space = search_space
        self.constraints = constraints or []

    def add_plot_layout(self, name: Optional[str] = None, title: Union[str, bool] = True) -> None:
        """Add plot layout configuration.
        
        Args:
            name: Output filename. Auto-generated if None.
            title: Plot title or True for auto-generated title
        """
        if name is None and self.optimizer is not None:
            name = f"{self.optimizer._name_}.gif"
        
        self.name = name or "optimization.gif"
        self.title = title

    def _validate_configuration(self) -> None:
        """Validate that all required configuration is set."""
        if self.optimizer is None:
            raise ValueError("Optimizer must be set using add_optimizer()")
        if self.objective_function is None:
            raise ValueError("Objective function must be set using add_test_function()")
        if not self.search_space:
            raise ValueError("Search space must be set using add_test_function()")
        if self.n_iter <= 0:
            raise ValueError("Number of iterations must be positive")
    
    def _create_plots(self) -> Path:
        """Generate individual plot frames."""
        plots_dir = self.path / "_plots"
        plots_dir.mkdir(exist_ok=True)
        
        plot_search_paths(
            path=str(self.path),
            optimizer=self.optimizer,
            opt_para=self.opt_para,
            n_iter_max=self.n_iter,
            objective_function=self.objective_function,
            search_space=self.search_space,
            constraints=self.constraints,
            initialize=self.initialize,
            random_state=self.random_state,
            title=self.title,
        )
        
        return plots_dir
    
    def _create_gif_with_ffmpeg(self, plots_dir: Path) -> None:
        """Create GIF using ffmpeg."""
        framerate = max(1, self.n_iter / 10)
        
        input_pattern = plots_dir / f"{self.optimizer._name_}_%03d.jpg"
        output_path = self.path / self.name
        
        cmd = [
            "ffmpeg",
            "-hide_banner",
            "-loglevel", "error",
            "-y",
            "-framerate", str(framerate),
            "-i", str(input_pattern),
            "-vf", "scale=1200:-1:flags=lanczos",
            str(output_path)
        ]
        
        try:
            result = subprocess.run(cmd, check=True, capture_output=True, text=True)
            print(f"Successfully created {self.name}")
        except subprocess.CalledProcessError as e:
            raise RuntimeError(f"FFmpeg failed: {e.stderr}") from e
        except FileNotFoundError:
            raise RuntimeError("FFmpeg not found. Please install FFmpeg.") from None
    
    def _cleanup_temp_files(self, plots_dir: Path) -> None:
        """Remove temporary plot files."""
        try:
            for jpg_file in plots_dir.glob("*.jpg"):
                jpg_file.unlink()
            plots_dir.rmdir()
        except OSError as e:
            warnings.warn(f"Failed to clean up temporary files: {e}")
    
    def create(self) -> None:
        """Generate the animated GIF of optimization search paths.
        
        Raises:
            ValueError: If required configuration is missing
            RuntimeError: If FFmpeg execution fails
        """
        self._validate_configuration()
        
        print(f"\nGenerating {self.name}...")
        plots_dir = self._create_plots()
        
        try:
            self._create_gif_with_ffmpeg(plots_dir)
        finally:
            self._cleanup_temp_files(plots_dir)

    def __repr__(self) -> str:
        """String representation of SearchPathGif."""
        optimizer_name = self.optimizer._name_ if self.optimizer else "None"
        return f"SearchPathGif(path='{self.path}', optimizer='{optimizer_name}', n_iter={self.n_iter})"
