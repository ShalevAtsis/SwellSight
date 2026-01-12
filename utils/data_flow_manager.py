"""
Data Flow Management for SwellSight Pipeline
Handles standardized data saving/loading and dependency validation between notebook stages
"""

import json
import os
from pathlib import Path
from typing import Dict, Any, List, Optional, Union
import datetime
import logging
import numpy as np

logger = logging.getLogger(__name__)

class DataFlowManager:
    """Manages data flow and dependencies between pipeline stages"""
    
    def __init__(self, base_output_dir: str = "./outputs"):
        self.base_output_dir = Path(base_output_dir)
        self.base_output_dir.mkdir(parents=True, exist_ok=True)
        
        # Stage dependency mapping
        self.stage_dependencies = {
            'setup': [],
            'data_preprocessing': ['setup'],
            'depth_extraction': ['data_preprocessing'],
            'augmentation': ['depth_extraction'],
            'synthetic_generation': ['augmentation'],
            'training': ['synthetic_generation'],
            'analysis': ['training'],
            'evaluation': ['training']
        }
    
    def save_stage_results(self, data: Dict[str, Any], stage_name: str, 
                          metadata: Optional[Dict[str, Any]] = None) -> bool:
        """
        Save stage results in standardized format
        
        Args:
            data: Dictionary containing stage results
            stage_name: Name of the pipeline stage
            metadata: Optional metadata about the processing
            
        Returns:
            True if save successful, False otherwise
        """
        try:
            # Create stage output directory
            stage_dir = self.base_output_dir / stage_name
            stage_dir.mkdir(parents=True, exist_ok=True)
            
            # Prepare result structure
            timestamp = datetime.datetime.now().isoformat()
            result_structure = {
                'stage_name': stage_name,
                'timestamp': timestamp,
                'status': 'completed',
                'data': data,
                'metadata': metadata or {}
            }
            
            # Add processing statistics if available
            if 'processing_time_seconds' not in result_structure['metadata']:
                result_structure['metadata']['processing_time_seconds'] = 0
            
            # Save main results file
            results_file = stage_dir / f"{stage_name}_results.json"
            with open(results_file, 'w') as f:
                json.dump(
                    self._make_json_serializable(result_structure), 
                    f, 
                    indent=2
                )
            
            # Save individual data components if they are large
            for key, value in data.items():
                if isinstance(value, (list, np.ndarray)) and len(str(value)) > 10000:
                    # Save large data separately
                    data_file = stage_dir / f"{key}.json"
                    with open(data_file, 'w') as f:
                        json.dump(self._make_json_serializable(value), f)
                    
                    # Update main results to reference the file
                    result_structure['data'][key] = f"{key}.json"
            
            # Create stage completion marker
            completion_marker = stage_dir / ".completed"
            with open(completion_marker, 'w') as f:
                f.write(timestamp)
            
            logger.info(f"Stage results saved: {stage_name} -> {results_file}")
            return True
            
        except Exception as e:
            logger.error(f"Error saving stage results for {stage_name}: {e}")
            return False
    
    def load_previous_results(self, stage_name: str, 
                            required_files: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Load results from a previous pipeline stage with validation
        
        Args:
            stage_name: Name of the stage to load results from
            required_files: Optional list of required files to validate
            
        Returns:
            Dictionary containing loaded results
            
        Raises:
            FileNotFoundError: If required files are missing
            ValueError: If data validation fails
        """
        try:
            stage_dir = self.base_output_dir / stage_name
            
            # Check if stage completed
            completion_marker = stage_dir / ".completed"
            if not completion_marker.exists():
                raise FileNotFoundError(f"Stage {stage_name} has not completed successfully")
            
            # Load main results file
            results_file = stage_dir / f"{stage_name}_results.json"
            if not results_file.exists():
                raise FileNotFoundError(f"Results file not found: {results_file}")
            
            with open(results_file, 'r') as f:
                results = json.load(f)
            
            # Load referenced data files
            if 'data' in results:
                for key, value in results['data'].items():
                    if isinstance(value, str) and value.endswith('.json'):
                        data_file = stage_dir / value
                        if data_file.exists():
                            with open(data_file, 'r') as f:
                                results['data'][key] = json.load(f)
            
            # Validate required files if specified
            if required_files:
                self._validate_required_files(stage_dir, required_files)
            
            # Validate data format
            if not self.validate_data_format(results, stage_name):
                raise ValueError(f"Data format validation failed for stage: {stage_name}")
            
            logger.info(f"Successfully loaded results from stage: {stage_name}")
            return results
            
        except Exception as e:
            logger.error(f"Error loading results from {stage_name}: {e}")
            raise
    
    def validate_data_format(self, data: Dict[str, Any], expected_stage: str) -> bool:
        """
        Validate data format against expected schema
        
        Args:
            data: Data to validate
            expected_stage: Expected stage name
            
        Returns:
            True if validation passes, False otherwise
        """
        try:
            # Check required top-level keys
            required_keys = ['stage_name', 'timestamp', 'status', 'data']
            for key in required_keys:
                if key not in data:
                    logger.error(f"Missing required key: {key}")
                    return False
            
            # Check stage name matches
            if data['stage_name'] != expected_stage:
                logger.error(f"Stage name mismatch: expected {expected_stage}, got {data['stage_name']}")
                return False
            
            # Check status
            if data['status'] != 'completed':
                logger.error(f"Stage status is not 'completed': {data['status']}")
                return False
            
            # Validate timestamp format
            try:
                datetime.datetime.fromisoformat(data['timestamp'])
            except ValueError:
                logger.error(f"Invalid timestamp format: {data['timestamp']}")
                return False
            
            # Check data section exists and is not empty
            if not isinstance(data['data'], dict):
                logger.error("Data section must be a dictionary")
                return False
            
            return True
            
        except Exception as e:
            logger.error(f"Error validating data format: {e}")
            return False
    
    def check_dependencies(self, stage_name: str) -> Dict[str, Any]:
        """
        Check if all dependencies for a stage are satisfied
        
        Args:
            stage_name: Name of the stage to check dependencies for
            
        Returns:
            Dictionary with dependency status and missing dependencies
        """
        try:
            dependencies = self.stage_dependencies.get(stage_name, [])
            missing_dependencies = []
            satisfied_dependencies = []
            
            for dep_stage in dependencies:
                completion_marker = self.base_output_dir / dep_stage / ".completed"
                if completion_marker.exists():
                    satisfied_dependencies.append(dep_stage)
                else:
                    missing_dependencies.append(dep_stage)
            
            return {
                'stage_name': stage_name,
                'all_satisfied': len(missing_dependencies) == 0,
                'satisfied_dependencies': satisfied_dependencies,
                'missing_dependencies': missing_dependencies,
                'total_dependencies': len(dependencies)
            }
            
        except Exception as e:
            logger.error(f"Error checking dependencies for {stage_name}: {e}")
            return {
                'stage_name': stage_name,
                'all_satisfied': False,
                'error': str(e)
            }
    
    def get_pipeline_status(self) -> Dict[str, Any]:
        """
        Get overall pipeline status and progress
        
        Returns:
            Dictionary with pipeline status information
        """
        try:
            status = {
                'completed_stages': [],
                'pending_stages': [],
                'total_stages': len(self.stage_dependencies),
                'completion_percentage': 0.0
            }
            
            for stage_name in self.stage_dependencies.keys():
                completion_marker = self.base_output_dir / stage_name / ".completed"
                if completion_marker.exists():
                    status['completed_stages'].append(stage_name)
                else:
                    status['pending_stages'].append(stage_name)
            
            # Calculate completion percentage
            if status['total_stages'] > 0:
                status['completion_percentage'] = (
                    len(status['completed_stages']) / status['total_stages']
                ) * 100.0
            
            return status
            
        except Exception as e:
            logger.error(f"Error getting pipeline status: {e}")
            return {'error': str(e)}
    
    def _validate_required_files(self, stage_dir: Path, required_files: List[str]) -> None:
        """Validate that required files exist in stage directory"""
        missing_files = []
        
        for filename in required_files:
            file_path = stage_dir / filename
            if not file_path.exists():
                missing_files.append(filename)
        
        if missing_files:
            raise FileNotFoundError(f"Missing required files: {missing_files}")
    
    def _make_json_serializable(self, obj: Any) -> Any:
        """Convert objects to JSON-serializable format"""
        if isinstance(obj, dict):
            return {k: self._make_json_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._make_json_serializable(item) for item in obj]
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (np.integer, np.floating)):
            return obj.item()
        elif isinstance(obj, Path):
            return str(obj)
        elif isinstance(obj, datetime.datetime):
            return obj.isoformat()
        else:
            return obj
    
    def cleanup_stage_outputs(self, stage_name: str) -> bool:
        """
        Clean up outputs from a specific stage
        
        Args:
            stage_name: Name of the stage to clean up
            
        Returns:
            True if cleanup successful, False otherwise
        """
        try:
            stage_dir = self.base_output_dir / stage_name
            
            if stage_dir.exists():
                import shutil
                shutil.rmtree(stage_dir)
                logger.info(f"Cleaned up outputs for stage: {stage_name}")
                return True
            else:
                logger.info(f"No outputs found for stage: {stage_name}")
                return True
                
        except Exception as e:
            logger.error(f"Error cleaning up stage {stage_name}: {e}")
            return False


# Convenience functions for direct use in notebooks
def save_stage_results(data: Dict[str, Any], stage_name: str, 
                      metadata: Optional[Dict[str, Any]] = None) -> bool:
    """Save stage results in standardized format"""
    manager = DataFlowManager()
    return manager.save_stage_results(data, stage_name, metadata)

def load_previous_results(stage_name: str, 
                         required_files: Optional[List[str]] = None) -> Dict[str, Any]:
    """Load results from a previous pipeline stage with validation"""
    manager = DataFlowManager()
    return manager.load_previous_results(stage_name, required_files)

def check_dependencies(stage_name: str) -> Dict[str, Any]:
    """Check if all dependencies for a stage are satisfied"""
    manager = DataFlowManager()
    return manager.check_dependencies(stage_name)