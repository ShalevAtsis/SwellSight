"""
Complete script to update notebook 02 to use production src.swellsight modules.
Task 3.5: Replace inline implementations with production module calls.
"""

import json
from pathlib import Path

# Read the original notebook
notebook_path = Path("02_Data_Import_and_Preprocessing.ipynb")
with open(notebook_path, 'r', encoding='utf-8') as f:
    notebook = json.load(f)

print(f"Original notebook has {len(notebook['cells'])} cells")

# Cell 1: Update imports to use production modules
cell_1_source = [
    "import sys\n",
    "import os\n",
    "import json\n",
    "import logging\n",
    "from pathlib import Path\n",
    "from datetime import datetime\n",
    "from typing import Dict, Any, List, Optional, Tuple\n",
    "import warnings\n",
    "warnings.filterwarnings('ignore')\n",
    "\n",
    "# Import core libraries\n",
    "import numpy as np\n",
    "import pandas as pd\n",
    "import matplotlib.pyplot as plt\n",
    "import seaborn as sns\n",
    "from PIL import Image\n",
    "import cv2\n",
    "from tqdm.auto import tqdm\n",
    "import ipywidgets as widgets\n",
    "from IPython.display import display, HTML, clear_output\n",
    "from collections import Counter\n",
    "\n",
    "# Add src directory to Python path for production modules\n",
    "src_path = Path.cwd() / \"src\"\n",
    "if str(src_path) not in sys.path:\n",
    "    sys.path.insert(0, str(src_path))\n",
    "\n",
    "# Import SwellSight production modules\n",
    "try:\n",
    "    from swellsight.evaluation.data_quality import DataQualityAssessor, create_quality_visualization\n",
    "    from swellsight.utils.error_handler import ErrorHandler, retry_with_backoff, RetryConfig, safe_execute\n",
    "    from swellsight.utils.performance import PerformanceOptimizer, OptimizationConfig\n",
    "    from swellsight.utils.hardware import HardwareManager\n",
    "    \n",
    "    print(\"✅ SwellSight production modules loaded successfully\")\n",
    "    PRODUCTION_MODULES_AVAILABLE = True\n",
    "    \n",
    "    # Initialize production utilities\n",
    "    error_handler = ErrorHandler()\n",
    "    hardware_manager = HardwareManager()\n",
    "    perf_optimizer = PerformanceOptimizer(OptimizationConfig(target_latency_ms=200.0))\n",
    "    \n",
    "except ImportError as e:\n",
    "    print(f\"⚠️ Could not import production modules: {e}\")\n",
    "    print(\"Loading fallback functions...\")\n",
    "    PRODUCTION_MODULES_AVAILABLE = False\n",
    "    error_handler = None\n",
    "    hardware_manager = None\n",
    "    perf_optimizer = None\n",
    "\n",
    "# Check environment\n",
    "IN_COLAB = 'google.colab' in sys.modules\n",
    "\n",
    "# Mount Google Drive if in Colab with retry logic\n",
    "if IN_COLAB:\n",
    "    from google.colab import drive\n",
    "    print(\"Mounting Google Drive...\")\n",
    "    \n",
    "    def mount_drive():\n",
    "        drive.mount('/content/drive')\n",
    "        return True\n",
    "    \n",
    "    if PRODUCTION_MODULES_AVAILABLE:\n",
    "        # Use production error handler with retry logic\n",
    "        try:\n",
    "            retry_config = RetryConfig(max_attempts=2, base_delay=1.0)\n",
    "            retry_with_backoff(retry_config=retry_config, component=\"colab\", operation=\"mount_drive\")(mount_drive)()\n",
    "            print(\"✓ Google Drive mounted successfully\")\n",
    "        except Exception as e:\n",
    "            print(f\"❌ Critical failure mounting drive: {e}\")\n",
    "            raise\n",
    "    else:\n",
    "        # Fallback to basic retry\n",
    "        try:\n",
    "            mount_drive()\n",
    "            print(\"✓ Google Drive mounted successfully\")\n",
    "        except Exception as e:\n",
    "            print(f\"Drive mount failed: {e}\")\n",
    "            try:\n",
    "                drive.mount('/content/drive', force_remount=True, timeout_ms=300000)\n",
    "                print(\"✓ Force remount successful\")\n",
    "            except Exception as e2:\n",
    "                print(f\"❌ Critical failure mounting drive: {e2}\")\n",
    "                raise\n",
    "\n",
    "# Configure logging\n",
    "logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')\n",
    "logger = logging.getLogger(__name__)\n"
]

notebook['cells'][0]['source'] = cell_1_source

# Cell 2: Load configuration (keep mostly the same, add error handling)
cell_2_source = [
    "# Load configuration from previous notebook\n",
    "try:\n",
    "    # Try to get from global variables first\n",
    "    CONFIG = globals().get('SWELLSIGHT_CONFIG')\n",
    "    logger_obj = globals().get('SWELLSIGHT_LOGGER')\n",
    "\n",
    "    if CONFIG is None:\n",
    "        # Load from file\n",
    "        if IN_COLAB:\n",
    "            config_file = Path('/content/drive/MyDrive/SwellSight/data/metadata/pipeline_config.json')\n",
    "        else:\n",
    "            config_file = Path('SwellSight/data/metadata/pipeline_config.json')\n",
    "\n",
    "        if config_file.exists():\n",
    "            with open(config_file, 'r') as f:\n",
    "                CONFIG = json.load(f)\n",
    "            print(f\"✓ Configuration loaded from {config_file}\")\n",
    "        else:\n",
    "            raise FileNotFoundError(f\"Configuration file not found at {config_file}\")\n",
    "\n",
    "    # Set up paths\n",
    "    DRIVE_PATH = Path(CONFIG['paths']['drive_path'])\n",
    "    DATA_PATH = Path(CONFIG['paths']['data_path'])\n",
    "    REAL_IMAGES_PATH = Path(CONFIG['paths']['real_images_path'])\n",
    "    METADATA_PATH = Path(CONFIG['paths']['metadata_path'])\n",
    "\n",
    "    print(f\"✓ Setup completed successfully\")\n",
    "    print(f\"  Session ID: {CONFIG['session']['session_id']}\")\n",
    "    print(f\"  Real images path: {REAL_IMAGES_PATH}\")\n",
    "    print(f\"  Metadata path: {METADATA_PATH}\")\n",
    "\n",
    "except Exception as e:\n",
    "    if PRODUCTION_MODULES_AVAILABLE:\n",
    "        error_handler.handle_error(e, \"DataImport\", \"load_configuration\")\n",
    "    print(f\"❌ Failed to load configuration: {e}\")\n",
    "    print(\"Please run 01_Setup_and_Installation.ipynb first\")\n",
    "    raise\n"
]

notebook['cells'][1]['source'] = cell_2_source

print("✅ Updated cells 1-2 (imports and configuration)")

# Save the updated notebook
output_path = Path("02_Data_Import_and_Preprocessing_Enhanced.ipynb")
with open(output_path, 'w', encoding='utf-8') as f:
    json.dump(notebook, f, indent=2, ensure_ascii=False)

print(f"✅ Updated notebook saved to {output_path}")
print(f"Updated notebook has {len(notebook['cells'])} cells")
