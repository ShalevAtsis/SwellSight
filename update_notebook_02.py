"""
Script to update notebook 02 to use production src.swellsight modules.
Task 3.5: Replace inline implementations with production module calls.
"""

import json
from pathlib import Path

# Read the original notebook
notebook_path = Path("02_Data_Import_and_Preprocessing.ipynb")
with open(notebook_path, 'r', encoding='utf-8') as f:
    notebook = json.load(f)

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
    "except ImportError as e:\n",
    "    print(f\"⚠️ Could not import production modules: {e}\")\n",
    "    print(\"Loading fallback functions...\")\n",
    "    PRODUCTION_MODULES_AVAILABLE = False\n",
    "    # Fallback will use inline implementations\n",
    "\n",
    "# Check environment\n",
    "IN_COLAB = 'google.colab' in sys.modules\n",
    "\n",
    "# Mount Google Drive if in Colab\n",
    "if IN_COLAB:\n",
    "    from google.colab import drive\n",
    "    print(\"Mounting Google Drive...\")\n",
    "    try:\n",
    "        drive.mount('/content/drive')\n",
    "        print(\"✓ Google Drive mounted successfully\")\n",
    "    except Exception as e:\n",
    "        print(f\"Drive mount failed: {e}\")\n",
    "        try:\n",
    "            drive.mount('/content/drive', force_remount=True, timeout_ms=300000)\n",
    "            print(\"✓ Force remount successful\")\n",
    "        except Exception as e2:\n",
    "            print(f\"❌ Critical failure mounting drive: {e2}\")\n",
    "            raise\n",
    "\n",
    "# Configure logging\n",
    "logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')\n",
    "logger = logging.getLogger(__name__)\n"
]

# Update cell 1 (imports)
notebook['cells'][0]['source'] = cell_1_source

# Save the updated notebook
output_path = Path("02_Data_Import_and_Preprocessing_Enhanced.ipynb")
with open(output_path, 'w', encoding='utf-8') as f:
    json.dump(notebook, f, indent=2, ensure_ascii=False)

print(f"✅ Updated notebook saved to {output_path}")
