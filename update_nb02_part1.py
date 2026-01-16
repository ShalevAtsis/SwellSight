"""Part 1: Update notebook 02 cells 1-5"""
import json
from pathlib import Path

notebook_path = Path("02_Data_Import_and_Preprocessing.ipynb")
with open(notebook_path, 'r', encoding='utf-8') as f:
    notebook = json.load(f)

# Cell 1: Imports with production modules
cell_1 = [
    "import sys\n", "import os\n", "import json\n", "import logging\n",
    "from pathlib import Path\n", "from datetime import datetime\n",
    "from typing import Dict, Any, List, Optional, Tuple\n",
    "import warnings\n", "warnings.filterwarnings('ignore')\n", "\n",
    "# Core libraries\n", "import numpy as np\n", "import pandas as pd\n",
    "import matplotlib.pyplot as plt\n", "import seaborn as sns\n",
    "from PIL import Image\n", "import cv2\n", "from tqdm.auto import tqdm\n",
    "import ipywidgets as widgets\n",
    "from IPython.display import display, HTML, clear_output\n",
    "from collections import Counter\n", "\n",
    "# Add src to path\n", "src_path = Path.cwd() / \"src\"\n",
    "if str(src_path) not in sys.path:\n",
    "    sys.path.insert(0, str(src_path))\n", "\n",
    "# Import production modules\n", "try:\n",
    "    from swellsight.evaluation.data_quality import DataQualityAssessor, create_quality_visualization\n",
    "    from swellsight.utils.error_handler import ErrorHandler, retry_with_backoff, RetryConfig\n",
    "    from swellsight.utils.performance import PerformanceOptimizer, OptimizationConfig\n",
    "    from swellsight.utils.hardware import HardwareManager\n",
    "    print(\"✅ Production modules loaded\")\n",
    "    PRODUCTION_MODULES_AVAILABLE = True\n",
    "    error_handler = ErrorHandler()\n",
    "    hardware_manager = HardwareManager()\n",
    "    perf_optimizer = PerformanceOptimizer()\n",
    "except ImportError as e:\n",
    "    print(f\"⚠️ Production modules unavailable: {e}\")\n",
    "    PRODUCTION_MODULES_AVAILABLE = False\n",
    "    error_handler = hardware_manager = perf_optimizer = None\n",
    "\n", "IN_COLAB = 'google.colab' in sys.modules\n",
    "logging.basicConfig(level=logging.INFO)\n",
    "logger = logging.getLogger(__name__)\n"
]
notebook['cells'][0]['source'] = cell_1

with open("02_Data_Import_and_Preprocessing_Enhanced.ipynb", 'w') as f:
    json.dump(notebook, f, indent=2)
print("✅ Part 1 complete")
