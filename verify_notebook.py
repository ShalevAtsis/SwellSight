import json
from pathlib import Path

nb_path = Path("02_Data_Import_and_Preprocessing_Enhanced.ipynb")
with open(nb_path) as f:
    nb = json.load(f)

print(f"✅ Notebook has {len(nb['cells'])} cells")
print(f"✅ Cell 0 uses swellsight: {'swellsight' in ''.join(nb['cells'][0]['source'])}")
print(f"✅ Cell 2 uses DataQualityAssessor: {'DataQualityAssessor' in ''.join(nb['cells'][2]['source'])}")
print(f"✅ Cell 0 uses ErrorHandler: {'ErrorHandler' in ''.join(nb['cells'][0]['source'])}")
print(f"✅ Cell 0 uses PerformanceOptimizer: {'PerformanceOptimizer' in ''.join(nb['cells'][0]['source'])}")
