"""Execute all numbered notebooks in place and ensure all output cells are saved."""
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
NOTEBOOKS_DIR = ROOT / "notebooks"

notebooks = sorted(NOTEBOOKS_DIR.glob("0*.ipynb"))

print(f"Found {len(notebooks)} notebooks to execute:")
for nb in notebooks:
    print(f"  - {nb.name}")

for nb in notebooks:
    print(f"\n[Executing] {nb.name} ...")
    cmd = [
        sys.executable,
        "-m", "nbconvert",
        "--to", "notebook",
        "--execute",
        "--inplace",
        str(nb),
        "--ExecutePreprocessor.kernel_name=day19",
        "--ExecutePreprocessor.timeout=900",
    ]
    res = subprocess.run(cmd, cwd=str(ROOT), capture_output=True, text=True)
    if res.returncode == 0:
        print(f"[PASS] {nb.name}")
    else:
        print(f"[FAIL] {nb.name}")
        print("STDOUT:\n", res.stdout[-1000:])
        print("STDERR:\n", res.stderr[-1000:])
        sys.exit(1)

print("\nAll notebooks executed successfully!")
