"""Generate frontend contracts without opening or migrating the user's memory."""

import json
import os
import subprocess
import tempfile
from pathlib import Path

with tempfile.TemporaryDirectory(prefix="grounded-contracts-") as directory:
    os.environ["GM_DATA_DIR"] = directory
    from grounded_memory.api import app

    schema = Path(directory) / "openapi.json"
    schema.write_text(json.dumps(app.openapi()))
    subprocess.run(
        ["node", "node_modules/openapi-typescript/bin/cli.js", str(schema), "-o", "src/api.d.ts"],
        cwd=Path(__file__).resolve().parents[1] / "web",
        check=True,
    )
