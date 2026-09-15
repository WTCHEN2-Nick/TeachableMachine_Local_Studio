from __future__ import annotations

import os

os.environ.setdefault("TF_USE_LEGACY_KERAS", "1")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

from tm_local.app import create_app

app = create_app()


if __name__ == "__main__":
    import uvicorn

    # Direct object startup avoids a second string-based module import.
    uvicorn.run(app, host="127.0.0.1", port=8765, reload=False, access_log=True)
