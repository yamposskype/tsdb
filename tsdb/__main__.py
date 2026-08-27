"""Entry point: python -m tsdb starts the HTTP server on port 9090."""

import uvicorn

from tsdb.api import app

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=9090)
