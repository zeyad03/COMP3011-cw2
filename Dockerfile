# Reproducible runtime image for the COMP3011-CW2 search engine.
#
# Build:    docker build -t cw2-search .
# Run:      docker run --rm -it -v "$(pwd)/data:/app/data" cw2-search
#
# The image only includes runtime dependencies; the test/dev tooling
# (pytest, mypy, hypothesis, scikit-learn) is intentionally excluded
# to keep the image small. For the dev loop, run pytest on the host.

FROM python:3.13-slim

WORKDIR /app

# Install runtime deps first so the layer is cacheable across code changes.
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

COPY src/ ./src/
COPY data/ ./data/

# The interactive shell needs a TTY: pass `-it` when running.
CMD ["python", "-m", "src.main"]
