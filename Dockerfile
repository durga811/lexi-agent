# Hugging Face Spaces (Docker SDK) — runs the Streamlit app.
FROM python:3.11-slim

# HF Spaces run containers as a non-root user (uid 1000); create it so the app
# can write the vector index + model cache into its home.
RUN useradd -m -u 1000 user
USER user
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH \
    HF_HOME=/home/user/.cache/huggingface \
    TOKENIZERS_PARALLELISM=false

WORKDIR /home/user/app

# Install deps first so this layer caches across code changes.
COPY --chown=user requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# App code + the extracted corpus (data/corpus.jsonl). The vector index is built
# from it on first launch.
COPY --chown=user . .

# HF routes traffic to $app_port (7860, set in README). Streamlit must bind there
# on all interfaces, headless (no browser).
EXPOSE 7860
CMD ["streamlit", "run", "app.py", \
     "--server.port=7860", "--server.address=0.0.0.0", "--server.headless=true"]
