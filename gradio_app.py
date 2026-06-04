"""Gradio app for the obstacle-detection pipeline.

Thin UI layer only -- all the actual logic lives in the `obstacle_detection`
package (config / models / car / detector / visualize). Upload a single image;
the app runs SAM + the vehicle model + Depth-Anything-3 and shows the 5-panel
view (Original | SAM | Vehicle | Depth | Obstacle Mask).

Run on the GPU server:
    python gradio_app.py
then open the printed URL (use an SSH tunnel, e.g. `ssh -L 7860:localhost:7860 ...`).

NOTE: do NOT rename this file to `gradio.py` - that shadows the `gradio` package.
"""

import os
import sys
import logging
from datetime import datetime

import gradio as gr

from obstacle_detection import (
    run_pipeline,
    render_result_figure,
    get_models,
    OBSTACLE_IMAGES_DIR,
)

# Live log file at the project root. Everything the app and pipeline emit
# (pipeline `print()`s, Gradio/uvicorn logging, tracebacks) is teed here so you
# can follow it live with `tail -f log.log` while the app runs.
LOG_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "log.log")


class _Tee:
    """Mirror a stream's writes to the real stream AND the log file, flushing
    each write so the log updates live (no buffering lag for `tail -f`)."""

    def __init__(self, stream, logfile):
        self._stream = stream
        self._logfile = logfile

    def write(self, data):
        self._stream.write(data)
        self._stream.flush()
        self._logfile.write(data)
        self._logfile.flush()
        return len(data)

    def flush(self):
        self._stream.flush()
        self._logfile.flush()

    def isatty(self):
        return getattr(self._stream, "isatty", lambda: False)()


def setup_logging():
    """Tee stdout/stderr into log.log and route the logging module there too.

    Because the logging StreamHandler writes to the (already teed) stdout, the
    file gets timestamped log records AND raw prints without duplicate handlers.
    """
    logfile = open(LOG_FILE, "a", buffering=1)  # line-buffered text mode
    sys.stdout = _Tee(sys.stdout, logfile)
    sys.stderr = _Tee(sys.stderr, logfile)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
        force=True,  # override any handlers Gradio/uvicorn already installed
    )
    logging.info("===== Gradio app started %s | logging to %s =====",
                 datetime.now().isoformat(timespec="seconds"), LOG_FILE)
    return logfile


def run(image_path):
    if image_path is None:
        logging.warning("Run requested with no image uploaded.")
        return None, "Please upload an image."
    logging.info("Running pipeline on %s", image_path)
    result = run_pipeline(image_path)
    fig_img = render_result_figure(result)
    status = "⚠️ Obstacle detected" if result["obstacle_exist"] else "✅ No obstacle"
    logging.info("Pipeline done: %s", status)
    return fig_img, status


with gr.Blocks(title="Obstacle Detection Pipeline") as demo:
    gr.Markdown("# Obstacle Detection Pipeline\nUpload an image to run SAM + vehicle + Depth-Anything-3.")
    with gr.Row():
        inp = gr.Image(type="filepath", label="Input image", height=320)
        status = gr.Textbox(label="Result", interactive=False)
    btn = gr.Button("Run pipeline", variant="primary")
    out = gr.Image(type="pil", label="Pipeline result", height=700)

    btn.click(fn=run, inputs=inp, outputs=[out, status])

    if os.path.isdir(OBSTACLE_IMAGES_DIR):
        examples = [os.path.join(OBSTACLE_IMAGES_DIR, f)
                    for f in sorted(os.listdir(OBSTACLE_IMAGES_DIR))
                    if f.lower().endswith((".png", ".jpg", ".jpeg"))]
        if examples:
            gr.Examples(examples=examples, inputs=inp)


if __name__ == "__main__":
    setup_logging()  # start teeing all output to log.log
    get_models()  # warm up so the first request is fast
    demo.launch(server_name="0.0.0.0", server_port=7860)
