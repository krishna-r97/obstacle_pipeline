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

import gradio as gr

from obstacle_detection import (
    run_pipeline,
    render_result_figure,
    get_models,
    OBSTACLE_IMAGES_DIR,
)


def run(image_path):
    if image_path is None:
        return None, "Please upload an image."
    result = run_pipeline(image_path)
    fig_img = render_result_figure(result)
    status = "⚠️ Obstacle detected" if result["obstacle_exist"] else "✅ No obstacle"
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
    get_models()  # warm up so the first request is fast
    demo.launch(server_name="0.0.0.0", server_port=7860)
