"""
Gradio UI for Image Restoration and Super-Resolution

Provides a simple web interface to:
  1. Upload an image
  2. Choose a model (SRCNN, Denoising, SRResNet, SRGAN)
  3. Run inference
  4. Preview before/after side by side
  5. Download the result

Run with:
    python ui/app.py

Then open http://localhost:7860 in your browser.
"""

import os
import sys
import gradio as gr
from PIL import Image
import numpy as np

# Add project root to Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from inference.inference_pipeline import InferencePipeline


# ----------------------------------------------------------------
# Available Models — update checkpoint paths as you train models
# ----------------------------------------------------------------
AVAILABLE_MODELS = {
    "SRCNN (Super-Resolution x2)": {
        "type": "srcnn",
        "checkpoint": "checkpoints/srcnn/srcnn_best.pth",
        "description": "Fast super-resolution. Bicubic upscaling refined by CNN. Best for quick inference.",
    },
    "Denoising Autoencoder": {
        "type": "denoising",
        "checkpoint": "checkpoints/denoising/denoising_best.pth",
        "description": "Removes Gaussian noise and JPEG artifacts from images.",
    },
    "SRResNet (Super-Resolution x2)": {
        "type": "srresnet",
        "checkpoint": "checkpoints/srresnet/srresnet_best.pth",
        "description": "Deep residual network for higher quality super-resolution.",
    },
    "SRGAN (Super-Resolution x2)": {
        "type": "srgan",
        "checkpoint": "checkpoints/srgan/srgan_gen_epoch_0100.pth",
        "description": "GAN-based super-resolution for perceptually sharp results.",
    },
}

# Cache loaded pipelines to avoid reloading on every click
_pipeline_cache = {}


def get_pipeline(model_name: str) -> InferencePipeline:
    """Load pipeline (with caching so we don't reload on every inference)."""
    if model_name not in _pipeline_cache:
        model_info = AVAILABLE_MODELS[model_name]

        checkpoint_path = model_info["checkpoint"]
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(
                f"Checkpoint not found: {checkpoint_path}\n"
                f"Please train the model first."
            )

        _pipeline_cache[model_name] = InferencePipeline(
            model_type=model_info["type"],
            checkpoint_path=checkpoint_path,
            config_path="configs/config.yaml",
            device='auto',
        )

    return _pipeline_cache[model_name]


def run_inference(input_image, model_name: str, target_image=None):
    """
    Main inference function called by the Gradio interface.
    
    Args:
        input_image:  PIL Image from Gradio upload
        model_name:   Selected model name (string)
        target_image: Optional reference image for PSNR/SSIM
    
    Returns:
        Tuple of (output_image, metrics_text)
    """
    if input_image is None:
        return None, "Please upload an image first."

    if model_name not in AVAILABLE_MODELS:
        return None, "Please select a valid model."

    try:
        pipeline = get_pipeline(model_name)
    except FileNotFoundError as e:
        return None, f"Error: {str(e)}"

    # Convert numpy array from Gradio to PIL
    if isinstance(input_image, np.ndarray):
        input_pil = Image.fromarray(input_image)
    else:
        input_pil = input_image

    # Run inference
    try:
        result = pipeline.run(
            input_image=input_pil,
            target_image=target_image,
        )
    except Exception as e:
        return None, f"Inference error: {str(e)}"

    output_pil = result['output']

    # Build metrics text
    metrics_text = f"Model: {model_name}\n"
    metrics_text += f"Input size: {input_pil.size[0]}×{input_pil.size[1]} px\n"
    metrics_text += f"Output size: {output_pil.size[0]}×{output_pil.size[1]} px\n"

    if 'psnr' in result:
        metrics_text += f"\nPSNR: {result['psnr']:.2f} dB"
        metrics_text += f"\nSSIM: {result['ssim']:.4f}"

    return output_pil, metrics_text


def get_model_description(model_name: str) -> str:
    """Return description for selected model."""
    if model_name in AVAILABLE_MODELS:
        return AVAILABLE_MODELS[model_name]["description"]
    return ""


# ----------------------------------------------------------------
# Build Gradio Interface
# ----------------------------------------------------------------
def build_interface():
    model_names = list(AVAILABLE_MODELS.keys())

    with gr.Blocks(title="Image Restoration & Super-Resolution") as demo:

        gr.Markdown("""
        # Image Restoration & Super-Resolution
        Upload a low-resolution or degraded image and apply AI enhancement.
        - **SRCNN / SRResNet / SRGAN**: Upscale image by 2x
        - **Denoising Autoencoder**: Remove noise and JPEG artifacts
        """)

        with gr.Row():
            # Left column: inputs
            with gr.Column():
                input_image = gr.Image(
                    label="Input Image",
                    type="pil",
                    height=300,
                )

                model_selector = gr.Dropdown(
                    choices=model_names,
                    value=model_names[0],
                    label="Select Model",
                )

                model_description = gr.Textbox(
                    label="Model Description",
                    value=AVAILABLE_MODELS[model_names[0]]["description"],
                    interactive=False,
                    lines=2,
                )

                target_image = gr.Image(
                    label="Reference Image (optional — for PSNR/SSIM metrics)",
                    type="pil",
                    height=200,
                )

                run_button = gr.Button("Run Enhancement", variant="primary")

            # Right column: outputs
            with gr.Column():
                output_image = gr.Image(
                    label="Enhanced Output",
                    height=300,
                )

                metrics_box = gr.Textbox(
                    label="Results",
                    lines=6,
                    interactive=False,
                )

                download_button = gr.Button("Save Output Image")

        # Update model description when selection changes
        model_selector.change(
            fn=get_model_description,
            inputs=model_selector,
            outputs=model_description,
        )

        # Run inference
        run_button.click(
            fn=run_inference,
            inputs=[input_image, model_selector, target_image],
            outputs=[output_image, metrics_box],
        )

        gr.Markdown("""
        ---
        **Tips:**
        - For super-resolution: upload a small/low-resolution image
        - For denoising: upload a noisy or compressed image
        - Upload a reference HR image to compute PSNR/SSIM quality metrics
        - Results are automatically shown in the output panel
        """)

    return demo


if __name__ == "__main__":
    demo = build_interface()

    demo.launch(
        server_name="0.0.0.0",  # Accept connections from all interfaces
        server_port=7860,
        share=True,            # Set to True to get a public Gradio link
        show_error=True,
    )