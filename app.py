# import spaces
import time
import torch
import gradio as gr
from PIL import Image
from huggingface_hub import hf_hub_download
from src_inference.pipeline import FluxPipeline
from src_inference.lora_helper import set_single_lora
import random

base_path = "black-forest-labs/FLUX.1-dev"
    
# Download OmniConsistency LoRA using hf_hub_download
omni_consistency_path = hf_hub_download(repo_id="showlab/OmniConsistency", 
                                        filename="OmniConsistency.safetensors", 
                                        local_dir="./Model")

# Initialize the pipeline with the model
pipe = FluxPipeline.from_pretrained(base_path, torch_dtype=torch.bfloat16).to("cuda")

# Set LoRA weights
set_single_lora(pipe.transformer, omni_consistency_path, lora_weights=[1], cond_size=512)

# Function to clear cache
def clear_cache(transformer):
    for name, attn_processor in transformer.attn_processors.items():
        attn_processor.bank_kv.clear()

# Function to download all LoRAs in advance
def download_all_loras():
    lora_names = [
        "3D_Chibi", "American_Cartoon", "Chinese_Ink", 
        "Clay_Toy", "Fabric", "Ghibli", "Irasutoya",
        "Jojo", "LEGO", "Line", "Macaron",
        "Oil_Painting", "Origami", "Paper_Cutting", 
        "Picasso", "Pixel", "Poly", "Pop_Art", 
        "Rick_Morty", "Snoopy", "Van_Gogh", "Vector"
    ]
    for lora_name in lora_names:
        hf_hub_download(repo_id="showlab/OmniConsistency", 
                        filename=f"LoRAs/{lora_name}_rank128_bf16.safetensors", 
                        local_dir="./LoRAs")

# Download all LoRAs in advance before the interface is launched
download_all_loras()

# Main function to generate the image
# @spaces.GPU()
def generate_image(lora_name, prompt, uploaded_image, width, height, guidance_scale, num_inference_steps, seed):
    # Download specific LoRA based on selection (use local directory as LoRAs are already downloaded)
    lora_path = f"./LoRAs/LoRAs/{lora_name}_rank128_bf16.safetensors"

    # Load the specific LoRA weights
    pipe.unload_lora_weights()
    pipe.load_lora_weights("./LoRAs/LoRAs", weight_name=f"{lora_name}_rank128_bf16.safetensors")

    # Prepare input image
    spatial_image = [uploaded_image.convert("RGB")]
    subject_images = []

    start_time = time.time()

    # Generate the image
    image = pipe(
        prompt,
        height=(int(height) // 8) * 8,
        width=(int(width) // 8) * 8,
        guidance_scale=guidance_scale,
        num_inference_steps=num_inference_steps,
        max_sequence_length=512,
        generator=torch.Generator("cpu").manual_seed(seed),
        spatial_images=spatial_image,
        subject_images=subject_images,
        cond_size=512,
    ).images[0]

    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"code running time: {elapsed_time} s")

    # Clear cache after generation
    clear_cache(pipe.transformer)

    return image

# Example data
examples = [
    ["3D_Chibi", "3D Chibi style",                  Image.open("./test_imgs/00.png"), 680, 1024, 3.5, 24, 42],
    ["Origami", "Origami style",                    Image.open("./test_imgs/01.png"), 560, 1024, 3.5, 24, 42],
    ["American_Cartoon", "American Cartoon style",  Image.open("./test_imgs/02.png"), 568, 1024, 3.5, 24, 42],
    ["Origami", "Origami style",                    Image.open("./test_imgs/03.png"), 768, 672, 3.5, 24, 42],
    ["Paper_Cutting", "Paper Cutting style",        Image.open("./test_imgs/04.png"), 696, 1024, 3.5, 24, 42]
]

# Gradio interface setup
def create_gradio_interface():
    lora_names = [
        "3D_Chibi", "American_Cartoon", "Chinese_Ink", 
        "Clay_Toy", "Fabric", "Ghibli", "Irasutoya",
        "Jojo", "LEGO", "Line", "Macaron",
        "Oil_Painting", "Origami", "Paper_Cutting", 
        "Picasso", "Pixel", "Poly", "Pop_Art", 
        "Rick_Morty", "Snoopy", "Van_Gogh", "Vector"
    ]

    with gr.Blocks() as demo:
        gr.Markdown("# OmniConsistency LoRA Image Generation")
        gr.Markdown("Select a LoRA, enter a prompt, and upload an image to generate a new image with OmniConsistency.")
        with gr.Row():
            with gr.Column(scale=1):
                lora_dropdown = gr.Dropdown(lora_names, label="Select LoRA")
                prompt_box = gr.Textbox(label="Prompt", placeholder="Enter a prompt...")
                image_input = gr.Image(type="pil", label="Upload Image")
            with gr.Column(scale=1):
                width_box = gr.Textbox(label="Width", value="1024")
                height_box = gr.Textbox(label="Height", value="1024")
                guidance_slider = gr.Slider(minimum=0.1, maximum=20, value=3.5, step=0.1, label="Guidance Scale")
                steps_slider = gr.Slider(minimum=1, maximum=50, value=25, step=1, label="Inference Steps")
                seed_slider = gr.Slider(minimum=1, maximum=10000000000, value=42, step=1, label="Seed")
                generate_button = gr.Button("Generate")
                output_image = gr.Image(type="pil", label="Generated Image")
        # Add examples for Generation
        gr.Examples(
            examples=examples,
            inputs=[lora_dropdown, prompt_box, image_input, height_box, width_box, guidance_slider, steps_slider, seed_slider],
            outputs=output_image,
            fn=generate_image,
            cache_examples=False,
            label="Examples"
        )

        generate_button.click(
            fn=generate_image,
            inputs=[
                lora_dropdown, prompt_box, image_input,
                width_box, height_box, guidance_slider,
                steps_slider, seed_slider
            ],
            outputs=output_image
        )

    return demo


# Launch the Gradio interface
interface = create_gradio_interface()
interface.launch()
