# OmniConsistency

#### Xiang Demo
```python
import time
import torch
import os
from tqdm import tqdm
from PIL import Image
from src_inference.pipeline import FluxPipeline
from src_inference.lora_helper import set_single_lora

# 初始化基础模型
base_path = "black-forest-labs/FLUX.1-dev"
pipe = FluxPipeline.from_pretrained(base_path, torch_dtype=torch.bfloat16)

# 设置OmniConsistency LoRA
set_single_lora(pipe.transformer, 
                "../OmniConsistency/OmniConsistency.safetensors", 
                lora_weights=[1], cond_size=512)

# 定义所有LoRA风格
lora_names = [
    "3D_Chibi", "American_Cartoon", "Chinese_Ink", "Clay_Toy", "Fabric",
    "Ghibli", "Irasutoya", "Jojo", "LEGO", "Line", "Macaron", "Oil_Painting",
    "Origami", "Paper_Cutting", "Picasso", "Pixel", "Poly", "Pop_Art",
    "Rick_Morty", "Snoopy", "Van_Gogh", "Vector"
]

# 加载所有LoRA
lora_dir = "../OmniConsistency/LoRAs"
for name in lora_names:
    weight_file = f"{name}_rank128_bf16.safetensors"
    pipe.load_lora_weights(lora_dir, weight_name=weight_file, adapter_name=name)

# 启用CPU offload节省显存
pipe.enable_sequential_cpu_offload()

# 输入设置
image_path1 = "xiang_image.jpg"
base_prompt = "A young man with black hair and glasses sits outdoors in a white T-shirt, surrounded by lush green foliage. The natural backdrop contrasts with his relaxed posture, creating a serene vibe."
spatial_image = [Image.open(image_path1).convert("RGB")]
subject_images = []
width, height = 1024, 1024

# 创建输出目录
output_dir = "results/lora_styles"
os.makedirs(output_dir, exist_ok=True)

# 遍历所有LoRA风格生成图片
for name in tqdm(lora_names, desc="Generating style variations"):
    # 设置当前LoRA
    pipe.set_adapters(name)
    
    # 构建风格化提示词
    style_prompt = f"{name.replace('_', ' ')} style, " + base_prompt
    
    # 生成图片
    start_time = time.time()
    image = pipe(
        style_prompt,
        height=height,
        width=width,
        guidance_scale=3.5,
        num_inference_steps=25,
        max_sequence_length=512,
        generator=torch.Generator("cpu").manual_seed(5),
        spatial_images=spatial_image,
        subject_images=subject_images,
        cond_size=512,
    ).images[0]
    
    # 保存结果
    output_path = os.path.join(output_dir, f"{name}.png")
    image.save(output_path)
    
    # 打印耗时
    elapsed_time = time.time() - start_time
    tqdm.write(f"Generated {name} in {elapsed_time:.2f}s")

print("All style generations completed!")

from datasets import Dataset, Image
import os

# Scan directory and create dataset
def create_style_dataset(directory="results/lora_styles"):
    samples = []
    for filename in os.listdir(directory):
        if filename.endswith(".png"):
            img_path = os.path.join(directory, filename)
            style_name = filename.replace(".png", "").replace("_", " ")
            samples.append({"image": img_path, "style": style_name})
    
    # Convert to HuggingFace Dataset
    ds = Dataset.from_dict({
        "image": [x["image"] for x in samples],
        "style": [x["style"] for x in samples]
    })
    
    # Cast image column to Image type
    ds = ds.cast_column("image", Image())
    return ds

# Usage
style_dataset = create_style_dataset()
print(style_dataset)
```

> **OmniConsistency: Learning Style-Agnostic
Consistency from Paired Stylization Data**
> <br>
> [Yiren Song](https://scholar.google.com.hk/citations?user=L2YS0jgAAAAJ), 
> [Cheng Liu](https://scholar.google.com.hk/citations?hl=zh-CN&user=TvdVuAYAAAAJ), 
> and 
> [Mike Zheng Shou](https://sites.google.com/view/showlab)
> <br>
> [Show Lab](https://sites.google.com/view/showlab), National University of Singapore
> <br>

<a href="https://huggingface.co/spaces/yiren98/OmniConsistency"><img src="https://img.shields.io/badge/🤗_HuggingFace-Space-ffbd45.svg" alt="HuggingFace"></a>
<a href="https://huggingface.co/showlab/OmniConsistency"><img src="https://img.shields.io/badge/🤗_HuggingFace-Model-ffbd45.svg" alt="HuggingFace"></a>



<img src='./figure/teaser.png' width='100%' />

## Installation

We recommend using Python 3.10 and PyTorch with CUDA support. To set up the environment:

```bash
# Create a new conda environment
conda create -n omniconsistency python=3.10
conda activate omniconsistency

# Install other dependencies
pip install -r requirements.txt
```

## Download

You can download the OmniConsistency model and pretrained LoRAs directly from [Hugging Face](https://huggingface.co/showlab/OmniConsistency).
Or download using Python script:

### OmniConsistency Model

```python
from huggingface_hub import hf_hub_download
hf_hub_download(repo_id="showlab/OmniConsistency", filename="LoRAs/3D_Chibi_rank128_bf16.safetensors", local_dir="./LoRAs")
hf_hub_download(repo_id="showlab/OmniConsistency", filename="LoRAs/American_Cartoon_rank128_bf16.safetensors", local_dir="./LoRAs")
hf_hub_download(repo_id="showlab/OmniConsistency", filename="LoRAs/Chinese_Ink_rank128_bf16.safetensors", local_dir="./LoRAs")
hf_hub_download(repo_id="showlab/OmniConsistency", filename="LoRAs/Clay_Toy_rank128_bf16.safetensors", local_dir="./LoRAs")
hf_hub_download(repo_id="showlab/OmniConsistency", filename="LoRAs/Fabric_rank128_bf16.safetensors", local_dir="./LoRAs")
hf_hub_download(repo_id="showlab/OmniConsistency", filename="LoRAs/Ghibli_rank128_bf16.safetensors", local_dir="./LoRAs")
hf_hub_download(repo_id="showlab/OmniConsistency", filename="LoRAs/Irasutoya_rank128_bf16.safetensors", local_dir="./LoRAs")
hf_hub_download(repo_id="showlab/OmniConsistency", filename="LoRAs/Jojo_rank128_bf16.safetensors", local_dir="./LoRAs")
hf_hub_download(repo_id="showlab/OmniConsistency", filename="LoRAs/LEGO_rank128_bf16.safetensors", local_dir="./LoRAs")
hf_hub_download(repo_id="showlab/OmniConsistency", filename="LoRAs/Line_rank128_bf16.safetensors", local_dir="./LoRAs")
hf_hub_download(repo_id="showlab/OmniConsistency", filename="LoRAs/Macaron_rank128_bf16.safetensors", local_dir="./LoRAs")
hf_hub_download(repo_id="showlab/OmniConsistency", filename="LoRAs/Oil_Painting_rank128_bf16.safetensors", local_dir="./LoRAs")
hf_hub_download(repo_id="showlab/OmniConsistency", filename="LoRAs/Origami_rank128_bf16.safetensors", local_dir="./LoRAs")
hf_hub_download(repo_id="showlab/OmniConsistency", filename="LoRAs/Paper_Cutting_rank128_bf16.safetensors", local_dir="./LoRAs")
hf_hub_download(repo_id="showlab/OmniConsistency", filename="LoRAs/Picasso_rank128_bf16.safetensors", local_dir="./LoRAs")
hf_hub_download(repo_id="showlab/OmniConsistency", filename="LoRAs/Pixel_rank128_bf16.safetensors", local_dir="./LoRAs")
hf_hub_download(repo_id="showlab/OmniConsistency", filename="LoRAs/Poly_rank128_bf16.safetensors", local_dir="./LoRAs")
hf_hub_download(repo_id="showlab/OmniConsistency", filename="LoRAs/Pop_Art_rank128_bf16.safetensors", local_dir="./LoRAs")
hf_hub_download(repo_id="showlab/OmniConsistency", filename="LoRAs/Rick_Morty_rank128_bf16.safetensors", local_dir="./LoRAs")
hf_hub_download(repo_id="showlab/OmniConsistency", filename="LoRAs/Snoopy_rank128_bf16.safetensors", local_dir="./LoRAs")
hf_hub_download(repo_id="showlab/OmniConsistency", filename="LoRAs/Van_Gogh_rank128_bf16.safetensors", local_dir="./LoRAs")
hf_hub_download(repo_id="showlab/OmniConsistency", filename="LoRAs/Vector_rank128_bf16.safetensors", local_dir="./LoRAs")
```
### Pretrained LoRAs
```python
from huggingface_hub import hf_hub_download
hf_hub_download(repo_id="showlab/OmniConsistency", filename="OmniConsistency.safetensors", local_dir="./Model")
```

## Usage
Here's a basic example of using OmniConsistency:

### Model Initialization
```python
import time
import torch
from PIL import Image
from src_inference.pipeline import FluxPipeline
from src_inference.lora_helper import set_single_lora

def clear_cache(transformer):
    for name, attn_processor in transformer.attn_processors.items():
        attn_processor.bank_kv.clear()

# Initialize model
device = "cuda"
base_path = "/path/to/black-forest-labs/FLUX.1-dev"
pipe = FluxPipeline.from_pretrained(base_path, torch_dtype=torch.bfloat16).to("cuda")

# Load OmniConsistency model
set_single_lora(pipe.transformer, 
                "/path/to/OmniConsistency.safetensors", 
                lora_weights=[1], cond_size=512)

# Load external LoRA
pipe.unload_lora_weights()
pipe.load_lora_weights("/path/to/lora_folder", 
                       weight_name="lora_name.safetensors")
```

### Style Inference
```python
image_path1 = "figure/test.png"
prompt = "3D Chibi style, Three individuals standing together in the office."

subject_images = []
spatial_image = [Image.open(image_path1).convert("RGB")]

width, height = 1024, 1024

start_time = time.time()

image = pipe(
    prompt,
    height=height,
    width=width,
    guidance_scale=3.5,
    num_inference_steps=25,
    max_sequence_length=512,
    generator=torch.Generator("cpu").manual_seed(5),
    spatial_images=spatial_image,
    subject_images=subject_images,
    cond_size=512,
).images[0]

end_time = time.time()
elapsed_time = end_time - start_time
print(f"code running time: {elapsed_time} s")

# Clear cache after generation
clear_cache(pipe.transformer)

image.save("results/output.png")
```

<!-- ## Citation
```

``` -->
