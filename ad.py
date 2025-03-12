import torch
from diffusers import StableDiffusionPipeline
import imageio
import os

# Check for GPU availability
device = "cuda" if torch.cuda.is_available() else "cpu"

# Load AI model
pipe = StableDiffusionPipeline.from_pretrained("runwayml/stable-diffusion-v1-5").to(device)

# Get user input
user_prompt = input("Enter your advertisement idea: ")

# Create output folder
os.makedirs("frames", exist_ok=True)

# Generate frames
frames = []
for i in range(30):  # 10 FPS * 3 seconds = 30 frames
    print(f"Generating frame {i+1}/30...")
    image = pipe(user_prompt).images[0]
    frame_path = f"frames/frame_{i:03d}.png"
    image.save(frame_path)
    frames.append(frame_path)

# Convert frames to GIF
gif_path = "output_ad.gif"
imageio.mimsave(gif_path, [imageio.imread(frame) for frame in frames], duration=0.1)  # 10 FPS

print(f"GIF saved as {gif_path} 🎉")
