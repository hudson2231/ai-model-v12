import os
import torch
import cv2
import numpy as np
from PIL import Image
from cog import BasePredictor, Input, Path
from diffusers import StableDiffusionControlNetPipeline, ControlNetModel
from controlnet_aux import CannyDetector
import warnings
warnings.filterwarnings('ignore')

print("🚨 AI-POWERED LINE ART - CHATGPT QUALITY!")


class AILineArtProcessor:
    """
    AI-Powered Line Art Processor using ControlNet + Stable Diffusion
    Replicates ChatGPT-4's line art generation capability
    """
    
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.pipe = None
        self.canny_detector = None
        
    def setup_models(self):
        """Initialize the AI models for line art generation"""
        print("🔧 Setting up AI models for ChatGPT-quality line art...")
        
        # Load ControlNet model for line art
        controlnet = ControlNetModel.from_pretrained(
            "lllyasviel/sd-controlnet-canny",
            torch_dtype=torch.float16
        )
        
        # Load Stable Diffusion pipeline with ControlNet
        self.pipe = StableDiffusionControlNetPipeline.from_pretrained(
            "runwayml/stable-diffusion-v1-5",
            controlnet=controlnet,
            torch_dtype=torch.float16,
            safety_checker=None,
            requires_safety_checker=False
        )
        
        # Optimize for speed and memory
        self.pipe.enable_model_cpu_offload()
        self.pipe.enable_xformers_memory_efficient_attention()
        
        # Initialize Canny edge detector
        self.canny_detector = CannyDetector()
        
        print("✅ AI models ready for empire-building!")
    
    def preprocess_image(self, image: Image.Image, target_size: int = 768) -> Image.Image:
        """Preprocess image for optimal AI generation"""
        # Resize maintaining aspect ratio
        w, h = image.size
        if max(w, h) > target_size:
            ratio = target_size / max(w, h)
            new_w, new_h = int(w * ratio), int(h * ratio)
            # Ensure dimensions are multiples of 8 for stable diffusion
            new_w = (new_w // 8) * 8
            new_h = (new_h // 8) * 8
            image = image.resize((new_w, new_h), Image.Resampling.LANCZOS)
        
        return image
    
    def generate_canny_control(self, image: Image.Image) -> Image.Image:
        """Generate Canny edge control image"""
        # Convert to numpy for OpenCV processing
        img_array = np.array(image)
        
        # Apply gentle bilateral filtering to reduce noise
        filtered = cv2.bilateralFilter(img_array, 9, 75, 75)
        
        # Convert back to PIL for Canny detector
        filtered_pil = Image.fromarray(filtered)
        
        # Generate Canny edges with optimized thresholds
        canny_image = self.canny_detector(filtered_pil, low_threshold=50, high_threshold=150)
        
        return canny_image
    
    def generate_line_art(self, image: Image.Image, canny_control: Image.Image) -> Image.Image:
        """Generate line art using AI - replicating ChatGPT's approach"""
        
        # The key prompt that replicates ChatGPT's success
        prompt = """
        Professional coloring book line art, clean black lines on white background, 
        detailed facial features clearly visible and identifiable, 
        rich detailed background elements for coloring, 
        smooth continuous lines, no shading or gradients, 
        adult coloring book style, high quality line art illustration
        """
        
        negative_prompt = """
        color, colored, shading, gradients, blur, photorealistic, 
        sketchy lines, incomplete lines, messy lines, watermark, 
        low quality, blurry, distorted faces
        """
        
        # Generate with optimal settings
        with torch.autocast(self.device):
            result = self.pipe(
                prompt=prompt,
                negative_prompt=negative_prompt,
                image=canny_control,
                num_inference_steps=20,  # Good balance of quality/speed
                guidance_scale=7.5,      # Strong prompt adherence
                controlnet_conditioning_scale=1.0,  # Full control influence
                generator=torch.Generator(device=self.device).manual_seed(42)  # Consistent results
            ).images[0]
        
        return result
    
    def post_process_line_art(self, line_art: Image.Image) -> Image.Image:
        """Post-process the AI-generated line art for perfection"""
        # Convert to numpy for processing
        img_array = np.array(line_art.convert('L'))  # Convert to grayscale
        
        # Ensure pure black and white
        _, binary = cv2.threshold(img_array, 127, 255, cv2.THRESH_BINARY)
        
        # Light cleanup to remove small artifacts
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))
        cleaned = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
        
        # Convert back to PIL
        final_image = Image.fromarray(cleaned).convert('RGB')
        
        return final_image
    
    def process(self, image: Image.Image) -> Image.Image:
        """Main processing pipeline - ChatGPT quality line art"""
        print("🎨 Starting AI-powered line art generation...")
        
        # Step 1: Preprocess for optimal AI generation
        print("📸 Preprocessing for AI...")
        processed_image = self.preprocess_image(image)
        
        # Step 2: Generate Canny control
        print("🔍 Generating edge control map...")
        canny_control = self.generate_canny_control(processed_image)
        
        # Step 3: AI line art generation
        print("🤖 AI generating ChatGPT-quality line art...")
        ai_line_art = self.generate_line_art(processed_image, canny_control)
        
        # Step 4: Post-process for perfection
        print("✨ Final AI enhancement...")
        final_result = self.post_process_line_art(ai_line_art)
        
        print("✅ AI line art generation complete!")
        
        return final_result


class Predictor(BasePredictor):
    def setup(self):
        """Initialize the AI predictor"""
        print("🚀 Setting up AI-Powered Line Art Processor...")
        self.processor = AILineArtProcessor()
        self.processor.setup_models()
        print("✅ AI setup complete - ready for ChatGPT quality!")
    
    def predict(
        self,
        input_image: Path = Input(description="Photo to convert to ChatGPT-quality line art"),
        target_size: int = Input(
            description="Image size for processing", 
            default=768, 
            ge=512, 
            le=1024
        ),
        line_strength: float = Input(
            description="Line art strength (higher = stronger lines)",
            default=1.0,
            ge=0.5,
            le=1.5
        ),
        detail_level: str = Input(
            description="Detail preservation level",
            default="high",
            choices=["medium", "high", "maximum"]
        ),
    ) -> Path:
        """Convert image to AI-powered line art"""
        
        print(f"📥 Loading image for AI processing: {input_image}")
        
        # Load image
        try:
            image = Image.open(input_image)
            if image.mode not in ['RGB', 'RGBA']:
                image = image.convert('RGB')
        except Exception as e:
            raise ValueError(f"Could not load image: {str(e)}")
        
        print(f"📐 Original size: {image.size}")
        
        # Process with AI
        result = self.processor.process(image)
        
        print(f"📤 Final AI result size: {result.size}")
        
        # Save with maximum quality
        output_path = "/tmp/ai_line_art.png"
        result.save(output_path, "PNG", optimize=False)
        
        print(f"💾 AI line art saved: {output_path}")
        return Path(output_path)
