import torch
import torch.nn.functional as F
from diffusers import AutoencoderKL, UNet2DConditionModel, DDPMScheduler
from transformers import CLIPTextModel, CLIPTokenizer
from PIL import Image
from torchvision import transforms
from tqdm import tqdm
import os


class WorkingArtProtector:
    """
    Working version of adversarial image protector
    Fixes all the gradient flow issues
    """

    def __init__(self, model_id="runwayml/stable-diffusion-v1-5", device="cuda"):
        if device == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA not available! Install PyTorch with CUDA support.")

        self.device = device
        print(f"✓ Using GPU: {torch.cuda.get_device_name(0)}")

        print(f"\nLoading Stable Diffusion components...")

        # Load components
        self.vae = AutoencoderKL.from_pretrained(model_id, subfolder="vae").to(device)
        self.unet = UNet2DConditionModel.from_pretrained(model_id, subfolder="unet").to(device)
        self.text_encoder = CLIPTextModel.from_pretrained(model_id, subfolder="text_encoder").to(device)
        self.tokenizer = CLIPTokenizer.from_pretrained(model_id, subfolder="tokenizer")
        self.scheduler = DDPMScheduler.from_pretrained(model_id, subfolder="scheduler")

        # Set to eval and freeze (we don't train these models)
        self.vae.eval()
        self.unet.eval()
        self.text_encoder.eval()

        for param in self.vae.parameters():
            param.requires_grad = False
        for param in self.unet.parameters():
            param.requires_grad = False
        for param in self.text_encoder.parameters():
            param.requires_grad = False

        print("✓ Models loaded\n")

    def load_image(self, image_path, size=512):
        """Load and preprocess image"""
        image = Image.open(image_path).convert("RGB")
        transform = transforms.Compose([
            transforms.Resize((size, size)),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5])
        ])
        return transform(image).unsqueeze(0).to(self.device)

    def encode_text(self, prompt):
        """Encode text to embeddings"""
        text_input = self.tokenizer(
            prompt,
            padding="max_length",
            max_length=self.tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt"
        )
        with torch.no_grad():
            text_embeddings = self.text_encoder(text_input.input_ids.to(self.device))[0]
        return text_embeddings

    def protect_image(
            self,
            image_path,
            output_path,
            target_prompt="",
            epsilon=0.03,
            steps=50,
            learning_rate=0.01
    ):
        """
        Generate protected version of an image

        Args:
            image_path: Path to input image
            output_path: Where to save protected image
            target_prompt: What the image might be trained as
            epsilon: Max perturbation strength (0.03 ≈ 8/255)
            steps: Optimization iterations
            learning_rate: Step size
        """
        print(f"🛡️  Protecting: {image_path}")
        print(f"   Target prompt: '{target_prompt}'")
        print(f"   Epsilon: {epsilon} (~{epsilon * 255:.0f}/255)")
        print(f"   Steps: {steps}\n")

        # Load image
        clean_image = self.load_image(image_path)

        # Initialize perturbation
        perturbation = torch.zeros_like(clean_image, requires_grad=True, device=self.device)

        # Optimizer
        optimizer = torch.optim.Adam([perturbation], lr=learning_rate)

        # Encode text
        text_embeddings = self.encode_text(target_prompt if target_prompt else "")

        # Optimization loop
        pbar = tqdm(range(steps), desc="Optimizing")
        for step in pbar:
            # Zero gradients
            optimizer.zero_grad()

            # Apply perturbation
            perturbed_image = clean_image + perturbation
            perturbed_image = torch.clamp(perturbed_image, -1, 1)

            # Encode to latent (deterministic, no sampling)
            with torch.enable_grad():
                posterior = self.vae.encode(perturbed_image).latent_dist
                # Use mean instead of sample for deterministic gradients
                latents = posterior.mean * 0.18215

            # Random timestep
            t = torch.randint(0, self.scheduler.config.num_train_timesteps, (1,), device=self.device).long()

            # Add noise
            noise = torch.randn_like(latents)
            noisy_latents = self.scheduler.add_noise(latents, noise, t)

            # Predict noise
            with torch.enable_grad():
                noise_pred = self.unet(
                    noisy_latents,
                    t,
                    encoder_hidden_states=text_embeddings
                ).sample

            # Attack loss: maximize prediction error
            loss = -F.mse_loss(noise_pred, noise)

            # Backward
            loss.backward()

            # Update
            optimizer.step()

            # Project to epsilon ball
            with torch.no_grad():
                perturbation.data = torch.clamp(perturbation.data, -epsilon, epsilon)

            # Update progress
            if step % 10 == 0:
                pbar.set_postfix({
                    'loss': f'{loss.item():.4f}',
                    'max_pert': f'{perturbation.abs().max().item():.4f}'
                })

        # Save protected image
        with torch.no_grad():
            protected = clean_image + perturbation
            protected = torch.clamp(protected, -1, 1)

            # Denormalize
            protected = (protected + 1) / 2
            protected = protected.clamp(0, 1)

            # Convert to PIL and save
            protected_np = protected.squeeze(0).cpu().permute(1, 2, 0).numpy()
            protected_pil = Image.fromarray((protected_np * 255).astype('uint8'))
            protected_pil.save(output_path, quality=95)

        print(f"\n✅ Protected image saved to: {output_path}")
        print(f"   Max perturbation: {perturbation.abs().max().item():.4f}")
        print(f"   L2 norm: {torch.norm(perturbation).item():.4f}")

        return protected_pil


# Simple usage example
if __name__ == "__main__":
    # Configuration
    INPUT_IMAGE = "PFP JANE2025.jpg"  # ← CHANGE THIS
    OUTPUT_IMAGE = "protected_image_PFP.jpg"

    # Check if image exists
    if not os.path.exists(INPUT_IMAGE):
        print(f"❌ Error: '{INPUT_IMAGE}' not found!")
        print("   Please update INPUT_IMAGE to your actual image path.")
        exit(1)

    # Create protector
    protector = WorkingArtProtector()

    # Protect the image
    protector.protect_image(
        image_path=INPUT_IMAGE,
        output_path=OUTPUT_IMAGE,
        target_prompt="Hornet",
        epsilon=0.03,  # Subtle but effective
        steps=50,  # More steps = better protection
        learning_rate=0.01
    )

    print("\n🎉 Done! The protected image looks identical but will poison AI training.")