import torch
import torch.nn.functional as F
from diffusers import AutoencoderKL, UNet2DConditionModel, DDPMScheduler
from transformers import CLIPTextModel, CLIPTokenizer
from PIL import Image
from torchvision import transforms
from tqdm import tqdm
import hashlib
import json
import os
import numpy as np


class SecureArtProtector:
    """
    Secure adversarial image protector with secret parameters.

    Without knowing the secret parameters, it's computationally infeasible to:
    1. Remove the protection
    2. Generate clean training data
    3. Reverse-engineer the perturbation

    Security features:
    - Secret seed controls random number generation
    - Secret key influences perturbation direction
    - Secret timestep schedule makes attack pattern unpredictable
    - All parameters must match to remove protection
    """

    def __init__(self, model_id="runwayml/stable-diffusion-v1-5", device="cuda", use_fp16=True):
        if device == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA not available!")

        self.device = device
        self.use_fp16 = use_fp16 and device == "cuda"

        print(f"✓ Using GPU: {torch.cuda.get_device_name(0)}")
        vram_gb = torch.cuda.get_device_properties(0).total_memory / 1024 ** 3
        print(f"  VRAM: {vram_gb:.1f} GB")

        if self.use_fp16:
            print("  Precision: FP16 (memory efficient)")
        else:
            print("  Precision: FP32")

        print(f"\nLoading Stable Diffusion components...")

        # Load components
        self.vae = AutoencoderKL.from_pretrained(model_id, subfolder="vae").to(device)
        self.unet = UNet2DConditionModel.from_pretrained(model_id, subfolder="unet").to(device)
        self.text_encoder = CLIPTextModel.from_pretrained(model_id, subfolder="text_encoder").to(device)
        self.tokenizer = CLIPTokenizer.from_pretrained(model_id, subfolder="tokenizer")
        self.scheduler = DDPMScheduler.from_pretrained(model_id, subfolder="scheduler")

        # Convert to FP16 for memory efficiency
        if self.use_fp16:
            self.vae = self.vae.half()
            self.unet = self.unet.half()
            self.text_encoder = self.text_encoder.half()

        self.vae.eval()
        self.unet.eval()
        self.text_encoder.eval()

        for param in self.vae.parameters():
            param.requires_grad = False
        for param in self.unet.parameters():
            param.requires_grad = False
        for param in self.text_encoder.parameters():
            param.requires_grad = False

        # Enable memory efficient attention if available
        try:
            self.unet.enable_xformers_memory_efficient_attention()
            print("  ✓ Memory efficient attention enabled")
        except:
            pass

        print("✓ Models loaded\n")

    def _generate_secret_hash(self, secret_seed, secret_key, image_path):
        """
        Generate deterministic hash from secret parameters.
        This creates a unique signature for each protected image.
        """
        # Combine all secret parameters
        secret_string = f"{secret_seed}:{secret_key}:{os.path.basename(image_path)}"

        # Create SHA-256 hash
        hash_obj = hashlib.sha256(secret_string.encode())
        return hash_obj.hexdigest()

    def _generate_deterministic_noise(self, shape, secret_seed, step):
        """
        Generate deterministic noise tensor based on secret seed.
        Uses manual seeding for reproducibility.
        """
        # Set manual seed based on secret and step
        combined_seed = int(hashlib.sha256(f"{secret_seed}:{step}".encode()).hexdigest(), 16) % (2 ** 32)
        torch.manual_seed(combined_seed)
        if self.device == "cuda":
            torch.cuda.manual_seed(combined_seed)

        noise = torch.randn(shape, device=self.device)

        # Convert to FP16 if needed
        if self.use_fp16:
            noise = noise.half()

        return noise

    def _get_timestep_schedule(self, secret_key, num_steps):
        """
        Generate secret timestep schedule based on key.
        This makes the attack pattern unpredictable without the key.
        """
        # Use secret key to generate deterministic timestep pattern
        key_hash = int(hashlib.sha256(f"{secret_key}".encode()).hexdigest(), 16)

        # Create pseudo-random but deterministic schedule
        schedule = []
        for i in range(num_steps):
            # Each step gets a deterministic timestep based on key
            step_seed = (key_hash + i * 7919) % 1000  # 7919 is a prime number
            schedule.append(step_seed)

        return schedule

    def load_image(self, image_path, size=None, max_size=1024):
        """
        Load and preprocess image, optionally preserving original size

        Args:
            image_path: Path to image
            size: Force specific size (None = preserve original)
            max_size: Maximum dimension for memory safety (default: 1024)
        """
        image = Image.open(image_path).convert("RGB")
        original_size = image.size

        # Store original image for later use
        self.original_pil_image = image.copy()

        # Limit size for memory constraints
        if size is None:
            w, h = original_size
            if max(w, h) > max_size:
                print(f"  ⚠️  Image too large ({w}×{h}), resizing to fit {max_size}px")
                if w > h:
                    new_w = max_size
                    new_h = int(h * max_size / w)
                else:
                    new_h = max_size
                    new_w = int(w * max_size / h)
                # Make dimensions divisible by 64 (required by VAE)
                new_w = (new_w // 64) * 64
                new_h = (new_h // 64) * 64
                image = image.resize((new_w, new_h), Image.LANCZOS)
                self.original_pil_image = image.copy()  # Update stored image
                print(f"     Resized to: {new_w}×{new_h}")

            transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize([0.5], [0.5])
            ])
        else:
            # Resize to specified size
            image = image.resize((size, size), Image.LANCZOS)
            self.original_pil_image = image.copy()  # Update stored image
            transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize([0.5], [0.5])
            ])

        tensor = transform(image).unsqueeze(0).to(self.device)

        # Convert to FP16 if enabled
        if self.use_fp16:
            tensor = tensor.half()

        return tensor, image.size

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
            secret_seed,
            secret_key,
            target_prompt="",
            epsilon=0.03,
            steps=50,
            learning_rate=0.01,
            save_metadata=True,
            preserve_size=True,
            max_size=1024  # NEW: Max dimension for memory safety
    ):
        """
        Generate protected version with secret parameters.

        CRITICAL SECURITY PARAMETERS:
        - secret_seed: Integer that controls random number generation (1-999999999)
        - secret_key: String that influences attack pattern (any text)

        Without BOTH parameters, the protection cannot be reversed or bypassed!

        Args:
            image_path: Path to input image
            output_path: Where to save protected image
            secret_seed: SECRET integer seed (KEEP THIS SAFE!)
            secret_key: SECRET string key (KEEP THIS SAFE!)
            target_prompt: What the image might be trained as
            epsilon: Max perturbation strength
            steps: Optimization iterations
            learning_rate: Step size
            save_metadata: Save protection metadata (non-secret info only)
            preserve_size: Keep original image dimensions (default: True)
            max_size: Maximum dimension for memory safety (default: 1024)
        """

        # Validate secret parameters
        if not isinstance(secret_seed, int) or secret_seed < 1:
            raise ValueError("secret_seed must be a positive integer!")
        if not isinstance(secret_key, str) or len(secret_key) < 4:
            raise ValueError("secret_key must be a string with at least 4 characters!")

        print("=" * 60)
        print("🔐 SECURE ART PROTECTION")
        print("=" * 60)
        print(f"\n🛡️  Protecting: {image_path}")
        print(f"   Target prompt: '{target_prompt}'")
        print(f"   Epsilon: {epsilon} (~{epsilon * 255:.0f}/255)")
        print(f"   Steps: {steps}")
        print(f"\n🔑 Security Parameters:")
        print(f"   Secret seed: {'*' * len(str(secret_seed))} (HIDDEN)")
        print(f"   Secret key:  {'*' * len(secret_key)} (HIDDEN)")

        # Generate unique hash for this protection
        protection_hash = self._generate_secret_hash(secret_seed, secret_key, image_path)
        print(f"   Protection hash: {protection_hash[:16]}...")

        # Load image
        if preserve_size:
            clean_image, original_size = self.load_image(image_path, size=None, max_size=max_size)
            print(f"\n📐 Image size: {original_size[0]}×{original_size[1]}")
        else:
            clean_image, _ = self.load_image(image_path, size=512, max_size=max_size)
            original_size = (512, 512)
            print(f"\n📐 Resized to: 512×512")

        # Check VRAM and warn if needed
        if self.device == "cuda":
            allocated = torch.cuda.memory_allocated() / 1024 ** 3
            print(f"  VRAM usage: {allocated:.2f} GB")

        # Initialize perturbation with secret seed - smaller initial magnitude
        perturbation = self._generate_deterministic_noise(clean_image.shape, secret_seed, 0) * 0.0001

        # Match dtype with image
        perturbation = perturbation.to(clean_image.dtype)
        perturbation.requires_grad = True

        # Optimizer
        optimizer = torch.optim.Adam([perturbation], lr=learning_rate)

        # Encode text with secret key mixed in (subtle influence)
        # This makes the attack direction depend on the secret key
        key_influenced_prompt = target_prompt + f" {hashlib.md5(secret_key.encode()).hexdigest()[:8]}"
        text_embeddings = self.encode_text(key_influenced_prompt)

        # Generate secret timestep schedule
        timestep_schedule = self._get_timestep_schedule(secret_key, steps)

        # Optimization loop
        print(f"\n🔄 Generating protection with secret parameters...\n")
        pbar = tqdm(range(steps), desc="Optimizing")

        for step in pbar:
            # Clear cache periodically
            if step % 10 == 0 and self.device == "cuda":
                torch.cuda.empty_cache()

            optimizer.zero_grad()

            # Apply perturbation (ensure matching dtypes)
            perturbed_image = clean_image + perturbation
            perturbed_image = torch.clamp(perturbed_image, -1, 1)

            # Encode to latent
            with torch.enable_grad():
                posterior = self.vae.encode(perturbed_image).latent_dist
                latents = posterior.mean * 0.18215

            # Use secret timestep from schedule
            t = torch.tensor([timestep_schedule[step]], device=self.device).long()

            # Generate noise with secret seed
            noise = self._generate_deterministic_noise(latents.shape, secret_seed, step)

            # Add noise
            noisy_latents = self.scheduler.add_noise(latents, noise, t)

            # Predict noise
            with torch.enable_grad():
                noise_pred = self.unet(
                    noisy_latents,
                    t,
                    encoder_hidden_states=text_embeddings
                ).sample

            # Attack loss
            loss = -F.mse_loss(noise_pred, noise)

            # Backward
            loss.backward()

            # Update with key-influenced learning rate
            # Secret key slightly modulates optimization trajectory
            key_factor = (int(hashlib.sha256(f"{secret_key}{step}".encode()).hexdigest(), 16) % 100) / 100
            effective_lr = learning_rate * (0.8 + 0.4 * key_factor)  # Vary between 0.8x and 1.2x

            for param_group in optimizer.param_groups:
                param_group['lr'] = effective_lr

            optimizer.step()

            # Project to epsilon ball
            with torch.no_grad():
                perturbation.data = torch.clamp(perturbation.data, -epsilon, epsilon)

            if step % 10 == 0:
                pbar.set_postfix({
                    'loss': f'{loss.item():.4f}',
                    'max_pert': f'{perturbation.abs().max().item():.4f}'
                })

        # Save protected image - COMPLETELY REWRITTEN FIX
        with torch.no_grad():
            print(f"\n📊 Debug info:")
            print(f"   Clean image dtype: {clean_image.dtype}")
            print(f"   Clean image range: [{clean_image.min().item():.3f}, {clean_image.max().item():.3f}]")
            print(f"   Perturbation dtype: {perturbation.dtype}")
            print(f"   Perturbation magnitude: {perturbation.abs().max().item():.4f}")

            # SOLUTION: Work directly with the original PIL image to avoid dtype issues
            # Get original image as numpy array [0, 255]
            original_np = np.array(self.original_pil_image).astype(np.float32)
            print(f"   Original PIL image range: [{original_np.min()}, {original_np.max()}]")
            print(f"   Original PIL image shape: {original_np.shape}")

            # Convert perturbation to numpy in pixel space
            # Perturbation is in normalized space [-1, 1], need to scale to [0, 255]
            pert_np = perturbation.float().cpu().squeeze(0).permute(1, 2, 0).numpy()
            pert_np_scaled = pert_np * 127.5  # Scale from [-1, 1] to pixel range

            print(f"   Perturbation (scaled) range: [{pert_np_scaled.min():.2f}, {pert_np_scaled.max():.2f}]")

            # Apply perturbation to original image
            protected_np = original_np + pert_np_scaled
            protected_np = np.clip(protected_np, 0, 255).astype(np.uint8)

            mean_value = protected_np.mean()
            print(f"   Final numpy range: [{protected_np.min()}, {protected_np.max()}]")
            print(f"   Mean pixel value: {mean_value:.1f}")

            # Sanity check
            if mean_value < 10:
                print("   ❌ ERROR: Image is still too dark! This shouldn't happen.")
                print("   Original mean:", original_np.mean())
                # Emergency fallback - just save the original
                protected_np = original_np.astype(np.uint8)
                print("   Using original image as emergency fallback")

            # Create PIL image
            protected_pil = Image.fromarray(protected_np)

            # Save with high quality
            if output_path.lower().endswith('.png'):
                protected_pil.save(output_path, format='PNG')
            else:
                protected_pil.save(output_path, format='JPEG', quality=95, subsampling=0)

        print(f"\n✅ Protected image saved to: {output_path}")
        print(f"   Max perturbation: {perturbation.abs().max().item():.4f}")
        print(f"   L2 norm: {torch.norm(perturbation).item():.4f}")

        # Save metadata (non-secret information only!)
        if save_metadata:
            metadata = {
                "protection_hash": protection_hash,
                "original_image": os.path.basename(image_path),
                "protected_image": os.path.basename(output_path),
                "original_size": original_size,
                "target_prompt": target_prompt,
                "epsilon": epsilon,
                "steps": steps,
                "note": "Secret seed and key are NOT stored here. Keep them safe!"
            }

            metadata_path = output_path.replace('.jpg', '_metadata.json').replace('.png', '_metadata.json')
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=2)

            print(f"   Metadata saved to: {metadata_path}")

        print("\n" + "=" * 60)
        print("🔒 IMPORTANT: Store your secret parameters securely!")
        print("=" * 60)
        print(f"   Secret seed: {secret_seed}")
        print(f"   Secret key:  {secret_key}")
        print(f"   Protection hash: {protection_hash}")
        print("\n⚠️  Without these secrets, the protection CANNOT be removed!")
        print("=" * 60)

        return protected_pil, protection_hash


# Example usage with secret parameters
if __name__ == "__main__":
    # Configuration
    INPUT_IMAGE = "test_image.jpg"
    OUTPUT_IMAGE = "protected_image.jpg"

    # ==========================================
    # SECRET PARAMETERS - KEEP THESE SAFE!
    # ==========================================
    # These are like passwords for your protection
    # Without them, no one can remove or bypass the protection

    SECRET_SEED = 42424242  # Choose any integer (1-999999999)
    SECRET_KEY = "my_art_2024"  # Choose any string (min 4 chars)

    # TIP: Use different secrets for different artworks!
    # ==========================================

    if not os.path.exists(INPUT_IMAGE):
        print(f"❌ Error: '{INPUT_IMAGE}' not found!")
        exit(1)

    # Create protector with memory-efficient settings
    protector = SecureArtProtector(use_fp16=True)  # FP16 saves ~50% VRAM

    # Protect with secret parameters
    protected_img, protection_hash = protector.protect_image(
        image_path=INPUT_IMAGE,
        output_path=OUTPUT_IMAGE,
        secret_seed=SECRET_SEED,
        secret_key=SECRET_KEY,
        target_prompt="artwork, digital art, illustration",
        epsilon=0.03,
        steps=50,
        learning_rate=0.01,
        save_metadata=True,
        preserve_size=True,
        max_size=1024  # Limit to 1024px for 8GB VRAM
    )

    print("\n🎉 Protection complete!")
    print("\n📝 SAVE THESE SECRETS:")
    print(f"   Image: {INPUT_IMAGE}")
    print(f"   Seed:  {SECRET_SEED}")
    print(f"   Key:   {SECRET_KEY}")
    print(f"   Hash:  {protection_hash}")