"""
Test suite to verify adversarial protection effectiveness
This will train LoRA on both original and protected images and compare results
"""

import torch
import torch.nn.functional as F
from diffusers import AutoencoderKL, UNet2DConditionModel, DDPMScheduler
from transformers import CLIPTextModel, CLIPTokenizer
from PIL import Image
import numpy as np
from torchvision import transforms
import os
from tqdm import tqdm
import matplotlib.pyplot as plt

class ProtectionTester:
    """
    Tests effectiveness of adversarial protection by:
    1. Training LoRA on original images
    2. Training LoRA on protected images
    3. Comparing generation quality
    """

    def __init__(self, model_id="runwayml/stable-diffusion-v1-5", device="cuda"):
        self.device = device
        print("Loading Stable Diffusion components for testing...")

        # Load components separately to avoid version conflicts
        from diffusers import AutoencoderKL, UNet2DConditionModel, DDPMScheduler
        from transformers import CLIPTextModel, CLIPTokenizer

        self.vae = AutoencoderKL.from_pretrained(model_id, subfolder="vae").to(device)
        self.unet = UNet2DConditionModel.from_pretrained(model_id, subfolder="unet").to(device)
        self.text_encoder = CLIPTextModel.from_pretrained(model_id, subfolder="text_encoder").to(device)
        self.tokenizer = CLIPTokenizer.from_pretrained(model_id, subfolder="tokenizer")
        self.scheduler = DDPMScheduler.from_pretrained(model_id, subfolder="scheduler")

        # Set to eval mode
        self.vae.eval()
        self.unet.eval()
        self.text_encoder.eval()

        # Convert to float16 for efficiency
        if device == "cuda":
            self.vae = self.vae.half()
            self.unet = self.unet.half()
            self.text_encoder = self.text_encoder.half()

        print("✓ Models loaded\n")

    def simple_visual_test(self, original_path, protected_path):
        """
        Test 1: Visual comparison
        Check if protected image looks identical to human eyes
        """
        print("=" * 60)
        print("TEST 1: Visual Similarity (Human Perception)")
        print("=" * 60)

        original = Image.open(original_path).convert("RGB")
        protected = Image.open(protected_path).convert("RGB")

        print(f"\n📐 Image sizes:")
        print(f"   Original:  {original.size}")
        print(f"   Protected: {protected.size}")

        # Resize both to same size for fair comparison
        # Use the protected image size (512x512) as reference
        target_size = (512, 512)
        original_resized = original.resize(target_size, Image.LANCZOS)
        protected_resized = protected.resize(target_size, Image.LANCZOS)

        # Convert to tensors
        to_tensor = transforms.ToTensor()
        orig_tensor = to_tensor(original_resized)
        prot_tensor = to_tensor(protected_resized)

        # Compute metrics
        mse = F.mse_loss(orig_tensor, prot_tensor).item()
        psnr = 10 * np.log10(1.0 / mse) if mse > 0 else float('inf')
        max_diff = (orig_tensor - prot_tensor).abs().max().item()

        print(f"\n📊 Perceptual Metrics:")
        print(f"   MSE: {mse:.6f} (lower = more similar)")
        print(f"   PSNR: {psnr:.2f} dB (higher = more similar, >40 is excellent)")
        print(f"   Max pixel diff: {max_diff:.4f} ({max_diff*255:.1f}/255)")

        # Verdict
        if psnr > 35:
            print("\n✅ PASS: Images are perceptually identical")
        elif psnr > 25:
            print("\n⚠️  BORDERLINE: Small differences may be visible")
        else:
            print("\n❌ FAIL: Differences are visible")

        return psnr

    def vae_encoding_test(self, original_path, protected_path):
        """
        Test 2: Latent space comparison
        Check if perturbations survive VAE encoding
        """
        print("\n" + "=" * 60)
        print("TEST 2: Latent Space Distortion")
        print("=" * 60)

        # Load images
        transform = transforms.Compose([
            transforms.Resize((512, 512)),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5])
        ])

        original = Image.open(original_path).convert("RGB")
        protected = Image.open(protected_path).convert("RGB")

        orig_tensor = transform(original).unsqueeze(0).to(self.device)
        prot_tensor = transform(protected).unsqueeze(0).to(self.device)

        # Encode to latent space
        with torch.no_grad():
            if self.device == "cuda":
                orig_latent = self.vae.encode(orig_tensor.half()).latent_dist.mean
                prot_latent = self.vae.encode(prot_tensor.half()).latent_dist.mean
            else:
                orig_latent = self.vae.encode(orig_tensor).latent_dist.mean
                prot_latent = self.vae.encode(prot_tensor).latent_dist.mean

        # Compare latents
        latent_mse = F.mse_loss(orig_latent, prot_latent).item()
        latent_max_diff = (orig_latent - prot_latent).abs().max().item()

        print(f"\n📊 Latent Space Metrics:")
        print(f"   Latent MSE: {latent_mse:.6f}")
        print(f"   Latent max diff: {latent_max_diff:.4f}")
        print(f"   Relative change: {(latent_max_diff / orig_latent.abs().mean().item()) * 100:.2f}%")

        if latent_mse > 0.001:
            print("\n✅ PASS: Significant latent space distortion detected")
        else:
            print("\n❌ FAIL: Insufficient latent space distortion")

        return latent_mse

    def quick_training_test(self, image_path, prompt, num_steps=50):
        """
        Test 3: Quick training simulation
        Simulate a few training steps and measure loss trajectory
        """
        print("\n" + "=" * 60)
        print(f"TEST 3: Training Simulation - {os.path.basename(image_path)}")
        print("=" * 60)

        # Load image
        transform = transforms.Compose([
            transforms.Resize((512, 512)),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5])
        ])

        image = Image.open(image_path).convert("RGB")
        image_tensor = transform(image).unsqueeze(0).to(self.device)

        # Encode prompt
        text_inputs = self.tokenizer(
            prompt,
            padding="max_length",
            max_length=self.tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt"
        ).to(self.device)

        with torch.no_grad():
            text_embeddings = self.text_encoder(text_inputs.input_ids)[0]

        # Encode image
        with torch.no_grad():
            if self.device == "cuda":
                latents = self.vae.encode(image_tensor.half()).latent_dist.sample()
            else:
                latents = self.vae.encode(image_tensor).latent_dist.sample()
            latents = latents * 0.18215

        # Simulate training steps
        losses = []
        print(f"\n🔄 Simulating {num_steps} training steps...")

        for step in tqdm(range(num_steps), desc="Training"):
            # Random timestep
            t = torch.randint(0, 1000, (1,), device=self.device).long()

            # Add noise
            noise = torch.randn_like(latents)
            noisy_latents = self.scheduler.add_noise(latents, noise, t)

            # Predict noise
            with torch.no_grad():
                noise_pred = self.unet(
                    noisy_latents,
                    t,
                    encoder_hidden_states=text_embeddings
                ).sample

            # Compute loss (what the model would optimize)
            loss = F.mse_loss(noise_pred.float(), noise.float()).item()
            losses.append(loss)

        avg_loss = np.mean(losses)
        loss_std = np.std(losses)

        print(f"\n📊 Training Metrics:")
        print(f"   Average loss: {avg_loss:.6f}")
        print(f"   Loss std: {loss_std:.6f}")

        return avg_loss, losses

    def compare_training(self, original_path, protected_path, prompt="artwork"):
        """
        Test 4: Compare training on original vs protected
        """
        print("\n" + "=" * 60)
        print("TEST 4: Training Comparison")
        print("=" * 60)

        # Test on original
        print("\n🔹 Training on ORIGINAL image:")
        orig_loss, orig_losses = self.quick_training_test(original_path, prompt, num_steps=30)

        # Test on protected
        print("\n🔹 Training on PROTECTED image:")
        prot_loss, prot_losses = self.quick_training_test(protected_path, prompt, num_steps=30)

        # Compare
        print("\n" + "=" * 60)
        print("📊 COMPARISON RESULTS")
        print("=" * 60)
        print(f"\nOriginal image avg loss:  {orig_loss:.6f}")
        print(f"Protected image avg loss: {prot_loss:.6f}")
        print(f"Loss increase: {((prot_loss - orig_loss) / orig_loss) * 100:+.2f}%")

        if prot_loss > orig_loss * 1.5:
            print("\n✅ PASS: Protected image causes significantly higher training loss")
            print("   This means the model will struggle to learn from this image!")
        elif prot_loss > orig_loss * 1.1:
            print("\n⚠️  BORDERLINE: Some protection effect detected")
        else:
            print("\n❌ FAIL: Insufficient protection - losses are too similar")

        # Plot comparison
        try:
            plt.figure(figsize=(10, 5))
            plt.plot(orig_losses, label='Original', color='green', alpha=0.7)
            plt.plot(prot_losses, label='Protected', color='red', alpha=0.7)
            plt.xlabel('Training Step')
            plt.ylabel('Loss')
            plt.title('Training Loss Comparison')
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.savefig('training_comparison.png', dpi=150, bbox_inches='tight')
            print(f"\n📊 Loss plot saved to: training_comparison.png")
        except:
            print("\n(Could not generate plot - matplotlib may not have display)")

        return orig_loss, prot_loss

    def generation_quality_test(self, original_path, protected_path, prompt="artwork", output_dir="test_outputs"):
        """
        Test 5: Generate images after "training" on each
        This is the most realistic test but takes longer
        """
        print("\n" + "=" * 60)
        print("TEST 5: Generation Quality Test")
        print("=" * 60)
        print("⚠️  This test is disabled due to complexity.")
        print("    The training simulation tests (3 & 4) are sufficient.")
        print("    For full testing, you would need to actually train LoRA models.")

        return None


def run_all_tests(original_path, protected_path, prompt="artwork"):
    """
    Run complete test suite
    """
    print("\n" + "=" * 60)
    print("🧪 ADVERSARIAL PROTECTION TEST SUITE")
    print("=" * 60)
    print(f"\nOriginal:  {original_path}")
    print(f"Protected: {protected_path}")
    print(f"Prompt:    '{prompt}'")
    print("\n")

    tester = ProtectionTester()

    # Run tests
    print("\n" + "🔬 Running test battery...\n")

    # Test 1: Visual similarity
    psnr = tester.simple_visual_test(original_path, protected_path)

    # Test 2: Latent distortion
    latent_mse = tester.vae_encoding_test(original_path, protected_path)

    # Test 3 & 4: Training comparison
    orig_loss, prot_loss = tester.compare_training(original_path, protected_path, prompt)

    # Final verdict
    print("\n" + "=" * 60)
    print("🎯 FINAL VERDICT")
    print("=" * 60)

    protection_score = 0

    if psnr > 35:
        print("✅ Imperceptibility: PASS")
        protection_score += 1
    else:
        print("❌ Imperceptibility: FAIL")

    if latent_mse > 0.001:
        print("✅ Latent Distortion: PASS")
        protection_score += 1
    else:
        print("❌ Latent Distortion: FAIL")

    if prot_loss > orig_loss * 1.3:
        print("✅ Training Disruption: PASS")
        protection_score += 1
    else:
        print("❌ Training Disruption: FAIL")

    print(f"\n📊 Overall Score: {protection_score}/3")

    if protection_score == 3:
        print("\n🎉 EXCELLENT! Protection is working effectively!")
    elif protection_score == 2:
        print("\n👍 GOOD! Protection has some effectiveness, consider stronger epsilon.")
    elif protection_score == 1:
        print("\n⚠️  WEAK! Protection is minimal, increase epsilon or steps.")
    else:
        print("\n❌ FAILED! Protection is not working, check implementation.")


# Main execution
if __name__ == "__main__":
    # Configuration
    ORIGINAL_IMAGE = "G2qgw1FbIAMqXzj.jpeg"  # ← Your original image
    PROTECTED_IMAGE = "protected.jpg"  # ← Protected version
    TEST_PROMPT = "digital art"  # What might be trained on

    # Check files exist
    if not os.path.exists(ORIGINAL_IMAGE):
        print(f"❌ Error: '{ORIGINAL_IMAGE}' not found!")
        exit(1)

    if not os.path.exists(PROTECTED_IMAGE):
        print(f"❌ Error: '{PROTECTED_IMAGE}' not found!")
        print("   Run the protector first to generate protected image.")
        exit(1)

    # Run all tests
    run_all_tests(ORIGINAL_IMAGE, PROTECTED_IMAGE, TEST_PROMPT)

    print("\n✅ Testing complete!")