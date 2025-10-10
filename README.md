# Adversarial Art Protection

A defensive security tool that adds imperceptible perturbations to images to protect artwork from unauthorized AI model training. By optimizing adversarial noise in the latent space of Stable Diffusion models, this system disrupts training processes while maintaining visual fidelity.

## 🎯 Project Purpose

This is a **defensive security tool** designed to protect intellectual property rights by preventing unauthorized use of artwork in AI training datasets. The protection works by introducing carefully crafted perturbations that:

- Are **visually imperceptible** to humans (PSNR > 35 dB)
- **Significantly disrupt** AI model training (30%+ loss increase)
- Operate in the **latent space** of diffusion models
- Can be **cryptographically secured** against removal attempts

**Important**: This tool is for defensive purposes only. It implements data poisoning techniques exclusively for protecting intellectual property.

---

## 🚀 Quick Start

### Prerequisites

- Python 3.8+
- CUDA-capable GPU (8GB+ VRAM recommended)
- Basic understanding of deep learning concepts

### Installation

```bash
# Clone the repository
git clone https://github.com/icantdo/adversarial-art-protection.git
cd adversarial-art-protection

# Install dependencies
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
pip install diffusers transformers pillow tqdm numpy matplotlib
```

### Basic Usage

**Simple Protection** (main.py):
```python
python main.py
```

Edit the configuration in `main.py`:
```python
INPUT_IMAGE = "your_artwork.jpg"
OUTPUT_IMAGE = "protected_artwork.png"
```

**Secure Protection** (recommended):
```python
python secure_protector.py
```

Configure in the `__main__` section:
```python
SECRET_SEED = 42  # Change this to your own secret!
SECRET_KEY = "your-secret-key-here"  # Keep this secure!
image_path = "your_artwork.jpg"
output_path = "protected_artwork.png"
```

**Testing Protection**:
```python
python test_protection.py
```

---

## 📚 For Users

### Understanding the Protection Levels

#### 1. **WorkingArtProtector** (main.py)
Basic adversarial protection without cryptographic security.

**Pros**:
- Simple to understand and use
- Good for learning how the system works
- Minimal configuration required

**Cons**:
- No secret parameters
- Perturbations are deterministic
- Can potentially be reversed with full knowledge of the algorithm

**Use case**: Educational purposes, basic protection

#### 2. **SecureArtProtector** (secure_protector.py)
Advanced protection with cryptographic security.

**Pros**:
- Secret seed and key make removal computationally infeasible
- Memory-efficient FP16 support
- Preserves original image dimensions
- Protection hash for verification

**Cons**:
- More complex setup
- Requires secure storage of secrets
- **Secrets cannot be recovered if lost**

**Use case**: Production use, protecting valuable artwork

### Parameter Configuration Guide

#### Epsilon (Perturbation Strength)
Controls how much noise is added to the image.

| Value | Visual Impact | Protection Level | Pixel Difference |
|-------|---------------|------------------|------------------|
| 0.01  | Imperceptible | Weak | 3/255 |
| 0.03  | **Recommended** | Good | 8/255 |
| 0.05  | Barely visible | Strong | 13/255 |
| 0.10  | Slightly visible | Maximum | 25/255 |

#### Steps (Optimization Iterations)
Number of gradient descent iterations.

- **20-30**: Fast, moderate protection
- **50**: Recommended default
- **100+**: Stronger protection (diminishing returns)

#### Target Prompt
Describes how the image might be used in training.

Examples:
- `"digital art, high quality"`
- `"portrait photography"`
- `"anime style illustration"`
- `"concept art, fantasy"`

**Tip**: More specific prompts create more targeted protection.

### Memory Requirements

| GPU VRAM | Settings |
|----------|----------|
| 6GB | `use_fp16=True`, `max_size=768` |
| 8GB | `use_fp16=True`, `max_size=1024` |
| 12GB+ | `use_fp16=False`, `max_size=2048` |

### Testing Your Protection

Use `test_protection.py` to validate effectiveness:

```python
ORIGINAL_IMAGE = "original.jpg"
PROTECTED_IMAGE = "protected.png"
TEST_PROMPT = "digital art"
```

**Success Criteria**:
- ✅ PSNR > 35 dB (imperceptible)
- ✅ Latent MSE > 0.001 (significant distortion)
- ✅ Training loss increase > 30% (disrupts learning)

**Score Interpretation**:
- **3/3**: Excellent protection
- **2/3**: Good protection (consider increasing epsilon)
- **1/3**: Weak protection (increase epsilon or steps)
- **0/3**: Failed (check configuration)

### Important Security Notes

#### For SecureArtProtector Users:

1. **Never lose your secrets**: The SECRET_SEED and SECRET_KEY cannot be recovered if lost
2. **Use unique secrets per artwork**: Don't reuse the same secrets across multiple images
3. **Store secrets securely**: Use a password manager or secure vault
4. **Verify protection**: Use the generated metadata file to verify protection hash

---

## 🛠️ For Developers

### Architecture Overview

The protection system consists of three main components:

```
┌─────────────────────────────────────────────────────┐
│                  Input Image                         │
└────────────────────┬────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────┐
│              VAE Encoder (SD 1.5)                    │
│  Converts pixel space → latent space (4×64×64)       │
└────────────────────┬────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────┐
│           Perturbation Optimizer                     │
│  - Initialize random perturbation (ε-bounded)        │
│  - Forward pass through UNet noise predictor         │
│  - Compute loss (MSE to maximize prediction error)   │
│  - Backward pass to compute gradients                │
│  - Update perturbation via gradient ascent           │
│  - Repeat for N steps                                │
└────────────────────┬────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────┐
│              VAE Decoder (SD 1.5)                    │
│  Converts latent space → pixel space                 │
└────────────────────┬────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────┐
│              Protected Image Output                  │
└─────────────────────────────────────────────────────┘
```

### Key Technical Decisions

#### Why `.mean` instead of `.sample()`?

The VAE encoder returns a distribution. We use `.mean` instead of `.sample()` because:
- **Deterministic**: Same input → same latent encoding
- **Gradient flow**: Enables backpropagation through the encoder
- **Reproducibility**: Critical for secret-based protection

```python
# Bad: Non-deterministic, breaks gradients
latents = vae.encode(image).latent_dist.sample()

# Good: Deterministic, preserves gradients
latents = vae.encode(image).latent_dist.mean
```

#### Gradient Flow Architecture

Both implementations carefully manage gradient contexts:

```python
with torch.no_grad():
    # Load models, prepare inputs (no gradients needed)
    latents = encode_image(image)

perturbation = torch.zeros_like(image, requires_grad=True)

for step in range(steps):
    with torch.enable_grad():
        # Apply perturbation
        perturbed_image = image + perturbation

        # Encode to latent space (gradients flow!)
        perturbed_latents = vae.encode(perturbed_image).latent_dist.mean

        # Predict noise with UNet
        noise_pred = unet(perturbed_latents, timestep, encoder_hidden_states).sample

        # Compute loss (maximize prediction error)
        loss = F.mse_loss(noise_pred, target_noise)

        # Backprop through entire chain
        loss.backward()

    # Update perturbation (gradient ascent)
    perturbation.data += lr * perturbation.grad.sign()
    perturbation.grad.zero_()
```

#### Cryptographic Security Model

The `SecureArtProtector` implements computational security:

1. **Secret Seed**: Controls all random number generation
   - Noise initialization: 2^32 possibilities
   - Timestep sampling: Different randomness per step

2. **Secret Key**: Influences optimization process
   - Timestep schedule modulation: 1000^steps possibilities
   - Learning rate perturbations
   - Text prompt mixing ratios

**Combined search space**: Computationally infeasible to brute-force

```python
# Seed controls RNG
rng = torch.Generator(device=device).manual_seed(secret_seed)
noise = torch.randn(..., generator=rng)

# Key influences optimization
timestep = (1000 * hash(secret_key + str(step))) % 1000
lr_factor = 1.0 + 0.1 * (hash(secret_key) % 100) / 100
```

### Development Setup

```bash
# Clone the repository
git clone https://github.com/yourusername/adversarial-art-protection.git
cd adversarial-art-protection

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install development dependencies
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
pip install diffusers transformers pillow tqdm numpy matplotlib
pip install pytest black flake8  # Optional: for testing and linting
```

### Project Structure

```
adversarial-art-protection/
├── main.py                  # WorkingArtProtector (basic implementation)
├── secure_protector.py      # SecureArtProtector (cryptographic version)
├── test_protection.py       # ProtectionTester (validation suite)
├── test.py                  # Quick usage example
├── README.md                # This file
└── examples/                # (Optional) Example images and outputs
```

### Contributing

We welcome contributions! Here are some areas where you can help:

#### 🐛 Bug Fixes
- Memory optimization improvements
- FP16 compatibility issues
- Image format handling edge cases

#### ✨ Feature Ideas
- Support for other diffusion models (SD 2.x, SDXL)
- Batch processing capabilities
- GUI application
- CLI argument parsing
- Alternative attack strategies (targeted, transferable)
- Integration with image editing software

#### 📚 Documentation
- Tutorial notebooks
- Video walkthroughs
- Translations
- Use case examples

#### 🧪 Testing
- Unit tests for core components
- Integration tests
- Performance benchmarks
- Effectiveness studies against real models

### Development Guidelines

1. **Code Style**: Follow PEP 8 conventions
2. **Documentation**: Add docstrings to all public functions
3. **Testing**: Include tests for new features
4. **Security**: Never commit actual secret seeds/keys
5. **Commits**: Use descriptive commit messages

### Common Development Tasks

#### Adding a New Protection Method

1. Inherit from base protector pattern
2. Implement `protect_image()` method
3. Handle gradient flow carefully
4. Add memory optimizations (FP16 support)
5. Include metadata generation
6. Write tests

Example:
```python
class CustomProtector:
    def __init__(self, device="cuda", use_fp16=False):
        self.device = device
        self.use_fp16 = use_fp16
        self._load_models()

    def protect_image(self, image_path, output_path, **kwargs):
        # Your implementation here
        pass
```

#### Testing Your Changes

```python
# Run the test suite
python test_protection.py

# Check memory usage
nvidia-smi

# Validate visual quality
# PSNR should be > 35 dB for imperceptibility
```

#### Debugging Tips

1. **Enable debug prints**: Uncomment print statements in the code
2. **Check tensor shapes**: Add `print(tensor.shape)` to verify dimensions
3. **Monitor VRAM**: Use `nvidia-smi` to track memory usage
4. **Visualize perturbations**: Save intermediate results
5. **Test with small images**: Use 512×512 for faster iteration

### Known Issues and Limitations

1. **GPU Memory**: Requires CUDA GPU with 6GB+ VRAM
2. **Processing Time**: ~30-60 seconds per image (depends on steps and resolution)
3. **Model Specificity**: Optimized for Stable Diffusion 1.5 architecture
4. **Format Support**: Best results with JPEG/PNG, may need adjustments for other formats

### Research and Technical Background

This implementation is based on adversarial machine learning research:

- **Adversarial Examples**: Perturbations that fool neural networks
- **Data Poisoning**: Corrupting training data to degrade model performance
- **Latent Space Attacks**: Operating in compressed representations
- **Gradient-based Optimization**: Using backpropagation to craft perturbations

**Key Papers** (for reference):
- "Intriguing properties of neural networks" (Szegedy et al., 2014)
- "Explaining and Harnessing Adversarial Examples" (Goodfellow et al., 2015)
- "Poison Frogs! Targeted Clean-Label Poisoning Attacks" (Shafahi et al., 2018)

### Roadmap

**v1.0 (Current)**:
- ✅ Basic adversarial protection
- ✅ Cryptographic security layer
- ✅ FP16 memory optimization
- ✅ Validation suite

**v2.0 (Planned)**:
- ⬜ Batch processing support
- ⬜ CLI interface with argparse
- ⬜ Support for SD 2.x / SDXL
- ⬜ Transferability across model architectures
- ⬜ GUI application

**v3.0 (Future)**:
- ⬜ Plugin system for custom attacks
- ⬜ Cloud processing support
- ⬜ Watermark integration
- ⬜ Real-time protection API

---

## 📄 License

MIT License

Copyright (c) 2025 icantdo

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

## 🙏 Acknowledgments

- Hugging Face for the `diffusers` library
- Stability AI for Stable Diffusion models
- The adversarial ML research community

## 📧 Contact

[Add contact information, Discord server, or discussion forum]

## ⚠️ Ethical Considerations

This tool is designed for **defensive purposes only**. Users should:

- Only protect their own artwork or artwork they have rights to
- Understand local laws regarding data protection and AI training
- Use responsibly and ethically
- Not attempt to attack or damage AI systems maliciously

**Remember**: The goal is to protect intellectual property, not to harm research or legitimate AI development.

---

## 🔗 Related Projects

- [Glaze](https://glaze.cs.uchicago.edu/) - Style mimicry protection
- [Nightshade](https://nightshade.cs.uchicago.edu/) - Concept poisoning
- [Mist](https://mist.cs.uchicago.edu/) - Watermarking for diffusion models

---

**Star this repository if you find it useful! Contributions and feedback are welcome.**
