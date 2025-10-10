from secure_protector import SecureArtProtector

protector = SecureArtProtector(use_fp16=True)

protector.protect_image(
    image_path="G2qgw1FbIAMqXzj.jpeg",
    output_path="protected.jpg",
    secret_seed=87654321,
    secret_key="MyArt_2024",
    target_prompt="digital art",
    epsilon=0.03,
    steps=50,
    max_size=1300
)