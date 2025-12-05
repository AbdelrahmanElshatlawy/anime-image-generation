import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
import streamlit as st
from pathlib import Path

# -----------------------
# Config & Utilities
# -----------------------

LATENT_DIM = 128
BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "generator_model.keras"   # put file next to app.py
ICON_PATH  = BASE_DIR / "image.jpg"              # your anime icon

@st.cache_resource
def load_generator():
    """Load the trained generator model once and cache it."""
    generator = load_model(MODEL_PATH, compile=False)
    return generator

def generate_images(generator, num_samples: int, latent_dim: int = LATENT_DIM,
                    use_cpu: bool = True, seed: int | None = None):
    """Generate anime images using the generator."""
    if seed is not None:
        tf.random.set_seed(seed)

    noise = tf.random.normal([num_samples, latent_dim])

    device = "/CPU:0" if use_cpu else "/GPU:0"
    with tf.device(device):
        imgs = generator(noise, training=False)

    imgs = 0.5 * imgs + 0.5
    imgs = tf.clip_by_value(imgs, 0.0, 1.0)
    return imgs.numpy()

# -----------------------
# Streamlit UI
# -----------------------

def main():
    st.set_page_config(
        page_title="Anime Character Generation using GAN",
        page_icon="🎨",
        layout="wide",
    )

    left_h, right_h = st.columns([0.12, 0.87])
    with left_h:
        st.image(str(ICON_PATH), width=65, clamp=True)
    with right_h:
        st.title("Anime Character Generation using GAN")

    st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)

    st.markdown(
        """
Welcome to the **Anime Character Generation using GAN** app!  

Here we use a trained **Generative Adversarial Network (GAN)** to create synthetic anime character images.

- The **generator** learns to produce images that *look* like real anime faces.
- The **discriminator** learns to distinguish between real and fake.
- Over training, the generator gets better and better until its images are hard to distinguish from real ones.
        """
    )

    st.sidebar.header("⚙️ Generation Settings")

    num_samples = st.sidebar.slider("Number of images", 1, 25, 9, 1)
    cols_per_row = st.sidebar.slider("Images per row", 1, 5, 3, 1)
    use_cpu = st.sidebar.checkbox(
        "Force CPU (recommended on most deployments)",
        value=True,
    )
    seed = st.sidebar.number_input(
        "Random seed (optional, -1 = random)",
        value=-1,
        step=1,
        help="Use a fixed seed for reproducible images.",
    )

    st.sidebar.markdown("---")
    generate_btn = st.sidebar.button("🎲 Generate Anime Characters")

    if not MODEL_PATH.exists():
        st.error(f"Model not found at:\n`{MODEL_PATH}`")
        st.stop()

    with st.spinner("Loading generator model..."):
        generator = load_generator()

    if generate_btn:
        st.subheader("Generated Anime Characters")

        seed_value = None if seed < 0 else int(seed)
        with st.spinner("Generating images..."):
            imgs = generate_images(
                generator,
                num_samples=num_samples,
                latent_dim=LATENT_DIM,
                use_cpu=use_cpu,
                seed=seed_value,
            )

        rows = int(np.ceil(num_samples / cols_per_row))
        idx = 0
        for _ in range(rows):
            cols = st.columns(cols_per_row)
            for col in cols:
                if idx >= num_samples:
                    break
                img = imgs[idx]
                if img.shape[-1] == 1:
                    img = np.repeat(img, 3, axis=-1)
                col.image(img, use_container_width=True)
                idx += 1

        st.success("Done! Scroll up to see the generated characters 💫")
    else:
        st.info("Use the controls in the sidebar and click **Generate Anime Characters** to start.")


if __name__ == "__main__":
    main()