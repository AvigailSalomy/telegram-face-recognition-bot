#!/usr/bin/env python3
"""
Face Map Debug Script
Visualizes all face encodings using TSNE dimensionality reduction.
Similar faces should appear closer together on the map.
"""

import pickle
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.manifold import TSNE

# Fix for face_recognition_models compatibility with newer setuptools
# Add pkg_resources to sys.modules before face_recognition_models tries to import it
import importlib.resources
class _FallbackResourceFinder:
    """Fallback for pkg_resources.resource_filename when setuptools >= 70 is used."""
    @staticmethod
    def resource_filename(package, resource):
        pkg_path = Path(importlib.resources.files(package).joinpath(resource))
        return str(pkg_path)

# Add the fallback to sys.modules
sys.modules['pkg_resources'] = type(sys)('pkg_resources')
sys.modules['pkg_resources'].resource_filename = _FallbackResourceFinder.resource_filename

# Now safe to import face_recognition
import face_recognition

# Constants from telegram_bot.py
FACES_DATA_FILE = "faces_data.pkl"
CELEBS_DIR = Path("celebs")

# Global storage (matching telegram_bot.py structure)
known_faces = {}  # {name: encoding}
celeb_faces_cache = {}  # {celeb_name: [{"encoding": encoding, "image_path": path}, ...]}


def load_known_faces():
    """Load known faces from pickle file."""
    global known_faces
    try:
        if Path(FACES_DATA_FILE).exists():
            with open(FACES_DATA_FILE, 'rb') as f:
                known_faces = pickle.load(f)
            print(f"Loaded {len(known_faces)} known faces")
    except Exception as e:
        print(f"Error loading known faces: {e}")
        known_faces = {}
    return known_faces


def load_celeb_faces():
    """Load celeb faces from the celebs folder - matches telegram_bot.py structure."""
    global celeb_faces_cache
    
    if celeb_faces_cache:
        print(f"Celeb cache already loaded with {len(celeb_faces_cache)} celebs")
        return celeb_faces_cache
    
    celeb_faces_cache = {}
    
    if not CELEBS_DIR.exists():
        print(f"Celebs directory does not exist: {CELEBS_DIR}")
        return celeb_faces_cache
    
    # Image extensions to look for (matching telegram_bot.py)
    image_extensions = {'.jpg', '.jpeg', '.png'}
    
    # Iterate through each celeb subfolder
    for celeb_folder in CELEBS_DIR.iterdir():
        if celeb_folder.is_dir():
            celeb_name = celeb_folder.name.replace('_', ' ').title()
            
            # Find all images in the celeb folder
            celeb_images = [f for f in celeb_folder.iterdir() if f.suffix.lower() in image_extensions]
            
            if not celeb_images:
                print(f"No images found for celeb: {celeb_name}")
                continue
            
            # Load all valid images (exactly one face)
            valid_images = []
            for img_path in celeb_images:
                try:
                    image = face_recognition.load_image_file(img_path)
                    encodings = face_recognition.face_encodings(image)
                    
                    # Only include images with exactly one face
                    if len(encodings) == 1:
                        valid_images.append({
                            "encoding": encodings[0],
                            "image_path": img_path  # Keep as Path object (like telegram_bot.py)
                        })
                    elif len(encodings) > 1:
                        print(f"Skipping {img_path}: multiple faces found")
                    else:
                        print(f"Skipping {img_path}: no face found")
                except Exception as e:
                    print(f"Error loading {img_path}: {e}")
            
            if valid_images:
                celeb_faces_cache[celeb_name] = valid_images
                print(f"Loaded celeb: {celeb_name} with {len(valid_images)} valid images")
            else:
                print(f"No valid images for celeb: {celeb_name}")
    
    total_images = sum(len(v) for v in celeb_faces_cache.values())
    print(f"Total celebs: {len(celeb_faces_cache)} with {total_images} images")
    return celeb_faces_cache

def create_face_map(known_faces, celeb_faces_cache):
    """Create a 2D visualization of face similarities using TSNE."""
    
    all_encodings = []
    labels = []
    image_paths = []
    sources = []

    # Step 1: Collect known user faces
    for name, encoding in known_faces.items():
        all_encodings.append(encoding)
        labels.append(name)
        image_paths.append(None)
        sources.append('known')

    # Step 2: Collect celebrity faces
    for celeb_name, celeb_data_list in celeb_faces_cache.items():
        for idx, celeb_data in enumerate(celeb_data_list):
            all_encodings.append(celeb_data["encoding"])
            labels.append(f"{celeb_name}_{idx}")
            image_paths.append(celeb_data["image_path"])
            sources.append('celebrity')

    if len(all_encodings) < 2:
        print("Not enough faces to create map.")
        return None

    print(f"[DEBUG] Total faces: {len(all_encodings)}")

    # Convert to numpy array for TSNE
    all_encodings = np.array(all_encodings)

    # TSNE settings tuned to keep similar faces grouped into visible clusters
    perplexity = min(8, len(all_encodings) - 1)
    print(f"Using perplexity: {perplexity}")

    tsne = TSNE(
        n_components=2,
        perplexity=perplexity,
        learning_rate=200,
        early_exaggeration=16.0,
        init='pca',
        random_state=42,
        max_iter=1500
    )

    embeddings_2d = tsne.fit_transform(all_encodings)

    return embeddings_2d, labels, image_paths, sources


def plot_face_map(embeddings_2d, labels, image_paths, sources):
    """Plot the 2D face embeddings with image thumbnails."""
    from matplotlib.offsetbox import OffsetImage, AnnotationBbox
    from PIL import Image

    fig, ax = plt.subplots(figsize=(18, 12))

    coords = embeddings_2d.astype(float).copy()

    # Keep TSNE geometry, only normalize gently to plotting space
    min_coords = coords.min(axis=0)
    max_coords = coords.max(axis=0)
    span = max_coords - min_coords
    span[span == 0] = 1.0

    coords = (coords - min_coords) / span
    coords[:, 0] = coords[:, 0] * 84 + 8
    coords[:, 1] = coords[:, 1] * 84 + 8

    # Gentle local anti-overlap only inside crowded areas
    # Strong enough to reduce collisions, but weak enough to preserve cluster structure
    min_dist = 3.4
    for _ in range(180):
        moved = False
        for i in range(len(coords)):
            for j in range(i + 1, len(coords)):
                dx = coords[j, 0] - coords[i, 0]
                dy = coords[j, 1] - coords[i, 1]
                dist = np.hypot(dx, dy)

                if dist < 1e-6:
                    angle = np.random.uniform(0, 2 * np.pi)
                    dx = np.cos(angle) * 0.01
                    dy = np.sin(angle) * 0.01
                    dist = np.hypot(dx, dy)

                if dist < min_dist:
                    push = (min_dist - dist) / 2.0
                    ux = dx / dist
                    uy = dy / dist

                    coords[i, 0] -= ux * push
                    coords[i, 1] -= uy * push
                    coords[j, 0] += ux * push
                    coords[j, 1] += uy * push
                    moved = True

        coords[:, 0] = np.clip(coords[:, 0], 5, 95)
        coords[:, 1] = np.clip(coords[:, 1], 5, 95)

        if not moved:
            break

    images_loaded = 0
    thumb_size = (36, 36)
    thumb_zoom = 0.68

    for i in range(len(coords)):
        x, y = coords[i]
        image_path = image_paths[i]

        path_obj = Path(image_path) if image_path else None
        if path_obj and path_obj.exists():
            try:
                img = Image.open(path_obj).convert("RGB")
                img.thumbnail(thumb_size, Image.Resampling.LANCZOS)
                img_array = np.array(img)

                imagebox = OffsetImage(img_array, zoom=thumb_zoom)
                imagebox.image.axes = ax

                ab = AnnotationBbox(
                    imagebox,
                    (x, y),
                    frameon=False,
                    pad=0.0,
                    boxcoords="data",
                    zorder=2
                )
                ax.add_artist(ab)
                images_loaded += 1
                continue

            except Exception as e:
                print(f"[DEBUG] Error loading image {image_path}: {e}")

        # Fallback for known faces or missing images
        color = 'blue' if sources[i] == 'known' else 'red'
        ax.scatter(x, y, c=color, s=90, marker='o', zorder=2)

    print(f"[DEBUG] Images loaded: {images_loaded}/{len([p for p in image_paths if p])}")

    # Put labels a bit lower so they won't sit on the image itself
    for i in range(len(coords)):
        x, y = coords[i]

        ax.annotate(
            labels[i],
            (x, y),
            xytext=(0, -13),
            textcoords='offset points',
            fontsize=5.2,
            ha='center',
            va='top',
            zorder=3,
            bbox=dict(
                boxstyle='round,pad=0.10',
                facecolor='white',
                edgecolor='gray',
                linewidth=0.4,
                alpha=0.9
            ),
            color='black'
        )

    ax.set_title('Face Similarity Map (similar faces are positioned closer)', fontsize=12)
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.grid(False)

    plt.tight_layout(pad=0.8)
    return plt

def main():
    print("=" * 50)
    print("Face Map Debug Tool")
    print("=" * 50)
    
    # Step 1: Load all face data
    print("\n[1] Loading face data...")
    load_known_faces()
    load_celeb_faces()
    
    # Step 2: Check if we have data
    total_known = len(known_faces)
    total_celeb = sum(len(v) for v in celeb_faces_cache.values())
    
    if total_known == 0 and total_celeb == 0:
        print("\nNo faces found! Please add faces first.")
        print("  - Add known faces using the Telegram bot")
        print("  - Add celeb images in the 'celebs' folder")
        sys.exit(1)
    
    # Step 3: Create the face map using TSNE
    print("\n[2] Creating face map with TSNE...")
    result = create_face_map(known_faces, celeb_faces_cache)
    
    if result is None:
        print("Failed to create face map")
        sys.exit(1)
    
    embeddings_2d, labels, image_paths, sources = result
    
    # Step 4: Plot and display
    print("\n[3] Displaying face map...")
    plot = plot_face_map(embeddings_2d, labels, image_paths, sources)
    plot.show()
    
    print("\nDone! Close the plot window to exit.")


if __name__ == "__main__":
    main()

