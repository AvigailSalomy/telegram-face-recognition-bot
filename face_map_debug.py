#!/usr/bin/env python3
"""
Face Map Debug Script
Visualizes all face encodings using TSNE dimensionality reduction.
Similar faces should appear closer together on the map.

Features:
- Persistent caching for face encodings (skips recomputation of unchanged images)
- TSNE result caching (reuses coordinates when face set is unchanged)
- Adaptive visualization for large datasets
- Celebrity cluster labels instead of per-image labels
"""

import hashlib
import pickle
import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.manifold import TSNE

# Fix for face_recognition_models compatibility with newer setuptools
import importlib.resources


class _FallbackResourceFinder:
    """Fallback for pkg_resources.resource_filename when setuptools >= 70 is used."""

    @staticmethod
    def resource_filename(package, resource):
        pkg_path = Path(importlib.resources.files(package).joinpath(resource))
        return str(pkg_path)


sys.modules['pkg_resources'] = type(sys)('pkg_resources')
sys.modules['pkg_resources'].resource_filename = _FallbackResourceFinder.resource_filename

import face_recognition

# Constants from telegram_bot.py
FACES_DATA_FILE = "faces_data.pkl"
CELEBS_DIR = Path("celebs_images")

# Cache files
CELEB_ENCODINGS_CACHE_FILE = "celeb_encodings_cache.pkl"
TSNE_CACHE_FILE = "tsne_cache.pkl"

# Global storage
known_faces = {}
celeb_faces_cache = {}


def _compute_file_hash(file_path: Path) -> str:
    """Compute a quick hash based on file path and modification time."""
    try:
        mtime = file_path.stat().st_mtime
        return f"{file_path}:{mtime}"
    except Exception:
        return str(file_path)


def _compute_faces_hash(known_faces: dict, celeb_faces_cache: dict) -> str:
    """Compute a hash that uniquely identifies the current face set."""
    # Hash known faces
    known_hash = hashlib.md5()
    for name in sorted(known_faces.keys()):
        known_hash.update(name.encode())

    # Hash celeb images
    celeb_hash = hashlib.md5()
    for celeb_name in sorted(celeb_faces_cache.keys()):
        for img_data in celeb_faces_cache[celeb_name]:
            path = str(img_data["image_path"])
            celeb_hash.update(path.encode())

    return f"{known_hash.hexdigest()}_{celeb_hash.hexdigest()}"


def load_celeb_faces_cached():
    """Load celeb faces with persistent caching for encodings."""
    global celeb_faces_cache

    if celeb_faces_cache:
        print(f"Celeb cache already loaded with {len(celeb_faces_cache)} celebs")
        return celeb_faces_cache

    start_time = time.time()
    celeb_faces_cache = {}

    if not CELEBS_DIR.exists():
        print(f"Celebs directory does not exist: {CELEBS_DIR}")
        return celeb_faces_cache

    print(f"Using celebs dataset: {CELEBS_DIR}")

    # Load existing cache if available
    cache = {}
    if Path(CELEB_ENCODINGS_CACHE_FILE).exists():
        try:
            with open(CELEB_ENCODINGS_CACHE_FILE, 'rb') as f:
                cache = pickle.load(f)
            print(f"Loaded encoding cache with {len(cache)} entries")
        except Exception as e:
            print(f"Could not load encoding cache: {e}")
            cache = {}

    # Track what's in the cache and what's new
    cache_hits = 0
    cache_misses = 0
    cache_stale = 0
    active_paths = set()

    # Image extensions to look for
    image_extensions = {'.jpg', '.jpeg', '.png'}

    # Iterate through each celeb subfolder
    for celeb_folder in CELEBS_DIR.iterdir():
        if celeb_folder.is_dir():
            celeb_name = celeb_folder.name.replace('_', ' ').title()
            celeb_images = [f for f in celeb_folder.iterdir() if f.suffix.lower() in image_extensions]

            if not celeb_images:
                print(f"No images found for celeb: {celeb_name}")
                continue

            valid_images = []
            for img_path in celeb_images:
                active_paths.add(str(img_path))
                cache_key = str(img_path)

                try:
                    # Check if we have a valid cached encoding
                    if cache_key in cache:
                        cached = cache[cache_key]
                        try:
                            current_mtime = img_path.stat().st_mtime
                            if cached.get("mtime") == current_mtime and cached.get("celeb_name") == celeb_name:
                                # Cache hit - verify encoding is still valid
                                if cached.get("encoding") is not None:
                                    valid_images.append({
                                        "encoding": cached["encoding"],
                                        "image_path": img_path,
                                        "celeb_name": celeb_name
                                    })
                                    cache_hits += 1
                                    continue
                        except Exception:
                            pass
                        # Cache stale (file modified or encoding invalid)
                        cache_stale += 1

                    # Cache miss or stale - compute encoding
                    image = face_recognition.load_image_file(img_path)
                    encodings = face_recognition.face_encodings(image)

                    if len(encodings) == 1:
                        mtime = img_path.stat().st_mtime
                        cache[cache_key] = {
                            "encoding": encodings[0],
                            "mtime": mtime,
                            "celeb_name": celeb_name,
                            "path": str(img_path)
                        }
                        valid_images.append({
                            "encoding": encodings[0],
                            "image_path": img_path,
                            "celeb_name": celeb_name
                        })
                        cache_misses += 1
                    elif len(encodings) > 1:
                        print(f"Skipping {img_path}: multiple faces found")
                        # Store in cache as None encoding so we don't re-check
                        cache[cache_key] = {
                            "encoding": None,
                            "mtime": img_path.stat().st_mtime,
                            "celeb_name": celeb_name,
                            "path": str(img_path)
                        }
                    else:
                        print(f"Skipping {img_path}: no face found")
                        cache[cache_key] = {
                            "encoding": None,
                            "mtime": img_path.stat().st_mtime,
                            "celeb_name": celeb_name,
                            "path": str(img_path)
                        }

                except Exception as e:
                    print(f"Error loading {img_path}: {e}")

            if valid_images:
                celeb_faces_cache[celeb_name] = valid_images
                print(f"Loaded celeb: {celeb_name} with {len(valid_images)} valid images")
            elif not celeb_images:
                print(f"No valid images for celeb: {celeb_name}")

    # Remove cached entries for deleted files
    deleted_entries = 0
    for cache_key in list(cache.keys()):
        if cache_key not in active_paths:
            del cache[cache_key]
            deleted_entries += 1

    # Save updated cache
    try:
        with open(CELEB_ENCODINGS_CACHE_FILE, 'wb') as f:
            pickle.dump(cache, f)
        print(f"Saved encoding cache ({len(cache)} entries)")
    except Exception as e:
        print(f"Could not save encoding cache: {e}")

    total_images = sum(len(v) for v in celeb_faces_cache.values())
    elapsed = time.time() - start_time

    print(f"\n[ENCODING CACHE STATS]")
    print(f"  Cache hits: {cache_hits} images")
    print(f"  Cache misses: {cache_misses} images (newly encoded)")
    print(f"  Stale entries refreshed: {cache_stale}")
    print(f"  Deleted entries removed: {deleted_entries}")
    print(f"  Total celebs: {len(celeb_faces_cache)} with {total_images} images")
    print(f"  Encoding time: {elapsed:.2f}s")

    return celeb_faces_cache


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
    """Wrapper that uses cached version."""
    return load_celeb_faces_cached()


def create_face_map(known_faces, celeb_faces_cache):
    """Create a 2D visualization of face similarities using TSNE."""
    start_time = time.time()

    all_encodings = []
    labels = []
    image_paths = []
    sources = []
    celeb_names = []  # Track which celebrity each face belongs to

    # Step 1: Collect known user faces
    for name, encoding in known_faces.items():
        all_encodings.append(encoding)
        labels.append(name)
        image_paths.append(None)
        sources.append('known')
        celeb_names.append(None)

    # Step 2: Collect celebrity faces
    for celeb_name, celeb_data_list in celeb_faces_cache.items():
        for idx, celeb_data in enumerate(celeb_data_list):
            all_encodings.append(celeb_data["encoding"])
            labels.append(f"{celeb_name}_{idx}")
            image_paths.append(celeb_data["image_path"])
            sources.append('celebrity')
            celeb_names.append(celeb_name)

    if len(all_encodings) < 2:
        print("Not enough faces to create map.")
        return None

    print(f"\n[TSNE] Total faces: {len(all_encodings)}")

    # Convert to numpy array for TSNE
    all_encodings = np.array(all_encodings)

    # Compute hash of current face set for TSNE cache
    faces_hash = _compute_faces_hash(known_faces, celeb_faces_cache)

    # Check for cached TSNE result
    tsne_cache = {}
    cached_tsne = None
    if Path(TSNE_CACHE_FILE).exists():
        try:
            with open(TSNE_CACHE_FILE, 'rb') as f:
                tsne_cache = pickle.load(f)
            if tsne_cache.get("hash") == faces_hash:
                cached_tsne = tsne_cache.get("embeddings")
                print(f"[TSNE] Loaded cached TSNE result (hash match)")
        except Exception as e:
            print(f"Could not load TSNE cache: {e}")

    if cached_tsne is not None:
        embeddings_2d = cached_tsne
        print("[TSNE] Using cached TSNE coordinates")
    else:
        # Compute TSNE
        perplexity = min(8, len(all_encodings) - 1)
        print(f"[TSNE] Computing with perplexity: {perplexity}")

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

        # Save to cache
        try:
            tsne_cache = {
                "hash": faces_hash,
                "embeddings": embeddings_2d
            }
            with open(TSNE_CACHE_FILE, 'wb') as f:
                pickle.dump(tsne_cache, f)
            print("[TSNE] Saved computed result to cache")
        except Exception as e:
            print(f"Could not save TSNE cache: {e}")

    tsne_time = time.time() - start_time
    print(f"[TSNE] Time: {tsne_time:.2f}s")

    return embeddings_2d, labels, image_paths, sources, celeb_names

def plot_face_map(embeddings_2d, labels, image_paths, sources, celeb_names):
    """Plot the 2D face embeddings with image thumbnails."""
    from matplotlib.offsetbox import OffsetImage, AnnotationBbox
    from PIL import Image

    start_time = time.time()

    n_images = len([p for p in image_paths if p])

    # Figure sizes for HIGH-RESOLUTION output
    if n_images < 50:
        fig_width, fig_height = 24, 18
    elif n_images < 100:
        fig_width, fig_height = 36, 26
    elif n_images < 200:
        fig_width, fig_height = 48, 34
    elif n_images < 400:
        fig_width, fig_height = 60, 44
    else:
        fig_width, fig_height = 72, 52

    fig, ax = plt.subplots(figsize=(fig_width, fig_height))

    coords = embeddings_2d.astype(float).copy()

    # Normalize TSNE coordinates to [0, 1]
    min_coords = coords.min(axis=0)
    max_coords = coords.max(axis=0)
    span = max_coords - min_coords
    span[span == 0] = 1.0
    coords = (coords - min_coords) / span

    # Scale to compact plotting area
    margin = 20
    plot_range = 60
    coords[:, 0] = coords[:, 0] * plot_range + margin
    coords[:, 1] = coords[:, 1] * plot_range + margin

    # Gentle image overlap resolution
    min_dist = 0.3
    max_iterations = 40

    for iteration in range(max_iterations):
        moved = False
        for i in range(len(coords)):
            for j in range(i + 1, len(coords)):
                dist_x = coords[j, 0] - coords[i, 0]
                dist_y = coords[j, 1] - coords[i, 1]
                dist = np.hypot(dist_x, dist_y)

                if dist < min_dist and dist > 1e-6:
                    push = (min_dist - dist) * 0.15
                    ux = dist_x / dist
                    uy = dist_y / dist

                    coords[i, 0] -= ux * push
                    coords[i, 1] -= uy * push
                    coords[j, 0] += ux * push
                    coords[j, 1] += uy * push
                    moved = True

        if not moved:
            break

    coords[:, 0] = np.clip(coords[:, 0], 10, 90)
    coords[:, 1] = np.clip(coords[:, 1], 10, 90)

    # Thumbnail sizes
    if n_images < 50:
        thumb_size = (80, 80)
        thumb_zoom = 0.85
    elif n_images < 100:
        thumb_size = (60, 60)
        thumb_zoom = 0.75
    elif n_images < 200:
        thumb_size = (48, 48)
        thumb_zoom = 0.65
    elif n_images < 400:
        thumb_size = (38, 38)
        thumb_zoom = 0.55
    else:
        thumb_size = (30, 30)
        thumb_zoom = 0.45

    images_loaded = 0

    # Collect celebrity positions
    celeb_positions = {}

    for i in range(len(coords)):
        x, y = coords[i]
        image_path = image_paths[i]
        celeb_name = celeb_names[i]

        if celeb_name:
            celeb_positions.setdefault(celeb_name, []).append((x, y))

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

        color = 'blue' if sources[i] == 'known' else 'red'
        ax.scatter(x, y, c=color, s=90, marker='o', zorder=2)

    # -------- Label placement --------
    cluster_data = []
    for celeb_name, positions in celeb_positions.items():
        if len(positions) < 1:
            continue

        positions_arr = np.array(positions, dtype=float)
        centroid = positions_arr.mean(axis=0)

        dists_to_centroid = np.hypot(
            positions_arr[:, 0] - centroid[0],
            positions_arr[:, 1] - centroid[1]
        )
        anchor_idx = np.argmin(dists_to_centroid)
        anchor_x, anchor_y = positions_arr[anchor_idx]

        cluster_data.append({
            "name": celeb_name,
            "positions_arr": positions_arr,
            "anchor_x": float(anchor_x),
            "anchor_y": float(anchor_y),
        })

    cluster_data.sort(key=lambda c: len(c["positions_arr"]), reverse=True)

    def label_box_dims(text):
        width = max(2.2, 0.20 * len(text) + 0.7)
        height = 0.85
        return width, height

    def make_label_box(x, y, text):
        w, h = label_box_dims(text)
        left = x - w / 2
        right = x + w / 2
        bottom = y
        top = y + h
        return left, right, bottom, top

    def boxes_overlap(a, b, pad=0.16):
        a_left, a_right, a_bottom, a_top = a
        b_left, b_right, b_bottom, b_top = b
        return not (
            a_right + pad < b_left or
            a_left - pad > b_right or
            a_top + pad < b_bottom or
            a_bottom - pad > b_top
        )

    def box_overlaps_points(box, positions_arr, clearance=1.05):
        left, right, bottom, top = box
        for px, py in positions_arr:
            nearest_x = min(max(px, left), right)
            nearest_y = min(max(py, bottom), top)
            dist = np.hypot(px - nearest_x, py - nearest_y)
            if dist < clearance:
                return True
        return False

    def min_dist_to_other_clusters(x, y, self_name):
        best = float("inf")
        for c in cluster_data:
            if c["name"] == self_name:
                continue
            positions_arr = c["positions_arr"]
            distances = np.hypot(positions_arr[:, 0] - x, positions_arr[:, 1] - y)
            d = float(distances.min()) if len(distances) else float("inf")
            if d < best:
                best = d
        return best

    placed_label_boxes = []
    labels_placed = 0

    for cluster in cluster_data:
        name = cluster["name"]
        positions_arr = cluster["positions_arr"]
        anchor_x = cluster["anchor_x"]
        anchor_y = cluster["anchor_y"]

        # Simple approach: place label slightly above the cluster center
        # Try a few fixed positions in order of preference
        label_x = anchor_x
        label_y = anchor_y + 0.5  # small fixed offset above

        # Check if this position overlaps own images
        box = make_label_box(label_x, label_y, name)
        if box_overlaps_points(box, positions_arr, clearance=0.8):
            # Try slightly to the right
            label_x = anchor_x + 0.4
            label_y = anchor_y + 0.4
            box = make_label_box(label_x, label_y, name)

        if box_overlaps_points(box, positions_arr, clearance=0.8):
            # Try slightly to the left
            label_x = anchor_x - 0.4
            label_y = anchor_y + 0.4
            box = make_label_box(label_x, label_y, name)

        # Ensure within bounds
        label_x = min(max(label_x, 4), 96)
        label_y = min(max(label_y, 4), 96)
        label_box = make_label_box(label_x, label_y, name)

        placed_label_boxes.append(label_box)

        ax.annotate(
            name,
            (label_x, label_y),
            fontsize=4,
            fontweight='bold',
            ha='center',
            va='bottom',
            zorder=10,
            color='white',
            bbox=dict(
                boxstyle='round,pad=0.15',
                facecolor='#333333',
                edgecolor='#666666',
                linewidth=0.5,
                alpha=0.85
            )
        )
        labels_placed += 1

    plot_time = time.time() - start_time

    print(f"\n[VISUALIZATION STATS]")
    print(f"  Figure size: {fig_width}x{fig_height}")
    print(f"  Thumbnails: {thumb_size[0]}x{thumb_size[1]}, zoom={thumb_zoom}")
    print(f"  Images loaded: {images_loaded}/{n_images}")
    print(f"  Celebrities: {len(celeb_positions)}, Labels shown: {labels_placed}")
    print(f"  Overlap iterations: {iteration + 1} (gentle mode)")
    print(f"  Plotting time: {plot_time:.2f}s")

    ax.set_title(
        f'Face Similarity Map - {n_images} images from {len(celeb_positions)} celebrities',
        fontsize=16,
        pad=20
    )

    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_facecolor('#f0f0f0')
    for spine in ax.spines.values():
        spine.set_visible(False)

    plt.tight_layout()
    return fig

def main():
    total_start = time.time()

    print("=" * 50)
    print("Face Map Debug Tool")
    print("=" * 50)

    # Step 1: Load all face data
    print("\n[1] Loading face data...")
    stage_start = time.time()
    load_known_faces()
    load_celeb_faces()
    stage_time = time.time() - stage_start
    print(f"[1] Face loading completed in {stage_time:.2f}s")

    # Step 2: Check if we have data
    total_known = len(known_faces)
    total_celeb = sum(len(v) for v in celeb_faces_cache.values())

    if total_known == 0 and total_celeb == 0:
        print("\nNo faces found! Please add faces first.")
        print("  - Add known faces using the Telegram bot")
        print("  - Add celeb images in the 'celebs_images' folder")
        sys.exit(1)

    # Step 3: Create the face map using TSNE
    print("\n[2] Creating face map with TSNE...")
    stage_start = time.time()
    result = create_face_map(known_faces, celeb_faces_cache)
    stage_time = time.time() - stage_start
    print(f"[2] TSNE completed in {stage_time:.2f}s")

    if result is None:
        print("Failed to create face map")
        sys.exit(1)

    embeddings_2d, labels, image_paths, sources, celeb_names = result

    # Step 4: Plot and display
    print("\n[3] Displaying face map...")
    fig = plot_face_map(embeddings_2d, labels, image_paths, sources, celeb_names)

    # Force a draw to ensure all artists are rendered before saving
    fig.canvas.draw()

    # Save HIGH-RESOLUTION PNG and PDF for zooming/detail viewing
    png_filename = "face_map_output.png"
    pdf_filename = "face_map_output.pdf"
    print(f"\n[Saving] Saving high-resolution files...")
    print(f"  - PNG at 200 DPI: {png_filename}")
    print(f"  - PDF vector: {pdf_filename}")

    # PNG with high DPI - do NOT use bbox_inches='tight' with AnnotationBbox (causes blank images)
    # Use explicit figure size instead
    fig.savefig(png_filename, dpi=200, facecolor='white', bbox_inches=None)

    # PDF for infinite zoom quality - same issue with bbox_inches
    fig.savefig(pdf_filename, format='pdf', facecolor='white', bbox_inches=None)

    print(f"[Saved] Both files saved successfully")

    total_time = time.time() - total_start
    print(f"\n[TOTAL RUNTIME: {total_time:.2f}s]")
    print("\nDone! Close the plot window to exit.")

    fig.show()


if __name__ == "__main__":
    main()

