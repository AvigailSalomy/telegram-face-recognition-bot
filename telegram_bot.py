import os
import re
import pickle
import traceback
import asyncio
from pathlib import Path
from dotenv import load_dotenv
from telegram import Update, KeyboardButton, ReplyKeyboardMarkup, InputFile
from telegram.ext import Application, CommandHandler, MessageHandler, filters, ContextTypes, ConversationHandler
import face_recognition
from PIL import Image, ImageDraw, ImageFont

# Import map generation from face_map
from face_map import generate_and_save_map

# Load environment variables
load_dotenv()

# Get token from .env
TOKEN = os.getenv("TELEGRAM_TOKEN")

# Face storage paths
FACES_DIR = Path("stored_faces")
FACES_DATA_FILE = FACES_DIR / "faces_data.pkl"

# Celeb faces paths
CELEBS_DIR = Path("celebs_images")
celeb_faces_cache = {}  # {celeb_name: {"encoding": encoding, "image_path": path}}

# Conversation states
WAITING_FOR_NAME = 1
WAITING_FOR_ADD_IMAGE = 2
WAITING_FOR_RECOGNIZE_IMAGE = 3
WAITING_FOR_SIMILAR_CELEB_IMAGE = 4
WAITING_FOR_GUESS = 5
MAIN_MENU = 6

# Similarity threshold (distance below this = match)
SIMILARITY_THRESHOLD = 0.6

# Global storage for known faces
known_faces = {}  # {name: encoding}

# Game state for "Guess the Celeb"
guess_the_celeb_games = {}  # {user_id: {"celeb_name": str, "image_path": str, "stage": int, "options": [names]}}


def sanitize_filename(name: str) -> str:
    """Sanitize name for use in filename - supports non-English characters."""
    # Keep alphanumeric, spaces, and common characters from any language
    cleaned = re.sub(r'[^\w\s]', '', name).strip()
    # Replace spaces with underscores
    result = cleaned.replace(' ', '_')
    # If empty or only underscores, use a hash of the original name
    if not result or result == '_' * len(result):
        result = f"person_{abs(hash(name)) % 100000}"
    return result


def ensure_faces_dir():
    """Ensure the faces directory exists."""
    FACES_DIR.mkdir(exist_ok=True)


async def download_file_with_retry(bot, file_id: str, dest_path: Path, max_retries: int = 3) -> bool:
    """Download a file with retry logic."""
    for attempt in range(max_retries):
        try:
            file = await bot.get_file(file_id)
            await file.download_to_drive(dest_path)
            return True
        except Exception as e:
            if attempt < max_retries - 1:
                print(f"Download attempt {attempt + 1} failed, retrying...")
                await asyncio.sleep(1)  # Wait 1 second before retry
            else:
                print(f"Download failed after {max_retries} attempts")
                raise
    return False


def load_known_faces() -> dict:
    """Load known faces from file with error handling."""
    global known_faces
    try:
        ensure_faces_dir()
        if FACES_DATA_FILE.exists():
            with open(FACES_DATA_FILE, 'rb') as f:
                known_faces = pickle.load(f)
    except Exception:
        known_faces = {}
    return known_faces


def save_known_faces() -> bool:
    """Save known faces to file with error handling."""
    global known_faces
    try:
        ensure_faces_dir()
        with open(FACES_DATA_FILE, 'wb') as f:
            pickle.dump(known_faces, f)
        return True
    except Exception:
        return False


def reset_all_faces():
    """Delete all stored faces and invalidate map caches."""
    global known_faces
    known_faces = {}
    try:
        ensure_faces_dir()
        if FACES_DATA_FILE.exists():
            FACES_DATA_FILE.unlink()
        # Delete from stored_faces directory
        for file in FACES_DIR.glob("*.png"):
            file.unlink()
        # Also delete from faces directory (legacy location)
        faces_legacy = Path("faces")
        if faces_legacy.exists():
            for file in faces_legacy.glob("*.png"):
                file.unlink()
            for file in faces_legacy.glob("*.jpg"):
                file.unlink()
        # Invalidate map caches so map regenerates without old faces
        try:
            tsne_cache = FACES_DIR / "tsne_cache.pkl"
            if tsne_cache.exists():
                tsne_cache.unlink()
            map_cache = FACES_DIR / "map_cache.pkl"
            if map_cache.exists():
                map_cache.unlink()
            temp_map = FACES_DIR / "temp_map.png"
            if temp_map.exists():
                temp_map.unlink()
            # Also clear legacy cache locations
            for cache_file in ["tsne_cache.pkl", "map_cache.pkl"]:
                legacy_cache = Path(cache_file)
                if legacy_cache.exists():
                    legacy_cache.unlink()
        except Exception:
            pass
    except Exception:
        pass

def load_celeb_faces(force_refresh: bool = False):
    global celeb_faces_cache

    # reuse memory cache
    if celeb_faces_cache and not force_refresh:
        print(f"Celeb cache already loaded with {len(celeb_faces_cache)} celebs, skipping reload")
        return celeb_faces_cache

    celeb_faces_cache = {}

    if not CELEBS_DIR.exists():
        print(f"Celebs directory does not exist: {CELEBS_DIR}")
        return celeb_faces_cache

    cache_file = FACES_DIR / "celeb_encodings_cache.pkl"

    # fast path: load directly from disk
    if cache_file.exists() and not force_refresh:
        try:
            with open(cache_file, 'rb') as f:
                cached_data = pickle.load(f)

            for celeb_name, image_entries in cached_data.items():
                valid_images = []
                for img_key, entry in image_entries.items():
                    img_path = Path(img_key)
                    if not img_path.exists():  # skip broken paths
                        continue
                    valid_images.append({
                        "encoding": entry["encoding"],
                        "image_path": img_path
                    })

                if valid_images:
                    celeb_faces_cache[celeb_name] = valid_images

            print(f"Loaded {len(celeb_faces_cache)} celebs from cache (FAST)")
            return celeb_faces_cache

        except Exception as e:
            print(f"Cache load failed: {e}")

    # slow path: full scan
    cached_data = {}
    if cache_file.exists():
        try:
            with open(cache_file, 'rb') as f:
                cached_data = pickle.load(f)
        except Exception:
            cached_data = {}

    image_extensions = {'.jpg', '.jpeg', '.png'}
    updated_cache = {}

    for celeb_folder in CELEBS_DIR.iterdir():
        if celeb_folder.is_dir():
            celeb_name = celeb_folder.name.replace('_', ' ').title()
            celeb_images = [f for f in celeb_folder.iterdir() if f.suffix.lower() in image_extensions]

            valid_images = []
            for img_path in celeb_images:
                try:
                    img_key = str(img_path)
                    current_mtime = img_path.stat().st_mtime

                    cached_entry = cached_data.get(celeb_name, {}).get(img_key)

                    if cached_entry and cached_entry.get('mtime') == current_mtime:
                        valid_images.append({
                            "encoding": cached_entry["encoding"],
                            "image_path": img_path
                        })
                    else:
                        image = face_recognition.load_image_file(img_path)
                        encodings = face_recognition.face_encodings(image)

                        if len(encodings) == 1:
                            valid_images.append({
                                "encoding": encodings[0],
                                "image_path": img_path
                            })

                            if celeb_name not in updated_cache:
                                updated_cache[celeb_name] = {}

                            updated_cache[celeb_name][img_key] = {
                                "encoding": encodings[0],
                                "mtime": current_mtime
                            }

                except Exception as e:
                    print(f"Error loading {img_path}: {e}")

            if valid_images:
                celeb_faces_cache[celeb_name] = valid_images

    for celeb_name, img_entries in cached_data.items():
        if celeb_name not in updated_cache:
            updated_cache[celeb_name] = img_entries
        else:
            for img_key, entry in img_entries.items():
                if img_key not in updated_cache[celeb_name]:
                    updated_cache[celeb_name][img_key] = entry

    try:
        cache_file.parent.mkdir(exist_ok=True)
        with open(cache_file, 'wb') as f:
            pickle.dump(updated_cache, f)
    except Exception as e:
        print(f"Could not save cache: {e}")

    print(f"Loaded {len(celeb_faces_cache)} celebs (slow path)")
    return celeb_faces_cache

def find_most_similar_celeb(face_encoding):
    """Find the most similar celeb to the given face encoding.
    
    Compares against ALL loaded celeb face encodings and returns the best match.
    """
    global celeb_faces_cache
    
    if not celeb_faces_cache:
        load_celeb_faces()
    
    if not celeb_faces_cache:
        return None, None, float('inf')
    
    best_celeb = None
    best_image_path = None
    best_distance = float('inf')
    
    # Compare against ALL celeb images
    for celeb_name, celeb_images in celeb_faces_cache.items():
        for celeb_data in celeb_images:
            distance = face_recognition.face_distance([celeb_data["encoding"]], face_encoding)[0]
            
            if distance < best_distance:
                best_distance = distance
                best_celeb = celeb_name
                best_image_path = celeb_data["image_path"]
    
    return best_celeb, best_image_path, best_distance


def get_random_celebrities(exclude_name: str, count: int) -> list:
    """Get random celeb names from the cache, excluding one."""
    global celeb_faces_cache
    
    available = [name for name in celeb_faces_cache.keys() if name != exclude_name]
    if len(available) < count:
        return available
    
    import random
    return random.sample(available, count)


def crop_face_for_stage(image_path: str, stage: int) -> str:
    """Crop celebrity image for different game stages.
    
    Stage 1: narrow band around eyes using face landmarks
    Stage 2: eyes + nose (20% to 65% of face height)
    Stage 3: almost full face (10% to 95% of face height)
    
    Returns path to cropped image.
    """
    try:
        # Load image and get face location
        image = face_recognition.load_image_file(image_path)
        face_locations = face_recognition.face_locations(image)
        
        if not face_locations:
            return image_path
        
        # Get the first face (largest)
        top, right, bottom, left = face_locations[0]
        face_height = bottom - top
        face_width = right - left
        
        # Calculate padding (30% of face dimensions)
        padding_h = int(face_height * 0.3)
        padding_w = int(face_width * 0.3)
        
        if stage == 1:
            # Stage 1: Use face landmarks to crop around eyes
            landmarks = face_recognition.face_landmarks(image, face_locations)
            if not landmarks:
                # Fallback to percentage if landmarks fail
                crop_top = top + int(face_height * 0.30)
                crop_bottom = top + int(face_height * 0.40)
            else:
                # Get eye coordinates
                left_eye = landmarks[0]['left_eye']
                right_eye = landmarks[0]['right_eye']
                all_eye_points = left_eye + right_eye
                
                # Find min and max y coordinates of eyes
                eye_ys = [point[1] for point in all_eye_points]
                min_eye_y = min(eye_ys)
                max_eye_y = max(eye_ys)
                
                # Add small vertical padding around eyes
                eye_padding = int((max_eye_y - min_eye_y) * 0.3)
                crop_top = max(0, min_eye_y - eye_padding)
                crop_bottom = min(image.shape[0], max_eye_y + eye_padding)
        else:
            # Stages 2 and 3: Use percentage-based cropping
            crop_ranges = {
                2: (0.20, 0.65),  # eyes + nose
                3: (0.10, 0.95)   # almost full face
            }
            start_fraction, end_fraction = crop_ranges.get(stage, (0.20, 0.65))
            
            # Vertical crop: from start_fraction to end_fraction of face height (no vertical padding)
            crop_top = top + int(face_height * start_fraction)
            crop_bottom = top + int(face_height * end_fraction)
        
        # Horizontal crop with padding (same for all stages)
        crop_left = max(0, left - padding_w)
        crop_right = min(image.shape[1], right + padding_w)
        
        # Crop using PIL
        pil_image = Image.open(image_path)
        cropped = pil_image.crop((crop_left, crop_top, crop_right, crop_bottom))
        
        # Save to temp location
        ensure_faces_dir()
        temp_path = FACES_DIR / f"game_crop_stage{stage}.png"
        cropped.save(temp_path, "PNG")
        
        return str(temp_path)
    
    except Exception as e:
        print(f"Error cropping image for stage {stage}: {e}")
        return image_path


def get_main_keyboard():
    """Create the main keyboard with 6 buttons."""
    keyboard = [
        [KeyboardButton("Add face"), KeyboardButton("Recognize faces")],
        [KeyboardButton("Similar celebs"), KeyboardButton("Guess the Celeb")],
        [KeyboardButton("Map"), KeyboardButton("Reset faces")]
    ]
    return ReplyKeyboardMarkup(keyboard, resize_keyboard=True)


async def start_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handle the /start command and show the custom keyboard."""
    load_known_faces()
    await update.message.reply_text(
        "👋 Welcome to the Face Recognition Bot!\n👇 Choose an option from the main menu:",
        reply_markup=get_main_keyboard()
    )
    return MAIN_MENU


async def handle_menu_choice(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handle main menu button choices."""
    text = update.message.text
    
    if text == "Add face":
        await update.message.reply_text("📸 Upload an image with a single face")
        return WAITING_FOR_ADD_IMAGE
        
    elif text == "Recognize faces":
        load_known_faces()
        if not known_faces:
            await update.message.reply_text(
                "🚫 No faces stored yet. Please add a face first."
            )
            await update.message.reply_text(
                "👇 Choose an option from the main menu:",
                reply_markup=get_main_keyboard()
            )
            return MAIN_MENU
        await update.message.reply_text(
            "👁️ Upload an image with at least one face and I will recognize who is in this image"
        )
        return WAITING_FOR_RECOGNIZE_IMAGE
        
    elif text == "Similar celebs":
        # Load celeb faces only if cache is empty
        if not celeb_faces_cache:
            load_celeb_faces()
        if not celeb_faces_cache:
            await update.message.reply_text(
                "🚫 No celeb faces available. Please check the celebs_images folder."
            )
            await update.message.reply_text(
                "👇 Choose an option from the main menu:",
                reply_markup=get_main_keyboard()
            )
            return MAIN_MENU
        await update.message.reply_text(
            "🌟 Upload me a picture of a single person and I will find which celebs are similar to that person"
        )
        return WAITING_FOR_SIMILAR_CELEB_IMAGE
        
    elif text == "Guess the Celeb":
        # Load celeb faces only if cache is empty
        if not celeb_faces_cache:
            load_celeb_faces()
        if not celeb_faces_cache:
            await update.message.reply_text(
                "🚫 No celeb faces available. Please check the celebs_images folder."
            )
            await update.message.reply_text(
                "👇 Choose an option from the main menu:",
                reply_markup=get_main_keyboard()
            )
            return MAIN_MENU
        
        # Start the guessing game
        import random
        celeb_name = random.choice(list(celeb_faces_cache.keys()))
        celeb_data = random.choice(celeb_faces_cache[celeb_name])
        image_path = str(celeb_data["image_path"])
        
        # Get 5 other celebs as options
        other_celebs = get_random_celebrities(celeb_name, 5)
        options = [celeb_name] + other_celebs
        random.shuffle(options)
        
        # Store game state
        user_id = update.effective_user.id
        guess_the_celeb_games[user_id] = {
            "celeb_name": celeb_name,
            "image_path": image_path,
            "stage": 1,
            "options": options
        }
        
        # Get cropped image for stage 1
        cropped_path = crop_face_for_stage(image_path, 1)
        
        # Create keyboard with options
        keyboard = [[KeyboardButton(name)] for name in options]
        reply_markup = ReplyKeyboardMarkup(keyboard, resize_keyboard=True, one_time_keyboard=True)
        
        # Send game image and options
        try:
            with open(cropped_path, "rb") as photo_file:
                await update.message.reply_photo(
                    photo=photo_file,
                    caption="🎮 Stage 1 of 3: Who is this celebrity? (You get 6 options)"
                )
        except Exception as e:
            print(f"Error sending game image: {e}")
            await update.message.reply_text("I see a face, but who is it? Pick one!")
        
        await update.message.reply_text(
            "👇 Choose your answer:",
            reply_markup=reply_markup
        )
        
        return WAITING_FOR_GUESS
        
    elif text == "Map":
        await update.message.reply_text("🗺️ Generating face similarity map...")
        
        # Generate the map using the original face_map logic
        ensure_faces_dir()
        map_output_path = str(FACES_DIR / "cached_face_map.png")
        
        try:
            success = generate_and_save_map(map_output_path)
            
            if success:
                # Verify image dimensions are valid for Telegram
                try:
                    with Image.open(map_output_path) as img:
                        width, height = img.size
                        print(f"[MAP] Generated image size: {width}x{height}")
                        
                        # Ensure max dimension is under 10000px for Telegram
                        if width > 10000 or height > 10000:
                            print(f"[MAP] Image too large, resizing...")
                            max_dim = 9500
                            ratio = min(max_dim / width, max_dim / height)
                            new_size = (int(width * ratio), int(height * ratio))
                            img = img.resize(new_size, Image.Resampling.LANCZOS)
                            img.save(map_output_path, "PNG")
                            print(f"[MAP] Resized to: {new_size[0]}x{new_size[1]}")
                except Exception as e:
                    print(f"[MAP] Could not verify/resize image: {e}")
                
                # Send as document to avoid compression (best for high-quality maps)
                try:
                    with open(map_output_path, "rb") as doc_file:
                        await update.message.reply_document(
                            document=InputFile(doc_file, filename="face_map.png"),
                            caption="Face Similarity Map"
                        )
                except Exception as doc_error:
                    # Fallback: try as photo
                    print(f"[MAP] Could not send as document, trying as photo: {doc_error}")
                    with open(map_output_path, "rb") as photo_file:
                        await update.message.reply_photo(
                            photo=photo_file,
                            caption="Face Similarity Map"
                        )
                
                
            else:
                await update.message.reply_text(
                    "I don't have any faces to display on the map yet.\n\n"
                    "Please add some faces first using the 'Add face' option, "
                    "or check that the celebs_images folder contains images."
                )
        except Exception as e:
            print(f"Error generating face map: {e}")
            traceback.print_exc()
            await update.message.reply_text(
                "⚠️ Sorry, an error occurred while generating the map."
            )
        
        await update.message.reply_text(
            "👇 Choose an option from the main menu:",
            reply_markup=get_main_keyboard()
        )
        return MAIN_MENU
        
    elif text == "Reset faces":
        reset_all_faces()
        await update.message.reply_text(
            "♻️ All faces have been deleted. The bot now knows no one."
        )
        await update.message.reply_text(
            "👇 Choose an option from the main menu:",
            reply_markup=get_main_keyboard()
        )
        return MAIN_MENU
        
    else:
        await update.message.reply_text(
            "👇 Choose an option from the main menu:",
            reply_markup=get_main_keyboard()
        )
        return MAIN_MENU


async def handle_add_image(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handle image uploaded for adding a face."""
    if not update.message.photo:
        await update.message.reply_text("📸 Please send an image with a single face.")
        return WAITING_FOR_ADD_IMAGE
    
    ensure_faces_dir()
    
    photo = update.message.photo[-1]
    
    try:
        temp_path = FACES_DIR / f"temp_{update.effective_user.id}.jpg"
        await download_file_with_retry(context.bot, photo.file_id, temp_path)
    except Exception:
        traceback.print_exc()
        await update.message.reply_text(
            "⚠️ Sorry, there was a network problem while downloading the image. Please try again."
        )
        return MAIN_MENU
    
    try:
        image = face_recognition.load_image_file(temp_path)
        # Get face locations AND encodings
        face_locations = face_recognition.face_locations(image)
        encodings = face_recognition.face_encodings(image, face_locations)
        
        if len(encodings) == 0:
            temp_path.unlink(missing_ok=True)
            await update.message.reply_text(
                "❌ No face found in this image. Please send an image with a single face."
            )
            return WAITING_FOR_ADD_IMAGE

        if len(encodings) > 1:
            temp_path.unlink(missing_ok=True)
            await update.message.reply_text(
                "❌ Multiple faces found in this image. Please send an image with a single face."
            )
            return WAITING_FOR_ADD_IMAGE

        await update.message.reply_text("✅ Great. What's the name of the person in this image?")
        context.user_data['temp_encoding'] = encodings[0]
        context.user_data['temp_image_path'] = temp_path
        context.user_data['temp_face_location'] = face_locations[0]  # Store face location for cropping
        return WAITING_FOR_NAME
        
    except Exception:
        traceback.print_exc()
        temp_path.unlink(missing_ok=True)
        await update.message.reply_text(
            "⚠️ Sorry, I couldn't process that image. Please try again."
        )
        return MAIN_MENU


async def handle_add_image_text(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handle plain text in WAITING_FOR_ADD_IMAGE."""
    await update.message.reply_text("📸 Please upload an image with a single face.")
    return WAITING_FOR_ADD_IMAGE


async def handle_recognize_image_text(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handle plain text in WAITING_FOR_RECOGNIZE_IMAGE."""
    await update.message.reply_text("👁️ Please upload an image with at least one face.")
    return WAITING_FOR_RECOGNIZE_IMAGE


async def handle_name_input(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handle name input after face image is uploaded."""
    name = update.message.text.strip()
    normalized_name = name.strip().lower()

    # 1. Validate name: must be at least 2 characters and contain at least one letter
    if len(name) < 2 or not any(c.isalpha() for c in name):
        await update.message.reply_text("✏️ Please enter a valid name (at least 2 characters and must include letters).")
        return WAITING_FOR_NAME

    encoding = context.user_data.get('temp_encoding')
    temp_path = context.user_data.get('temp_image_path')
    face_location = context.user_data.get('temp_face_location')

    if encoding is None or temp_path is None:
        await update.message.reply_text(
            "⚠️ Something went wrong. Please start again by pressing 'Add face'.",
            reply_markup=get_main_keyboard()
        )
        return ConversationHandler.END

    global known_faces

    # 2. Check if this face already exists in known_faces
    existing_name = None
    for known_name, known_encoding in known_faces.items():
        distance = face_recognition.face_distance([known_encoding], encoding)[0]
        if distance < SIMILARITY_THRESHOLD:
            existing_name = known_name
            break

    if existing_name is not None:
        # Face already exists
        if existing_name.strip().lower() == normalized_name:
            # 3. Same face, same name
            await update.message.reply_text(f"✅ I already know this face as {existing_name}.")
            # Clean up temp file
            if temp_path.exists():
                temp_path.unlink(missing_ok=True)
            context.user_data.pop('temp_encoding', None)
            context.user_data.pop('temp_image_path', None)
            context.user_data.pop('temp_face_location', None)
            await update.message.reply_text(
                "👇 Choose an option from the main menu:",
                reply_markup=get_main_keyboard()
            )
            return MAIN_MENU
        else:
            # 4. Same face, different name - update the name
            old_name = existing_name
            known_faces[name] = known_faces.pop(old_name)
            
            if not save_known_faces():
                await update.message.reply_text(
                    "⚠️ Sorry, I couldn't update the face. Please try again.",
                    reply_markup=get_main_keyboard()
                )
                context.user_data.pop('temp_encoding', None)
                context.user_data.pop('temp_image_path', None)
                context.user_data.pop('temp_face_location', None)
                if temp_path.exists():
                    temp_path.unlink(missing_ok=True)
                return MAIN_MENU
            
            # Update the face image file name
            ensure_faces_dir()
            old_safe_name = sanitize_filename(old_name)
            old_face_image_path = FACES_DIR / f"{old_safe_name}.png"
            new_safe_name = sanitize_filename(name)
            new_face_image_path = FACES_DIR / f"{new_safe_name}.png"
            
            if old_face_image_path.exists():
                old_face_image_path.rename(new_face_image_path)
            
            # Clean up temp file
            if temp_path.exists():
                temp_path.unlink(missing_ok=True)
            
            context.user_data.pop('temp_encoding', None)
            context.user_data.pop('temp_image_path', None)
            context.user_data.pop('temp_face_location', None)
            
            await update.message.reply_text(f"✅ This face already exists as {old_name}. I updated the name to {name}.")
            await update.message.reply_text(
                "👇 Choose an option from the main menu:",
                reply_markup=get_main_keyboard()
            )
            return MAIN_MENU

    # Face is new - check if name already exists
    normalized_existing = {existing.strip().lower(): existing for existing in known_faces.keys()}
    if normalized_name in normalized_existing:
        # 5. Different face + existing name
        await update.message.reply_text("❗ This name already exists. Please choose a different name.")
        return WAITING_FOR_NAME

    # 6. New face + new valid name - save normally
    known_faces[name] = encoding

    if not save_known_faces():
        await update.message.reply_text(
            "⚠️ Sorry, I couldn't save the face. Please try again.",
            reply_markup=get_main_keyboard()
        )
        context.user_data.pop('temp_encoding', None)
        context.user_data.pop('temp_image_path', None)
        context.user_data.pop('temp_face_location', None)
        if temp_path.exists():
            temp_path.unlink(missing_ok=True)
        return MAIN_MENU

    ensure_faces_dir()
    safe_name = sanitize_filename(name)
    face_image_path = FACES_DIR / f"{safe_name}.png"
    
    # Crop face with padding for better thumbnail quality on map
    if temp_path.exists() and face_location:
        try:
            # Load full image
            pil_image = Image.open(temp_path)
            img_width, img_height = pil_image.size
            
            # Get face bounding box (top, right, bottom, left)
            top, right, bottom, left = face_location
            
            # Calculate padding (30% of face height on each side)
            face_height = bottom - top
            face_width = right - left
            padding_h = int(face_height * 0.3)
            padding_w = int(face_width * 0.3)
            
            # Calculate crop coordinates with padding
            crop_left = max(0, left - padding_w)
            crop_top = max(0, top - padding_h)
            crop_right = min(img_width, right + padding_w)
            crop_bottom = min(img_height, bottom + padding_h)
            
            # Crop and save at high resolution (400x400) for crisp thumbnails on map
            cropped = pil_image.crop((crop_left, crop_top, crop_right, crop_bottom))
            cropped = cropped.resize((400, 400), Image.Resampling.LANCZOS)
            cropped.save(face_image_path, "PNG")
            
            # Clean up temp file
            temp_path.unlink(missing_ok=True)
        except Exception as e:
            print(f"Could not crop face: {e}")
            # Fallback: just rename the temp file
            if temp_path.exists():
                temp_path.rename(face_image_path)
    elif temp_path.exists():
        temp_path.rename(face_image_path)
    
    context.user_data.pop('temp_encoding', None)
    context.user_data.pop('temp_image_path', None)
    context.user_data.pop('temp_face_location', None)
    
    await update.message.reply_text("✅ Great! I will now remember this face")
    await update.message.reply_text(
        "👇 Choose an option from the main menu:",
        reply_markup=get_main_keyboard()
    )
    return MAIN_MENU


async def handle_recognize_image(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handle image uploaded for recognizing faces."""
    load_known_faces()
    
    if not update.message.photo:
        await update.message.reply_text("👁️ Please send an image with at least one face.")
        return WAITING_FOR_RECOGNIZE_IMAGE
    
    ensure_faces_dir()
    
    photo = update.message.photo[-1]
    
    try:
        temp_path = FACES_DIR / f"recognize_{update.effective_user.id}.jpg"
        await download_file_with_retry(context.bot, photo.file_id, temp_path)
    except Exception:
        traceback.print_exc()
        await update.message.reply_text(
            "⚠️ Sorry, there was a network problem while downloading the image. Please try again.",
            reply_markup=get_main_keyboard()
        )
        return ConversationHandler.END
    
    try:
        image = face_recognition.load_image_file(temp_path)
        face_locations = face_recognition.face_locations(image)
        face_encodings = face_recognition.face_encodings(image, face_locations)
        
        if not face_encodings:
            await update.message.reply_text(
                "❌ I don't recognize anyone in this image.",
                reply_markup=get_main_keyboard()
            )
            temp_path.unlink(missing_ok=True)
            return MAIN_MENU
        
        pil_image = Image.fromarray(image)
        draw = ImageDraw.Draw(pil_image)
        
        try:
            font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 20)
        except:
            font = ImageFont.load_default()
        
        results = []
        
        for face_encoding in face_encodings:
            best_match = None
            best_confidence = 0
            
            for known_name, known_encoding in known_faces.items():
                distance = face_recognition.face_distance([known_encoding], face_encoding)[0]
                
                if distance < SIMILARITY_THRESHOLD:
                    confidence = (1 - distance) * 100
                    if confidence > best_confidence:
                        best_match = known_name
                        best_confidence = confidence
            
            results.append(best_match)
        
        # Draw bounding boxes only for recognized faces
        for idx, (top, right, bottom, left) in enumerate(face_locations):
            name = results[idx]
            
            if name is not None:
                confidence = (1 - face_recognition.face_distance([known_faces[name]], face_encodings[idx])[0]) * 100
                draw.rectangle([(left, top), (right, bottom)], outline="blue", width=3)
                label = f"{name}: {confidence:.1f}%"
                text_bbox = draw.textbbox((left, top), label, font=font)
                text_width = text_bbox[2] - text_bbox[0]
                text_height = text_bbox[3] - text_bbox[1]
                draw.rectangle([(left, top - text_height - 4), (left + text_width + 4, top)], fill="blue")
                draw.text((left + 2, top - text_height - 2), label, fill="white", font=font)
        
        output_path = FACES_DIR / f"result_{update.effective_user.id}.jpg"
        pil_image.save(output_path)
        
        recognized = [r for r in results if r is not None]
        unknown_count = len(results) - len(recognized)
        
        if recognized and unknown_count > 0:
            names_str = ", ".join(recognized)
            face_word = "face" if len(face_locations) == 1 else "faces"
            unknown_word = "face" if unknown_count == 1 else "faces"
            message = f"I found {len(face_locations)} {face_word} in this image. I recognized {names_str}, and {unknown_count} {unknown_word} I don't know."
        elif recognized:
            names_str = ", ".join(recognized)
            if len(recognized) == 1:
                message = f"I found {len(face_locations)} face in this image and the person is {names_str}"
            else:
                message = f"I found {len(face_locations)} faces in this image and the people are {names_str}"
        else:
            message = "❌ I don't recognize anyone in this image."
        
        with open(output_path, "rb") as photo_file:
            await update.message.reply_photo(
                photo=photo_file,
                caption=message
            )
        
        await update.message.reply_text(
            "👇 Choose an option from the main menu:",
            reply_markup=get_main_keyboard()
        )
        
        temp_path.unlink(missing_ok=True)
        output_path.unlink(missing_ok=True)
        return MAIN_MENU
        
    except Exception:
        traceback.print_exc()
        await update.message.reply_text(
            "⚠️ Sorry, I couldn't process that image. Please try again.",
            reply_markup=get_main_keyboard()
        )
        temp_path.unlink(missing_ok=True)
        return MAIN_MENU


async def handle_similar_celeb_image(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handle image uploaded for finding similar celebs."""
    if not update.message.photo:
        await update.message.reply_text("📸 Please send an image with a single face.")
        return WAITING_FOR_SIMILAR_CELEB_IMAGE
    
    ensure_faces_dir()
    
    photo = update.message.photo[-1]
    temp_path = None
    
    try:
        temp_path = FACES_DIR / f"similar_celeb_{update.effective_user.id}.jpg"
        await download_file_with_retry(context.bot, photo.file_id, temp_path)
    except Exception:
        traceback.print_exc()
        await update.message.reply_text(
            "⚠️ Sorry, there was a network problem while downloading the image. Please try again.",
            reply_markup=get_main_keyboard()
        )
        return ConversationHandler.END
    
    try:
        image = face_recognition.load_image_file(temp_path)
        encodings = face_recognition.face_encodings(image)
        
        if len(encodings) == 0:
            temp_path.unlink(missing_ok=True)
            await update.message.reply_text(
                "❌ No face found in this image. Please send an image with a single face."
            )
            return WAITING_FOR_SIMILAR_CELEB_IMAGE

        if len(encodings) > 1:
            temp_path.unlink(missing_ok=True)
            await update.message.reply_text(
                "❌ Multiple faces found in this image. Please send an image with a single face."
            )
            return WAITING_FOR_SIMILAR_CELEB_IMAGE

        # Find the most similar celeb
        best_celeb, celeb_image_path, distance = find_most_similar_celeb(encodings[0])
        
        # Clean up temp file
        temp_path.unlink(missing_ok=True)
        
        if best_celeb is None:
            await update.message.reply_text(
                "⚠️ Sorry, I couldn't find any similar celebs.",
                reply_markup=get_main_keyboard()
            )
            return ConversationHandler.END
        
        # Check if the best match is above the similarity threshold
        # distance < threshold means good match (face_distance returns lower = more similar)
        if distance >= SIMILARITY_THRESHOLD:
            await update.message.reply_text(
                "❌ I couldn't find a similar celeb for this image.",
                reply_markup=get_main_keyboard()
            )
            return ConversationHandler.END
        
        # Send the result with the celeb image
        try:
            with open(celeb_image_path, "rb") as celeb_photo:
                await update.message.reply_photo(
                    photo=celeb_photo,
                    caption=f"The celeb that the person is most similar to is {best_celeb}"
                )
        except Exception as e:
            print(f"Error sending celeb photo: {e}")
            await update.message.reply_text(
                f"The celeb that the person is most similar to is {best_celeb}"
            )
        
        await update.message.reply_text(
            "👇 Choose an option from the main menu:",
            reply_markup=get_main_keyboard()
        )
        
        return MAIN_MENU
        
    except Exception:
        traceback.print_exc()
        if temp_path:
            temp_path.unlink(missing_ok=True)
        await update.message.reply_text(
            "⚠️ Sorry, I couldn't process that image. Please try again.",
            reply_markup=get_main_keyboard()
        )
        return MAIN_MENU


async def handle_guess(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handle user's guess in the Guess the Celeb game."""
    user_id = update.effective_user.id
    guess = update.message.text.strip()
    
    if user_id not in guess_the_celeb_games:
        await update.message.reply_text(
            "No active game. 👇 Choose an option from the main menu:",
            reply_markup=get_main_keyboard()
        )
        return MAIN_MENU
    
    game = guess_the_celeb_games[user_id]
    correct_name = game["celeb_name"]
    stage = game["stage"]
    
    if guess == correct_name:
        # Correct guess
        await update.message.reply_text("🎉 Correct!")
        
        # Send full image
        try:
            with open(game["image_path"], "rb") as photo_file:
                await update.message.reply_photo(
                    photo=photo_file,
                    caption=f"This is {correct_name}! 🌟"
                )
        except Exception as e:
            print(f"Error sending full image: {e}")
            await update.message.reply_text(f"This is {correct_name}! 🌟")
        
        # Clean up game state
        del guess_the_celeb_games[user_id]
        
        await update.message.reply_text(
            "👇 Choose an option from the main menu:",
            reply_markup=get_main_keyboard()
        )
        return MAIN_MENU
    
    # Wrong guess
    if stage < 3:
        # Send wrong answer message
        await update.message.reply_text("Not quite! Here's a bigger reveal. Try again. 🤔")
        
        # Move to next stage
        next_stage = stage + 1
        game["stage"] = next_stage
        
        # Get new options for next stage
        stage_options_count = {2: 3, 3: 1}  # 3 others for stage 2 (total 4), 1 other for stage 3 (total 2)
        other_celebs = get_random_celebrities(correct_name, stage_options_count.get(next_stage, 1))
        options = [correct_name] + other_celebs
        import random
        random.shuffle(options)
        game["options"] = options
        
        # Get cropped image for next stage
        cropped_path = crop_face_for_stage(game["image_path"], next_stage)
        
        # Create keyboard with new options
        keyboard = [[KeyboardButton(name)] for name in options]
        reply_markup = ReplyKeyboardMarkup(keyboard, resize_keyboard=True, one_time_keyboard=True)
        
        # Send next stage image and options
        stage_messages = {
            2: "🎮 Stage 2 of 3: A bit more visible... Who is this? (You get 4 options)",
            3: "🎮 Stage 3 of 3: Almost full face! Last chance. Who is this? (You get 2 options)"
        }
        
        try:
            with open(cropped_path, "rb") as photo_file:
                await update.message.reply_photo(
                    photo=photo_file,
                    caption=stage_messages.get(next_stage, "Next stage!")
                )
        except Exception as e:
            print(f"Error sending stage image: {e}")
            await update.message.reply_text(f"Stage {next_stage}!")
        
        await update.message.reply_text(
            "👇 Choose your answer:",
            reply_markup=reply_markup
        )
        
        return WAITING_FOR_GUESS
    else:
        # Stage 3 failed - game over
        await update.message.reply_text(f"❌ The correct answer is {correct_name}.")
        
        # Send full image
        try:
            with open(game["image_path"], "rb") as photo_file:
                await update.message.reply_photo(
                    photo=photo_file,
                    caption=f"This was {correct_name}! 😎"
                )
        except Exception as e:
            print(f"Error sending full image: {e}")
            await update.message.reply_text(f"This was {correct_name}! 😎")
        
        # Clean up game state
        del guess_the_celeb_games[user_id]
        
        await update.message.reply_text(
            "Choose an option from the main menu:",
            reply_markup=get_main_keyboard()
        )
        return MAIN_MENU


async def invalid_input_handler(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handle invalid inputs by showing the main menu."""
    await update.message.reply_text(
        "👇 Choose an option from the main menu:",
        reply_markup=get_main_keyboard()
    )
    return MAIN_MENU


def main():
    """Run the bot."""
    load_known_faces()
    application = Application.builder().token(TOKEN).build()
    
    conv_handler = ConversationHandler(
        entry_points=[
            CommandHandler("start", start_command),
            MessageHandler(filters.TEXT & ~filters.COMMAND, handle_menu_choice)
        ],
        states={
            MAIN_MENU: [
                MessageHandler(filters.TEXT & ~filters.COMMAND, handle_menu_choice),
                MessageHandler(filters.ALL, invalid_input_handler)
            ],
            WAITING_FOR_ADD_IMAGE: [
                MessageHandler(filters.PHOTO, handle_add_image),
                MessageHandler(filters.TEXT & ~filters.COMMAND, handle_add_image_text),
                MessageHandler(filters.ALL, invalid_input_handler)
            ],
            WAITING_FOR_NAME: [
                MessageHandler(filters.TEXT & ~filters.COMMAND, handle_name_input),
                MessageHandler(filters.ALL, invalid_input_handler)
            ],
            WAITING_FOR_RECOGNIZE_IMAGE: [
                MessageHandler(filters.PHOTO, handle_recognize_image),
                MessageHandler(filters.TEXT & ~filters.COMMAND, handle_recognize_image_text),
                MessageHandler(filters.ALL, invalid_input_handler)
            ],
            WAITING_FOR_SIMILAR_CELEB_IMAGE: [
                MessageHandler(filters.PHOTO, handle_similar_celeb_image),
                MessageHandler(filters.TEXT & ~filters.COMMAND, handle_menu_choice),
                MessageHandler(filters.ALL, invalid_input_handler)
            ],
            WAITING_FOR_GUESS: [
                MessageHandler(filters.TEXT & ~filters.COMMAND, handle_guess),
                MessageHandler(filters.ALL, invalid_input_handler)
            ],
        },
        fallbacks=[],
        name="face_conversation"
    )
    
    application.add_handler(conv_handler)
    print("Bot is running...")
    application.run_polling(poll_interval=3)


if __name__ == "__main__":
    main()

