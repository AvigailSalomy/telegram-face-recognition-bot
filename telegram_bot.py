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

# Load environment variables
load_dotenv()

# Get token from .env
TOKEN = os.getenv("TELEGRAM_TOKEN")

# Face storage paths
FACES_DIR = Path("stored_faces")
FACES_DATA_FILE = FACES_DIR / "faces_data.pkl"

# Conversation states
WAITING_FOR_NAME = 1
WAITING_FOR_ADD_IMAGE = 2
WAITING_FOR_RECOGNIZE_IMAGE = 3

# Similarity threshold (distance below this = match)
SIMILARITY_THRESHOLD = 0.6

# Global storage for known faces
known_faces = {}  # {name: encoding}


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
    """Delete all stored faces."""
    global known_faces
    known_faces = {}
    try:
        ensure_faces_dir()
        if FACES_DATA_FILE.exists():
            FACES_DATA_FILE.unlink()
        for file in FACES_DIR.glob("*.png"):
            file.unlink()
    except Exception:
        pass


def get_main_keyboard():
    """Create the main keyboard with 3 buttons."""
    keyboard = [
        [KeyboardButton("Add face"), KeyboardButton("Recognize faces"), KeyboardButton("Reset faces")]
    ]
    return ReplyKeyboardMarkup(keyboard, resize_keyboard=True)


async def start_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handle the /start command and show the custom keyboard."""
    load_known_faces()
    await update.message.reply_text(
        "Welcome to the Face Recognition Bot!\nChoose an option from the main menu:",
        reply_markup=get_main_keyboard()
    )


async def handle_menu_choice(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handle main menu button choices."""
    text = update.message.text
    
    if text == "Add face":
        await update.message.reply_text("Upload an image with a single face")
        return WAITING_FOR_ADD_IMAGE
        
    elif text == "Recognize faces":
        load_known_faces()
        if not known_faces:
            await update.message.reply_text(
                "No faces stored yet. Please add a face first."
            )
            await update.message.reply_text(
                "Choose an option from the main menu:",
                reply_markup=get_main_keyboard()
            )
            return ConversationHandler.END
        await update.message.reply_text(
            "Upload an image with at least one face and I will recognize who is in this image"
        )
        return WAITING_FOR_RECOGNIZE_IMAGE
        
    elif text == "Reset faces":
        reset_all_faces()
        await update.message.reply_text(
            "All faces have been deleted. The bot now knows no one."
        )
        await update.message.reply_text(
            "Choose an option from the main menu:",
            reply_markup=get_main_keyboard()
        )
        return ConversationHandler.END
        
    else:
        await update.message.reply_text(
            "Please choose an option from the menu:",
            reply_markup=get_main_keyboard()
        )
        return ConversationHandler.END


async def handle_add_image(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handle image uploaded for adding a face."""
    if not update.message.photo:
        await update.message.reply_text("Please send an image with a single face.")
        return WAITING_FOR_ADD_IMAGE
    
    ensure_faces_dir()
    
    photo = update.message.photo[-1]
    
    try:
        temp_path = FACES_DIR / f"temp_{update.effective_user.id}.jpg"
        await download_file_with_retry(context.bot, photo.file_id, temp_path)
    except Exception:
        traceback.print_exc()
        await update.message.reply_text(
            "Sorry, there was a network problem while downloading the image. Please try again."
        )
        return ConversationHandler.END
    
    try:
        image = face_recognition.load_image_file(temp_path)
        encodings = face_recognition.face_encodings(image)
        
        if len(encodings) == 0:
            await update.message.reply_text(
                "No face found in this image. Please send an image with a single face."
            )
            return WAITING_FOR_ADD_IMAGE
        
        if len(encodings) > 1:
            await update.message.reply_text(
                "Multiple faces found in this image. Please send an image with a single face."
            )
            return WAITING_FOR_ADD_IMAGE
        
        await update.message.reply_text("Great. What's the name of the person in this image?")
        context.user_data['temp_encoding'] = encodings[0]
        context.user_data['temp_image_path'] = temp_path
        return WAITING_FOR_NAME
        
    except Exception:
        traceback.print_exc()
        await update.message.reply_text(
            "Sorry, I couldn't process that image. Please try again."
        )
        return ConversationHandler.END


async def handle_name_input(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handle name input after face image is uploaded."""
    name = update.message.text.strip()
    
    if not name:
        await update.message.reply_text("Please enter a valid name.")
        return WAITING_FOR_NAME
    
    encoding = context.user_data.get('temp_encoding')
    temp_path = context.user_data.get('temp_image_path')
    
    if encoding is None or temp_path is None:
        await update.message.reply_text(
            "Something went wrong. Please start again by pressing 'Add face'.",
            reply_markup=get_main_keyboard()
        )
        return ConversationHandler.END
    
    global known_faces
    known_faces[name] = encoding
    
    if not save_known_faces():
        await update.message.reply_text(
            "Sorry, I couldn't save the face. Please try again.",
            reply_markup=get_main_keyboard()
        )
        context.user_data.pop('temp_encoding', None)
        context.user_data.pop('temp_image_path', None)
        if temp_path.exists():
            temp_path.unlink(missing_ok=True)
        return ConversationHandler.END
    
    ensure_faces_dir()
    safe_name = sanitize_filename(name)
    face_image_path = FACES_DIR / f"{safe_name}.png"
    if temp_path.exists():
        temp_path.rename(face_image_path)
    
    context.user_data.pop('temp_encoding', None)
    context.user_data.pop('temp_image_path', None)
    
    await update.message.reply_text("Great. I will now remember this face")
    await update.message.reply_text(
        "Choose an option from the main menu:",
        reply_markup=get_main_keyboard()
    )
    return ConversationHandler.END


async def handle_recognize_image(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handle image uploaded for recognizing faces."""
    load_known_faces()
    
    if not update.message.photo:
        await update.message.reply_text("Please send an image with at least one face.")
        return WAITING_FOR_RECOGNIZE_IMAGE
    
    ensure_faces_dir()
    
    photo = update.message.photo[-1]
    
    try:
        temp_path = FACES_DIR / f"recognize_{update.effective_user.id}.jpg"
        await download_file_with_retry(context.bot, photo.file_id, temp_path)
    except Exception:
        traceback.print_exc()
        await update.message.reply_text(
            "Sorry, there was a network problem while downloading the image. Please try again.",
            reply_markup=get_main_keyboard()
        )
        return ConversationHandler.END
    
    try:
        image = face_recognition.load_image_file(temp_path)
        face_locations = face_recognition.face_locations(image)
        face_encodings = face_recognition.face_encodings(image, face_locations)
        
        if not face_encodings:
            await update.message.reply_text(
                "I don't recognize anyone in this image",
                reply_markup=get_main_keyboard()
            )
            temp_path.unlink(missing_ok=True)
            return ConversationHandler.END
        
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
            message = "I don't recognize anyone in this image"
        
        with open(output_path, "rb") as photo_file:
            await update.message.reply_photo(
                photo=photo_file,
                caption=message
            )
        
        await update.message.reply_text(
            "Choose an option from the main menu:",
            reply_markup=get_main_keyboard()
        )
        
        temp_path.unlink(missing_ok=True)
        output_path.unlink(missing_ok=True)
        return ConversationHandler.END
        
    except Exception:
        traceback.print_exc()
        await update.message.reply_text(
            "Sorry, I couldn't process that image. Please try again.",
            reply_markup=get_main_keyboard()
        )
        temp_path.unlink(missing_ok=True)
        return ConversationHandler.END


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
            WAITING_FOR_ADD_IMAGE: [
                MessageHandler(filters.PHOTO, handle_add_image),
                MessageHandler(filters.TEXT & ~filters.COMMAND, handle_menu_choice)
            ],
            WAITING_FOR_NAME: [
                MessageHandler(filters.TEXT & ~filters.COMMAND, handle_name_input)
            ],
            WAITING_FOR_RECOGNIZE_IMAGE: [
                MessageHandler(filters.PHOTO, handle_recognize_image),
                MessageHandler(filters.TEXT & ~filters.COMMAND, handle_menu_choice)
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
