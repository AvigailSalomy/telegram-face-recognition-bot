import os
from dotenv import load_dotenv
from telegram import Update, KeyboardButton, ReplyKeyboardMarkup
from telegram.ext import Application, CommandHandler, MessageHandler, filters, ContextTypes

# Load environment variables
load_dotenv()

# Get token from .env
TOKEN = os.getenv("TELEGRAM_TOKEN")


async def start_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handle the /start command and show the custom keyboard."""
    keyboard = [
        [KeyboardButton("Hello"), KeyboardButton("World"), KeyboardButton("Telegram")]
    ]
    reply_markup = ReplyKeyboardMarkup(keyboard, resize_keyboard=True)
    
    await update.message.reply_text(
        "Welcome! Choose an option:",
        reply_markup=reply_markup
    )


async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handle incoming messages and reply with the same text."""
    text = update.message.text
    
    # Reply with the same text (lowercase)
    await update.message.reply_text(text.lower())


def main():
    """Run the bot."""
    # Create the Application
    application = Application.builder().token(TOKEN).build()
    
    # Add handlers
    application.add_handler(CommandHandler("start", start_command))
    application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))
    
    # Start polling
    print("Bot is running...")
    application.run_polling(poll_interval=3)


if __name__ == "__main__":
    main()
