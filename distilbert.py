import re
import torch
import demoji
import pandas as pd
import unicodedata
import numpy as np
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification


PREDEFINED_ASPECTS = {
    "Battery": ["battery", "charge", "charging", "lifespan", "power", "capacity", "fast charge", "quick charge", "rapid charge", "wireless charge", "reverse wireless charging", "battery life", "battery drain", "power consumption", "energy efficiency", "charging speed", "full charge", "low battery", "battery health", "optimization", "power saving mode"],
    "Display": ["screen", "display", "resolution", "brightness", "peak brightness", "refresh rate", "adaptive refresh rate", "panel", "oled", "amoled", "super amoled", "poled", "ltpo", "lcd", "ips lcd", "hdr", "hdr10", "hdr10+", "dolby vision", "touchscreen", "multi-touch", "bezels", "thin bezels", "notch", "hole-punch", "under-display camera", "color accuracy", "color gamut", "viewing angles", "outdoor visibility", "sunlight legibility", "screen size", "aspect ratio", "pixel density", "protection", "gorilla glass", "ceramic shield"],
    "Performance": ["speed", "performance", "lag", "stutter", "jank", "processor", "chipset", "soc", "cpu", "cores", "clock speed", "gpu", "graphics", "ram", "memory", "lpddr5", "thermal", "cooling", "heat management", "overheating", "throttling", "fps", "frame rate", "benchmark", "antutu", "geekbench", "responsiveness", "smoothness", "multitasking", "app loading times", "system stability"],
    "Camera": ["camera", "photos", "pictures", "videos", "lens", "focal length", "aperture", "sensor size", "pixel size", "zoom", "optical zoom", "digital zoom", "hybrid zoom", "megapixel", "image", "quality", "detail", "sharpness", "noise", "dynamic range", "hdr", "auto hdr", "night mode", "low light performance", "portrait mode", "bokeh", "depth effect", "wide-angle", "ultrawide", "telephoto", "macro", "optical image stabilization", "ois", "electronic image stabilization", "eis", "selfie", "front camera", "videography", "video recording", "resolution", "frame rate", "video stabilization", "cinematic mode"],
    "Design": ["design", "build", "aesthetic", "form factor", "size", "dimensions", "weight", "ergonomic", "materials", "aluminum", "glass", "ceramic", "plastic", "polycarbonate", "stainless steel", "frame", "back panel", "compact", "sleek", "premium feel", "ruggedness", "durability", "look and feel", "handling", "one-handed use", "button placement", "port placement", "camera bump", "color options", "finish", "texture"],
    "Audio": ["sound", "speaker", "stereo speakers", "mono speaker", "microphone", "audio", "bass", "treble", "clarity", "volume", "loudness", "dolby atmos", "spatial audio", "noise cancellation", "active noise cancellation", "anc", "passive noise cancellation", "headphone jack", "3.5mm jack", "audio quality", "soundstage", "immersive audio", "surround sound", "call quality", "voice clarity"],
    "Connectivity": ["wifi", "wi-fi", "wireless internet", "bluetooth", "version", "nfc", "near-field communication", "usb", "usb-c", "thunderbolt", "ethernet", "hdmi", "wireless", "cellular", "mobile network", "5g", "sub-6ghz", "mmwave", "lte", "4g", "signal strength", "network speed", "data transfer speed", "connectivity issues", "dropped connections", "pairing", "hotspot", "casting", "screen mirroring"],
    "Storage": ["storage", "internal storage", "ssd", "nvme", "ufs", "hdd", "capacity", "expandable", "memory", "microsd", "sd card", "cloud storage", "read speed", "write speed", "file transfer", "available storage", "storage management"],
    "Software": ["software", "os", "operating system", "android", "ios", "harmonyos", "update", "software update", "os update", "security patch", "bug", "glitch", "issue", "feature", "new features", "app", "application", "pre-installed apps", "bloatware", "interface", "ui", "user interface", "ux", "user experience", "customization", "themes", "launcher", "user-friendly", "intuitive", "stability", "performance", "responsiveness"],
    "Security": ["security", "fingerprint sensor", "in-display fingerprint sensor", "side-mounted fingerprint sensor", "rear-mounted fingerprint sensor", "face unlock", "facial recognition", "encryption", "data encryption", "privacy", "vpn", "virtual private network", "firewall", "biometric", "authentication", "password", "pin", "pattern", "data protection", "permissions", "app permissions", "security features", "malware protection"],
    "Gaming": ["gaming", "game", "fps", "frame rate", "refresh rate", "touch latency", "input lag", "ray tracing", "mobile gaming", "game performance", "graphics", "visuals", "smoothness", "optimization", "game modes", "controller support", "vr", "virtual reality", "streaming", "game streaming", "esports", "competitive gaming", "cooling for gaming", "thermal management for gaming"],
    "Durability": ["durability", "waterproof", "water resistance", "ip rating", "ip68", "ip67", "splashproof", "dustproof", "shockproof", "drop resistance", "scratch resistance", "rugged", "build quality", "resistance to damage", "longevity", "wear and tear"],
    "Compatibility": ["compatibility", "integration", "sync", "ecosystem", "apple ecosystem", "android ecosystem", "windows", "linux", "device compatibility", "accessory compatibility", "software compatibility", "cross-platform", "interoperability"],
    "Price": ["price", "cost", "value", "expensive", "affordable", "worth", "budget", "overpriced", "underpriced", "deal", "pricing", "value for money", "cost-effective", "flagship price", "mid-range price", "entry-level price", "return on investment"],
    "Features": ["features", "innovations", "unique selling points", "gimmicks", "functionality", "capabilities", "special features", "advanced features", "key features", "notable features"], 
    "Accessories": ["accessories", "charger", "power adapter", "wall adapter", "charging cable", "usb cable", "case", "phone case", "protective case", "screen protector", "tempered glass", "stylus", "headphones", "earphones", "buds", "smartwatch compatibility", "other accessories"],
}

def clean_text_for_distilbert(text):
    """Cleans and normalizes text for DistilBERT preprocessing."""
    
    # 1. Remove URLs
    text = re.sub(r"http\S+", "", text)
    
    # 2. Normalize Unicode (fixes special characters)
    text = unicodedata.normalize("NFKC", text)
    
    # 3. Convert emojis to text descriptions
    emojis = demoji.findall(text)
    for emoji in emojis:
        text = text.replace(emoji, " " + emojis[emoji].split(":")[0])
    
    # 4. Lowercasing (optional: helps with noisy data)
    text = text.lower()
    
    # 5. Trim unnecessary spaces
    return text.strip()

def load_distilbert():
    # Load the DistilBERT tokenizer and model
    model_path = "danctl/tech_distilbert_fyp"
    tokenizer = DistilBertTokenizer.from_pretrained(model_path)
    model = DistilBertForSequenceClassification.from_pretrained(model_path)
    model.eval()

    # Return all models
    return tokenizer, model


def analyze_sentiment_bert(tokenizer, model, text):
    """Analyze sentiment using DistilBERT model."""
    # Tokenize and encode the input text
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512)

    # Get the model outputs (logits)
    with torch.no_grad():
        outputs = model(**inputs)

    logits = outputs.logits  # Get logits from the model's output

    # Apply softmax to get probabilities
    probabilities = torch.nn.functional.softmax(logits, dim=-1)

    # Calculate the sentiment score based on probabilities (assuming a 0-4 range)
    sentiment_score = np.dot(probabilities.numpy(), np.array([0, 1, 2, 3, 4]))  # Weighted sentiment score
    
    # Scale sentiment score to range [0, 1]
    scaled_sentiment_score = sentiment_score / 4
    
    return scaled_sentiment_score

def extract_aspect_sentiment(tokenizer, model, text):
    """Extracts sentiment scores for predefined aspects using DistilBERT."""
    aspect_sentiments = {aspect: [] for aspect in PREDEFINED_ASPECTS.keys()}

    contrast_split = re.split(r'\b(but|and|however|although|,)\b', text, flags=re.IGNORECASE)

    for segment in contrast_split:
        segment = segment.strip()

        for aspect, keywords in PREDEFINED_ASPECTS.items():
            if any(keyword in segment.lower() for keyword in keywords):
                aspect_score = analyze_sentiment_bert(tokenizer, model, segment)
                aspect_sentiments[aspect].append(aspect_score)

    return {
        aspect: np.mean(scores) if scores else None  # Keep None if no aspect is mentioned
        for aspect, scores in aspect_sentiments.items()
    }

