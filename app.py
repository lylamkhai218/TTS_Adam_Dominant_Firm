#!/usr/bin/env python3
"""
Flask Web Backend for ElevenLabs TTS and DSP Enhancer
Provides API endpoints to convert text to speech and apply local DSP filters.
"""

import os
import sys
import glob
import time
import uuid
from flask import Flask, request, jsonify, render_template, send_from_directory
from dotenv import load_dotenv

# Ensure we can import from tts_adam_enhance
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
try:
    from tts_adam_enhance import enhance_audio, save_wav, ADAM_VOICE_ID
    from elevenlabs.client import ElevenLabs
    from elevenlabs import VoiceSettings
except ImportError as e:
    print(f"Error importing modules: {e}")
    sys.exit(1)

app = Flask(__name__)

# Ensure static directory exists
STATIC_DIR = os.path.join(app.root_path, 'static')
os.makedirs(STATIC_DIR, exist_ok=True)


def cleanup_old_files():
    """
    Deletes any WAV files in the static folder that are older than 15 minutes
    to prevent disk space bloating.
    """
    now = time.time()
    cutoff = now - 900  # 15 minutes ago
    pattern = os.path.join(STATIC_DIR, "*.wav")
    for filepath in glob.glob(pattern):
        try:
            if os.path.getmtime(filepath) < cutoff:
                os.remove(filepath)
        except OSError as e:
            print(f"Error removing temp file {filepath}: {e}")


@app.route('/')
def index():
    """Renders the main web interface."""
    return render_template('index.html')


@app.route('/static/<path:filename>')
def serve_static(filename):
    """Serves generated WAV files from the static directory."""
    return send_from_directory(STATIC_DIR, filename)


@app.route('/api/generate', methods=['POST'])
def generate_speech():
    """
    API endpoint that accepts text and synthesis configurations,
    makes a request to ElevenLabs, applies DSP enhancement,
    and returns URLs to the raw and enhanced audio.
    """
    # Clean up old files first
    cleanup_old_files()

    # Load API key
    load_dotenv(override=True)
    api_key = os.getenv("ELEVENLABS_API_KEY")
    if not api_key:
        return jsonify({"error": "Missing ElevenLabs API Key in server configuration. Please check your .env file."}), 500

    data = request.json or {}
    text = data.get("text", "").strip()
    if not text:
        return jsonify({"error": "Văn bản nhập vào không được trống."}), 400

    # Get voice settings with defaults
    stability = float(data.get("stability", 0.75))
    similarity = float(data.get("similarity", 0.85))
    style = float(data.get("style", 0.0))
    use_speaker_boost = bool(data.get("use_speaker_boost", True))

    # Get DSP settings with defaults
    warmth_gain = float(data.get("warmth_gain", 3.5))
    clarity_gain = float(data.get("clarity_gain", 2.5))
    threshold_db = float(data.get("threshold", -18.0))
    ratio = float(data.get("ratio", 3.0))
    skip_enhance = bool(data.get("skip_enhance", False))

    print(f"Web request - Text length: {len(text)}, Stability={stability}, Similarity={similarity}, Warmth={warmth_gain}, Clarity={clarity_gain}")

    try:
        # Initialize ElevenLabs Client
        client = ElevenLabs(api_key=api_key)

        # Output formats to try
        pcm_formats = [
            {"format": "pcm_44100", "rate": 44100},
            {"format": "pcm_24000", "rate": 24000},
            {"format": "pcm_22050", "rate": 22050},
            {"format": "pcm_16000", "rate": 16000}
        ]

        raw_audio_bytes = None
        selected_rate = None

        # Request loop
        for fmt_info in pcm_formats:
            fmt = fmt_info["format"]
            rate = fmt_info["rate"]
            try:
                print(f"API Call - Format: {fmt} ({rate}Hz)...")
                response_gen = client.text_to_speech.convert(
                    text=text,
                    voice_id=ADAM_VOICE_ID,
                    model_id="eleven_v3",
                    output_format=fmt,
                    voice_settings=VoiceSettings(
                        stability=stability,
                        similarity_boost=similarity,
                        style=style,
                        use_speaker_boost=use_speaker_boost
                    )
                )

                # Consume generator stream
                bytes_list = []
                for chunk in response_gen:
                    if chunk:
                        bytes_list.append(chunk)
                
                raw_audio_bytes = b"".join(bytes_list)
                selected_rate = rate
                break
            except Exception as e:
                print(f"Format {fmt} failed: {e}")

        if not raw_audio_bytes:
            return jsonify({"error": "Không thể kết nối ElevenLabs API. Vui lòng kiểm tra lại API Key hoặc quyền truy cập của gói tài khoản."}), 500

        # Create unique file identifiers
        file_id = str(uuid.uuid4())
        raw_filename = f"raw_{file_id}.wav"
        enhanced_filename = f"enhanced_{file_id}.wav"

        raw_filepath = os.path.join(STATIC_DIR, raw_filename)
        enhanced_filepath = os.path.join(STATIC_DIR, enhanced_filename)

        # Save raw audio
        save_wav(raw_audio_bytes, raw_filepath, selected_rate)

        # Apply DSP enhancement
        if not skip_enhance:
            try:
                enhanced_bytes = enhance_audio(
                    pcm_bytes=raw_audio_bytes,
                    sample_rate=selected_rate,
                    warmth_gain=warmth_gain,
                    clarity_gain=clarity_gain,
                    threshold_db=threshold_db,
                    ratio=ratio
                )
                save_wav(enhanced_bytes, enhanced_filepath, selected_rate)
                has_enhanced = True
            except Exception as dsp_err:
                print(f"DSP error: {dsp_err}")
                has_enhanced = False
        else:
            has_enhanced = False

        # Build response
        response_data = {
            "success": True,
            "sample_rate": selected_rate,
            "raw_url": f"/static/{raw_filename}"
        }
        if has_enhanced:
            response_data["enhanced_url"] = f"/static/{enhanced_filename}"
        else:
            response_data["enhanced_url"] = None

        return jsonify(response_data)

    except Exception as err:
        print(f"General error: {err}")
        return jsonify({"error": f"Lỗi máy chủ: {str(err)}"}), 500


@app.route('/api/enhance', methods=['POST'])
def enhance_text_api():
    """
    API endpoint that accepts text and enhances it with emotional tags
    using Gemini API or a local keyword-based fallback system.
    """
    data = request.json or {}
    text = data.get("text", "").strip()
    if not text:
        return jsonify({"error": "Văn bản không được trống."}), 400

    # Load Gemini API Key
    dotenv_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '.env')
    load_dotenv(dotenv_path=dotenv_path, override=True)
    gemini_key = os.getenv("GEMINI_API_KEY")
    
    # Check if Gemini key exists and is not a placeholder
    use_gemini = False
    if gemini_key and gemini_key.strip() and gemini_key.strip() != "your_gemini_key_here":
        use_gemini = True

    enhanced_text = None
    method = "local_fallback"

    if use_gemini:
        try:
            print("Enhancing text using Google Gemini API...")
            import requests
            url = "https://generativelanguage.googleapis.com/v1beta/models/gemini-flash-latest:generateContent"
            headers = {
                "Content-Type": "application/json",
                "X-goog-api-key": gemini_key.strip()
            }
            
            prompt = (
                "You are an expert TTS voice director. Analyze the following Vietnamese text and insert emotional/expressive voice direction tags in square brackets to guide the ElevenLabs v3 voice synthesis engine.\n"
                "Supported tags: [excited], [sad], [angry], [sorrowful], [whispers], [shouts], [calm], [laughs], [sighs], [gasp].\n"
                "Rules:\n"
                "1. Keep the original text content exactly identical. Only add emotional/expressive tags in square brackets at appropriate places (usually at the beginning of clauses or sentences).\n"
                "2. Insert tags only when the context clearly suggests an emotion or vocal style (e.g. laughter, whispering, sadness, anger, sighing).\n"
                "3. Do not over-use tags. Add at most 1 or 2 tags for short text, and only where it adds value.\n"
                "4. Output ONLY the resulting enhanced text. Do NOT write any introductions, explanations, markdown formatting (like ```), or other text.\n"
                f"Text: {text}"
            )
            
            payload = {
                "contents": [{
                    "parts": [{"text": prompt}]
                }],
                "generationConfig": {
                    "temperature": 0.2
                }
            }
            
            res = requests.post(url, json=payload, headers=headers, timeout=10)
            if res.status_code == 200:
                res_data = res.json()
                enhanced_text = res_data["candidates"][0]["content"]["parts"][0]["text"].strip()
                # Clean up potential markdown formatting if Gemini ignored the prompt rules
                if enhanced_text.startswith("```"):
                    lines = enhanced_text.split("\n")
                    if len(lines) >= 3:
                        # Remove markdown code block symbols
                        enhanced_text = "\n".join(lines[1:-1]).strip()
                method = "gemini_api"
                print(f"Gemini enhancement successful: '{enhanced_text}'")
            else:
                print(f"Gemini API returned error code {res.status_code}: {res.text}")
        except Exception as e:
            print(f"Error calling Gemini API: {e}")

    # Fallback to local rule-based enhancement
    if not enhanced_text:
        print("Using local keyword-based fallback text enhancement...")
        enhanced_text = local_enhance_text(text)

    return jsonify({
        "success": True,
        "original_text": text,
        "enhanced_text": enhanced_text,
        "method": method
    })


def local_enhance_text(text):
    """
    Applies Vietnamese keyword-based rules to add emotion tags.
    """
    enhanced = text
    lower_text = enhanced.lower()
    
    # Comprehensive mapping of ElevenLabs tags to list of Vietnamese keywords
    mappings = {
        "happy": ["vui vẻ", "vui sướng", "hạnh phúc"],
        "excited": ["hào hứng", "phấn khích", "tuyệt vời", "yeah", "tuyệt hảo"],
        "cheerful": ["tươi vui", "vui tươi", "hân hoan"],
        "energetic": ["đầy năng lượng", "mạnh mẽ", "sung sức"],
        "calm": ["bình tĩnh", "điềm tĩnh", "nhẹ nhõm"],
        "relaxed": ["thư giãn", "thong thả", "thoải mái"],
        "soft": ["nhẹ nhàng", "êm ái"],
        "gentle": ["dịu dàng", "ôn hòa"],
        "sad": ["buồn", "buồn bã", "đau lòng", "thất vọng", "khổ sở"],
        "emotional": ["xúc động", "cảm động"],
        "serious": ["nghiêm túc", "đứng đắn", "nghiêm nghị"],
        "dramatic": ["kịch tính", "gay cấn"],
        "angry": ["tức giận", "giận dữ", "tức điên", "đáng ghét"],
        "frustrated": ["khó chịu", "bực bội", "bực mình"],
        "scared": ["sợ hãi", "hoảng sợ", "khiếp sợ"],
        "nervous": ["lo lắng", "bồn chồn", "hồi hộp"],
        "confident": ["tự tin", "quyết đoán"],
        "mysterious": ["bí ẩn", "kỳ bí", "huyền bí"],
        "sarcastic": ["mỉa mai", "châm biếm"],
        "playful": ["tinh nghịch", "đùa giỡn", "nhí nhảnh"],
        "seductive": ["quyến rũ", "gợi cảm"],
        "romantic": ["lãng mạn", "tình tứ"],
        "inspirational": ["truyền cảm hứng"],
        "motivational": ["tạo động lực", "cố lên"],
        "professional": ["chuyên nghiệp"],
        "authoritative": ["uy quyền", "quyền lực"],
        "cinematic": ["điện ảnh"],
        "epic": ["hùng tráng", "epic"],
        "funny": ["hài hước", "dí dỏm"],
        "awkward": ["ngượng ngùng", "lúng túng", "ngượng ngịu"],
        "whispering": ["thì thầm", "nói nhỏ", "suỵt", "nói khẽ", "nói thì thầm"],
        "crying": ["khóc", "sụt sùi", "khóc lóc", "nức nở", "khóc sụt sịt"],
        "shouting": ["hét lớn", "hét lên", "la hét", "yelling"],
        
        # Kiểu nói / cách thể hiện
        "laughing": ["cười lớn", "haha", "hihi", "hehe", "kaka", "cười", "a đang làm ai"],
        "chuckles": ["cười khúc khích", "chuckle"],
        "giggles": ["cười khanh khách", "giggle"],
        "sighs": ["thở dài", "haizz", "ôi chao"],
        "gasping": ["thở hổn hển"],
        "breathing heavily": ["thở nặng nhọc", "thở gấp"],
        "stuttering": ["nói lắp", "lắp bắp"],
        "pausing": ["ngắt nghỉ", "tạm dừng"],
        "yelling": ["la hét", "yell"],
        "murmuring": ["lẩm bẩm", "thì thào"],
        "speaking fast": ["nói nhanh", "nói lẹ"],
        "speaking slowly": ["nói chậm", "nói rề rề"],
        "monotone": ["giọng đều đều", "monotone"],
        "storytelling": ["kể chuyện", "ngữ điệu kể chuyện"],
        "conversational": ["trò chuyện tự nhiên", "hội thoại"],
        "announcer style": ["kiểu phát thanh viên", "phát thanh viên"],
        "trailer voice": ["giọng trailer phim", "trailer phim"],
        "podcast tone": ["giọng podcast", "podcast"],
        "ASMR style": ["phong cách asmr", "asmr"],
        
        # Âm thanh biểu cảm không lời
        "haha": ["ha ha", "haha"],
        "hmm": ["hừm", "hmm"],
        "ahh": ["ahh", "à", "ah"],
        "oh": ["ô", "oh"],
        "uh": ["ừm", "uh"],
        "um": ["umm", "um"],
        "eh": ["eh"],
        "tch": ["chậc", "tch"],
        "mhm": ["ừ hử", "mhm"],
        "gasp": ["hít mạnh vì bất ngờ", "gasp"],
        "sniff": ["sụt sịt", "sniff"],
        "sob": ["nức nở", "sob"],
        "cough": ["ho", "cough"],
        "clearing throat": ["hắng giọng", "clearing throat"],
        "lip smack": ["chép miệng", "lip smack"],
        "exhale": ["thở ra", "exhale"],
        "inhale": ["hít vào", "inhale"]
    }

    # Match each category and insert at most 2 distinct tags at the beginning
    added_tags = []
    for tag, keywords in mappings.items():
        if any(kw in lower_text for kw in keywords):
            tag_str = f"[{tag}]"
            if tag_str not in enhanced:
                added_tags.append(tag_str)
                if len(added_tags) >= 2:
                    break
                    
    if added_tags:
        enhanced = " ".join(added_tags) + " " + enhanced
        
    return enhanced


if __name__ == '__main__':
    # Run locally on port 5000
    app.run(host='127.0.0.1', port=5000, debug=True)
