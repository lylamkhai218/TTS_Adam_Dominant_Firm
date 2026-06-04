#!/usr/bin/env python3
"""
ElevenLabs Text-to-Speech (TTS) with Local DSP Audio Enhancement
Model: Eleven v3
Voice: Adam (Voice ID: pNInz6obpgDQGcFmaJgB) - Configured for a Dominant, Firm tone.
Enhanced using: NumPy/SciPy Digital Signal Processing (Highpass, Peaking EQs, Compressor, Normalizer).
"""

import os
import sys
import wave
import argparse
from dotenv import load_dotenv

# Optional packages
try:
    from elevenlabs import VoiceSettings
    from elevenlabs.client import ElevenLabs
except ImportError:
    print("Error: The 'elevenlabs' package is not installed. Please run: pip install elevenlabs")
    sys.exit(1)

try:
    import numpy as np
    import scipy.signal as signal
except ImportError:
    print("Error: 'numpy' or 'scipy' is not installed. Please run: pip install numpy scipy")
    sys.exit(1)


# Adam Voice ID on ElevenLabs
ADAM_VOICE_ID = "pNInz6obpgDQGcFmaJgB"

def get_peaking_eq_sos(fs, f0, gain_db, q=0.8):
    """
    Computes biquad peaking equalizer coefficients in Second-Order Sections (SOS) format.
    Uses Robert Bristow-Johnson's Audio EQ Cookbook formulas.
    """
    A = 10.0 ** (gain_db / 40.0)
    w0 = 2.0 * np.pi * f0 / fs
    alpha = np.sin(w0) / (2.0 * q)
    
    b0 = 1.0 + alpha * A
    b1 = -2.0 * np.cos(w0)
    b2 = 1.0 - alpha * A
    a0 = 1.0 + alpha / A
    a1 = -2.0 * np.cos(w0)
    a2 = 1.0 - alpha / A
    
    # Normalize by a0 and return as SOS shape (1, 6)
    return np.array([[b0/a0, b1/a0, b2/a0, 1.0, a1/a0, a2/a0]], dtype=np.float32)


def enhance_audio(pcm_bytes, sample_rate, warmth_gain=3.5, clarity_gain=2.5, threshold_db=-18.0, ratio=3.0):
    """
    Applies a DSP vocal enhancement pipeline to the raw PCM data:
    1. High-Pass Filter (removes rumble below 80 Hz)
    2. Warmth EQ (boosts 150 Hz to add chest resonance and firmness)
    3. Presence/Clarity EQ (boosts 3.2 kHz for crispness and articulation)
    4. Dynamic Range Compressor (even loudness, dense tone)
    5. Normalization (brings peaks to -1 dB)
    """
    print(f"Applying DSP enhancement (Sample Rate: {sample_rate} Hz)...")
    
    # Convert raw signed 16-bit PCM bytes to floating-point array normalized to [-1.0, 1.0]
    audio_data = np.frombuffer(pcm_bytes, dtype=np.int16).astype(np.float32)
    audio_data /= 32768.0
    
    if len(audio_data) == 0:
        return pcm_bytes

    # --- 1. High-Pass Filter (cutoff 80 Hz, 4th order) ---
    sos_hp = signal.butter(4, 80.0, btype='highpass', fs=sample_rate, output='sos')
    audio_processed = signal.sosfilt(sos_hp, audio_data)

    # --- 2. Warmth EQ (Peaking at 150 Hz) ---
    if warmth_gain != 0:
        sos_warmth = get_peaking_eq_sos(sample_rate, f0=150.0, gain_db=warmth_gain, q=0.8)
        audio_processed = signal.sosfilt(sos_warmth, audio_processed)

    # --- 3. Presence/Clarity EQ (Peaking at 3200 Hz) ---
    if clarity_gain != 0:
        sos_clarity = get_peaking_eq_sos(sample_rate, f0=3200.0, gain_db=clarity_gain, q=1.0)
        audio_processed = signal.sosfilt(sos_clarity, audio_processed)

    # --- 4. Dynamic Range Compressor ---
    # 4.1 RMS Level detection with exponential smoothing (30 ms window)
    tau_env = 0.030
    alpha_env = np.exp(-1.0 / (sample_rate * tau_env))
    b_env = [1.0 - alpha_env]
    a_env = [1.0, -alpha_env]
    rms_squared = signal.lfilter(b_env, a_env, audio_processed ** 2)
    rms = np.sqrt(np.maximum(rms_squared, 1e-10))
    
    # 4.2 Gain Computer
    rms_db = 20.0 * np.log10(rms)
    over_threshold = rms_db - threshold_db
    target_gain_db = np.where(over_threshold > 0, over_threshold * (1.0 / ratio - 1.0), 0.0)
    
    # 4.3 Smooth the gain change (50 ms time constant) to prevent clicking/distortion
    tau_smooth = 0.050
    alpha_smooth = np.exp(-1.0 / (sample_rate * tau_smooth))
    b_smooth = [1.0 - alpha_smooth]
    a_smooth = [1.0, -alpha_smooth]
    smoothed_gain_db = signal.lfilter(b_smooth, a_smooth, target_gain_db)
    
    # Apply calculated gain envelope
    gain = 10.0 ** (smoothed_gain_db / 20.0)
    audio_compressed = audio_processed * gain

    # --- 5. Peak Normalization (-1 dB = 10^(-1/20) ~= 0.891) ---
    peak = np.max(np.abs(audio_compressed))
    target_peak = 10.0 ** (-1.0 / 20.0)
    if peak > 0:
        audio_normalized = audio_compressed * (target_peak / peak)
    else:
        audio_normalized = audio_compressed

    # Clip to avoid overflow and convert back to 16-bit integer PCM
    audio_clipped = np.clip(audio_normalized, -1.0, 1.0)
    audio_out = (audio_clipped * 32767.0).astype(np.int16)
    
    return audio_out.tobytes()


def save_wav(pcm_bytes, file_path, sample_rate):
    """
    Saves raw mono 16-bit PCM bytes to a standard WAV file.
    """
    try:
        with wave.open(file_path, 'wb') as wav_file:
            wav_file.setnchannels(1)  # Mono
            wav_file.setsampwidth(2)  # 16-bit (2 bytes)
            wav_file.setframerate(sample_rate)
            wav_file.writeframes(pcm_bytes)
        print(f"Successfully saved WAV: {file_path}")
    except Exception as e:
        print(f"Error saving WAV file {file_path}: {e}")


def main():
    # Load .env file
    load_dotenv(override=True)

    # Parse arguments
    parser = argparse.ArgumentParser(
        description="Text-to-Speech using ElevenLabs Eleven v3 model and local audio enhancement."
    )
    
    # Input options
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("-t", "--text", type=str, help="Text to convert to speech.")
    group.add_argument("-f", "--text-file", type=str, help="File path to read text from.")
    
    # Output options
    parser.add_argument("-o", "--output", type=str, default="output_enhanced.wav", 
                        help="Output path for the enhanced WAV file.")
    parser.add_argument("-r", "--output-raw", type=str, default="output_raw.wav",
                        help="Output path for the raw unenhanced WAV file.")
    parser.add_argument("--skip-enhance", action="store_true", 
                        help="Skip the local audio enhancement pipeline.")
    
    # ElevenLabs API Configs
    parser.add_argument("--api-key", type=str, help="ElevenLabs API Key (overrides env var).")
    
    # ElevenLabs Voice Settings
    # Default settings configured to sound Dominant & Firm (High stability & similarity boost)
    parser.add_argument("--stability", type=float, default=0.75,
                        help="Voice stability (0.0 to 1.0). Default is 0.75 (Firm/Steady).")
    parser.add_argument("--similarity", type=float, default=0.85,
                        help="Voice similarity boost (0.0 to 1.0). Default is 0.85.")
    parser.add_argument("--style", type=float, default=0.0,
                        help="Voice style exaggeration (0.0 to 1.0). Default is 0.0.")
    parser.add_argument("--disable-speaker-boost", action="store_true",
                        help="Disable speaker similarity boost.")

    # Local DSP settings
    parser.add_argument("--warmth-gain", type=float, default=3.5,
                        help="Warmth boost at 150 Hz in dB. Default is 3.5 dB.")
    parser.add_argument("--clarity-gain", type=float, default=2.5,
                        help="Clarity/Presence boost at 3.2 kHz in dB. Default is 2.5 dB.")
    parser.add_argument("--threshold", type=float, default=-18.0,
                        help="Compressor threshold in dB. Default is -18.0 dB.")
    parser.add_argument("--ratio", type=float, default=3.0,
                        help="Compressor ratio. Default is 3.0.")

    args = parser.parse_args()

    # Get API key
    api_key = args.api_key or os.getenv("ELEVENLABS_API_KEY")
    if not api_key:
        print("Error: ElevenLabs API Key is missing.")
        print("Please configure it in a '.env' file as ELEVENLABS_API_KEY=your_key or pass it using --api-key.")
        sys.exit(1)

    # Read input text
    if args.text_file:
        try:
            with open(args.text_file, 'r', encoding='utf-8') as f:
                text_to_speak = f.read().strip()
        except Exception as e:
            print(f"Error reading file {args.text_file}: {e}")
            sys.exit(1)
    else:
        text_to_speak = args.text.strip()

    if not text_to_speak:
        print("Error: Input text is empty.")
        sys.exit(1)

    # Automatically optimize settings for expression if emotional tags are detected in square brackets
    import re
    has_tags = bool(re.search(r'\[[a-zA-Z\s]+\]', text_to_speak))
    if has_tags:
        adjusted_params = []
        if args.stability == 0.75:
            args.stability = 0.35
            adjusted_params.append("stability=0.35")
        if args.style == 0.0:
            args.style = 0.25
            adjusted_params.append("style=0.25")
        if adjusted_params:
            print(f"Notice: Detected emotional tags in text. Automatically optimized voice settings for expressiveness: {', '.join(adjusted_params)}")

    # Initialize ElevenLabs Client
    client = ElevenLabs(api_key=api_key)

    # Define formats to try (highest quality first)
    # Output formats that are PCM (mono, 16-bit)
    pcm_formats = [
        {"format": "pcm_44100", "rate": 44100},
        {"format": "pcm_24000", "rate": 24000},
        {"format": "pcm_22050", "rate": 22050},
        {"format": "pcm_16000", "rate": 16000}
    ]

    raw_audio_bytes = None
    selected_format = None
    selected_rate = None

    print(f"Synthesizing speech using ElevenLabs (Model: eleven_v3, Voice: Adam)...")
    print(f"Voice parameters: Stability={args.stability}, Similarity={args.similarity}, Style={args.style}, SpeakerBoost={not args.disable_speaker_boost}")

    # Try requested output formats in order of quality
    for fmt_info in pcm_formats:
        fmt = fmt_info["format"]
        rate = fmt_info["rate"]
        try:
            print(f"Attempting API request with format: {fmt} ({rate}Hz)...")
            
            # Generate speech
            response_generator = client.text_to_speech.convert(
                text=text_to_speak,
                voice_id=ADAM_VOICE_ID,
                model_id="eleven_v3",
                output_format=fmt,
                voice_settings=VoiceSettings(
                    stability=args.stability,
                    similarity_boost=args.similarity,
                    style=args.style,
                    use_speaker_boost=not args.disable_speaker_boost
                )
            )

            # Consume the stream generator to get raw bytes
            audio_bytes_list = []
            for chunk in response_generator:
                if chunk:
                    audio_bytes_list.append(chunk)
            
            raw_audio_bytes = b"".join(audio_bytes_list)
            selected_format = fmt
            selected_rate = rate
            print(f"Synthesis successful using format: {fmt}")
            break
        except Exception as e:
            print(f"Format {fmt} failed or is unsupported: {e}")
            print("Retrying with next format...")

    if not raw_audio_bytes:
        print("\nError: All ElevenLabs PCM API attempts failed. Please verify:")
        print("1. Your API key is correct and valid.")
        print("2. Your ElevenLabs subscription plan supports the requested model (eleven_v3).")
        sys.exit(1)

    # Save raw WAV file
    print("\nSaving raw (unenhanced) audio...")
    save_wav(raw_audio_bytes, args.output_raw, selected_rate)

    # Apply DSP enhancement
    if not args.skip_enhance:
        try:
            enhanced_bytes = enhance_audio(
                pcm_bytes=raw_audio_bytes,
                sample_rate=selected_rate,
                warmth_gain=args.warmth_gain,
                clarity_gain=args.clarity_gain,
                threshold_db=args.threshold,
                ratio=args.ratio
            )
            print("Saving enhanced audio...")
            save_wav(enhanced_bytes, args.output, selected_rate)
        except Exception as e:
            print(f"Error during audio enhancement DSP pipeline: {e}")
            print("WAV file enhancement skipped, raw WAV file is still available.")
    else:
        print("Audio enhancement skipped by user request.")

    print("\nProcess completed successfully!")

if __name__ == "__main__":
    main()
