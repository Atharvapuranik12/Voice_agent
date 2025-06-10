# I have used the gemini-1.5-flash. You can use any model of your choice.
import asyncio
import os
import time
import requests
import threading
import io
from dotenv import load_dotenv
from livekit import rtc
from livekit.agents import AgentCallbacks, JobContext, Worker, WorkerOptions, JobType
from livekit.agents.utils import AudioBuffer
import google.generativeai as genai
import whisper
import numpy as np
import soundfile as sf

load_dotenv()

GEMINI_API_KEY = os.getenv("Gemini-API")
CARTESIA_API_KEY = os.getenv("CARTESIA_API_KEY")
LIVEKIT_URL = os.getenv("LIVEKIT_URL")
LIVEKIT_API_KEY = os.getenv("LIVEKIT-API-KEY")
LIVEKIT_API_SECRET = os.getenv("LIVEKIT_API-SECRET")

GEMINI_MODEL_NAME = "gemini-1.5-flash"
WHISPER_MODEL_SIZE = "base"
CARTESIA_TTS_URL = "https://api.cartesia.ai/tts/bytes"

gemini_model = None
if GEMINI_API_KEY:
    genai.configure(api_key=GEMINI_API_KEY)
    gemini_model = genai.GenerativeModel(GEMINI_MODEL_NAME)
    print(f"Gemini model '{GEMINI_MODEL_NAME}' initialized.")
else:
    print("Warning: GEMINI_API_KEY not found.")

if not CARTESIA_API_KEY:
    print("Warning: CARTESIA_API_KEY not found.")

whisper_model = None
print(f"Loading Whisper model...")
try:
    whisper_model = whisper.load_model(WHISPER_MODEL_SIZE, device="cpu", download_root="models", in_memory=True)
    print("Whisper model loaded.")
except Exception as e:
    print(f"Error loading Whisper: {e}")
    whisper_model = None


class VoiceAgentSession(AgentCallbacks):
    def __init__(self, room):
        self.room = room
        self.chat_history = []
        self.interrupted = False
        self.response_queue = asyncio.Queue()
        self.speaking_task = None
        self.stt_processing_task = None
        self.audio_input_buffer = AudioBuffer(frame_rate=16000, num_channels=1)

    async def _process_stt_audio(self):
        if not whisper_model:
            print("Whisper model not loaded.")
            return

        while True:
            await asyncio.sleep(0.2)
            if self.audio_input_buffer.samples_per_channel == 0:
                continue

            print("Processing audio...")
            audio_data_int16 = self.audio_input_buffer.to_numpy()
            audio_data_float32 = audio_data_int16.astype(np.float32) / 32768.0
            temp_audio_file = f"temp_audio_{self.room.sid}_{int(time.time() * 1000)}.wav"

            try:
                sf.write(temp_audio_file, audio_data_float32, self.audio_input_buffer.frame_rate)
                result = await asyncio.to_thread(whisper_model.transcribe, temp_audio_file, fp16=False)
                text = result["text"].strip()

                if text:
                    print(f"User said: '{text}'")
                    asyncio.create_task(self._process_llm_request(text))
                else:
                    print("No text detected.")

                self.audio_input_buffer.clear()
            except Exception as e:
                print(f"Error during transcription: {e}")
                self.audio_input_buffer.clear()
            finally:
                if os.path.exists(temp_audio_file):
                    os.remove(temp_audio_file)

    async def _process_llm_request(self, user_utterance):
        if not gemini_model:
            print("Gemini not initialized.")
            return

        print(f"Processing: '{user_utterance}'")
        self.interrupted = False
        self.chat_history.append({"role": "user", "parts": [user_utterance]})

        agent_response_text = ""
        try:
            response_stream = gemini_model.generate_content(
                self.chat_history,
                stream=True,
                generation_config=genai.types.GenerationConfig(
                    temperature=0.7,
                    max_output_tokens=200,
                )
            )

            async for chunk in response_stream:
                if self.interrupted:
                    print("LLM interrupted.")
                    break

                if chunk.text:
                    agent_response_text += chunk.text
                    await self.response_queue.put(chunk.text)

        except Exception as e:
            print(f"Error during LLM: {e}")
            error_message = "Sorry, I'm having trouble right now."
            await self.response_queue.put(error_message)
            agent_response_text = error_message
        finally:
            self.chat_history.append({"role": "model", "parts": [agent_response_text]})
            await self.response_queue.put(None)
            print(f"LLM finished: '{agent_response_text[:100]}...'")

    async def _speak_response(self):
        if not CARTESIA_API_KEY:
            print("Cartesia API key not set.")
            return
        if not CARTESIA_TTS_URL:
            print("Cartesia URL not set.")
            return

        while True:
            text_chunk = await self.response_queue.get()

            if text_chunk is None:
                print("End of response.")
                self.response_queue.task_done()
                continue

            if self.interrupted:
                print("TTS interrupted.")
                self.response_queue.task_done()
                continue

            print(f"Generating speech for: '{text_chunk[:50]}...'")
            try:
                headers = {
                    "Authorization": f"Bearer {CARTESIA_API_KEY}",
                    "Content-Type": "application/json",
                    "Cartesia-Version": "2025-04-16"
                }
                payload = {
                    "model_id": "sonic-2",
                    "transcript": text_chunk,
                    "voice": {
                        "mode": "id",
                        "id": "694f9389-aac1-45b6-b726-9d9369183238"
                    },
                    "output_format": {
                        "container": "mp3",
                        "bit_rate": 128000,
                        "sample_rate": 44100
                    },
                    "language": "en"
                }

                response = await asyncio.to_thread(
                    requests.post, CARTESIA_TTS_URL, headers=headers, json=payload
                )
                response.raise_for_status()

                audio_content = response.content
                audio_io = io.BytesIO(audio_content)
                audio_np, sample_rate = sf.read(audio_io, dtype='int16')

                if audio_np.ndim == 1:
                    audio_np = audio_np.reshape(-1, 1)
                num_channels = audio_np.shape[1]

                frame_duration_seconds = 0.020
                frame_samples = int(sample_rate * frame_duration_seconds)

                for i in range(0, len(audio_np), frame_samples):
                    if self.interrupted:
                        print("TTS streaming interrupted.")
                        break

                    chunk_to_send = audio_np[i: i + frame_samples].flatten()

                    frame = rtc.AudioFrame(
                        data=chunk_to_send,
                        sample_rate=sample_rate,
                        num_channels=num_channels,
                        samples_per_channel=len(chunk_to_send) // num_channels
                    )
                    await self.room.local_participant.publish_audio_track.write_frame(frame)

                self.response_queue.task_done()
            except requests.exceptions.RequestException as e:
                print(f"Error during TTS: {e}")
                if response.status_code:
                    print(f"Status Code: {response.status_code}")
                    print(f"Response: {response.text}")
                self.response_queue.task_done()
            except Exception as e:
                print(f"Error during TTS: {e}")
                self.response_queue.task_done()
            continue

    async def on_connected(self):
        print(f"Connected to room: {self.room.name}")
        await self.room.local_participant.publish_audio_track()
        self.speaking_task = asyncio.create_task(self._speak_response())
        self.stt_processing_task = asyncio.create_task(self._process_stt_audio())
        print("Tasks started.")

    async def on_user_speaking(self, speaking):
        if speaking:
            print("User started speaking.")
            self.interrupted = True

            while not self.response_queue.empty():
                try:
                    self.response_queue.get_nowait()
                    self.response_queue.task_done()
                except asyncio.QueueEmpty:
                    pass
            self.audio_input_buffer.clear()
            print("Response interrupted, buffer cleared.")
        else:
            print("User stopped speaking.")

    async def on_audio_frame(self, participant, track, frame):
        if participant.identity != self.room.local_participant.identity:
            if self.interrupted and self.audio_input_buffer.samples_per_channel > 0:
                self.audio_input_buffer.clear()
            self.audio_input_buffer.write_frame(frame)

    async def on_closed(self):
        print("Session closed.")
        if self.speaking_task:
            self.speaking_task.cancel()
        if self.stt_processing_task:
            self.stt_processing_task.cancel()

        try:
            await asyncio.gather(self.speaking_task, self.stt_processing_task, return_exceptions=True)
        except asyncio.CancelledError:
            pass
        print("Cleanup complete.")


async def start_agent_worker():
    print("Starting worker...")
    worker = Worker(
        connect_opts=rtc.ConnectOptions(
            api_key=LIVEKIT_API_KEY,
            api_secret=LIVEKIT_API_SECRET,
            auto_subscribe=True,
            simulate_speakers=True,
        ),
        worker_options=WorkerOptions(
            url=LIVEKIT_URL,
            rtc_config=rtc.RTCConfiguration(),
        ),
    )

    @worker.agent_handler(JobType.ROOM)
    async def handle_room_job(ctx):
        print(f"Got room job: {ctx.room.name}")
        session = VoiceAgentSession(ctx.room)
        await ctx.room.run(session)
        print(f"Room job completed.")

    print("Worker started.")
    await worker.run()


if __name__ == "__main__":
    if not all([LIVEKIT_URL, LIVEKIT_API_KEY, LIVEKIT_API_SECRET, GEMINI_API_KEY, CARTESIA_API_KEY]):
        print("ERROR: Please set all environment variables in .env file")
        exit(1)

    try:
        asyncio.run(start_agent_worker())
    except KeyboardInterrupt:
        print("Stopped by user.")
    except Exception as e:
        print(f"Error: {e}")
