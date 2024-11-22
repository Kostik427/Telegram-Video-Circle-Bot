import cv2
import numpy as np
from telegram import Update
from telegram.ext import CallbackContext
import os
import asyncio
from concurrent.futures import ThreadPoolExecutor
from typing import Tuple, List
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

PATH_CONFIG = {
    "rounds_dir": "rounds",
    "output_filename": "output_video.mp4",
    "final_output_filename": "final_output.mp4",
    "temp_frames_dir": "temp_frames",
}

VIDEO_CONFIG = {
    "circle_size": 360,
    "fourcc": "mp4v",
    "preset": "fast",
    "bitrate": "5M",
    "chunk_size": 30,
    "max_workers": 4,
}

FFMPEG_CONFIG = {
    "video_codec": "libx264",
    "audio_codec": "aac",
}

MESSAGE_CONFIG = {
    "start_message": "😛Отправьте мне видео, и я преобразую его в видеокружок.",
    "error_message": "😛Произошла ошибка при обработке видео: {}",
    "processing_message": "😛Обработка видео началась...",
    "success_message": "😛Видео успешно обработано!",
}


def create_circular_mask(size: int) -> np.ndarray:
    mask = np.zeros((size, size), dtype=np.uint8)
    center = (size // 2, size // 2)
    cv2.circle(mask, center, size // 2, 255, -1)
    return mask


def process_frame(
    frame_data: Tuple[np.ndarray, int, int, int]
) -> Tuple[np.ndarray, int]:
    frame, width, height, circle_size = frame_data
    aspect_ratio = width / height

    if width > height:
        new_w = int(circle_size * aspect_ratio)
        new_h = circle_size
    else:
        new_w = circle_size
        new_h = int(circle_size / aspect_ratio)

    try:
        resized = cv2.resize(frame, (new_w, new_h))
        y_start = (new_h - circle_size) // 2
        x_start = (new_w - circle_size) // 2
        cropped = resized[
            y_start : y_start + circle_size, x_start : x_start + circle_size
        ]

        mask = create_circular_mask(circle_size)
        result = cv2.bitwise_and(cropped, cropped, mask=mask)

        return result, 1
    except Exception as e:
        logger.error(f"Ошибка при обработке кадра: {str(e)}")
        return None, 0


async def process_frame_chunk(
    frames: List[np.ndarray],
    width: int,
    height: int,
    circle_size: int,
    executor: ThreadPoolExecutor,
) -> List[np.ndarray]:
    frame_data = [(frame, width, height, circle_size) for frame in frames]
    loop = asyncio.get_event_loop()

    results = await loop.run_in_executor(
        executor, lambda: list(map(process_frame, frame_data))
    )

    return [frame for frame, success in results if success]


async def process_video(update: Update, context: CallbackContext):
    cap = None
    out = None
    temp_dir = None

    try:
        status_message = await update.message.reply_text(
            MESSAGE_CONFIG["processing_message"]
        )

        video_file = await context.bot.getFile(update.message.video.file_id)
        chat_id = update.message.chat_id

        round_path = os.path.join(PATH_CONFIG["rounds_dir"], str(chat_id))
        temp_dir = os.path.join(round_path, PATH_CONFIG["temp_frames_dir"])
        os.makedirs(round_path, exist_ok=True)
        os.makedirs(temp_dir, exist_ok=True)

        file_name = (
            update.message.video.file_name or f"{update.message.video.file_id}.mp4"
        )
        input_path = os.path.join(round_path, file_name)
        output_path = os.path.join(round_path, PATH_CONFIG["output_filename"])
        final_output_path = os.path.join(
            round_path, PATH_CONFIG["final_output_filename"]
        )

        await video_file.download_to_drive(input_path)

        cap = cv2.VideoCapture(input_path)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        circle_size = VIDEO_CONFIG["circle_size"]
        fourcc = cv2.VideoWriter_fourcc(*VIDEO_CONFIG["fourcc"])
        out = cv2.VideoWriter(output_path, fourcc, fps, (circle_size, circle_size))

        with ThreadPoolExecutor(max_workers=VIDEO_CONFIG["max_workers"]) as executor:
            frames_buffer = []
            processed_frames = 0

            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break

                frames_buffer.append(frame)
                processed_frames += 1

                if len(frames_buffer) >= VIDEO_CONFIG["chunk_size"]:
                    processed_chunk = await process_frame_chunk(
                        frames_buffer, width, height, circle_size, executor
                    )

                    for processed_frame in processed_chunk:
                        if processed_frame is not None:
                            out.write(processed_frame)

                    frames_buffer = []

                    progress = int((processed_frames / frame_count) * 100)
                    await status_message.edit_text(f"Обработано {progress}% видео...")

            if frames_buffer:
                processed_chunk = await process_frame_chunk(
                    frames_buffer, width, height, circle_size, executor
                )
                for processed_frame in processed_chunk:
                    if processed_frame is not None:
                        out.write(processed_frame)

        cap.release()
        out.release()

        ffmpeg_command = [
            "ffmpeg",
            "-y",
            "-i",
            output_path,
            "-i",
            input_path,
            "-map",
            "0:v:0",
            "-map",
            "1:a:0",
            "-c:v",
            FFMPEG_CONFIG["video_codec"],
            "-preset",
            VIDEO_CONFIG["preset"],
            "-b:v",
            VIDEO_CONFIG["bitrate"],
            "-c:a",
            FFMPEG_CONFIG["audio_codec"],
            "-shortest",
            final_output_path,
        ]

        process = await asyncio.create_subprocess_exec(
            *ffmpeg_command,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        await process.communicate()

        duration = frame_count / fps
        await status_message.edit_text(MESSAGE_CONFIG["success_message"])

        try:
            file_size = os.path.getsize(final_output_path)
            if file_size > 50 * 1024 * 1024:
                await update.message.reply_text(
                    "Видео слишком большое для отправки. Максимальный размер: 50MB"
                )
                return

            with open(final_output_path, "rb") as video:
                await asyncio.wait_for(
                    context.bot.send_video_note(
                        chat_id=update.message.chat_id,
                        video_note=video,
                        duration=int(duration),
                        length=circle_size,
                        read_timeout=30,
                        write_timeout=30,
                        connect_timeout=30,
                        pool_timeout=30,
                    ),
                    timeout=60,
                )
        except asyncio.TimeoutError:
            await update.message.reply_text(
                "Превышено время ожидания при отправке видео. Попробуйте еще раз или уменьшите размер видео."
            )
        except Exception as e:
            logger.error(f"Ошибка при отправке видео: {str(e)}")
            await update.message.reply_text(f"Ошибка при отправке видео: {str(e)}")

    except Exception as e:
        logger.error(f"Ошибка при обработке видео: {str(e)}")
        await update.message.reply_text(MESSAGE_CONFIG["error_message"].format(str(e)))

    finally:
        if cap is not None:
            cap.release()
        if out is not None:
            out.release()
        if temp_dir and os.path.exists(temp_dir):
            try:
                for file in os.listdir(temp_dir):
                    os.remove(os.path.join(temp_dir, file))
                os.rmdir(temp_dir)
            except Exception as e:
                logger.error(f"Ошибка при очистке временных файлов: {str(e)}")


async def start(update: Update, context: CallbackContext):
    await update.message.reply_text(MESSAGE_CONFIG["start_message"])
