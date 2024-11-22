# Telegram-Video-Circle-Bot
Этот проект представляет собой открытый код бота для Telegram, который преобразует видео, отправленное пользователем, в видеокружок (круглое видео).

![alt text](https://leonardo.osnova.io/7506e765-ca5c-53b2-abfa-d1f27f0e4102/-/preview/2100/-/format/webp/)

## Установка
1. Клонируйте репозиторий:
``` git clone https://github.com/Dizro/Telegram-Video-Circle-Bot.git ```

2. Перейдите в директорию проекта:
``` cd Telegram-Video-Circle-Bot ```

3. Установите необходимые зависимости:
``` pip install -r requirements.txt ```

## Настройка
* Откройте файл config.py и замените API_TOKEN на ваш токен API от BotFather.

## Запуск

1. Запустите бота:
``` python main.py ```

## Использование
1. Откройте Telegram и найдите вашего бота по имени, которое вы указали при создании бота в BotFather.
   
3. Отправьте команду `/start` боту.
   
5. Отправьте видео, которое вы хотите преобразовать в кружок.
   
7. Бот скачает видео, преобразует его в видеокружок и отправит обратно вам.

## Technical Details

### Performance Optimizations
- Chunk-based video processing with parallel execution
- Efficient memory management with frame buffering
- OpenCV-based frame processing for optimal performance
- Asynchronous operations for better responsiveness

### Features
- Supports videos of various formats and aspect ratios
- Automatic video resizing and circular masking
- Progress tracking during processing
- File size validation
- Proper error handling and user feedback
- Temporary file cleanup

### Configuration
The bot uses several configuration dictionaries that can be easily modified:
- `PATH_CONFIG`: File and directory paths
- `VIDEO_CONFIG`: Video processing parameters
- `FFMPEG_CONFIG`: Video encoding settings
- `MESSAGE_CONFIG`: User interface messages

### System Requirements
- Python 3.7+
- OpenCV (cv2)
- FFmpeg
- Telegram Bot API library
- Sufficient disk space for temporary files
- Memory according to video size being processed

### Error Handling
The bot includes comprehensive error handling for:
- File size limitations
- Processing errors
- Network timeouts
- Resource cleanup
- Invalid video formats

### Logging
Implemented detailed logging system for:
- Processing progress

- Error tracking
- System diagnostics
- Performance monitoring

Лицензия
Этот проект лицензирован под MIT License - подробности смотрите в файле `LICENSE`.

Контакты

Если у вас есть вопросы или предложения пишите на почту указанную в профиле.
