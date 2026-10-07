import os
import argparse
import yt_dlp

def download_youtube_video(url: str, output_path: str = "input_video.mp4") -> str:
    """
    YouTube 영상을 cookies.txt 기반으로 안전하게 다운로드합니다.
    """
    ydl_opts = {
        'format': 'bestvideo[ext=mp4]+bestaudio[ext=m4a]/best[ext=mp4]/best',
        'outtmpl': output_path,
        'cookiefile': 'cookies.txt',  # 비밀 변수로 생성된 쿠키 파일 적용
        'quiet': False,
        'no_warnings': True,
    }

    print(f"[*] Downloading video from: {url}")
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        ydl.download([url])
    
    print(f"[+] Download completed: {output_path}")
    return output_path
