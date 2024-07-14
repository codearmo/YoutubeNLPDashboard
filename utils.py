

def convert_url_to_video_id(url: str) -> str:
    return url.split('?')[-1][2:]


def video_id_to_url(video_id: str) -> str:
    return f'https://www.youtube.com/watch?v={video_id}'