import requests
import json
import os
import time
from bs4 import BeautifulSoup

# 使用提供的 cookie 字典
COOKIES = {
    "__utma": "223695111.1514497507.1741570557.1741662840.1741773703.4",
    "__utmb": "223695111.0.10.1741773703",
    "__utmc": "223695111",
    "__utmv": "30149280.28765",
    "__utmz": "30149280.1741773699.8.5.utmcsr=bing|utmccn=(organic)|utmcmd=organic|utmctr=(not%20provided)",
    "__yadk_uid": "XqdPW7ff2nYPW7G2GODFD2biLvEngZeo",
    "_pk_id.100001.4cf6": "833d49dba14bf72f.1741570557.",
    "_pk_ref.100001.4cf6": "%5B%22%22%2C%22%22%2C1741773703%2C%22https%3A%2F%2Fwww.douban.com%2F%22%5D",
    "_pk_ses.100001.4cf6": "1",
    "_vwo_uuid_v2": "DFD17C60FEE86B8DAD912F72316FFEE13|fba001d66dbb25d38f1c760b7a8227d4",
    "bid": "3CptwmIm83Y",
    "ck": "UnAW",
    "dbcl2": "\"287659272:LKclQwq8vtQ\"",
    "frodotk_db": "\"da7ae835ee6cc9b38f70b8a1cb465d19\"",
    "ll": "\"118254\"",
    "push_doumail_num": "0",
    "push_noty_num": "0"
}

# 固定的 User-Agent
USER_AGENT = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"

# 读取 JSON 文件，获取电影名称
def load_movie_titles(filename="douban_movies.json"):
    with open(filename, "r", encoding="utf-8") as f:
        movies = json.load(f)
    return [movie['title'] for movie in movies]

# 获取豆瓣即将上映的电影链接列表
def get_coming_movies(session):
    url = "https://movie.douban.com/coming"
    headers = {
        "User-Agent": USER_AGENT,
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,image/apng,*/*;q=0.8",
        "Accept-Encoding": "gzip, deflate, br",
        "Connection": "keep-alive",
        "Referer": "https://movie.douban.com/",
        "Host": "movie.douban.com"
    }

    response = session.get(url, headers=headers, cookies=COOKIES)
    response.encoding = "utf-8"

    if response.status_code != 200:
        print("Failed to retrieve the coming movies page.")
        return []

    soup = BeautifulSoup(response.text, "html.parser")
    movie_links = []

    # 获取所有即将上映的电影链接
    for movie in soup.find_all("tr")[1:]:  # 跳过表头
        movie_info = movie.find_all("td")
        if len(movie_info) > 1:
            movie_title = movie_info[1].text.strip()
            movie_link = movie_info[1].find("a")["href"] if movie_info[1].find("a") else ""
            if movie_link:
                movie_links.append((movie_title, movie_link))

    return movie_links

# 获取电影详情页中的演职人员图片 URL
def get_movie_cast_images(movie_url, session):
    headers = {
        "User-Agent": USER_AGENT,
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,image/apng,*/*;q=0.8",
        "Accept-Encoding": "gzip, deflate, br",
        "Connection": "keep-alive",
        "Referer": movie_url,
        "Host": "movie.douban.com"
    }

    response = session.get(movie_url, headers=headers, cookies=COOKIES)
    response.encoding = "utf-8"

    if response.status_code != 200:
        print(f"Failed to retrieve movie details: {movie_url}")
        return []

    soup = BeautifulSoup(response.text, "html.parser")

    # 获取导演和演员
    directors = soup.find_all("a", rel="v:directedBy")
    actors = soup.find_all("a", rel="v:starring")

    # 输出 HTML 以调试
    if not directors and not actors:
        print(f"Couldn't find directors or actors for {movie_url}")
        print(soup.prettify())  # 打印整个 HTML 页面，查看结构

    cast_images = []

    # 获取导演头像 URL
    for director in directors:
        director_name = director.text.strip()
        director_image = director.find("img")
        if director_image:
            img_url = director_image["src"]
            cast_images.append((director_name, img_url))

    # 获取演员头像 URL
    for actor in actors:
        actor_name = actor.text.strip()
        actor_image = actor.find("img")
        if actor_image:
            img_url = actor_image["src"]
            cast_images.append((actor_name, img_url))

    return cast_images

# 保存演职人员图片 URL 到文本文件
def save_cast_images_to_text(movie_title, cast_images, output_file="movie_cast_images.txt"):
    with open(output_file, "a", encoding="utf-8") as f:
        movie_info = f"{movie_title}: "  # 格式：电影名:
        movie_info += " ".join([f"{name}:{img_url.split('?')[0]}" for name, img_url in cast_images])  # 每个人员用空格隔开
        f.write(movie_info + "\n")
    print(f"Saved cast images for {movie_title}.")

# 模拟正常用户访问间隔
def random_delay():
    time.sleep(2)  # 固定等待时间，避免频繁请求

# 主函数
def main():
    # 创建会话
    session = requests.Session()

    # 获取即将上映电影的链接
    movie_links = get_coming_movies(session)

    # 输出结果到文本文件
    for title, movie_link in movie_links:
        print(f"Processing movie: {title} ({movie_link})")
        # 获取演职人员的头像图片 URL
        cast_images = get_movie_cast_images(movie_link, session)

        if cast_images:
            save_cast_images_to_text(title, cast_images)
        else:
            print(f"No cast images found for {title}")

        # 加入延迟
        random_delay()

if __name__ == "__main__":
    main()
