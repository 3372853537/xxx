import requests
import json
import time
from bs4 import BeautifulSoup


# 获取豆瓣电影信息
def get_douban_movies():
    url = "https://movie.douban.com/coming"

    # 增加请求头，模拟真实的浏览器行为
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,image/apng,*/*;q=0.8",
        "Accept-Language": "zh-CN,zh;q=0.9,en-US;q=0.8,en;q=0.7",
        "Accept-Encoding": "gzip, deflate, br",
        "Connection": "keep-alive",
        "Upgrade-Insecure-Requests": "1",
        "Referer": "https://movie.douban.com/",
        "Host": "movie.douban.com",
        "Dnt": "1",  # 防止追踪
        "Cache-Control": "max-age=0",
        "TE": "Trailers",
        "Cookie": "__utma=30149280.475061059.1741570556.1741662840.1741664240.7; __utmb=30149280.6.10.1741664240; __utmc=30149280; __utmt=1; __utmv=30149280.28765; __utmz=30149280.1741664240.7.4.utmcsr=bing|utmccn=(organic)|utmcmd=organic|utmctr=(not%20provided); __yadk_uid=SrBCzAX80iAM68PGLFHV705Y4txUTWrR; _pk_id.100001.8cb4=d025034f23e461c2.1741570553.; _pk_ref.100001.8cb4=%5B%22%22%2C%22%22%2C1741664240%2C%22https%3A%2F%2Fwww.bing.com%2F%22%5D; _pk_ses.100001.8cb4=1; _vwo_uuid_v2=DFD17C60FEE86B8DAD912F72316FFEE13|fba001d66dbb25d38f1c760b7a8227d4; ap_v=0,6.0; bid=3CptwmIm83Y; ck=UnAW; dbcl2=\"287659272:LKclQwq8vtQ\"; frodotk_db=\"b396c4efff1165001ac44ecffd342f00\"; ll=\"118254\"; push_doumail_num=0; push_noty_num=0" # 在这里填入你从浏览器复制的 cookie 信息
    }

    # 使用 session 保持会话，自动处理 cookies
    session = requests.Session()
    session.headers.update(headers)

    # 发送请求
    response = session.get(url)
    response.encoding = "utf-8"

    # 如果请求失败（状态码不是 200），打印状态码
    if response.status_code != 200:
        print(f"Failed to retrieve the webpage. Status code: {response.status_code}")
        return []

    soup = BeautifulSoup(response.text, "html.parser")
    movie_list = []

    # 找到电影列表所在的表格
    table = soup.find("table", class_="coming_list")
    if table:
        for row in table.find_all("tr")[1:]:  # 跳过表头行
            cols = row.find_all("td")
            movie_link = cols[1].find("a")["href"] if cols[1].find("a") else ""
            movie_data = {
                "id": "",
                "title": cols[1].text.strip(),
                "releaseTime": cols[0].text.strip(),
                "whis_count": cols[4].text.strip(),
                "poster": "",
                "plot": "",
                "directors": [],
                "actors": []
            }

            # 获取电影详情
            if movie_link:
                movie_details = get_movie_details(movie_link, session)
                movie_data.update(movie_details)

            movie_list.append(movie_data)

            # 等待 1 秒钟，避免被服务器识别为过于频繁的请求
            time.sleep(1)

    return movie_list


# 获取电影详情
def get_movie_details(url, session):
    response = session.get(url)
    response.encoding = "utf-8"

    if response.status_code != 200:
        return {}

    soup = BeautifulSoup(response.text, "html.parser")

    image_url = soup.find("img", class_="")
    summary = soup.find("span", property="v:summary")
    director = [a.text.strip() for a in soup.find_all("a", rel="v:directedBy")]
    actors = [{"name": a.text.strip(), "role": "未知"} for a in soup.find_all("a", rel="v:starring")]

    return {
        "poster": image_url["src"] if image_url else "",
        "plot": summary.text.strip() if summary else "",
        "directors": director,
        "actors": actors
    }


# 保存数据到 JSON 文件
def save_to_json(data, filename="douban_movies.json"):
    with open(filename, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=4)
    print(f"Data successfully saved to {filename}")


# 主函数
if __name__ == "__main__":
    movies = get_douban_movies()
    if movies:
        save_to_json(movies)
