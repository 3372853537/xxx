import requests
import json
from bs4 import BeautifulSoup


# 获取豆瓣100部经典电影信息
def get_douban_movies():
    base_url = "https://movie.douban.com/top250"  # 豆瓣Top 250页面
    movie_list = []

    for page in range(0, 100, 25):  # 每页25个电影，爬取4页
        url = f"{base_url}?start={page}&filter="

        # 增加请求头，模拟真实的浏览器行为
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
            "Accept-Language": "zh-CN,zh;q=0.9,en-US;q=0.8,en;q=0.7",
            "Accept-Encoding": "gzip, deflate, br",
            "Connection": "keep-alive",
            "Upgrade-Insecure-Requests": "1",
            "Referer": "https://movie.douban.com/",
            "Host": "movie.douban.com",
            "Cookie": "__utma=30149280.475061059.1741570556.1741662840.1741664240.7; __utmb=30149280.6.10.1741664240; __utmc=30149280; __utmt=1; __utmv=30149280.28765; __utmz=30149280.1741664240.7.4.utmcsr=bing|utmccn=(organic)|utmcmd=organic|utmctr=(not%20provided); __yadk_uid=SrBCzAX80iAM68PGLFHV705Y4txUTWrR; _pk_id.100001.8cb4=d025034f23e461c2.1741570553.; _pk_ref.100001.8cb4=%5B%22%22%2C%22%22%2C1741664240%2C%22https%3A%2F%2Fwww.bing.com%2F%22%5D; _pk_ses.100001.8cb4=1; _vwo_uuid_v2=DFD17C60FEE86B8DAD912F72316FFEE13|fba001d66dbb25d38f1c760b7a8227d4; ap_v=0,6.0; bid=3CptwmIm83Y; ck=UnAW; dbcl2=\"287659272:LKclQwq8vtQ\"; frodotk_db=\"b396c4efff1165001ac44ecffd342f00\"; ll=\"118254\"; push_doumail_num=0; push_noty_num=0" # 在这里填入你从浏览器复制的 cookie 信息  # 请将你的cookie粘贴在这里
        }

        response = requests.get(url, headers=headers)
        response.encoding = "utf-8"

        if response.status_code != 200:
            print(f"Failed to retrieve the webpage. Status code: {response.status_code}")
            return []

        soup = BeautifulSoup(response.text, "html.parser")

        # 找到电影列表
        movie_items = soup.find_all("div", class_="item")
        for movie in movie_items:
            title = movie.find("span", class_="title").text.strip()

            # 获取电影上映时间
            release_time_tag = movie.find("span", class_="year")
            release_time = release_time_tag.text.strip("()") if release_time_tag else "未知"

            # 获取电影评分
            rating_tag = movie.find("span", class_="rating_num")
            rating = rating_tag.text.strip() if rating_tag else "未知"

            # 获取电影海报链接
            poster = movie.find("img")["src"]

            # 获取电影详情链接
            movie_link = movie.find("a")["href"]

            # 构造电影基本信息
            movie_data = {
                "id": "",  # 豆瓣Top 250电影没有直接的ID字段
                "title": title,
                "releaseTime": release_time,
                "whis_count": rating,
                "poster": poster,
                "plot": "",
                "directors": [],
                "actors": []
            }

            # 获取电影的详细信息（导演、演员、简介等）
            movie_details = get_movie_details(movie_link, headers)
            movie_data.update(movie_details)

            # 添加电影到列表
            movie_list.append(movie_data)

    return movie_list


# 获取电影详细信息
def get_movie_details(url, headers):
    response = requests.get(url, headers=headers)
    response.encoding = "utf-8"

    if response.status_code != 200:
        return {}

    soup = BeautifulSoup(response.text, "html.parser")

    # 获取电影海报（可能为空）
    image_url = soup.find("img", class_="")["src"] if soup.find("img", class_="") else ""

    # 获取电影简介
    summary = soup.find("span", property="v:summary")
    plot = summary.text.strip() if summary else "暂无简介"

    # 获取导演
    directors = [a.text.strip() for a in soup.find_all("a", rel="v:directedBy")]

    # 获取演员
    actors = [{"name": a.text.strip(), "role": "未知"} for a in soup.find_all("a", rel="v:starring")]

    return {
        "poster": image_url,
        "plot": plot,
        "directors": directors,
        "actors": actors
    }


# 保存数据到JSON文件
def save_to_json(data, filename="douban_movies_top_100.json"):
    with open(filename, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=4)
    print(f"Data successfully saved to {filename}")


# 主函数
if __name__ == "__main__":
    movies = get_douban_movies()
    if movies:
        save_to_json(movies)
