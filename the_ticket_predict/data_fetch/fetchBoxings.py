#爬取已上映电影每日票房数据，票房占比，排片占比，周几（自行量化），场均人次
#将每日票房数据写入BoxingData.txt文件中
#同时爬取的数据传入数据库  字段可以是Name Day1 Day2 Day3.....   每一条数据都是一部电影上映的每日票房数据
#数据的格式例如： 美国队长 [28556.72,0.784,0.4383,0.5,58] [56556.2,0.734,0.4283,0.2,46] ......
#其中[28556.72,0.784,0.4383,0.5,58]分别对应[每日票房数据，票房占比，排片占比，周几（自行量化），场均人次]
#从https://zgdypf.zgdypw.cn/爬取

#***当前我保存两个文件all_movies.txt为按天数爬取的票房信息，filtered_movies_data.txt是基于前者进行筛选的信息***#

from lxml import etree
import requests
from datetime import datetime, timedelta
import json

# 初始化起始日期(不含当天前30天)
current_date = datetime.now()
date_list = [(current_date - timedelta(days=i)).strftime('%Y-%m-%d') for i in range(1, 31)]
# print(date_list)
all_movies_data = []

for date in date_list:
    singalday_movies_data = []
    url = f"https://zgdypf.zgdypw.cn/getDayData?date={date}&withSvcFee=true"
    headers = {
        "User-Agent": "Mozilla/5.0 (Linux; Android 6.0; Nexus 5 Build/MRA58N) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/133.0.0.0 Mobile Safari/537.36"
    }

    resp = requests.get(url, headers=headers)
    if resp.status_code != 200:
        print(f"请求失败，状态码：{resp.status_code}")
        continue

    try:
        data = resp.json()
    except json.JSONDecodeError:
        print(f"解析 JSON 失败，日期: {date}")
        continue

    movies = data.get('list', [])
    for movie in movies:
        movies_dict = {
            '日期':date,
            '代码': movie.get('code'),
            '片名': movie.get('name'),
            '上映天数': movie.get('releaseDays'),
            '每日票房': movie.get('salesInWanDesc'),
            '每日票房占比': movie.get('salesRateDesc'),
            '每日排片占比': movie.get('sessionRateDesc')
        }
        singalday_movies_data.append(movies_dict)

    all_movies_data.append(singalday_movies_data)

# print(movies_data)
# print(type(all_movies_data))
# print(len( all_movies_data))
# print(type(all_movies_data[0]))
# print(all_movies_data[0])
# print(type(all_movies_data[0][0]))
# print(all_movies_data[0][0])

output_file = 'all_movies.txt'

day =0#30

with open(output_file, 'w', encoding='utf-8') as fobj:
    for movies_list in all_movies_data:
        record_list = [f"{date_list[day]}："]
        for movie_dict in movies_list:
            record_list.append(
                f"日期：{movie_dict['日期']},代码: {movie_dict['代码']}, 片名: {movie_dict['片名']}, 上映天数: {movie_dict['上映天数']}, 每日票房: {movie_dict['每日票房']}, 每日票房占比: {movie_dict['每日票房占比']}, 每日排片占比: {movie_dict['每日排片占比']}"
            )
        day += 1
        fobj.write("\n".join(record_list) + "\n")



# 新的筛选电影名单
filtered_movies = []

# 从第三十天开始筛选符合条件的电影
for day, day_movies_data in enumerate(all_movies_data[0:], start=0):
    for movie in day_movies_data:
        release_days = int(movie['上映天数'])
        movie_code = movie['代码']
        movie_name = movie['片名']

        if release_days >= 30 and release_days < 100:
            # 上映时间大于等于30天，小于100天的电影，直接获取30天的数据
            if movie_code not in [m['代码'] for m in filtered_movies]:
                filtered_movies.append({
                    '代码': movie_code,
                    '片名': movie_name,
                    '上映天数': release_days,
                    'movie_data': []  # 用于存储每一天的数据
                })
        elif release_days > 15 and release_days < 30:
            # 上映时间大于15天，小于30天的电影，复制前15天数据到后15天
            if movie_code not in [m['代码'] for m in filtered_movies]:
                filtered_movies.append({
                    '代码': movie_code,
                    '片名': movie_name,
                    '上映天数': release_days,
                    'movie_data': []  # 用于存储每一天的数据
                })

# 定义一个函数来量化周几（0=周一, 6=周日）
def get_weekday(date_str):
    date_obj = datetime.strptime(date_str, '%Y-%m-%d')
    return date_obj.weekday()  # 0 = Monday, 6 = Sunday

# 汇总筛选电影的数据
for day, day_movies_data in enumerate(all_movies_data[0:], start=0):
    for movie in day_movies_data:
        movie_code = movie['代码']
        for filtered_movie in filtered_movies:
            if filtered_movie['代码'] == movie_code:
                # 获取电影每日数据
                daily_data = [
                    movie.get('每日票房'),
                    movie.get('每日票房占比'),
                    movie.get('每日排片占比'),
                    get_weekday(date_list[day])  # 量化为周几
                ]
                filtered_movie['movie_data'].append(daily_data)

# 对于上映天数小于30但大于15的电影，复制前15天的数据
for filtered_movie in filtered_movies:
    release_days = filtered_movie['上映天数']
    if release_days > 15 and release_days < 30:
        # 复制前15天数据
        initial_data = filtered_movie['movie_data'][:15]
        for i in range(15, 30):
            filtered_movie['movie_data'].append(initial_data[i - 15])



# 输出数据到txt文件
output_file = 'filtered_movies_data.txt'

with open(output_file, 'w', encoding='utf-8') as fobj:
    for filtered_movie in filtered_movies:
        record_list = [
            f"电影代码: {filtered_movie['代码']}, 电影名: {filtered_movie['片名']}, 上映天数: {filtered_movie['上映天数']}"
        ]
        for daily_data in filtered_movie['movie_data']:
            daily_record = f"[{', '.join(map(str, daily_data))}]"
            record_list.append(daily_record)
        fobj.write("\n".join(record_list) + "\n")

print(f"筛选并处理后的电影数据已保存到 {output_file}")
