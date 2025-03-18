from flask import Flask
import pickle
import mysql.connector
import re


db_config = {
    'host': '127.0.0.1',
    'user': 'root',
    'password': 'zhanglude2004',
    'database': 'dataclass'
}

# 读取数据文件内容
with open('MoviesData\dailyPredictData\BoxingData.txt', 'r', encoding='utf-8') as f:
    content = f.read()

# 正则表达式匹配电影数据
pattern = re.compile(r'^(.+?)\s+\[.*?\]', re.MULTILINE)
movies = pattern.findall(content)

# 连接到数据库
conn = mysql.connector.connect(**db_config)
cursor = conn.cursor()

# 遍历每部电影生成插入语句
for movie in movies:
    # 提取电影名称和每日数据
    movie_name = movie.strip()
    daily_data = re.findall(r'\[([\d.]+),\s*([\d.]+),\s*([\d.]+),\s*(\d+)\]', content.split(movie)[1].strip())

    # 初始化30天数据为NULL
    days = [ (None, None, None, None) ] * 30

    # 填充实际数据
    for i, data in enumerate(daily_data[:30]):
        daily, share, screen, week = data
        days[i] = (
            float(daily),
            float(share) ,  # 转换为小数形式
            float(screen),  # 转换为小数形式
            int(week)
        )

    # 生成 SQL 语句
    columns = [f'day{i + 1}_{col}' for i in range(30) for col in ['daily', 'share', 'screen', 'week']]
    values = []
    for day in days:
        for value in day:
            if value is None:
                values.append('NULL')
            else:
                values.append(str(value))

    columns_str = ', '.join(['movie_name'] + columns)
    values_str = ', '.join([f"'{movie_name}'"] + values)

    sql = f"INSERT INTO movie_box_office_30days ({columns_str}) VALUES ({values_str});"

    try:
        cursor.execute(sql)
        conn.commit()
        print(f"成功插入 {movie_name} 的数据")
    except Exception as e:
        print(f"插入 {movie_name} 的数据时出错: {e}")

# 关闭数据库连接
conn.close()