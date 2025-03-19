# from flask import Flask
# import pickle
# import mysql.connector
# import re


# db_config = {
#     'host': '127.0.0.1',
#     'user': 'root',
#     'password': 'root',
#     'database': 'project'
# }

# # 读取数据文件内容
# with open('MoviesData\dailyPredictData\BoxingData.txt', 'r', encoding='utf-8') as f:
#     content = f.read()

# # 正则表达式匹配电影数据
# pattern = re.compile(r'^(.+?)\s+\[.*?\]', re.MULTILINE)
# movies = pattern.findall(content)

# # 连接到数据库
# conn = mysql.connector.connect(**db_config)
# cursor = conn.cursor()

# # 遍历每部电影生成插入语句
# for movie in movies:
#     # 提取电影名称和每日数据
#     movie_name = movie.strip()
#     daily_data = re.findall(r'\[([\d.]+),\s*([\d.]+),\s*([\d.]+),\s*(\d+)\]', content.split(movie)[1].strip())

#     # 初始化30天数据为NULL
#     days = [ (None, None, None, None) ] * 30

#     # 填充实际数据
#     for i, data in enumerate(daily_data[:30]):
#         daily, share, screen, week = data
#         days[i] = (
#             float(daily),
#             float(share) ,  # 转换为小数形式
#             float(screen),  # 转换为小数形式
#             int(week)
#         )

#     # 生成 SQL 语句
#     columns = [f'day{i + 1}_{col}' for i in range(30) for col in ['daily', 'share', 'screen', 'week']]
#     values = []
#     for day in days:
#         for value in day:
#             if value is None:
#                 values.append('NULL')
#             else:
#                 values.append(str(value))

#     columns_str = ', '.join(['movie_name'] + columns)
#     values_str = ', '.join([f"'{movie_name}'"] + values)

#     sql = f"INSERT INTO movie_box_office_30days ({columns_str}) VALUES ({values_str});"

#     try:
#         cursor.execute(sql)
#         conn.commit()
#         print(f"成功插入 {movie_name} 的数据")
#     except Exception as e:
#         print(f"插入 {movie_name} 的数据时出错: {e}")

# # 关闭数据库连接
# conn.close()

import pandas as pd
import mysql.connector
from mysql.connector import Error

# 1. 数据库配置（根据实际修改）
db_config = {
    'host': '127.0.0.1',
    'user': 'root',
    'password': 'root',
    'database': 'project',  # 已存在的数据库
    'charset': 'utf8mb4'    # 支持特殊字符
}

# 2. 读取txt文件（关键：处理多空格+清洗列名）
df = pd.read_csv(
    'test.txt',       # 替换为实际文件名
    sep='\s+',             # 匹配1个或多个空格（处理列间多余空格）
    engine='python',
    na_values=['nan'],
    on_bad_lines='warn'    # 忽略格式错误行（调试用）
)

# 3. 数据清洗（针对你的表结构优化）
def clean_column(col):
    return col.replace('  ', '_').replace(' ', '_').strip()  # 合并连续空格

df.columns = [clean_column(col) for col in df.columns]  # 清洗列名（如"language_한  국어"转"language_한국어"）

# 4. 类型转换（重点处理布尔和tinyint列）
boolean_cols = [col for col in df.columns if col.startswith(('genre_', 'production_', 'keyword_', 'has_'))]
df[boolean_cols] = df[boolean_cols].astype(bool)  # 转换布尔列（0/1转True/False）

# 处理tinyint(1)的has_homepage（确保0/1或None）
if 'has_homepage' in df.columns:
    df['has_homepage'] = df['has_homepage'].apply(lambda x: x if x in {0, 1} else None)

# 5. NaN转None（数据库NULL）
df = df.where(pd.notna(df), None)

# 6. 数据库插入（分批执行防超时）
try:
    conn = mysql.connector.connect(**db_config)
    cursor = conn.cursor()

    # 验证表存在（可选）
    cursor.execute("SHOW TABLES LIKE 'movie_features'")
    if not cursor.fetchone():
        raise ValueError("表不存在，请检查表名")

    # 生成插入语句（自动匹配表列）
    cols = df.columns.tolist()
    placeholders = ', '.join(['%s'] * len(cols))
    insert_sql = f"INSERT INTO movie_features ({', '.join(cols)}) VALUES ({placeholders})"

    # 分批插入（每批100条，根据数据量调整）
    batch_size = 100
    for i in range(0, len(df), batch_size):
        batch = df.iloc[i:i+batch_size].values.tolist()
        cursor.executemany(insert_sql, batch)
        conn.commit()
        print(f"已插入 {i+len(batch)}/{len(df)} 条")

    print(f"\n成功插入 {len(df)} 条数据！")

except Error as e:
    print(f"\n数据库错误: {e}")
    print("错误行示例：", df.iloc[0].to_dict())  # 打印第一条数据辅助排查
    conn.rollback()

finally:
    if 'cursor' in locals():
        cursor.close()
    if 'conn' in locals():
        conn.close()