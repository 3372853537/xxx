import json
import mysql.connector
from mysql.connector import Error
import re

# 数据库配置（请根据实际修改）
db_config = {
    'host': '127.0.0.1',
    'user': 'root',
    'password': 'root',
    'database': 'project'
}

def insert_movie_data(json_file='zhengzaishangying.json'):
    try:
        conn = mysql.connector.connect(**db_config)
        cursor = conn.cursor()

        with open(json_file, 'r', encoding='utf-8') as f:
            movies_data = json.load(f)

        for idx, movie in enumerate(movies_data, 1):
            try:
                # 主表字段清洗
                title = movie['title'].strip()
                release_time = movie.get('release_time', '未知').strip()
                poster = movie.get('poster', '').strip()
                plot = movie.get('plot', '').strip()
                directors_poster = movie.get('directors_poster', '').strip()
                score = movie.get('score', '暂无评分').strip()

                # 插入主表（忽略重复标题）
                insert_movie_sql = """
                INSERT INTO moviesnow 
                (title, release_time, poster, plot, directors_poster, score)
                VALUES (%s, %s, %s, %s, %s, %s)
                ON DUPLICATE KEY UPDATE title=VALUES(title)
                """
                cursor.execute(insert_movie_sql, (
                    title, release_time, poster, plot, directors_poster, score
                ))
                movie_id = cursor.lastrowid if cursor.rowcount > 0 else None

                if not movie_id:
                    print(f"警告：电影《{title}》已存在，跳过")
                    continue

                # 处理导演（支持多导演）
                for director in movie.get('directors', []):
                    if isinstance(director, str) and director.strip():
                        cursor.execute("""
                        INSERT INTO directorsnow (movie_id, director)
                        VALUES (%s, %s)
                        """, (movie_id, director.strip()))

                # 处理演员（支持海报和角色）
                for actor_info in movie.get('actors', []):
                    if not isinstance(actor_info, dict):
                        continue
                    actor = actor_info.get('name', '').strip()
                    role = actor_info.get('role', '未知').strip()
                    actor_poster = actor_info.get('actor_poster', '').strip()
                    if actor:
                        cursor.execute("""
                        INSERT INTO actorsnow (movie_id, actor, role, actor_poster)
                        VALUES (%s, %s, %s, %s)
                        """, (movie_id, actor, role, actor_poster))

                print(f"已导入 [{idx}/{len(movies_data)}] {title}")

            except Exception as e:
                conn.rollback()
                print(f"错误：第{idx}部电影《{title}》导入失败 - {str(e)}")
                continue

        conn.commit()
        print(f"\n成功导入{len(movies_data)}部电影数据（含重复跳过）")

    except Error as e:
        print(f"数据库连接错误: {e}")
    finally:
        if conn.is_connected():
            cursor.close()
            conn.close()

if __name__ == '__main__':
    insert_movie_data()