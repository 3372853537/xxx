from flask import Flask, jsonify, request,send_file, make_response
import mysql.connector 
from mysql.connector import Error
import torch
import torch.nn as nn
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import matplotlib.pyplot as plt
import io
import json
import joblib
import pandas as pd
import xgboost as xgb
from openai import OpenAI
import requests
import os
import datetime 
from datetime import datetime as datetimes
from wsgiref.handlers import format_date_time
from time import mktime
import hashlib
import base64
import hmac
from urllib.parse import urlencode
import json
from PIL import Image
from io import BytesIO
import time
import hashlib
import hmac
import json
from urllib.parse import urlparse
import ssl
from time import mktime
from urllib.parse import urlencode
from wsgiref.handlers import format_date_time
from werkzeug.security import generate_password_hash, check_password_hash
import websocket
import openpyxl
from concurrent.futures import ThreadPoolExecutor, as_completed
import _thread as thread
import threading

from flask_cors import CORS  # 导入 CORS
#——————————————————————————————————————————————————————————————————————————————
class Ws_Param(object):
    # 初始化
    def __init__(self, APPID, APIKey, APISecret, gpt_url):
        self.APPID = APPID
        self.APIKey = APIKey
        self.APISecret = APISecret
        self.host = urlparse(gpt_url).netloc
        self.path = urlparse(gpt_url).path
        self.gpt_url = gpt_url

    # 生成url
    def create_url(self):
        # 生成RFC1123格式的时间戳
        now = datetimes.now()
        date = format_date_time(mktime(now.timetuple()))

        # 拼接字符串
        signature_origin = "host: " + self.host + "\n"
        signature_origin += "date: " + date + "\n"
        signature_origin += "GET " + self.path + " HTTP/1.1"

        # 进行hmac-sha256进行加密
        signature_sha = hmac.new(self.APISecret.encode('utf-8'), signature_origin.encode('utf-8'),
                                 digestmod=hashlib.sha256).digest()

        signature_sha_base64 = base64.b64encode(signature_sha).decode(encoding='utf-8')

        authorization_origin = f'api_key="{self.APIKey}", algorithm="hmac-sha256", headers="host date request-line", signature="{signature_sha_base64}"'

        authorization = base64.b64encode(authorization_origin.encode('utf-8')).decode(encoding='utf-8')

        # 将请求的鉴权参数组合为字典
        v = {
            "authorization": authorization,
            "date": date,
            "host": self.host
        }
        # 拼接鉴权参数，生成url
        url = self.gpt_url + '?' + urlencode(v)
        # 此处打印出建立连接时候的url,参考本demo的时候可取消上方打印的注释，比对相同参数时生成的url与自己代码生成的url是否一致
        return url

aiMessage = ""
event = threading.Event()
# 收到websocket错误的处理
def on_error(ws, error):
    print("### error:", error)


def on_close(ws, close_status_code, close_msg):
    print("### closed ###")
    # 可以根据需要打印关闭状态码和关闭消息
    print(f"关闭状态码: {close_status_code}")
    print(f"关闭消息: {close_msg}")


# 收到websocket连接建立的处理
def on_open(ws):
    thread.start_new_thread(run, (ws,))


def run(ws, *args):
    data = json.dumps(gen_params(appid=ws.appid, query=ws.query, domain=ws.domain))
    ws.send(data)


# 收到websocket消息的处理
def on_message(ws, message):
    # print(message)
    global aiMessage
    data = json.loads(message)
    code = data['header']['code']
    if code != 0:
        print(f'请求错误: {code}, {data}')
        ws.close()
    else:
        choices = data["payload"]["choices"]
        status = choices["status"]
        content = choices["text"][0]["content"]
        aiMessage += content  # 追加新内容到 aiMessage
        print(content, end='')
        if status == 2:
            print("#### 关闭会话")
            ws.close()
            event.set()


def gen_params(appid, query, domain):
    """
    通过appid和用户的提问来生成请参数
    """

    data = {
        "header": {
            "app_id": appid,
            "uid": "1234",           
            # "patch_id": []    #接入微调模型，对应服务发布后的resourceid          
        },
        "parameter": {
            "chat": {
                "domain": domain,
                "temperature": 0.5,
                "max_tokens": 4096,
                "auditing": "default",
            }
        },
        "payload": {
            "message": {
                "text": [{"role": "user", "content": query}]
            }
        }
    }
    return data


def text_main(appid, api_secret, api_key, Spark_url, domain, query):
    wsParam = Ws_Param(appid, api_key, api_secret, Spark_url)
    websocket.enableTrace(False)
    wsUrl = wsParam.create_url()

    ws = websocket.WebSocketApp(wsUrl, on_message=on_message, on_error=on_error, on_close=on_close, on_open=on_open)
    ws.appid = appid
    ws.query = query
    ws.domain = domain
    ws.run_forever(sslopt={"cert_reqs": ssl.CERT_NONE})

#————————————————————————————————————————————————————————————————————————————————


# 后续代码可以正常使用 datetime 模块
now = datetime.datetime.now()

app = Flask(__name__)
app.json.ensure_ascii = False # 解决中文乱码问题

# 添加全局响应头设置
@app.after_request
def after_request(response):
    response.headers.add('Access-Control-Allow-Credentials', 'true')
    response.headers.add('Access-Control-Allow-Headers', 'Content-Type,Authorization,X-Requested-With')
    response.headers.add('Access-Control-Allow-Methods', 'GET,PUT,POST,DELETE,OPTIONS')
    response.headers.add('Access-Control-Allow-Origin', 'http://localhost:8080')
    return response


class LSTM(nn.Module):
    def __init__(self, input_size=4, hidden_layer_size=1500, output_size=1):
        super().__init__()
        self.hidden_layer_size = hidden_layer_size
        #定义LSTM层
        self.lstm = nn.LSTM(input_size, hidden_layer_size)
        #定义全连接层
        self.linear = nn.Linear(hidden_layer_size, output_size)

    def forward(self, input_seq):
        # 获取批量大小
        batch_size = input_seq.size(1)
        # 初始化隐藏状态和细胞状态，形状为 (num_layers, batch_size, hidden_layer_size)
        self.hidden_cell = (
            torch.zeros(1, batch_size, self.hidden_layer_size, device=input_seq.device),
            torch.zeros(1, batch_size, self.hidden_layer_size, device=input_seq.device)
        )
        # 输入序列已经是 (时间步, 批量大小, 输入特征维度) 的形状，无需调整
        lstm_out, self.hidden_cell = self.lstm(input_seq, self.hidden_cell)
        # 调整 LSTM 输出的形状，以便输入到全连接层
        # lstm_out 的形状是 (时间步, 批量大小, 隐藏层维度)
        # 调整为 (时间步 * 批量大小, 隐藏层维度)
        lstm_out_reshaped = lstm_out.view(-1, self.hidden_layer_size)
        # 通过全连接层得到预测结果
        # predictions 的形状是 (时间步 * 批量大小, 输出维度)
        
        predictions = self.linear(lstm_out_reshaped)
        # 重新调整预测结果的形状为 (时间步, 批量大小, 输出维度)
        
        predictions = predictions.view(input_seq.size(0), batch_size, -1)
        # 返回每个批次最后一个时间步的预测结果
        # 形状为 (批量大小, 输出维度)
        return predictions[-1]

feature_order = ["budget", "original_language", "popularity", "runtime", "collection_name", "has_collection", "num_genres", "all_genres", "genre_Drama", "genre_Comedy", "genre_Thriller", "genre_Action", "genre_Romance", "genre_Crime", "genre_Adventure", "genre_Horror", "genre_Science_Fiction", "genre_Family", "genre_Fantasy", "genre_Mystery", "genre_Animation", "genre_History", "genre_Music", "num_companies", "production_company_Warner_Bros", "production_company_Universal_Pictures", "production_company_Paramount_Pictures", "production_company_Twentieth_Century_Fox_Film_Corporation", "production_company_Columbia_Pictures", "production_company_Metro_Goldwyn_Mayer_MGM", "production_company_New_Line_Cinema", "production_company_Touchstone_Pictures", "production_company_Walt_Disney_Pictures", "production_company_Columbia_Pictures_Corporation", "production_company_TriStar_Pictures", "production_company_Relativity_Media", "production_company_Canal_", "production_company_United_Artists", "production_company_Miramax_Films", "production_company_Village_Roadshow_Pictures", "production_company_Regency_Enterprises", "production_company_BBC_Films", "production_company_Dune_Entertainment", "production_company_Working_Title_Films", "production_company_Fox_Searchlight_Pictures", "production_company_StudioCanal", "production_company_Lionsgate", "production_company_DreamWorks_SKG", "production_company_Fox_2000_Pictures", "production_company_Summit_Entertainment", "production_company_Hollywood_Pictures", "production_company_Orion_Pictures", "production_company_Amblin_Entertainment", "production_company_Dimension_Films", "num_countries", "production_country_United_States_of_America", "production_country_United_Kingdom", "production_country_France", "production_country_Germany", "production_country_Canada", "production_country_India", "production_country_Italy", "production_country_Japan", "production_country_Australia", "production_country_Russia", "production_country_Spain", "production_country_China", "production_country_Hong_Kong", "production_country_Ireland", "production_country_Belgium", "production_country_South_Korea", "production_country_Mexico", "production_country_Sweden", "production_country_New_Zealand", "production_country_Netherlands", "production_country_Czech_Republic", "production_country_Denmark", "production_country_Brazil", "production_country_Luxembourg", "production_country_South_Africa", "num_languages", "language_English", "language_Français", "language_Español", "language_Deutsch", "language_Pусский", "language_Italiano", "language_日本語", "language_普通话", "language_हिन्दी", "language_Português", "language_العربية", "language_한국어_조선말", "language_广州话_廣州話", "language_தமிழ்", "language_Polski", "language_Magyar", "language_Latin", "language_svenska", "language_ภาษาไทย", "language_Český", "language_עִבְרִית", "language_ελληνικά", "language_Türkçe", "language_Dansk", "language_Nederlands", "language_فارسی", "language_Tiếng_Việt", "language_اردو", "language_Română", "num_Keywords", "keyword_woman_director", "keyword_independent_film", "keyword_duringcreditsstinger", "keyword_murder", "keyword_based_on_novel", "keyword_violence", "keyword_sport", "keyword_biography", "keyword_aftercreditsstinger", "keyword_dystopia", "keyword_revenge", "keyword_friendship", "keyword_sex", "keyword_suspense", "keyword_sequel", "keyword_love", "keyword_police", "keyword_teenager", "keyword_nudity", "keyword_female_nudity", "keyword_drug", "keyword_prison", "keyword_musical", "keyword_high_school", "keyword_los_angeles", "keyword_new_york", "keyword_family", "keyword_father_son_relationship", "keyword_kidnapping", "keyword_investigation", "num_cast", "cast_name_Samuel_L_Jackson", "cast_name_Robert_De_Niro", "cast_name_Morgan_Freeman", "cast_name_J_K_Simmons", "cast_name_Bruce_Willis", "cast_name_Liam_Neeson", "cast_name_Susan_Sarandon", "cast_name_Bruce_McGill", "cast_name_John_Turturro", "cast_name_Forest_Whitaker", "cast_name_Willem_Dafoe", "cast_name_Bill_Murray", "cast_name_Owen_Wilson", "cast_name_Nicolas_Cage", "cast_name_Sylvester_Stallone", "genders_0_cast", "genders_1_cast", "genders_2_cast", "cast_character_Himself", "cast_character_Herself", "cast_character_Dancer", "cast_character_Additional_Voices_voice", "cast_character_Doctor", "cast_character_Reporter", "cast_character_Waitress", "cast_character_Nurse", "cast_character_Bartender", "cast_character_Jack", "cast_character_Debutante", "cast_character_Security_Guard", "cast_character_Paul", "cast_character_Frank", "num_crew", "crew_name_Avy_Kaufman", "crew_name_Robert_Rodriguez", "crew_name_Deborah_Aquila", "crew_name_James_Newton_Howard", "crew_name_Mary_Vernieu", "crew_name_Steven_Spielberg", "crew_name_Luc_Besson", "crew_name_Jerry_Goldsmith", "crew_name_Francine_Maisler", "crew_name_Tricia_Wood", "crew_name_James_Horner", "crew_name_Kerry_Barden", "crew_name_Bob_Weinstein", "crew_name_Harvey_Weinstein", "crew_name_Janet_Hirshenson", "genders_0_crew", "genders_1_crew", "genders_2_crew", "jobs_Producer", "jobs_Executive_Producer", "jobs_Director", "jobs_Screenplay", "jobs_Editor", "jobs_Casting", "jobs_Director_of_Photography", "jobs_Original_Music_Composer", "jobs_Art_Direction", "jobs_Production_Design", "jobs_Costume_Design", "jobs_Writer", "jobs_Set_Decoration", "jobs_Makeup_Artist", "jobs_Sound_Re_Recording_Mixer", "departments_Production", "departments_Sound", "departments_Art", "departments_Crew", "departments_Writing", "departments_Costume__Make_Up", "departments_Camera", "departments_Directing", "departments_Editing", "departments_Visual_Effects", "departments_Lighting", "departments_Actors", "log_budget", "has_homepage", "release_date_year", "release_date_weekday", "release_date_month", "release_date_day", "release_date_quarter", "release_date_weekofyear", "len_title", "words_title", "len_tagline", "words_tagline", "len_overview", "words_overview", "len_original_title", "words_original_title", "title_oof", "tagline_oof", "overview_oof", "original_title_oof", "budget_to_popularity", "budget_to_runtime", "_budget_year_ratio", "_releaseYear_popularity_ratio", "_releaseYear_popularity_ratio2", "runtime_to_mean_year", "popularity_to_mean_year", "budget_to_mean_year", "budget_to_year", "budget_to_mean_year_to_year", "popularity_to_mean_year_to_log_budget", "year_to_log_budget", "budget_to_runtime_to_year", "genders_1_cast_to_log_budget", "all_genres_to_popularity_to_mean_year", "genders_2_crew_to_budget_to_mean_year", "overview_oof_to_genders_2_crew", "popularity2", "rating", "totalVotes"]


db_config = {
    'host': '127.0.0.1',
    'user': 'root',
    'password': 'zhanglude2004',
    'database': 'dataclass'
}

# 加载模型的 state_dict
try:
    model1 = LSTM()
    # 修改加载模型方法，确保在 CPU-only 环境下运行
    model1.load_state_dict(torch.load('model1.pth', map_location=torch.device('cpu'), weights_only=True))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model1 = model1.to(device)
    model1.eval()
    
except Exception as e:
    print(f"加载模型时出错: {e}")
    model1 = None
Rawmodel1 = joblib.load('Rawmodel1.joblib')
Rawmodel2 = joblib.load('Rawmodel2.joblib')
Rawmodel3 = joblib.load('Rawmodel3.joblib')
Rawmodel4 = joblib.load('Rawmodel4.joblib')
Rawmodel5 = joblib.load('Rawmodel5.joblib')
model2 = joblib.load('lightgbm_model.joblib')

#这个端口可以拿到每部电影每日的票房数据
@app.route('/movies/<movie_name>', methods=['GET'])
def get_movie_data(movie_name):
    try:
        # 连接数据库
        conn = mysql.connector.connect(**db_config)
        cursor = conn.cursor(dictionary=True)

        # 构建 SQL 查询语句
        query = "SELECT * FROM movie_box_office_30days WHERE movie_name = %s"
        cursor.execute(query, (movie_name,))

        # 获取查询结果
        movie_data = cursor.fetchone()

        # 关闭数据库连接
        cursor.close()
        conn.close()

        if movie_data:
            return jsonify(movie_data)
        else:
            return jsonify({"message": "未找到该电影的数据"}), 404

    except mysql.connector.Error as err:
        return jsonify({"error": str(err)}), 500

# 解决中文字体缺失
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False
#这个端口可以通过过往的票房来对未来的票房进行预测
@app.route('/predict', methods=['POST'])
def prediction():
    try:
        movieInfo = request.get_json()
        name = movieInfo.get('name')
        step = movieInfo.get('step')  # 通过上映多少天的数据来进行预测
        flag = movieInfo.get('flag')  #true 代表预测新电影
        if not name or not step:
            return jsonify({"message": "请求数据不完整，请提供电影名称和预测步数"}), 400

        # 在数据库中查询电影
        conn = mysql.connector.connect(**db_config)
        cursor = conn.cursor(dictionary=True)
        try:
            query = "SELECT * FROM movie_box_office_30days WHERE movie_name = %s"
            cursor.execute(query, (name,))
            movie_data = cursor.fetchone()
        finally:
            cursor.close()
            conn.close()

        if not movie_data:
            return jsonify({"message": "未找到该电影的数据"}), 404

        # 初始化空列表存储每天的特征
        daily_features = []

        # 遍历 day1 到 day30
        for day in range(1, 31):
            daily = float(movie_data[f'day{day}_daily'])
            share = float(movie_data[f'day{day}_share'])
            screen = float(movie_data[f'day{day}_screen'])
            week = int(movie_data[f'day{day}_week'])
            daily_features.append([daily, share, screen, week])

        daily_features = np.array(daily_features)
        # 先转换为 PyTorch 张量
        daily_features = torch.from_numpy(daily_features)
        # 再使用 unsqueeze 方法
        daily_features = daily_features.unsqueeze(1)
        
        
        # 初始化 MinMaxScaler
        scaler = MinMaxScaler()

        # 将数据调整为二维数组进行归一化
        data_2d = daily_features.reshape(-1, daily_features.shape[-1])
        scaled_data_2d = scaler.fit_transform(data_2d)

        scaled_data_array = scaled_data_2d.reshape(daily_features.shape)

        # 将归一化后的 NumPy 数组转换为 PyTorch 张量
        tensor_data = torch.from_numpy(scaled_data_array).float().to(device)
        
        # 从多少天开始预测
        test_inputs = tensor_data[:step, :, :]
        # base_inputs 是对 test_inputs 的一个备份
        base_inputs = test_inputs

        # 将剩下的天数进行预测
        for i in range(30 - step):
            seq = test_inputs[-step:]
            with torch.no_grad():
                model1.hidden_cell = (
                    torch.zeros(1, seq.size(1), model1.hidden_layer_size, device=device),
                    torch.zeros(1, seq.size(1), model1.hidden_layer_size, device=device)
                )
                # 预测值形状为 (批量大小, 1)
                seq_pred = model1(seq)

                # 将真实的后面三个特征拼接进去，方便预测
                additional_features = tensor_data[i + step, :, 1:].unsqueeze(0)
                seq_pred = torch.cat((seq_pred, additional_features.squeeze(0)), dim=1)
                # 添加到 test_inputs 后面
                test_inputs = torch.cat((test_inputs, seq_pred.unsqueeze(0)), dim=0)


        # test_inputs前几天是真实数据，后是预测数据
        prediction_result = test_inputs[:, :, 0]

        # 反归一化预测结果
        # 提取票房特征在归一化前的最小和最大值
        min_val = scaler.data_min_[0]
        max_val = scaler.data_max_[0]
        denormalized_prediction = prediction_result.cpu().numpy() * (max_val - min_val) + min_val

        x = np.arange(denormalized_prediction.shape[0])
        Test_data = daily_features[:, :, 0]

        # 创建一个字节流来保存图像
        img_buffer = io.BytesIO()

    #     # 遍历每个批量
    #     for i in range(denormalized_prediction.shape[1]):
    #         # 提取当前批量的数据
    #         pred_batch = denormalized_prediction[:, i]
    #         test_batch = Test_data[:, i]
    #         # 创建一个新的图形
    #         plt.figure()
    #         plt.title(f'Comparison of Batch {i}')
    #         plt.xlabel('Time Step')
    #         plt.ylabel('Value')
    #         plt.grid(True)

    #         # 绘制预测结果和测试数据
    #         plt.plot(x, pred_batch, label='Prediction', color='blue')
    #         if(flag==0): plt.plot(x, test_batch, label='Test Data', color='red')

    #         # 添加图例
    #         plt.legend()

    #     # 保存图形到字节流
    #     plt.savefig(img_buffer, format='png')
    #     img_buffer.seek(0)

    #     return send_file(img_buffer, mimetype='image/png')

    # except Exception as e:
    #     return jsonify({"error": str(e)}), 500
        #————————————————————————————————————————————————
        # 创建一个包含两个子图的画布，一行两列
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))

        # 绘制折线图
        for i in range(denormalized_prediction.shape[1]):
            # 提取当前批量的数据
            pred_batch = denormalized_prediction[:, i]
            test_batch = Test_data[:, i]
            axes[0].set_title(f'票房预测')
            axes[0].set_xlabel('上映天数')
            axes[0].set_ylabel('票房')
            axes[0].grid(True)

            # 绘制预测结果和测试数据
            axes[0].plot(x, pred_batch, label='Prediction', color='blue')
            if flag == 0:
                axes[0].plot(x, test_batch, label='Test Data', color='red')

            # 添加图例
            axes[0].legend()

        # 绘制柱状图
        axes[1].bar(x, denormalized_prediction[:, 0], label='Prediction', color='green')
        axes[1].set_title('票房预测柱状图')
        axes[1].set_xlabel('上映天数')
        axes[1].set_ylabel('票房')
        axes[1].legend()

        # 调整子图布局
        plt.tight_layout()

        # 保存图形到字节流
        plt.savefig(img_buffer, format='png')
        img_buffer.seek(0)

        return send_file(img_buffer, mimetype='image/png')

    except Exception as e:
        return jsonify({"error": str(e)}), 500

def getAverage():
    try:
        conn = mysql.connector.connect(**db_config)

        # 执行查询
        query = """
            SELECT 
                budget, original_language, popularity, runtime, collection_name, has_collection, 
                num_genres, all_genres, genre_Drama, genre_Comedy, genre_Thriller, genre_Action, 
                genre_Romance, genre_Crime, genre_Adventure, genre_Horror, genre_Science_Fiction, 
                genre_Family, genre_Fantasy, genre_Mystery, genre_Animation, genre_History, genre_Music, 
                num_companies, 
                production_company_Warner_Bros, production_company_Universal_Pictures, 
                production_company_Paramount_Pictures, production_company_Twentieth_Century_Fox_Film_Corporation, 
                production_company_Columbia_Pictures, production_company_Metro_Goldwyn_Mayer_MGM, 
                production_company_New_Line_Cinema, production_company_Touchstone_Pictures, 
                production_company_Walt_Disney_Pictures, production_company_Columbia_Pictures_Corporation, 
                production_company_TriStar_Pictures, production_company_Relativity_Media, 
                production_company_Canal_, production_company_United_Artists, 
                production_company_Miramax_Films, production_company_Village_Roadshow_Pictures, 
                production_company_Regency_Enterprises, production_company_BBC_Films, 
                production_company_Dune_Entertainment, production_company_Working_Title_Films, 
                production_company_Fox_Searchlight_Pictures, production_company_StudioCanal, 
                production_company_Lionsgate, production_company_DreamWorks_SKG, 
                production_company_Fox_2000_Pictures, production_company_Summit_Entertainment, 
                production_company_Hollywood_Pictures, production_company_Orion_Pictures, 
                production_company_Amblin_Entertainment, production_company_Dimension_Films, 
                num_countries, 
                production_country_United_States_of_America, production_country_United_Kingdom, 
                production_country_France, production_country_Germany, production_country_Canada, 
                production_country_India, production_country_Italy, production_country_Japan, 
                production_country_Australia, production_country_Russia, production_country_Spain, 
                production_country_China, production_country_Hong_Kong, production_country_Ireland, 
                production_country_Belgium, production_country_South_Korea, production_country_Mexico, 
                production_country_Sweden, production_country_New_Zealand, production_country_Netherlands, 
                production_country_Czech_Republic, production_country_Denmark, production_country_Brazil, 
                production_country_Luxembourg, production_country_South_Africa, 
                num_languages, 
                language_English, language_Français, language_Español, language_Deutsch, 
                language_Pусский, language_Italiano, language_日本語, language_普通话, 
                language_हिन्दी, language_Português, language_العربية, language_한국어_조선말, 
                language_广州话_廣州話, language_தமிழ், language_Polski, language_Magyar, 
                language_Latin, language_svenska, language_ภาษาไทย, language_Český, 
                language_עִבְרִית, language_ελληνικά, language_Türkçe, language_Dansk, 
                language_Nederlands, language_فارسی, language_Tiếng_Việt, language_اردو, 
                language_Română, 
                num_Keywords, 
                keyword_woman_director, keyword_independent_film, keyword_duringcreditsstinger, 
                keyword_murder, keyword_based_on_novel, keyword_violence, keyword_sport, 
                keyword_biography, keyword_aftercreditsstinger, keyword_dystopia, 
                keyword_revenge, keyword_friendship, keyword_sex, keyword_suspense, 
                keyword_sequel, keyword_love, keyword_police, keyword_teenager, 
                keyword_nudity, keyword_female_nudity, keyword_drug, keyword_prison, 
                keyword_musical, keyword_high_school, keyword_los_angeles, keyword_new_york, 
                keyword_family, keyword_father_son_relationship, keyword_kidnapping, 
                keyword_investigation, 
                num_cast, 
                cast_name_Samuel_L_Jackson, cast_name_Robert_De_Niro, cast_name_Morgan_Freeman, 
                cast_name_J_K_Simmons, cast_name_Bruce_Willis, cast_name_Liam_Neeson, 
                cast_name_Susan_Sarandon, cast_name_Bruce_McGill, cast_name_John_Turturro, 
                cast_name_Forest_Whitaker, cast_name_Willem_Dafoe, cast_name_Bill_Murray, 
                cast_name_Owen_Wilson, cast_name_Nicolas_Cage, cast_name_Sylvester_Stallone, 
                genders_0_cast, genders_1_cast, genders_2_cast, 
                cast_character_Himself, cast_character_Herself, cast_character_Dancer, 
                cast_character_Additional_Voices_voice, cast_character_Doctor, 
                cast_character_Reporter, cast_character_Waitress, cast_character_Nurse, 
                cast_character_Bartender, cast_character_Jack, cast_character_Debutante, 
                cast_character_Security_Guard, cast_character_Paul, cast_character_Frank, 
                num_crew, 
                crew_name_Avy_Kaufman, crew_name_Robert_Rodriguez, crew_name_Deborah_Aquila, 
                crew_name_James_Newton_Howard, crew_name_Mary_Vernieu, crew_name_Steven_Spielberg, 
                crew_name_Luc_Besson, crew_name_Jerry_Goldsmith, crew_name_Francine_Maisler, 
                crew_name_Tricia_Wood, crew_name_James_Horner, crew_name_Kerry_Barden, 
                crew_name_Bob_Weinstein, crew_name_Harvey_Weinstein, crew_name_Janet_Hirshenson, 
                genders_0_crew, genders_1_crew, genders_2_crew, 
                jobs_Producer, jobs_Executive_Producer, jobs_Director, jobs_Screenplay, 
                jobs_Editor, jobs_Casting, jobs_Director_of_Photography, jobs_Original_Music_Composer, 
                jobs_Art_Direction, jobs_Production_Design, jobs_Costume_Design, jobs_Writer, 
                jobs_Set_Decoration, jobs_Makeup_Artist, jobs_Sound_Re_Recording_Mixer, 
                departments_Production, departments_Sound, departments_Art, departments_Crew, 
                departments_Writing, departments_Costume__Make_Up, departments_Camera, 
                departments_Directing, departments_Editing, departments_Visual_Effects, 
                departments_Lighting, departments_Actors, 
                log_budget, has_homepage, release_date_year, release_date_weekday, 
                release_date_month, release_date_day, release_date_quarter, release_date_weekofyear, 
                len_title, words_title, len_tagline, words_tagline, len_overview, words_overview, 
                len_original_title, words_original_title, title_oof, tagline_oof, 
                overview_oof, original_title_oof, 
                budget_to_popularity, budget_to_runtime, _budget_year_ratio, 
                _releaseYear_popularity_ratio, _releaseYear_popularity_ratio2, runtime_to_mean_year, 
                popularity_to_mean_year, budget_to_mean_year, budget_to_year, 
                budget_to_mean_year_to_year, popularity_to_mean_year_to_log_budget, 
                year_to_log_budget, budget_to_runtime_to_year, 
                genders_1_cast_to_log_budget, all_genres_to_popularity_to_mean_year, 
                genders_2_crew_to_budget_to_mean_year, overview_oof_to_genders_2_crew, 
                popularity2, rating, totalVotes
            FROM movie_features;
        """
        with conn.cursor() as cursor:
            cursor.execute(query)
            data = cursor.fetchall()
            # 获取列名
            column_names = [desc[0] for desc in cursor.description]

        # 将数据转换为 DataFrame
        df = pd.DataFrame(data, columns=column_names)
        if df.empty:
            print("警告：表中无数据！")
            return {}

        # 计算各列均值，将结果存储在字典中
        mean_dict = {}
        for col in df.columns:
            try:
                # 尝试将列转换为数值类型
                numeric_col = pd.to_numeric(df[col], errors='coerce')
                mean = numeric_col.mean()
                if pd.notna(mean):
                    mean_dict[col] = round(mean, 4)
                else:
                    # 如果均值为 NaN，添加 0
                    mean_dict[col] = 0
            except Exception:
                # 如果转换失败，添加 0
                mean_dict[col] = 0


        # 输出结果
        print(f"成功计算{len(mean_dict)}个数值字段和布尔字段的平均值：")

    except mysql.connector.Error as e:
        print(f"数据库错误：{e}")
        return []
    finally:
        if 'conn' in locals() and conn.is_connected():
            conn.close()

    return mean_dict

mean_dict =getAverage()

def dict_to_list(dictionary, order):
    """
    将字典按照指定顺序转换为列表
    :param dictionary: 要转换的字典
    :param order: 特征顺序列表
    :return: 转换后的列表
    """
    return [dictionary[key] for key in order]

#——————————————————————————————————————————————————————————————————————————
class AssembleHeaderException(Exception):
    def __init__(self, msg):
        self.message = msg
class Url:
    def __init__(this, host, path, schema):
        this.host = host
        this.path = path
        this.schema = schema
        pass
def sha256base64(data):
    sha256 = hashlib.sha256()
    sha256.update(data)
    digest = base64.b64encode(sha256.digest()).decode(encoding='utf-8')
    return digest
def parse_url(requset_url):
    stidx = requset_url.index("://")
    host = requset_url[stidx + 3:]
    schema = requset_url[:stidx + 3]
    edidx = host.index("/")
    if edidx <= 0:
        raise AssembleHeaderException("invalid request url:" + requset_url)
    path = host[edidx:]
    host = host[:edidx]
    u = Url(host, path, schema)
    return u
def assemble_ws_auth_url(requset_url, method="GET", api_key="", api_secret=""):
    u = parse_url(requset_url)
    host = u.host
    path = u.path
    now = datetimes.now()
    date = format_date_time(mktime(now.timetuple()))
    # print(date)
    # date = "Thu, 12 Dec 2019 01:57:27 GMT"
    signature_origin = "host: {}\ndate: {}\n{} {} HTTP/1.1".format(host, date, method, path)
    # print(signature_origin)
    signature_sha = hmac.new(api_secret.encode('utf-8'), signature_origin.encode('utf-8'),
                             digestmod=hashlib.sha256).digest()
    signature_sha = base64.b64encode(signature_sha).decode(encoding='utf-8')
    authorization_origin = "api_key=\"%s\", algorithm=\"%s\", headers=\"%s\", signature=\"%s\"" % (
        api_key, "hmac-sha256", "host date request-line", signature_sha)
    authorization = base64.b64encode(authorization_origin.encode('utf-8')).decode(encoding='utf-8')
    # print(authorization_origin)
    values = {
        "host": host,
        "date": date,
        "authorization": authorization
    }
    return requset_url + "?" + urlencode(values)
def getBody(appid,text):
    body= {
        "header": {
            "app_id": appid,
            "uid":"123456789"
        },
        "parameter": {
            "chat": {
                "domain": "general",
                "temperature":0.5,
                "max_tokens":4096
            }
        },
        "payload": {
            "message":{
                "text":[
                    {
                        "role":"user",
                        "content":text
                    }
                ]
            }
        }
    }
    return body
def main(text,appid,apikey,apisecret):
    host = 'http://spark-api.cn-huabei-1.xf-yun.com/v2.1/tti'
    url = assemble_ws_auth_url(host,method='POST',api_key=apikey,api_secret=apisecret)
    content = getBody(appid,text)
    response = requests.post(url,json=content,headers={'content-type': "application/json"}).text
    return response
def base64_to_image(base64_data):
    try:
        # 解码 base64 数据
        img_data = base64.b64decode(base64_data)
        # 将解码后的数据转换为图片
        img = Image.open(BytesIO(img_data))
        # 将图片保存到内存中
        img_byte = BytesIO()
        img.save(img_byte, format='JPEG')
        img_byte.seek(0)
        return img_byte
    except (base64.binascii.Error, OSError) as e:
        print(f"图片解码或打开失败: {e}")
        return None

def parser_Message(message):
    try:
        data = json.loads(message)
        code = data['header']['code']
        if code != 0:
            print(f'请求错误: {code}, {data}')
            return None
        else:
            text = data["payload"]["choices"]["text"]
            imageContent = text[0]
            imageBase = imageContent["content"]
            return base64_to_image(imageBase)
    except json.JSONDecodeError as e:
        print(f"JSON 解析失败: {e}")
        return None
#——————————————————————————————————————————————————————————————————————————————————————
@app.route('/Rawpredict', methods=['POST'])
def Rawprediction():
    # 获取前端传入的 JSON 数据
    data = request.get_json()
    # 提取所需字段
    name = data.get('name')
    budget = data.get('budget')         
    runtime = data.get('runtime')
    genres = data.get('genres')
    language = data.get('language')
    keyword1 = data.get('keyword1')
    keyword2 = data.get('keyword2')
    keyword3 = data.get('keyword3')
    overview = data.get('overview')
    
    #重要的属性
    totalVotes = data.get('totalVotes')  #1  直接读   市场对于此电影的讨论程度

    release_date_year = data.get('release_date_year') #2  直接读  上映年份
    
    rating = data.get('rating') #3  直接读   业内对于此电影的评分
    log_budget = np.log(budget)
    year_to_log_budget = release_date_year / log_budget #5 上映年份和预算之间的比例
    
    #以上共十个特征的输入 分别是budget runtime genres language keyword1 keyword2 keyword3
    #totalVotes release_date_year rating year_to_log_budget
    #——————————————————————————————————————————————————————————————————————
    # 修改 mean_dict 中的对应值
    new_mean_dict = mean_dict.copy()
    new_mean_dict["budget"] = budget
    new_mean_dict["runtime"] = runtime
    new_mean_dict["totalVotes"] = totalVotes
    new_mean_dict["release_date_year"] = release_date_year
    new_mean_dict["rating"] = rating
    new_mean_dict["year_to_log_budget"] = year_to_log_budget

    # 处理 genres
    genre_key = f"genre_{genres}"
    if genre_key in new_mean_dict:
        new_mean_dict[genre_key] = 1

    # 处理 language
    language_key = f"language_{language}"
    if language_key in new_mean_dict:
        new_mean_dict[language_key] = 1

    # 处理 keyword
    keyword_key1 = f"keyword_{keyword1}"
    if keyword_key1 in new_mean_dict:
        new_mean_dict[keyword_key1] = 1
        
    keyword_key2 = f"keyword_{keyword2}"
    if keyword_key2 in new_mean_dict:
        new_mean_dict[keyword_key2] = 1
        
    keyword_key3 = f"keyword_{keyword3}"
    if keyword_key3 in new_mean_dict:
        new_mean_dict[keyword_key3] = 1

    input_list = dict_to_list(new_mean_dict, feature_order)
    input_data = np.array(input_list).reshape(1, -1).astype(float)
    # 使用模型进行预测
    prediction1 = Rawmodel1.predict(input_data)
    input_dmatrix = xgb.DMatrix(input_data)
    prediction2 = Rawmodel2.predict(input_dmatrix)
    prediction3 = Rawmodel3.predict(input_data)
    prediction4 = Rawmodel4.predict(input_data)
    prediction5 = Rawmodel5.predict(input_data)

    input_data = np.array([prediction1,prediction2,prediction3,prediction4,prediction5]).reshape(1, -1).astype(float)
    prediction = model2.predict(input_data)
    
#——————————————————————————————————————————————————————————————————
    if np.expm1(prediction[0]) < budget : 
        return jsonify({"prediction": np.expm1(prediction[0])})

    APPID = 'bce33c37'
    APISecret = 'OGUzMDVmNzg3OWYyNTcwZjViNDNmMzM3'
    APIKEY = '580748ab65e6dfdd59ab3a89eed3be34'
    desc = "关于" + name + "的海报"
    print(desc)
    try:
        # 调用 main 函数生成图片
        res = main(desc, appid=APPID, apikey=APIKEY, apisecret=APISecret)
    except requests.RequestException as e:
        print(f"调用外部 API 失败: {e}")
        return make_response("调用外部 API 失败", 500)
    # 解析图片
    img = parser_Message(res)
    if img is None:
        return make_response("图片生成失败", 500)
#—————————————————————————————————————————————————————————————————
    # 重置全局变量和事件
    global aiMessage, event
    aiMessage = ""
    event.clear()

    text_main(
        appid="bce33c37",
        api_secret="OGUzMDVmNzg3OWYyNTcwZjViNDNmMzM3",
        api_key="580748ab65e6dfdd59ab3a89eed3be34",
        Spark_url="wss://spark-api.xf-yun.com/v4.0/chat",
        domain="4.0Ultra",
        query="我要拍一部电影，电影的简介是" + overview + "请给我一些建议"
    )
    print("hhhh")
    #messageContent = aiMessage
    # 等待消息处理完成
    event.wait()
    print("kkkk")
    # 构造响应：预测结果作为 HTTP 头，图片作为 Body
    response = make_response(send_file(img, mimetype='image/jpeg'))
    response.headers['Prediction'] = np.expm1(prediction[0])
    if aiMessage is not None:
        # 对 aiMessage 进行 Base64 编码
        encoded_aiMessage = base64.b64encode(aiMessage.encode('utf-8')).decode('ascii')
        response.headers['aiMessage'] = encoded_aiMessage

    # 新增：暴露自定义响应头
    response.headers['Access-Control-Expose-Headers'] = 'Prediction, aiMessage'
    
    print(response.headers)
    return response



@app.route('/')
def index():
    return 'Hello, Flask!'
#这个端口可以拿到100部经典影片
@app.route('/Moviesdata',methods=["GET"])
def getMoviesData():
    try:
        conn = mysql.connector.connect(**db_config)
        cursor = conn.cursor(dictionary=True)

        # 查询电影主表数据
        movie_query = "SELECT * FROM movies_top_100"
        cursor.execute(movie_query)
        movies = cursor.fetchall()

        for movie in movies:
            movie_id = movie['id']

            # 查询该电影的导演数据
            director_query = "SELECT director FROM directors_top_100 WHERE movie_id = %s"
            cursor.execute(director_query, (movie_id,))
            directors = [row['director'] for row in cursor.fetchall()]
            movie['directors'] = directors

            # 查询该电影的演员数据
            actor_query = "SELECT actor, role, actor_poster FROM actors_top_100 WHERE movie_id = %s"
            cursor.execute(actor_query, (movie_id,))
            actors = cursor.fetchall()
            movie['actors'] = actors

        return jsonify(movies)

    except Error as e:
        return jsonify({"error": f"数据库连接错误: {str(e)}"}), 500
    finally:
        if conn.is_connected():
            cursor.close()
            conn.close()


#这个端口可以拿到即将上映的影片
@app.route('/MoviesToDisPlay',methods=["GET"])
def getToDisPlay():
    try:
        conn = mysql.connector.connect(**db_config)
        cursor = conn.cursor(dictionary=True)

        # 查询电影主表数据
        movie_query = "SELECT * FROM movieswill"
        cursor.execute(movie_query)
        movies = cursor.fetchall()

        for movie in movies:
            movie_id = movie['id']

            # 查询该电影的导演数据
            director_query = "SELECT director FROM directorswill WHERE movie_id = %s"
            cursor.execute(director_query, (movie_id,))
            directors = [row['director'] for row in cursor.fetchall()]
            movie['directors'] = directors

            # 查询该电影的演员数据
            actor_query = "SELECT actor, role, actor_poster FROM actorswill WHERE movie_id = %s"
            cursor.execute(actor_query, (movie_id,))
            actors = cursor.fetchall()
            movie['actors'] = actors

        return jsonify(movies)

    except Error as e:
        return jsonify({"error": f"数据库连接错误: {str(e)}"}), 500
    finally:
        if conn.is_connected():
            cursor.close()
            conn.close()


@app.route('/MoviesNowDisPlay',methods=["GET"])
def getNowDisPlay():
    try:
        # 连接数据库
        conn = mysql.connector.connect(**db_config)
        cursor = conn.cursor(dictionary=True)

        # 查询电影主表数据
        movie_query = "SELECT * FROM moviesnow"
        cursor.execute(movie_query)
        movies = cursor.fetchall()

        for movie in movies:
            movie_id = movie['id']

            # 查询该电影的导演数据
            director_query = "SELECT director FROM directorsnow WHERE movie_id = %s"
            cursor.execute(director_query, (movie_id,))
            directors = [row['director'] for row in cursor.fetchall()]
            movie['directors'] = directors

            # 查询该电影的演员数据
            actor_query = "SELECT actor, role, actor_poster FROM actorsnow WHERE movie_id = %s"
            cursor.execute(actor_query, (movie_id,))
            actors = cursor.fetchall()
            movie['actors'] = actors

        return jsonify(movies)

    except mysql.connector.Error as err:
        # 处理数据库连接和查询过程中的错误
        error_message = f"数据库错误: {err}"
        return jsonify({"error": error_message}), 500
    finally:
        if conn.is_connected():
            # 关闭游标和数据库连接
            cursor.close()
            conn.close()


@app.route('/admin/login', methods=['POST'])
def admin_login():
    try:
        # 获取请求中的 JSON 数据
        data = request.get_json()
        if not data:
            return jsonify({"error": "请求数据不能为空"}), 400

        username = data.get('username')
        password = data.get('password')

        if not username or not password:
            return jsonify({"error": "用户名和密码不能为空"}), 400

        # 连接数据库
        conn = mysql.connector.connect(**db_config)
        cursor = conn.cursor(dictionary=True)

        # 查询数据库中是否存在匹配的管理员记录
        query = "SELECT * FROM administrators WHERE username = %s AND password = %s"
        cursor.execute(query, (username, password))
        admin = cursor.fetchone()

        if admin:
            return jsonify({"message": "登录成功", "admin": admin}), 200
        else:
            return jsonify({"error": "用户名或密码错误"}), 401

    except mysql.connector.Error as err:
        # 处理数据库连接和查询过程中的错误
        error_message = f"数据库错误: {err}"
        return jsonify({"error": error_message}), 500
    finally:
        if conn.is_connected():
            # 关闭游标和数据库连接
            cursor.close()
            conn.close()



#这个端口可以拿到类似于猫眼的当日票房数据
@app.route('/Boxingsdata', methods=["GET"])
def getBoxingsData():
    try:
        with open('movie_box_office_data.json', 'r', encoding='utf-8') as file:
            data = json.load(file)
        response = jsonify(data)
        #response.headers['Content-Type'] = 'application/json; charset=utf-8'  # 设置响应头
        return response
    except FileNotFoundError:
        return jsonify({"error": "JSON 文件未找到"}), 404
    except json.JSONDecodeError:
        return jsonify({"error": "JSON 解析出错"}), 500

# 注册接口
@app.route('/register', methods=['POST'])
def register():
    data = request.get_json()
    if not data or 'username' not in data or 'password' not in data:
        return jsonify({"message": "Missing username or password"}), 400

    username = data['username']
    password = data['password']

    try:
        # 连接数据库
        conn = mysql.connector.connect(**db_config)
        cursor = conn.cursor()

        # 检查用户名是否已存在
        cursor.execute("SELECT * FROM users WHERE username = %s", (username,))
        existing_user = cursor.fetchone()
        if existing_user:
            cursor.close()
            conn.close()
            return jsonify({"message": "Username already exists"}), 409

        # 对密码进行哈希处理
        hashed_password = generate_password_hash(password)

        # 插入新用户
        cursor.execute("INSERT INTO users (username, password) VALUES (%s, %s)", (username, hashed_password))
        conn.commit()

        cursor.close()
        conn.close()
        return jsonify({"message": "User registered successfully"}), 201

    except mysql.connector.Error as err:
        return jsonify({"message": f"Registration failed: {str(err)}"}), 500


# 登录接口
@app.route('/login', methods=['POST'])
def login():
    data = request.get_json()
    if not data or 'username' not in data or 'password' not in data:
        return jsonify({"message": "Missing username or password"}), 400

    username = data['username']
    password = data['password']

    try:
        # 连接数据库
        conn = mysql.connector.connect(**db_config)
        cursor = conn.cursor()

        # 查询用户信息
        cursor.execute("SELECT password FROM users WHERE username = %s", (username,))
        result = cursor.fetchone()

        cursor.close()
        conn.close()

        print(username)
        print(password)
        print(result[0])
        if result and check_password_hash(result[0], password):
            return jsonify({"message": "Login successful"}), 200
        else:
            return jsonify({"message": "Invalid username or password"}), 401

    except mysql.connector.Error as err:
        return jsonify({"message": f"Login failed: {str(err)}"}), 500

#————————————————————————————————————————————————————————————————————————————————————————
#————————————————————————————————————————————————————————————————————————————————————————
#正在上映的电影进行增删改查
# 查询所有电影信息  返回所有信息
@app.route('/admin/nowmovies', methods=['GET'])
def get_all_movies():
    try:
        conn = mysql.connector.connect(**db_config)
        cursor = conn.cursor(dictionary=True)

        # 查询电影主表数据
        movie_query = "SELECT * FROM moviesnow"
        cursor.execute(movie_query)
        movies = cursor.fetchall()

        for movie in movies:
            movie_id = movie['id']

            # 查询该电影的导演数据
            director_query = "SELECT director FROM directorsnow WHERE movie_id = %s"
            cursor.execute(director_query, (movie_id,))
            directors = [row['director'] for row in cursor.fetchall()]
            movie['directors'] = directors

            # 查询该电影的演员数据
            actor_query = "SELECT actor, role, actor_poster FROM actorsnow WHERE movie_id = %s"
            cursor.execute(actor_query, (movie_id,))
            actors = cursor.fetchall()
            movie['actors'] = actors

        return jsonify(movies)

    except mysql.connector.Error as err:
        return jsonify({"error": f"数据库错误: {err}"}), 500
    finally:
        if conn.is_connected():
            cursor.close()
            conn.close()

# 根据 ID 查询单个电影信息
@app.route('/admin/nowmovies/query/<int:movie_id>', methods=['GET'])
def get_movie(movie_id):
    try:
        conn = mysql.connector.connect(**db_config)
        cursor = conn.cursor(dictionary=True)

        # 查询电影主表数据
        movie_query = "SELECT * FROM moviesnow WHERE id = %s"
        cursor.execute(movie_query, (movie_id,))
        movie = cursor.fetchone()

        if movie:
            # 查询该电影的导演数据
            director_query = "SELECT director FROM directorsnow WHERE movie_id = %s"
            cursor.execute(director_query, (movie_id,))
            directors = [row['director'] for row in cursor.fetchall()]
            movie['directors'] = directors

            # 查询该电影的演员数据
            actor_query = "SELECT actor, role, actor_poster FROM actorsnow WHERE movie_id = %s"
            cursor.execute(actor_query, (movie_id,))
            actors = cursor.fetchall()
            movie['actors'] = actors

            return jsonify(movie)
        else:
            return jsonify({"error": "未找到该电影"}), 404

    except mysql.connector.Error as err:
        return jsonify({"error": f"数据库错误: {err}"}), 500
    finally:
        if conn.is_connected():
            cursor.close()
            conn.close()

# 添加新电影
@app.route('/admin/nowmovies/add', methods=['POST'])
def add_movie():
    data = request.get_json()
    if not data:
        return jsonify({"error": "请求数据不能为空"}), 400

    title = data.get('title')
    release_time = data.get('release_time')
    poster = data.get('poster')
    plot = data.get('plot')
    directors_poster = data.get('directors_poster')
    score = data.get('score', '暂无评分')
    directors = data.get('directors', [])
    actors = data.get('actors', [])

    if not title:
        return jsonify({"error": "电影标题不能为空"}), 400

    try:
        conn = mysql.connector.connect(**db_config)
        cursor = conn.cursor()

        # 插入电影主表数据
        movie_query = "INSERT INTO moviesnow (title, release_time, poster, plot, directors_poster, score) VALUES (%s, %s, %s, %s, %s, %s)"
        cursor.execute(movie_query, (title, release_time, poster, plot, directors_poster, score))
        movie_id = cursor.lastrowid

        # 插入导演数据
        for director in directors:
            director_query = "INSERT INTO directorsnow (movie_id, director) VALUES (%s, %s)"
            cursor.execute(director_query, (movie_id, director))

        # 插入演员数据
        for actor in actors:
            actor_name = actor.get('actor')
            role = actor.get('role', '未知')
            actor_poster = actor.get('actor_poster')
            actor_query = "INSERT INTO actorsnow (movie_id, actor, role, actor_poster) VALUES (%s, %s, %s, %s)"
            cursor.execute(actor_query, (movie_id, actor_name, role, actor_poster))

        conn.commit()
        return jsonify({"message": "电影添加成功", "movie_id": movie_id}), 201

    except mysql.connector.Error as err:
        conn.rollback()
        return jsonify({"error": f"数据库错误: {err}"}), 500
    finally:
        if conn.is_connected():
            cursor.close()
            conn.close()
# 根据 ID 更新电影信息
@app.route('/admin/nowmovies/update/<int:movie_id>', methods=['PUT'])
def update_movie(movie_id):
    data = request.get_json()
    if not data:
        return jsonify({"error": "请求数据不能为空"}), 400

    try:
        conn = mysql.connector.connect(**db_config)
        cursor = conn.cursor()

        # 检查电影是否存在
        movie_query = "SELECT * FROM moviesnow WHERE id = %s"
        cursor.execute(movie_query, (movie_id,))
        movie = cursor.fetchone()

        if not movie:
            return jsonify({"error": "未找到该电影"}), 404

        # 构建电影主表更新语句
        update_query = "UPDATE moviesnow SET "
        values = []
        fields = {
            'title': data.get('title'),
            'release_time': data.get('release_time'),
            'poster': data.get('poster'),
            'plot': data.get('plot'),
            'directors_poster': data.get('directors_poster'),
            'score': data.get('score')
        }
        set_clauses = []
        for field, value in fields.items():
            if value is not None:
                set_clauses.append(f"{field} = %s")
                values.append(value)

        if set_clauses:
            update_query += ", ".join(set_clauses)
            update_query += " WHERE id = %s"
            values.append(movie_id)
            cursor.execute(update_query, tuple(values))

        # 处理导演信息
        if 'directors' in data:
            # 删除原有的导演数据
            delete_director_query = "DELETE FROM directorsnow WHERE movie_id = %s"
            cursor.execute(delete_director_query, (movie_id,))

            # 插入新的导演数据
            for director in data['directors']:
                director_query = "INSERT INTO directorsnow (movie_id, director) VALUES (%s, %s)"
                cursor.execute(director_query, (movie_id, director))

        # 处理演员信息
        if 'actors' in data:
            # 删除原有的演员数据
            delete_actor_query = "DELETE FROM actorsnow WHERE movie_id = %s"
            cursor.execute(delete_actor_query, (movie_id,))

            # 插入新的演员数据
            for actor in data['actors']:
                actor_name = actor.get('actor')
                role = actor.get('role', '未知')
                actor_poster = actor.get('actor_poster')
                actor_query = "INSERT INTO actorsnow (movie_id, actor, role, actor_poster) VALUES (%s, %s, %s, %s)"
                cursor.execute(actor_query, (movie_id, actor_name, role, actor_poster))

        conn.commit()
        return jsonify({"message": "电影信息更新成功"}), 200

    except mysql.connector.Error as err:
        conn.rollback()
        return jsonify({"error": f"数据库错误: {err}"}), 500
    finally:
        if conn.is_connected():
            cursor.close()
            conn.close()

# 根据 ID 删除电影信息
@app.route('/admin/nowmovies/delete/<int:movie_id>', methods=['DELETE'])
def delete_movie(movie_id):
    try:
        conn = mysql.connector.connect(**db_config)
        cursor = conn.cursor()

        # 检查电影是否存在
        movie_query = "SELECT * FROM moviesnow WHERE id = %s"
        cursor.execute(movie_query, (movie_id,))
        movie = cursor.fetchone()

        if not movie:
            return jsonify({"error": "未找到该电影"}), 404

        # 删除导演数据
        delete_director_query = "DELETE FROM directorsnow WHERE movie_id = %s"
        cursor.execute(delete_director_query, (movie_id,))

        # 删除演员数据
        delete_actor_query = "DELETE FROM actorsnow WHERE movie_id = %s"
        cursor.execute(delete_actor_query, (movie_id,))

        # 删除电影主表数据
        delete_movie_query = "DELETE FROM moviesnow WHERE id = %s"
        cursor.execute(delete_movie_query, (movie_id,))

        conn.commit()
        return jsonify({"message": "电影信息删除成功"}), 200

    except mysql.connector.Error as err:
        conn.rollback()
        return jsonify({"error": f"数据库错误: {err}"}), 500
    finally:
        if conn.is_connected():
            cursor.close()
            conn.close()

#————————————————————————————————————————————————————————————————————————————————————————
#top100的电影进行增删改查
# 查询所有电影信息
@app.route('/admin/top100movies', methods=['GET'])
def get_all_movies1():
    try:
        conn = mysql.connector.connect(**db_config)
        cursor = conn.cursor(dictionary=True)

        # 查询电影主表数据
        movie_query = "SELECT * FROM movies_top_100"
        cursor.execute(movie_query)
        movies = cursor.fetchall()

        for movie in movies:
            movie_id = movie['id']

            # 查询该电影的导演数据
            director_query = "SELECT director FROM directors_top_100 WHERE movie_id = %s"
            cursor.execute(director_query, (movie_id,))
            directors = [row['director'] for row in cursor.fetchall()]
            movie['directors'] = directors

            # 查询该电影的演员数据
            actor_query = "SELECT actor, role FROM actors_top_100 WHERE movie_id = %s"
            cursor.execute(actor_query, (movie_id,))
            actors = cursor.fetchall()
            movie['actors'] = actors

        return jsonify(movies)

    except mysql.connector.Error as err:
        return jsonify({"error": f"数据库错误: {err}"}), 500
    finally:
        if conn.is_connected():
            cursor.close()
            conn.close()

# 根据 ID 查询单个电影信息
@app.route('/admin/top100movies/query/<int:movie_id>', methods=['GET'])
def get_movie1(movie_id):
    try:
        conn = mysql.connector.connect(**db_config)
        cursor = conn.cursor(dictionary=True)

        # 查询电影主表数据
        movie_query = "SELECT * FROM movies_top_100 WHERE id = %s"
        cursor.execute(movie_query, (movie_id,))
        movie = cursor.fetchone()

        if movie:
            # 查询该电影的导演数据
            director_query = "SELECT director FROM directors_top_100 WHERE movie_id = %s"
            cursor.execute(director_query, (movie_id,))
            directors = [row['director'] for row in cursor.fetchall()]
            movie['directors'] = directors

            # 查询该电影的演员数据
            actor_query = "SELECT actor, role FROM actors_top_100 WHERE movie_id = %s"
            cursor.execute(actor_query, (movie_id,))
            actors = cursor.fetchall()
            movie['actors'] = actors

            return jsonify(movie)
        else:
            return jsonify({"error": "未找到该电影"}), 404

    except mysql.connector.Error as err:
        return jsonify({"error": f"数据库错误: {err}"}), 500
    finally:
        if conn.is_connected():
            cursor.close()
            conn.close()

# 添加新电影
@app.route('/admin/top100movies/add', methods=['POST'])
def add_movie1():
    data = request.get_json()
    if not data:
        return jsonify({"error": "请求数据不能为空"}), 400

    title = data.get('title')
    release_time = data.get('release_time')
    whis_count = data.get('whis_count')
    poster = data.get('poster')
    plot = data.get('plot')
    directors_poster = data.get('directors_poster')
    directors = data.get('directors', [])
    actors = data.get('actors', [])

    if not title:
        return jsonify({"error": "电影标题不能为空"}), 400

    try:
        conn = mysql.connector.connect(**db_config)
        cursor = conn.cursor()

        # 插入电影主表数据
        movie_query = "INSERT INTO movies_top_100 (title, release_time, whis_count, poster, plot,directors_poster) VALUES (%s, %s, %s, %s, %s,%s)"
        cursor.execute(movie_query, (title, release_time, whis_count, poster, plot,directors_poster))
        movie_id = cursor.lastrowid

        # 插入导演数据
        for director in directors:
            director_query = "INSERT INTO directors_top_100 (movie_id, director) VALUES (%s, %s)"
            cursor.execute(director_query, (movie_id, director))

        # 插入演员数据
        for actor in actors:
            actor_name = actor.get('actor')
            role = actor.get('role', '未知')
            actor_query = "INSERT INTO actors_top_100 (movie_id, actor, role) VALUES (%s, %s, %s)"
            cursor.execute(actor_query, (movie_id, actor_name, role))

        conn.commit()
        return jsonify({"message": "电影添加成功", "movie_id": movie_id}), 201

    except mysql.connector.Error as err:
        conn.rollback()
        return jsonify({"error": f"数据库错误: {err}"}), 500
    finally:
        if conn.is_connected():
            cursor.close()
            conn.close()

# 根据 ID 更新电影信息
@app.route('/admin/top100movies/update/<int:movie_id>', methods=['PUT'])
def update_movie1(movie_id):
    data = request.get_json()
    if not data:
        return jsonify({"error": "请求数据不能为空"}), 400

    try:
        conn = mysql.connector.connect(**db_config)
        cursor = conn.cursor()

        # 检查电影是否存在
        movie_query = "SELECT * FROM movies_top_100 WHERE id = %s"
        cursor.execute(movie_query, (movie_id,))
        movie = cursor.fetchone()

        if not movie:
            return jsonify({"error": "未找到该电影"}), 404

        # 构建电影主表更新语句
        update_query = "UPDATE movies_top_100 SET "
        values = []
        fields = {
            'title': data.get('title'),
            'release_time': data.get('release_time'),
            'whis_count': data.get('whis_count'),
            'poster': data.get('poster'),
            'plot': data.get('plot')
        }
        set_clauses = []
        for field, value in fields.items():
            if value is not None:
                set_clauses.append(f"{field} = %s")
                values.append(value)

        if set_clauses:
            update_query += ", ".join(set_clauses)
            update_query += " WHERE id = %s"
            values.append(movie_id)
            cursor.execute(update_query, tuple(values))

        # 处理导演信息
        if 'directors' in data:
            # 删除原有的导演数据
            delete_director_query = "DELETE FROM directors_top_100 WHERE movie_id = %s"
            cursor.execute(delete_director_query, (movie_id,))
            # 插入新的导演数据
            for director in data['directors']:
                director_query = "INSERT INTO directors_top_100 (movie_id, director) VALUES (%s, %s)"
                cursor.execute(director_query, (movie_id, director))

        # 处理演员信息
        if 'actors' in data:
            # 删除原有的演员数据
            delete_actor_query = "DELETE FROM actors_top_100 WHERE movie_id = %s"
            cursor.execute(delete_actor_query, (movie_id,))
            # 插入新的演员数据
            for actor in data['actors']:
                actor_name = actor.get('actor')
                role = actor.get('role', '未知')
                actor_query = "INSERT INTO actors_top_100 (movie_id, actor, role) VALUES (%s, %s, %s)"
                cursor.execute(actor_query, (movie_id, actor_name, role))

        conn.commit()
        return jsonify({"message": "电影信息更新成功"}), 200

    except mysql.connector.Error as err:
        conn.rollback()
        return jsonify({"error": f"数据库错误: {err}"}), 500
    finally:
        if conn.is_connected():
            cursor.close()
            conn.close()

# 根据 ID 删除电影信息
@app.route('/admin/top100movies/delete/<int:movie_id>', methods=['DELETE'])
def delete_movie1(movie_id):
    try:
        conn = mysql.connector.connect(**db_config)
        cursor = conn.cursor()

        # 检查电影是否存在
        movie_query = "SELECT * FROM movies_top_100 WHERE id = %s"
        cursor.execute(movie_query, (movie_id,))
        movie = cursor.fetchone()

        if not movie:
            return jsonify({"error": "未找到该电影"}), 404

        # 删除导演数据
        delete_director_query = "DELETE FROM directors_top_100 WHERE movie_id = %s"
        cursor.execute(delete_director_query, (movie_id,))

        # 删除演员数据
        delete_actor_query = "DELETE FROM actors_top_100 WHERE movie_id = %s"
        cursor.execute(delete_actor_query, (movie_id,))

        # 删除电影主表数据
        delete_movie_query = "DELETE FROM movies_top_100 WHERE id = %s"
        cursor.execute(delete_movie_query, (movie_id,))

        conn.commit()
        return jsonify({"message": "电影信息删除成功"}), 200

    except mysql.connector.Error as err:
        conn.rollback()
        return jsonify({"error": f"数据库错误: {err}"}), 500
    finally:
        if conn.is_connected():
            cursor.close()
            conn.close()
#————————————————————————————————————————————————————————————————————————————————————
#即将上映的电影进行增删改查
# 查询所有电影信息
@app.route('/admin/willmovies', methods=['GET'])
def get_all_movies2():
    try:
        conn = mysql.connector.connect(**db_config)
        cursor = conn.cursor(dictionary=True)

        # 查询电影主表数据
        movie_query = "SELECT * FROM movieswill"
        cursor.execute(movie_query)
        movies = cursor.fetchall()

        for movie in movies:
            movie_id = movie['id']

            # 查询该电影的导演数据
            director_query = "SELECT director FROM directorswill WHERE movie_id = %s"
            cursor.execute(director_query, (movie_id,))
            directors = [row['director'] for row in cursor.fetchall()]
            movie['directors'] = directors

            # 查询该电影的演员数据
            actor_query = "SELECT actor, role FROM actorswill WHERE movie_id = %s"
            cursor.execute(actor_query, (movie_id,))
            actors = cursor.fetchall()
            movie['actors'] = actors

        return jsonify(movies)

    except mysql.connector.Error as err:
        return jsonify({"error": f"数据库错误: {err}"}), 500
    finally:
        if conn.is_connected():
            cursor.close()
            conn.close()

# 根据 ID 查询单个电影信息
@app.route('/admin/willmovies/query/<int:movie_id>', methods=['GET'])
def get_movie2(movie_id):
    try:
        conn = mysql.connector.connect(**db_config)
        cursor = conn.cursor(dictionary=True)

        # 查询电影主表数据
        movie_query = "SELECT * FROM movieswill WHERE id = %s"
        cursor.execute(movie_query, (movie_id,))
        movie = cursor.fetchone()

        if movie:
            # 查询该电影的导演数据
            director_query = "SELECT director FROM directorswill WHERE movie_id = %s"
            cursor.execute(director_query, (movie_id,))
            directors = [row['director'] for row in cursor.fetchall()]
            movie['directors'] = directors

            # 查询该电影的演员数据
            actor_query = "SELECT actor, role FROM actorswill WHERE movie_id = %s"
            cursor.execute(actor_query, (movie_id,))
            actors = cursor.fetchall()
            movie['actors'] = actors

            return jsonify(movie)
        else:
            return jsonify({"error": "未找到该电影"}), 404

    except mysql.connector.Error as err:
        return jsonify({"error": f"数据库错误: {err}"}), 500
    finally:
        if conn.is_connected():
            cursor.close()
            conn.close()

# 添加新电影
@app.route('/admin/willmovies/add', methods=['POST'])
def add_movie2():
    data = request.get_json()
    if not data:
        return jsonify({"error": "请求数据不能为空"}), 400

    title = data.get('title')
    release_time = data.get('release_time')
    whis_count = data.get('whis_count')
    poster = data.get('poster')
    plot = data.get('plot')
    directors_poster = data.get('directors_poster')
    directors = data.get('directors', [])
    actors = data.get('actors', [])

    if not title:
        return jsonify({"error": "电影标题不能为空"}), 400

    try:
        conn = mysql.connector.connect(**db_config)
        cursor = conn.cursor()

        # 插入电影主表数据
        movie_query = "INSERT INTO movieswill (title, release_time, whis_count, poster, plot,directors_poster) VALUES (%s, %s, %s, %s, %s,%s)"
        cursor.execute(movie_query, (title, release_time, whis_count, poster, plot,directors_poster))
        movie_id = cursor.lastrowid

        # 插入导演数据
        for director in directors:
            director_query = "INSERT INTO directorswill (movie_id, director) VALUES (%s, %s)"
            cursor.execute(director_query, (movie_id, director))

        # 插入演员数据
        for actor in actors:
            actor_name = actor.get('actor')
            role = actor.get('role', '未知')
            actor_query = "INSERT INTO actorswill (movie_id, actor, role) VALUES (%s, %s, %s)"
            cursor.execute(actor_query, (movie_id, actor_name, role))

        conn.commit()
        return jsonify({"message": "电影添加成功", "movie_id": movie_id}), 201

    except mysql.connector.Error as err:
        conn.rollback()
        return jsonify({"error": f"数据库错误: {err}"}), 500
    finally:
        if conn.is_connected():
            cursor.close()
            conn.close()

# 根据 ID 更新电影信息
@app.route('/admin/willmovies/update/<int:movie_id>', methods=['PUT'])
def update_movie2(movie_id):
    data = request.get_json()
    if not data:
        return jsonify({"error": "请求数据不能为空"}), 400

    try:
        conn = mysql.connector.connect(**db_config)
        cursor = conn.cursor()

        # 检查电影是否存在
        movie_query = "SELECT * FROM movieswill WHERE id = %s"
        cursor.execute(movie_query, (movie_id,))
        movie = cursor.fetchone()

        if not movie:
            return jsonify({"error": "未找到该电影"}), 404

        # 构建电影主表更新语句
        update_query = "UPDATE movieswill SET "
        values = []
        fields = {
            'title': data.get('title'),
            'release_time': data.get('release_time'),
            'whis_count': data.get('whis_count'),
            'poster': data.get('poster'),
            'plot': data.get('plot')
        }
        set_clauses = []
        for field, value in fields.items():
            if value is not None:
                set_clauses.append(f"{field} = %s")
                values.append(value)

        if set_clauses:
            update_query += ", ".join(set_clauses)
            update_query += " WHERE id = %s"
            values.append(movie_id)
            cursor.execute(update_query, tuple(values))

        # 处理导演信息
        if 'directors' in data:
            # 删除原有的导演数据
            delete_director_query = "DELETE FROM directorswill WHERE movie_id = %s"
            cursor.execute(delete_director_query, (movie_id,))
            # 插入新的导演数据
            for director in data['directors']:
                director_query = "INSERT INTO directorswill (movie_id, director) VALUES (%s, %s)"
                cursor.execute(director_query, (movie_id, director))

        # 处理演员信息
        if 'actors' in data:
            # 删除原有的演员数据
            delete_actor_query = "DELETE FROM actorswill WHERE movie_id = %s"
            cursor.execute(delete_actor_query, (movie_id,))
            # 插入新的演员数据
            for actor in data['actors']:
                actor_name = actor.get('actor')
                role = actor.get('role', '未知')
                actor_query = "INSERT INTO actorswill (movie_id, actor, role) VALUES (%s, %s, %s)"
                cursor.execute(actor_query, (movie_id, actor_name, role))

        conn.commit()
        return jsonify({"message": "电影信息更新成功"}), 200

    except mysql.connector.Error as err:
        conn.rollback()
        return jsonify({"error": f"数据库错误: {err}"}), 500
    finally:
        if conn.is_connected():
            cursor.close()
            conn.close()

# 根据 ID 删除电影信息
@app.route('/admin/willmovies/delete/<int:movie_id>', methods=['DELETE'])
def delete_movie2(movie_id):
    try:
        conn = mysql.connector.connect(**db_config)
        cursor = conn.cursor()

        # 检查电影是否存在
        movie_query = "SELECT * FROM movieswill WHERE id = %s"
        cursor.execute(movie_query, (movie_id,))
        movie = cursor.fetchone()

        if not movie:
            return jsonify({"error": "未找到该电影"}), 404

        # 删除导演数据
        delete_director_query = "DELETE FROM directorswill WHERE movie_id = %s"
        cursor.execute(delete_director_query, (movie_id,))

        # 删除演员数据
        delete_actor_query = "DELETE FROM actorswill WHERE movie_id = %s"
        cursor.execute(delete_actor_query, (movie_id,))

        # 删除电影主表数据
        delete_movie_query = "DELETE FROM movieswill WHERE id = %s"
        cursor.execute(delete_movie_query, (movie_id,))

        conn.commit()
        return jsonify({"message": "电影信息删除成功"}), 200

    except mysql.connector.Error as err:
        conn.rollback()
        return jsonify({"error": f"数据库错误: {err}"}), 500
    finally:
        if conn.is_connected():
            cursor.close()
            conn.close()
#————————————————————————————————————————————————————————————————————————————————————————
if __name__ == '__main__':
    #运行前请配置以下鉴权三要素，获取途径：https://console.xfyun.cn/services/tti
    app.run(debug=True)


    
    
    
    