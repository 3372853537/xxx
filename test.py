import requests
import base64
# 后端接口地址（确保与Flask路由一致）
url = "http://127.0.0.1:5000/Rawpredict"

# 测试数据（与后端期望的JSON结构一致）
data = {
    "name": "学链科技纪录片",
    "budget": 8000000,  # 故意设为小于预测值，触发纯JSON返回
    "runtime": 90,
    "genres": "Documentary",
    "language": "普通话",
    "keyword1": "learning",
    "keyword2": "technology",
    "keyword3": "campus",
    "totalVotes": 1200.5,
    "release_date_year": 2024,
    "rating": 7.8,
    "overview": "讲述大学生在学链科技实习的成长故事"
}

url = "http://127.0.0.1:5000/Rawpredict"
headers = {"Content-Type": "application/json"}

# 测试预测值大于预算的情况
response_1 = requests.post(url, json=data, headers=headers)

print(response_1.text)
print("测试预测值大于预算的情况：")
if response_1.status_code == 200:
    if response_1.headers.get('Content-Type') == 'image/jpeg':
        with open('output_1.jpg', 'wb') as f:
            f.write(response_1.content)
        print("图片已保存为 output_1.jpg")
        print("预测结果：", response_1.headers.get('Prediction'))
        # 获取 Base64 编码的 aiMessage
        encoded_ai_message = response_1.headers.get('aiMessage')
        if encoded_ai_message:
            # 解码 aiMessage
            ai_message = base64.b64decode(encoded_ai_message).decode('utf-8')
        print("返回的文字：", ai_message)
    else:
        print("响应类型不是图片，可能存在问题。")
else:
    print(f"请求失败，状态码: {response_1.status_code}，响应内容: {response_1.text}")