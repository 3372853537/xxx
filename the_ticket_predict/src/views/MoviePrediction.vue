<template>
  <div class="prediction-container">
    <div class="main-layout">
      <!-- 左侧面板 -->
      <div class="left-panel">
        <div class="button-group">
          <button 
            :class="['control-btn', { active: predictionType === 'released' }]"
            @click="predictionType = 'released'">
            已上映电影
          </button>
          <button 
            :class="['control-btn', { active: predictionType === 'upcoming' }]"
            @click="predictionType = 'upcoming'">
            未上映电影
          </button>
          <button 
            :class="['control-btn', { active: showWeightImage }]"
            @click="showWeightImage = !showWeightImage">
            特征权重图
          </button>
        </div>

        <div class="prediction-content">
          <!-- 已上映电影预测 -->
          <div v-if="predictionType === 'released'" class="prediction-form-container">
            <div class="form-group">
              <label>选择电影</label>
              <select v-model="selectedMovie" @change="updateMovieDetails">
                <option value="">请选择电影</option>
                <option v-for="movie in boxOfficeData" 
                        :key="movie['影片']" 
                        :value="movie['影片']">
                  {{ movie['影片'] }}
                </option>
              </select>
            </div>
            <!-- 其他表单元素 -->
            <div class="form-group">
              <label>预测天数</label>
              <select v-model="predictionDays">
                <option v-for="day in availableDays" 
                        :key="day" 
                        :value="day">
                  {{ day }}天
                </option>
              </select>
            </div>
            <button class="predict-btn" @click="predict" :disabled="!selectedMovie">
              开始预测
            </button>
          </div>

          <!-- 未上映电影预测 -->
          <div v-if="predictionType === 'upcoming'" class="prediction-form-container">
            <div class="form-grid">
              <div class="form-group">
                <label>电影名称</label>
                <input type="text" v-model="newMovie.name">
              </div>
              <div class="form-group">
                <label>预算 (万元)</label>
                <input type="number" v-model="newMovie.budget">
              </div>
              <div class="form-group">
                <label>片长 (分钟)</label>
                <input type="number" v-model="newMovie.runtime">
              </div>
              <div class="form-group">
                <label>类型</label>
                <select v-model="newMovie.genres">
                  <option value="Action">动作</option>
                  <option value="Comedy">喜剧</option>
                  <option value="Drama">剧情</option>
                  <!-- 添加更多类型选项 -->
                </select>
              </div>
              <div class="form-group">
                <label>语言</label>
                <select v-model="newMovie.language">
                  <option value="cn">中文</option>
                  <option value="en">英语</option>
                  <!-- 添加更多语言选项 -->
                </select>
              </div>
              <div class="form-group">
                <label>关键词1</label>
                <input type="text" v-model="newMovie.keyword1">
              </div>
              <div class="form-group">
                <label>关键词2</label>
                <input type="text" v-model="newMovie.keyword2">
              </div>
              <div class="form-group">
                <label>关键词3</label>
                <input type="text" v-model="newMovie.keyword3">
              </div>
              <div class="form-group">
                <label>讨论热度</label>
                <input type="number" v-model="newMovie.totalVotes">
              </div>
              <div class="form-group">
                <label>上映年份</label>
                <input type="number" v-model="newMovie.release_date_year">
              </div>
              <div class="form-group">
                <label>评分</label>
                <input type="number" v-model="newMovie.rating" step="0.1" min="0" max="10">
              </div>
              <div class="form-group">
                <label>电影简介</label>
                <textarea v-model="newMovie.overview" rows="4" class="form-textarea"></textarea>
              </div>
            </div>
            <button class="predict-btn" @click="predictNewMovie">预测票房</button>
          </div>
        </div>
      </div>

      <!-- 右侧结果展示区域 -->
      <div class="right-panel">
        <!-- 权重图显示区域 -->
        <div class="results-content" v-if="showWeightImage">
          <h4>特征权重分布图</h4>
          <img src=../assets/controlrate.png alt="特征权重分布" class="weight-image">
        </div>

        <!-- 原有的预测结果显示 -->
        <div class="results-content" v-else-if="showResults">
          <!-- 已上映电影预测结果 -->
          <div v-if="predictionImage && predictionType === 'released'" class="prediction-result">
            <h4>预测结果</h4>
            <img :src="predictionImage" alt="票房预测图" class="prediction-chart">
          </div>

          <!-- 未上映电影预测结果 -->
          <div v-if="predictionType === 'upcoming' && (rawPredictionResult || moviePoster)" class="prediction-result">
            <h4>预测结果</h4>
            <div v-if="rawPredictionResult" class="prediction-value">
              预测票房：{{ formatPrediction(rawPredictionResult) }} 美元
            </div>
            <div v-if="moviePoster" class="movie-poster">
              <h5>AI 生成的电影海报</h5>
              <img :src="moviePoster" alt="Movie Poster" class="poster-image">
            </div>
            <div v-if="aiMessage" class="ai-suggestion">
              <h5>AI 建议</h5>
              <p>{{ aiMessage }}</p>
            </div>
          </div>
        </div>
      </div>
    </div>
  </div>
</template>

<script>
import axios from "axios";

export default {
  name: "MoviePrediction",
  data() {
    return {
      currentDate: new Date().toLocaleDateString('zh-CN', {
        year: 'numeric',
        month: '2-digit',
        day: '2-digit',
        hour: '2-digit',
        minute: '2-digit'
      }),
      selectedMovie: '',
      boxOfficeData: [],
      selectedMovieDetails: null,
      predictionDays: 7,
      availableDays: [7, 14, 21],
      predictionImage: null,
      predictionType: 'released',
      newMovie: {
        budget: 0,
        runtime: 0,
        genres: '',
        language: '',
        keyword1: '',
        keyword2: '',
        keyword3: '',
        totalVotes: 0,
        release_date_year: new Date().getFullYear(),
        rating: 0,
        overview: '',  // 添加电影简介字段
        name: ''      // 添加电影名称字段
      },
      rawPredictionResult: null,
      aiMessage: null,      // 存储 AI 建议
      moviePoster: null,    // 存储电影海报
      showResults: false,    // 添加这行
      showWeightImage: false,
      weightImageUrl: '/path/to/weight/image.png' // 替换为实际的权重图路径
    };
  },
  mounted() {
    this.fetchBoxOfficeData();
  },
  methods: {
    fetchBoxOfficeData() {
      axios.get("http://127.0.0.1:5000/Boxingsdata")
        .then(response => {
          this.boxOfficeData = response.data;
        })
        .catch(error => {
          console.error("Error fetching box office data:", error);
        });
    },
    goToDetail(movie) {
      this.$router.push({
        name: "MovieDetail",
        params: { title: movie.title }
      });
    },
    updateMovieDetails() {
      if (this.selectedMovie) {
        this.selectedMovieDetails = this.boxOfficeData.find(
          movie => movie['影片'] === this.selectedMovie
        );
      } else {
        this.selectedMovieDetails = null;
      }
    },
    async predict() {
      if (!this.selectedMovie) {
        alert('请先选择电影');
        return;
      }

      try {
        const response = await axios.post('http://127.0.0.1:5000/predict', {
          name: this.selectedMovie,
          step: this.predictionDays,
          flag: false
        }, {
          responseType: 'blob'
        });

        // 创建图片URL
        const imageUrl = URL.createObjectURL(response.data);
        this.predictionImage = imageUrl;
        this.showResults = true;  // 添加这行
      } catch (error) {
        console.error('预测失败:', error.response || error);
        // 如果错误响应为 Blob，则读取错误详情
        if (error.response && error.response.data instanceof Blob) {
          const reader = new FileReader();
          reader.onload = () => {
            alert(`预测失败: ${reader.result}`);
          };
          reader.readAsText(error.response.data);
        } else {
          let errorMsg = '预测失败，请稍后重试';
          if (error.response && error.response.data && typeof error.response.data === 'string') {
            errorMsg = error.response.data;
          }
          alert(errorMsg);
        }
      }
    },
    async predictNewMovie() {
      try {
        const response = await axios.post('http://127.0.0.1:5000/Rawpredict', this.newMovie, {
          responseType: 'blob'
        });
        console.log("Response Headers:", response.headers);
        const prediction = response.headers['prediction'];
        const encodedAiMessage = response.headers['aimessage'];

        if (prediction) {
          this.rawPredictionResult = Number(prediction);
        }
        if (encodedAiMessage) {
          // 调用 this.b64DecodeUnicode 而非直接调用 b64DecodeUnicode
          this.aiMessage = this.b64DecodeUnicode(encodedAiMessage);
        }
        this.moviePoster = URL.createObjectURL(response.data);
        this.showResults = true;  // 添加这行
      } catch (error) {
        console.error('预测失败:', error);
        alert('预测失败，请稍后重试');
      }
    },
    formatPrediction(value) {
      return Number(value).toLocaleString('zh-CN', {
        maximumFractionDigits: 2
      });
    },
    // 辅助函数：正确解码 Base64 的 Unicode 字符串
    b64DecodeUnicode(str) {
      return decodeURIComponent(Array.prototype.map.call(atob(str), c =>
        '%' + ('00' + c.charCodeAt(0).toString(16)).slice(-2)
      ).join(''));
    }
  }
};
</script>

<style scoped>
.prediction-container {
  padding: 24px;
  background: #f5f5f5;
  min-height: 100vh;
}

.prediction-header {
  text-align: center;
  margin-bottom: 32px;
}

/* 修改预测类型选择器样式 */
.prediction-type-selector {
  display: flex;
  flex-direction: column;
  gap: 12px;
  margin-bottom: 24px;
}

.type-btn {
  width: 100%;
  padding: 12px;
}

.main-layout {
  display: flex;
  gap: 24px;
  max-width: 1600px;
  margin: 0 auto;
  padding: 0 20px;
}

/* 修改左侧面板样式 */
.left-panel {
  flex: 0 0 45%;  /* 增加左侧面板的宽度占比 */
  max-width: 800px;  /* 增加最大宽度 */
  display: flex;
  flex-direction: column;
  gap: 20px;
}

/* 修改右侧面板样式 */
.right-panel {
  flex: 0 0 770px; /* 固定宽度 */
  min-height: 500px; /* 最小高度 */
  height: calc(100vh - 100px); /* 固定高度，减去上下边距 */
  position: sticky;
  top: 24px;
}

.results-content {
  background: white;
  border-radius: 12px;
  padding: 32px;
  box-shadow: 0 2px 12px rgba(0,0,0,0.1);
  height: 100%;
  overflow-y: auto; /* 内容过多时可滚动 */
}

.weight-image,
.prediction-chart,
.movie-poster img {
  max-width: 100%;
  object-fit: contain;
  margin: 0 auto;
  display: block;
}

.prediction-content {
  background: white;
  border-radius: 12px;
  padding: 32px;
  box-shadow: 0 2px 12px rgba(0,0,0,0.1);
  margin: 0;
}

.prediction-form-container {
  max-width: none;
  margin: 0;
}

.form-grid {
  display: grid;
  grid-template-columns: repeat(2, 1fr);  /* 改为两列布局 */
  gap: 24px;  /* 增加间距 */
  margin-bottom: 24px;
}

.form-group label {
  font-size: 16px;  /* 增大标签字体 */
  margin-bottom: 10px;
}

.form-group input,
.form-group select {
  padding: 12px;  /* 增加输入框内边距 */
  font-size: 15px;  /* 增大输入框字体 */
}

.form-textarea {
  grid-column: 1 / -1;  /* 文本框占据整行 */
  min-height: 120px;  /* 增加文本框高度 */
}

.prediction-content {
  padding: 32px;  /* 增加内容区域内边距 */
}

.page-title {
  text-align: center;
  color: #333;
  margin-bottom: 30px;
}

.dashboard-container {
  display: flex;
  padding: 20px;
  background: #f5f5f5;
  min-height: 100vh;
  gap: 20px;
}

.main-box-office {
  flex: 1;
  background: #fff;
  border-radius: 8px;
  box-shadow: 0 2px 12px rgba(0, 0, 0, 0.1);
  padding: 20px;
  overflow-y: auto;
  height: calc(100vh - 40px);
}

.header-section {
  display: flex;
  justify-content: space-between;
  align-items: center;
  margin-bottom: 20px;
  padding: 0 10px;
}

.section-title {
  font-size: 24px;
  color: #333;
  font-weight: bold;
}

.date-info {
  color: #666;
  font-size: 14px;
}

.movie-table {
  width: 100%;
  border: 1px solid #eee;
  border-radius: 4px;
}

.table-header {
  background: #f7f7f7;
  font-weight: bold;
}

.table-header, .table-row {
  display: flex;
  align-items: center;
  border-bottom: 1px solid #eee;
}

.table-row:hover {
  background: #f9f9f9;
}

.table-cell {
  padding: 15px 10px;
  font-size: 14px;
}

.rank { width: 8%; }
.movie-name { width: 25%; }
.box-office { width: 15%; }
.percent { width: 12%; }
.shows { width: 15%; }
.days { width: 10%; }
.total { width: 15%; }

.rank-num {
  display: inline-block;
  width: 24px;
  height: 24px;
  line-height: 24px;
  text-align: center;
  border-radius: 4px;
}

.top-three {
  background: #ff5733;
  color: white;
}

.side-panel {
  width: 350px; 
  background: #fff;
  border-radius: 8px;
  box-shadow: 0 2px 12px rgba(0, 0, 0, 0.1);
  padding: 20px;
  position: sticky;
  top: 20px;
  height: fit-content;
  max-height: calc(100vh - 40px);
  overflow-y: auto;
}

.prediction-panel {
  margin-bottom: 20px;
}

.panel-title {
  font-size: 18px;
  color: #333;
  margin-bottom: 15px;
  padding-bottom: 10px;
  border-bottom: 1px solid #eee;
}

.prediction-form {
  padding: 15px 0;
}

.form-group {
  margin-bottom: 15px;
}

.form-group label {
  display: block;
  margin-bottom: 8px;
  color: #666;
}

.form-group select {
  width: 100%;
  padding: 8px;
  border: 1px solid #ddd;
  border-radius: 4px;
}

.predict-btn {
  width: 100%;
  padding: 12px;
  background: #ff5733;
  color: white;
  border: none;
  border-radius: 4px;
  cursor: pointer;
  font-size: 16px;
  transition: background 0.3s;
}

.predict-btn:hover {
  background: #ff4719;
}

.predict-btn {
  margin-top: 24px;
  height: 48px;
  font-size: 18px;
}

.movie-detail-panel {
  margin-top: 20px;
  padding: 20px;
  background: #fff;
  border-radius: 8px;
  box-shadow: 0 2px 12px rgba(0, 0, 0, 0.1);
}

.movie-detail-content {
  margin-top: 15px;
}

.detail-item {
  display: flex;
  justify-content: space-between;
  margin-bottom: 12px;
  padding: 8px 0;
  border-bottom: 1px solid #f0f0f0;
}

.detail-label {
  color: #666;
  font-size: 14px;
}

.detail-value {
  color: #333;
  font-size: 14px;
  font-weight: 500;
}

.detail-value.highlight {
  color: #ff5733;
  font-weight: bold;
}

/* 修改表格行悬停效果 */
.table-row {
  cursor: pointer;
}

.table-row:hover {
  background: #fff5f2;
}

/* 调整选择框样式 */
.form-group select {
  width: 100%;
  padding: 10px;
  border: 1px solid #ddd;
  border-radius: 4px;
  font-size: 14px;
  color: #333;
  cursor: pointer;
}

.form-group select:focus {
  border-color: #ff5733;
  outline: none;
}

/* 移除所有图表相关样式 */
.chart-prediction-panel,
.search-form,
.chart-container {
  display: none;
}

/* 其他样式保持不变 */

/* 添加预测相关样式 */
.prediction-result {
  margin-top: 0;
  padding: 0;
  border-top: none;
}

.prediction-chart {
  max-width: 100%;
  margin-top: 20px;
}

.predict-btn {
  margin-top: 15px;
}

.predict-btn:disabled {
  background: #ccc;
  cursor: not-allowed;
}

.prediction-type {
  margin-bottom: 20px;
}

.prediction-type select {
  width: 100%;
  padding: 10px;
  border: 1px solid #ddd;
  border-radius: 4px;
  font-size: 14px;
}

.prediction-value {
  margin-top: 20px;
  font-size: 24px;
}

.form-group input {
  width: 100%;
  padding: 8px;
  border: 1px solid #ddd;
  border-radius: 4px;
  font-size: 14px;
}

.form-group input:focus {
  border-color: #ff5733;
  outline: none;
}

.form-textarea {
  width: 100%;
  padding: 8px;
  border: 1px solid #ddd;
  border-radius: 4px;
  font-size: 14px;
  resize: vertical;
  min-height: 100px;
}

.movie-poster {
  margin-top: 30px;
  text-align: center;
}

.movie-poster img {
  max-width: 100%;
  height: auto;
  border-radius: 8px;
  box-shadow: 0 2px 8px rgba(0,0,0,0.1);
}

.ai-suggestion {
  margin-top: 30px;
  padding: 15px;
  background: #f9f9f9;
  border-radius: 8px;
}

.ai-suggestion h5 {
  color: #333;
  margin-bottom: 10px;
}

.ai-suggestion p {
  color: #666;
  line-height: 1.6;
  white-space: pre-line;
}

/* 添加权重图按钮样式 */
.weight-btn {
  width: 100%;
  padding: 12px;
  background: #4CAF50;
  color: white;
  border: none;
  border-radius: 4px;
  cursor: pointer;
  margin-top: 20px;
  font-size: 16px;
  transition: background 0.3s;
}

.weight-btn:hover {
  background: #45a049;
}

.weight-image {
  width: 100%;
  height: auto;
  border-radius: 8px;
  box-shadow: 0 2px 8px rgba(0,0,0,0.1);
  margin-top: 20px;
}

/* 修改按钮组样式 */
.button-group {
  display: grid;
  grid-template-columns: repeat(3, 1fr);
  gap: 12px;
  margin-bottom: 24px;
  padding: 16px;
  background: white;
  border-radius: 12px;
  box-shadow: 0 2px 12px rgba(0,0,0,0.1);
}

.control-btn {
  padding: 12px 24px;
  border: 2px solid #ff5733;
  border-radius: 8px;
  background: white;
  color: #ff5733;
  font-size: 16px;
  cursor: pointer;
  transition: all 0.3s ease;
  height: 48px;
  display: flex;
  align-items: center;
  justify-content: center;
  white-space: nowrap;
  font-weight: 500;
}

.control-btn:hover {
  background: #fff5f2;
  transform: translateY(-1px);
  box-shadow: 0 2px 8px rgba(255, 87, 51, 0.2);
}

.control-btn.active {
  background: #ff5733;
  color: white;
  box-shadow: 0 2px 8px rgba(255, 87, 51, 0.3);
}

/* 删除之前的按钮相关样式 */
.prediction-type-selector,
.type-btn,
.weight-btn {
  display: none;
}
</style>
