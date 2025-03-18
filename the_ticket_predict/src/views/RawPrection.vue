<template>
  <div class="raw-prediction-container">
    <div class="prediction-card">
      <h2>电影预测结果</h2>
      
      <!-- 电影基本信息 -->
      <div class="movie-info">
        <h3>{{ movieData.name }}</h3>
        <div class="info-grid">
          <div class="info-item">
            <label>预算:</label>
            <span>{{ movieData.budget }}万元</span>
          </div>
          <div class="info-item">
            <label>片长:</label>
            <span>{{ movieData.runtime }}分钟</span>
          </div>
          <div class="info-item">
            <label>类型:</label>
            <span>{{ movieData.genres }}</span>
          </div>
          <div class="info-item">
            <label>语言:</label>
            <span>{{ movieData.language }}</span>
          </div>
        </div>
      </div>

      <!-- 加载动画 -->
      <div v-if="loading" class="loading">
        <div class="spinner"></div>
        <p>正在生成预测结果...</p>
      </div>

      <!-- 预测结果 -->
      <div v-else class="prediction-results">
        <div v-if="predictionResult" class="result-section">
          <h3>票房预测</h3>
          <div class="prediction-value">
            {{ formatPrediction(predictionResult) }} 美元
          </div>
        </div>

        <div v-if="moviePoster" class="poster-section">
          <h3>AI 生成的电影海报</h3>
          <img :src="moviePoster" alt="Movie Poster" class="poster-image">
        </div>

        <div v-if="aiSuggestion" class="suggestion-section">
          <h3>AI 建议</h3>
          <p>{{ aiSuggestion }}</p>
        </div>
      </div>

      <!-- 返回按钮 -->
      <button class="back-button" @click="goBack">返回预测页面</button>
    </div>
  </div>
</template>

<script>
import axios from 'axios';

export default {
  name: 'RawPrection',
  data() {
    return {
      loading: true,
      movieData: {},
      predictionResult: null,
      moviePoster: null,
      aiSuggestion: null
    };
  },
  created() {
    this.movieData = this.$route.query;
    this.getPrediction();
  },
  methods: {
    async getPrediction() {
      try {
        const response = await axios.post('http://127.0.0.1:5000/Rawpredict', 
          this.movieData,
          { responseType: 'blob' }
        );

        // 获取响应头中的数据
        const prediction = response.headers['prediction'];
        const encodedAiMessage = response.headers['aimessage'];

        if (prediction) {
          this.predictionResult = Number(prediction);
        }

        if (encodedAiMessage) {
          this.aiSuggestion = atob(encodedAiMessage);
        }

        // 创建海报图片 URL
        this.moviePoster = URL.createObjectURL(response.data);
      } catch (error) {
        console.error('预测失败:', error);
      } finally {
        this.loading = false;
      }
    },
    formatPrediction(value) {
      return Number(value).toLocaleString('zh-CN', {
        maximumFractionDigits: 2
      });
    },
    goBack() {
      this.$router.push('/movie-prediction');
    }
  }
};
</script>

<style scoped>
.raw-prediction-container {
  padding: 40px;
  background: #f5f5f5;
  min-height: 100vh;
}

.prediction-card {
  max-width: 800px;
  margin: 0 auto;
  background: white;
  border-radius: 8px;
  box-shadow: 0 2px 12px rgba(0, 0, 0, 0.1);
  padding: 30px;
}

.movie-info {
  margin-bottom: 30px;
  padding: 20px;
  background: #f9f9f9;
  border-radius: 8px;
}

.info-grid {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
  gap: 20px;
  margin-top: 20px;
}

.info-item {
  display: flex;
  flex-direction: column;
}

.info-item label {
  color: #666;
  margin-bottom: 5px;
}

.loading {
  text-align: center;
  padding: 40px 0;
}

.spinner {
  border: 4px solid #f3f3f3;
  border-top: 4px solid #ff5733;
  border-radius: 50%;
  width: 40px;
  height: 40px;
  animation: spin 1s linear infinite;
  margin: 0 auto 20px;
}

@keyframes spin {
  0% { transform: rotate(0deg); }
  100% { transform: rotate(360deg); }
}

.result-section {
  margin-bottom: 30px;
}

.prediction-value {
  font-size: 24px;
  color: #ff5733;
  text-align: center;
  padding: 20px;
  background: #fff5f2;
  border-radius: 8px;
  margin-top: 10px;
}

.poster-section {
  margin-bottom: 30px;
  text-align: center;
}

.poster-image {
  max-width: 100%;
  height: auto;
  border-radius: 8px;
  box-shadow: 0 2px 8px rgba(0,0,0,0.1);
}

.suggestion-section {
  background: #f9f9f9;
  padding: 20px;
  border-radius: 8px;
  margin-bottom: 30px;
}

.suggestion-section p {
  color: #666;
  line-height: 1.6;
  white-space: pre-line;
}

.back-button {
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

.back-button:hover {
  background: #ff4719;
}

h2, h3 {
  color: #333;
  margin-bottom: 20px;
}

h2 {
  font-size: 24px;
  text-align: center;
}

h3 {
  font-size: 18px;
  margin-top: 20px;
}
</style>
