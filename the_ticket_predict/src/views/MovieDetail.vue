<template>
  <div>
    <div v-if="movie">
      <div class="movie-detail-page">
        <div class="movie-detail-content">
          <div class="movie-header">
            <h1 class="movie-title">{{ movie.title }}</h1>
            <div class="movie-info">
              <!-- <p class="movie-release-time">发布时间：{{ movie.releaseTime }}</p> -->
              <p class="movie-format">
              </p>
            </div>
          </div>
          <div class="movie-poster">
            <!-- 修改绑定字段为 movie.poster -->
            <img :src="getPoster(movie.poster)" alt="{{ movie.title }}海报" />
          </div>
          <div class="movie-summary">
            <h2>剧情简介</h2>
            <p>{{ movie.plot }}</p>
          </div>
          <div class="movie-credits">
            <h2>演职人员</h2>
            <div class="crew-section">
              <h3>导演</h3>
              <div class="crew-list">
                <div v-for="director in movie.directors" :key="director" class="crew-item">
                  <img :src="getPoster(movie.directors_poster)" :alt="director" class="crew-image">
                  <div class="crew-info">
                      <span class="crew-name">{{ director }}</span>
                  </div>
                </div>
              </div>
            </div>
            <div class="crew-section">
              <h3>演员</h3>
              <div class="crew-list">
                <div v-for="(actor, index) in movie.actors" :key="index" class="crew-item">
                  <img :src="getPoster(actor.actor_poster)" :alt="actor.name" class="crew-image">
                  <div class="crew-info">
                    <span class="crew-name">{{ actor.actor }}</span>
                    <span class="crew-role">{{ actor.role }}</span>
                  </div>
                </div>
              </div>
            </div>
          </div>
          <div class="movie-trailers">
            <div v-for="(trailer, index) in movie.trailers" :key="index">
              <a :href="trailer.url" target="_blank">{{ trailer.title }}</a>
            </div>
          </div>

          <!-- 新增票房数据展示区域 -->
          <!-- 修改票房数据展示区域 -->
          <div class="box-office-section">
            <h2>每日票房数据</h2>
            <div v-if="boxOfficeData" class="box-office-charts">
              <div ref="dailyBoxOfficeChart" class="chart"></div>
              <div ref="screenShareChart" class="chart"></div>
            </div>
          </div>
        </div>
      </div>
    </div>
    <div v-else>
      Loading...
    </div>
  </div>
</template>

<script>
import axios from 'axios'
import * as echarts from 'echarts'
const images = require.context('@/assets', true, /\.(png|jpe?g|svg)$/);
export default {
  name: 'MovieDetail',
  components: {
  },
  data() {
    return {
      boxOfficeData: null,
      boxOfficeError: null,
      movies: [], // 存储所有电影数据
      movie: null
    }
  },
  methods: {
getPoster(path) {
      if (path.startsWith('../assets/')) {
        let imgPath = './' + path.slice(10);
        // 如果扩展名前缺少点号，则添加点号（例如将 "3.1jpg" 转为 "3.1.jpg"）
        imgPath = imgPath.replace(/(\d+)(jpe?g|png|svg)$/i, '$1.$2');
        return images(imgPath);
      }
      return path;
    },
    async fetchAllMovies() {
      try {
        const [nowShowing, upcoming, topMovies] = await Promise.all([
          axios.get('http://127.0.0.1:5000/MoviesNowDisPlay'),
          axios.get('http://127.0.0.1:5000/MoviesToDisPlay'),
          axios.get('http://127.0.0.1:5000/Moviesdata')
        ]);

        // 合并所有电影数据并添加本地海报路径
        this.movies = [
          ...nowShowing.data,
          ...upcoming.data,
          ...topMovies.data
        ];
      } catch (error) {
        console.error('获取电影数据失败:', error);
      }
    },

    async fetchBoxOfficeData() {
      try {
        const response = await axios.get(`http://127.0.0.1:5000/movies/${this.movie.title}`)
        this.boxOfficeData = response.data
      } catch (error) {
        this.boxOfficeError = error.response?.data?.message || "获取票房数据失败" 
      }
    },
    initCharts() {
      if (!this.boxOfficeData) return
      
      // 初始化日票房图表
      const dailyChart = echarts.init(this.$refs.dailyBoxOfficeChart)
      const days = Array.from({length: 30}, (_, i) => `第${i + 1}天`)
      const dailyData = Array.from({length: 30}, (_, i) => this.boxOfficeData[`day${i + 1}_daily`])
      
      dailyChart.setOption({
        title: { text: '日票房走势（万元）' },
        tooltip: { trigger: 'axis' },
        xAxis: { type: 'category', data: days },
        yAxis: { type: 'value' },
        series: [{
          data: dailyData,
          type: 'line',
          smooth: true
        }]
      })
      
      // 初始化排片占比图表
      const screenChart = echarts.init(this.$refs.screenShareChart)
      const screenData = Array.from({length: 30}, (_, i) => this.boxOfficeData[`day${i + 1}_screen`])
      const shareData = Array.from({length: 30}, (_, i) => this.boxOfficeData[`day${i + 1}_share`])
      
      screenChart.setOption({
        title: { text: '排片场次与占比' },
        tooltip: { trigger: 'axis' },
        legend: { data: ['排片场次', '排片占比(%)'] },
        xAxis: { type: 'category', data: days },
        yAxis: [
          { type: 'value', name: '场次' },
          { type: 'value', name: '占比(%)', max: 100 }
        ],
        series: [
          {
            name: '排片场次',
            type: 'bar',
            data: screenData
          },
          {
            name: '排片占比(%)',
            type: 'line',
            yAxisIndex: 1,
            data: shareData
          }
        ]
      })
    }
  },
  watch: {
    boxOfficeData: {
      handler(val) {
        if (val) {
          this.$nextTick(() => {
            this.initCharts()
          })
        }
      },
      immediate: true
    }
  },
  async created() {
    await this.fetchAllMovies();
    const movieTitle = this.$route.params.title;
    this.movie = this.movies.find(movie => movie.title === movieTitle);
    
    if (!this.movie) {
      alert("电影不存在");
      this.$router.push('/');
    } else {
      this.fetchBoxOfficeData();
    }
  }
}
</script>

<style scoped>
/* 新增票房数据相关样式 */
.box-office-section {
  margin: 30px 0;
  padding: 20px;
  background: #fff;
  border-radius: 8px;
  box-shadow: 0 2px 12px rgba(0,0,0,0.1);
}

.box-office-chart {
  margin-top: 20px;
  background: #fff;
  border-radius: 8px;
  padding: 20px;
  box-shadow: 0 2px 12px rgba(0,0,0,0.1);
}

.error-message {
  color: #ff5733;
  text-align: center;
  padding: 20px;
}

.loading {
  text-align: center;
  padding: 20px;
  color: #666;
}

.movie-detail-page {
  padding: 20px;
}

.movie-detail-content {
  max-width: 1200px;
  margin: 0 auto;
}

.movie-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  margin-bottom: 20px;
}

.movie-title {
  font-size: 32px;
  color: #333;
}

.movie-info {
  font-size: 16px;
  color: #666;
}

.movie-poster {
  text-align: center;
  margin-bottom: 20px;
}

.movie-poster img {
  max-width: 100%;
  height: auto;
}

.movie-summary,
.movie-credits,
.movie-trailers {
  margin-bottom: 30px;
}

.movie-summary h2,
.movie-credits h2,
.movie-trailers h2 {
  font-size: 24px;
  color: #333;
  margin-bottom: 10px;
}

.movie-summary p {
  font-size: 16px;
  line-height: 1.6;
}

.crew-section {
  margin-bottom: 20px;
}

.crew-section h3 {
  font-size: 20px;
  color: #333;
  margin-bottom: 10px;
}

.crew-section ul {
  list-style-type: none;
  padding: 0;
}

.crew-section li {
  font-size: 16px;
  margin-bottom: 5px;
}

.movie-trailers a {
  display: block;
  font-size: 16px;
  color: #007bff;
  text-decoration: none;
  margin-bottom: 5px;
}

.movie-trailers a:hover {
  text-decoration: underline;
}
.box-office-charts {
  display: flex;
  flex-direction: column;
  gap: 20px;
}

.chart {
  width: 100%;
  height: 400px;
  background: #fff;
  border-radius: 8px;
  padding: 20px;
  box-shadow: 0 2px 12px rgba(0,0,0,0.1);
}

.crew-list {
  display: flex;
  flex-wrap: wrap;
  gap: 20px;
  padding: 0;
  list-style: none;
}

.crew-item {
  width: 150px;
  text-align: center;
  margin-bottom: 20px;
}

.crew-image {
  width: 120px;
  height: 160px;
  object-fit: cover;
  border-radius: 8px;
  margin-bottom: 8px;
  box-shadow: 0 2px 8px rgba(0,0,0,0.1);
}

.crew-info {
  display: flex;
  flex-direction: column;
  gap: 4px;
}

.crew-name {
  font-weight: bold;
  font-size: 14px;
}

.crew-role {
  color: #666;
  font-size: 12px;
}
</style>