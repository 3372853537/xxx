<template>
  <div class="news-container">
    <!-- 轮播图部分 -->
    <div class="carousel-section">
      <el-carousel height="400px">
        <el-carousel-item v-for="item in carouselItems" :key="item.id">
          <img :src="item.imageUrl" :alt="item.title" class="carousel-image">
          <div class="carousel-caption">{{ item.title }}</div>
        </el-carousel-item>
      </el-carousel>
    </div>

    <!-- 新闻资讯部分 -->
    <div class="news-section">
      <h2 class="section-title">今日资讯</h2>
      <div class="news-grid">
        <div v-for="news in newsItems" :key="news.id" class="news-item">
          <img :src="news.imageUrl" :alt="news.title">
          <div class="news-content">
            <h3>{{ news.title }}</h3>
            <p>{{ news.summary }}</p>
            <span class="news-time">{{ news.time }}</span>
          </div>
        </div>
      </div>
    </div>

    <!-- 每日佳片推荐部分 -->
    <div class="daily-movies-section">
      <h2 class="section-title">每日佳片</h2>
      <div class="movies-grid">
        <div v-for="movie in dailyMovies" :key="movie.id" class="movie-card">
          <img :src="getPoster(movie.poster)" :alt="movie.title">
          <h3>{{ movie.title }}</h3>
          <div class="movie-rating">评分：{{ movie.score }}</div>
        </div>
      </div>
    </div>
  </div>
</template>

<script>
import { ElCarousel, ElCarouselItem } from 'element-plus'
import 'element-plus/dist/index.css'
import axios from 'axios'
const images = require.context('@/assets', true, /\.(png|jpe?g|svg)$/);

export default {
  name: 'MovieNews',
  components: {
    ElCarousel,
    ElCarouselItem
  },
  data() {
    return {
      carouselItems: [
        { 
          id: 1, 
          imageUrl:'https://img5.mtime.cn/mg/2025/03/12/103252.75428708.jpg', 
        //   title: '用银幕的光影照亮希望' 
        },
        { 
          id: 2, 
          imageUrl:'https://img5.mtime.cn/mg/2025/02/27/174257.13583407.jpg', 
        //   title: '崩塌的"美国梦",塑料的"姐妹情""' 
        },
        { 
          id: 3, 
          imageUrl:'http://img5.mtime.cn/mg/2025/02/21/135048.18921224.jpg', 
        //   title: '2023暑期档观影指南' 
        },
        { 
          id: 4, 
          imageUrl:'http://img5.mtime.cn/mg/2025/02/17/172421.11125646.jpg', 
        //   title: '2023暑期档观影指南' 
        },
        { 
          id: 5, 
          imageUrl:'http://img5.mtime.cn/mg/2025/02/05/135634.83021766.jpg', 
        //   title: '2023暑期档观影指南' 
        },
        { 
          id: 6, 
          imageUrl:'http://img5.mtime.cn/mg/2025/02/04/100754.14115894.jpg', 
        //   title: '2023暑期档观影指南' 
        }
      ],
      newsItems: [
        {
          id: 1,
          imageUrl:require('../assets/will/10.jpg'),
          title: '《奥本海默》口碑爆棚',
          summary: '诺兰新作口碑创新高，烂番茄新鲜度94%',
          time: '2023-07-21'
        },
        {
          id: 2,
          imageUrl:require('../assets/will/10.jpg'),
          title: '《封神第一部》上映',
          summary: '暑期档最受期待大片终于来袭',
          time: '2023-07-22'
        }
      ],
      dailyMovies: []
    }
},
  created() {
    this.fetchCarouselItems();
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
    async fetchCarouselItems() {
      try {
        const response = await axios.get('http://localhost:5000/MoviesNowDisPlay');
        this.dailyMovies = response.data.slice(0, 5); // 只取前六部
      } catch (error) {
        console.error('Error fetching carousel items:', error);
      }
    }
  }
}
</script>

<style scoped>
.news-container {
  max-width: 1200px;
  margin: 0 auto;
  padding: 20px;
}

.carousel-section {
  margin-bottom: 40px;
}

.carousel-image {
  width: 100%;
  height: 100%;
  object-fit: cover;
}

.carousel-caption {
  position: absolute;
  bottom: 20px;
  left: 20px;
  color: white;
  font-size: 24px;
  text-shadow: 0 0 10px rgba(0,0,0,0.5);
}

.section-title {
  font-size: 24px;
  margin-bottom: 20px;
  padding-bottom: 10px;
  border-bottom: 2px solid #ffc107;
}

.news-grid {
  display: grid;
  grid-template-columns: repeat(2, 1fr);
  gap: 20px;
  margin-bottom: 40px;
}

.news-item {
  display: flex;
  gap: 15px;
  padding: 15px;
  border: 1px solid #eee;
  border-radius: 8px;
}

.news-item img {
  width: 200px;
  height: 120px;
  object-fit: cover;
  border-radius: 4px;
}

.news-content h3 {
  margin: 0 0 10px 0;
  font-size: 18px;
}

.news-time {
  color: #999;
  font-size: 14px;
}

.movies-grid {
  display: grid;
  grid-template-columns: repeat(5, 1fr);
  gap: 20px;
}

.movie-card {
  text-align: center;
}

.movie-card img {
  width: 100%;
  aspect-ratio: 2/3;
  object-fit: cover;
  border-radius: 8px;
  margin-bottom: 10px;
}

.movie-card h3 {
  margin: 5px 0;
  font-size: 16px;
}

.movie-rating {
  color: #ffc107;
  font-weight: bold;
}
</style>