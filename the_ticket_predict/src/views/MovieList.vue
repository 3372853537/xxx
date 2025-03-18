<template>
  <div class="movie-list-page">
    <TitleTabs :currentTab="currentTab" @tab-change="changeTab" />
    <div class="movie-list-container">
      <div v-for="(movie, index) in paginatedMovies" :key="index" class="movie-item">
        <div class="movie-cover">
          <img :src="getPoster(movie.poster)" alt="电影海报" class="movie-poster" @click="goToDetail(movie.title)" />
          <div class="movie-info">
            <span class="movie-title">{{ movie.title }}</span>
            <template v-if="currentTab === 'hot'">  
              <span class="movie-score" v-if="movie.score !== '暂无评分'">{{ movie.score }}分</span>
            </template>
            <template v-if="currentTab === 'coming'">
              <span class="movie-release-date">{{ movie.whis_count }}</span>
            </template>
            <template v-if="currentTab === 'top100'">
              <span class="movie-whis-count">{{ movie.whis_count }}</span>
            </template>
          </div>
        </div>
      </div>
    </div>
    <div class="pagination">
      <button @click="prevPage" :disabled="currentPage === 1">上一页</button>
      <span>{{ currentPage }} / {{ totalPages }}</span>
      <button @click="nextPage" :disabled="currentPage === totalPages">下一页</button>
    </div>
  </div>
</template>

<script>
import TitleTabs from "@/components/TitleTabs.vue";
import axios from 'axios';

// 添加动态加载静态资源的配置（递归查找 assets 目录下图片）
const images = require.context('@/assets', true, /\.(png|jpe?g|svg)$/);

export default {
  name: "MovieListPage",
  components: {
    TitleTabs,
  },
  data() {
    return {
      currentTab: "hot",
      currentPage: 1,
      hotMovies: [],
      comingSoonMovies: [],
      top100Movies: []  // 新增的 top100Movies 数据
    };
  },
  computed: {
    getCurrentMovies() {
      if (this.currentTab === "hot") return this.hotMovies;
      else if (this.currentTab === "coming") return this.comingSoonMovies;
      else return this.top100Movies;  // 新增的 top100Movies 数据
    },
    totalPages() {
      return Math.ceil(this.getCurrentMovies.length / 24) || 1;
    },
    paginatedMovies() {
      const start = (this.currentPage - 1) * 24;
      return this.getCurrentMovies.slice(start, start + 24);
    },
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
    changeTab(tab) {
      this.currentTab = tab;
      this.currentPage = 1; // 切换 Tab 重置页码
      if (tab === 'top100' && this.top100Movies.length === 0) {
        this.fetchTop100Movies();
      }else if(tab==='hot' && this.hotMovies.length===0){
        this.fetchHotMovies();
      }else if(tab==='coming' && this.comingSoonMovies.length===0){
        this.fetchComingMovies();
      }
    },
    goToDetail(title) {
      this.$router.push({
        name: 'MovieDetail',
        params: { title: title }
      })
    },
    prevPage() {
      if(this.currentPage > 1) this.currentPage--;
    },
    nextPage() {
      if(this.currentPage < this.totalPages) this.currentPage++;
    },
    fetchTop100Movies() {
      axios.get('http://127.0.0.1:5000/admin/top100movies')
        .then(response => {
          this.top100Movies = response.data
          })
        .catch(error => {
          console.error("Error fetching top 100 movies:", error);
        });
    },
    fetchHotMovies() {
      axios.get('http://127.0.0.1:5000/admin/nowmovies')
        .then(response => {
          this.hotMovies = response.data
        })
        .catch(error => {
          console.error("Error fetching hot  movies:", error);
        });
    },
    fetchComingMovies() {
      axios.get('http://127.0.0.1:5000/admin/willmovies')
        .then(response => {
          this.comingSoonMovies = response.data
        })
        .catch(error => {
          console.error("Error fetching will movies:", error);
        });
    }
  },
  mounted() {
    if (this.currentTab === 'top100') {
      this.fetchTop100Movies();
    }
    else if(this.currentTab==='hot'){
      this.fetchHotMovies();
    }
    else{
      this.fetchComingMovies();
    }

  }
};
</script>

<style scoped>
.movie-list-page {
  padding: 30px 20px;
}

.movie-list-container {
  display: grid;
  grid-template-columns: repeat(5, 1fr);
  gap: 16px;
  padding: 16px;
  max-height: 800px; /* 调整最大高度 */
  overflow-y: auto;
  width: 100%;
  margin: 0 auto;
}

.movie-item {
  width: 100%; /* 修改为100%适应网格布局 */
  position: relative;
  border-radius: 4px;
  overflow: hidden;
  background-color: #fff;
  box-shadow: 0 2px 8px rgba(0, 0, 0, 0.1);
  margin: 0; /* 移除margin */
}

.movie-cover {
  position: relative;
  width: 100%;
  padding-top: 140%; /* 保持固定的宽高比 */
  overflow: hidden;
}

.movie-poster {
  position: absolute;
  top: 0;
  left: 0;
  width: 100%;
  height: 100%;
  object-fit: cover; /* 确保图片填充整个容器 */
  display: block;
  transition: transform 0.3s ease;
}

.movie-info {
  position: absolute;
  bottom: 0;
  left: 0;
  width: 100%;
  background: rgba(0, 0, 0, 0.7);
  color: #fff;
  display: flex;
  justify-content: space-between;
  align-items: center;
  padding: 0px;
  font-size: 14px;
  height: 40px;
}

.movie-title {
  flex: 1;
  text-align: left;
  white-space: nowrap;
  overflow: hidden;
  text-overflow: ellipsis;
  margin-right: 8px;
  font-size: 14px;
}

.movie-score,
.movie-release-date,
.movie-whis-count {
  flex: none;
  text-align: right;
  min-width: 60px;
  white-space: nowrap;
  color: #ffb400;
  font-weight: 700;
  font-size: 14px;
}

.pagination {
  margin-top: 20px;
  text-align: center;
}

.pagination button {
  margin: 0 10px;
}
</style>