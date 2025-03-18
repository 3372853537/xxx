<template>
  <div class="main-container">
        <div class="content-container">
      <div class="left-content">
        <section class="now-movies">
          <div class="section-header">
            <h2 class="section-title">正在热映 ({{ nowMovies.length }}部)</h2>
          </div>
          <div class="movie-list">
            <div v-for="(movie, index) in nowMovies" :key="index" class="movie-item" @click='goToDetail(movie.title)'>
              <div class="movie-cover">
                <img :src="movie.poster" alt="电影海报" class="movie-poster" />
                <div class="movie-info">
                  <span class="movie-title">{{ movie.title }}</span>
                  <span class="movie-score" v-if="movie.score !== '暂无评分'">{{ movie.score }}分</span>
                </div>
              </div>
            </div>
          </div>
        </section>
        <section class="coming-soon-movies">
          <div class="section-header">
            <h2 class="section-title">即将上映 ({{ comingSoonMovies.length }}部)</h2>
          </div>
          <div class="movie-list">
            <div v-for="(movie, index) in comingSoonMovies" :key="index" class="movie-item" @click='goToDetail(movie.title)'>
              <div class="movie-cover">
                <img :src="movie.poster" alt="电影海报" class="movie-poster" />
                <div class="movie-info">
                  <span class="movie-title">{{ movie.title }}</span>
                  <span class="movie-release-date">{{ movie.whis_count }}</span>
                </div>
              </div>
            </div>
          </div>
        </section>
      </div>
      
      <!-- 修改右侧内容区域 -->
      <div class="right-content">
                <section class="box-office-module">
          <h2 class="section-title">今日榜单</h2>
          <ul class="box-office-list">
            <li v-for="(item, index) in boxOfficeData" :key="index" class="box-office-item">
              <span :class="['ranking', index < 3 ? 'top3' : '', index < 5 ? 'italic' : '']">{{ index + 1 }}</span>
              <img v-if="index === 0" :src="item.poster" class="movie-poster-small" />
              <div class="box-office-info">
                <span class="movie-title">{{ item.title }}</span>
                <div class="box-office-detail">
                  <span class="box-office">{{ item.amount }}</span>
                </div>
              </div>
            </li>
          </ul>
        </section>
        <section class="most-expected-module">
          <h2 class="section-title">最受期待</h2>
          <ul class="most-expected-list">
            <li v-for="(item, index) in mostExpectedData" :key="index" class="box-office-item">
              <span :class="['ranking', index < 3 ? 'top3' : '', index < 5 ? 'italic' : '']">{{ index + 1 }}</span>
              <img v-if="index === 0" :src="item.poster" class="movie-poster-small" />
              <div class="box-office-info">
                <span class="movie-title">{{ item.title }}</span>
                <div class="box-office-detail">
                  <span class="box-office">{{ item.expectation }}</span>
                </div>
              </div>
            </li>
          </ul>
        </section>
      </div>
    </div>
  </div>
</template>

<script>
export default {
  name: "HomePage", 
  data() {
    return {
      nowMovies: [
        {
        "title": "还有明天",
        "release_time": "2023",
        "poster": require("../assets/now/1.jpg"),
        // ...已删除 "plot" 与 "crew" ...
        "score": "9.4"
    },
    {
        "title": "初步举证",
        "release_time": "2022",
        "poster": require("../assets/now/2.jpg"),
        // ...已删除 "plot" 与 "crew" ...
        "score": "9.5"
    },
    {
        "title": "编号17",
        "release_time": "2025",
        "poster": require("../assets/now/3.jpg"),
        // ...已删除 "plot" 与 "crew" ...
        "score": "7.2"
    },
    {
        "title": "平原上的火焰",
        "release_time": "2021",
        "poster": require("../assets/now/4.jpg"),
        // ...已删除 "plot" 与 "crew" ...
        "score": "暂无评分"
    },
    {
        "title": "想飞的女孩",
        "release_time": "2025",
        "poster": require("../assets/now/5.jpg"),
        // ...已删除 "plot" 与 "crew" ...
        "score": "暂无评分"
    },
    {
        "title": "猫猫的奇幻漂流",
        "release_time": "2024",
        "poster": require("../assets/now/6.jpg"),
        // ...已删除 "plot" 与 "crew" ...
        "score": "8.4"
    },
    {
        "title": "天堂旅行团",
        "release_time": "2025",
        "poster": require("../assets/now/7.jpg"),
        // ...已删除 "plot" 与 "crew" ...
        "score": "暂无评分"
    },
    {
        "title": "诡才之道",
        "release_time": "2024",
        "poster": require("../assets/now/8.jpg"),
        // ...已删除 "plot" 与 "crew" ...
        "score": "7.8"
    },

      ],
      comingSoonMovies: [
       {
          title: "潮",
          releaseTime: "03月11日",
          whis_count: "606人\n                想看",
          poster: require("../assets/will/1.jpg"),
        },
        {
          title: "疾速追杀4",
          releaseTime: "03月14日",
          whis_count: "72454人\n                想看",
          poster: require("../assets/will/2.jpg"),
        },
        {
          title: "非标准恋爱",
          releaseTime: "03月14日",
          whis_count: "807人\n                想看",
          poster: require("../assets/will/3.jpg"),
        },
        {
          title: "午夜怨灵",
          releaseTime: "03月14日",
          whis_count: "200人\n                想看",
          poster: require("../assets/will/4.jpg"),
        },
        {
          title: "夜半凶宅",
          releaseTime: "03月14日",
          whis_count: "150人\n                想看",
          poster: require("../assets/will/5.jpg"),
        },
        {
          title: "真爱营业",
          releaseTime: "03月15日",
          whis_count: "1023人\n                想看",
          poster: require("../assets/will/6.jpg"),
        },
        {
          title: "久别·重逢",
          releaseTime: "03月15日",
          whis_count: "919人\n                想看",
          poster: require("../assets/will/7.jpg"),
        },
        {
          title: "援藏日记",
          releaseTime: "03月20日",
          whis_count: "3468人\n                想看",
          poster: require("../assets/will/8.jpg"),
        },
      ],
      boxOfficeData: [
        { title: "哪吒之魔童闹海", amount: "227.8万", poster: require("../assets/now/9.jpg") },
        { title: "极速追杀4", amount: "31.2万" },
        { title: "您的声音", amount: "27.5万" },
        { title: "还有明天", amount: "24.1万" },
        { title: "唐探1900", amount: "23.7万" }
      ],
      mostExpectedData: [
        { title: "倩女幽魂", expectation: "156142人想看", poster: require("../assets/will/9.jpg") },
        { title: "向阳·花", expectation: "75732人想看" },
        { title: "731", expectation: "496854人想看" },
        { title: "我的世界大电影", expectation: "108366人想看" },
        { title: "侏罗纪世界·重生", expectation: "70694人想看" }
      ],
    };
  },
  methods: {
    goToDetail(title) {
      this.$router.push({
        name: 'MovieDetail',
        params: { title: title }
      })
    }
  }
};
</script>

<style scoped>
.main-container {
  display: flex;
  flex-direction: column;
}

.content-container {
  display: flex;
  justify-content: space-between;
  padding: 30px 20px;
  width: 1200px;
  margin: 0 auto;
  padding: 30px 0;
}

.left-content {
  flex: 3; /* 增加左侧内容的占比，从2改为3 */
}

.right-content {
  flex: 1; /* 增加右侧内容的占比，从0.8改为1 */
  margin-left: 30px; /* 增加左边距 */
  width: 400px; /* 增加宽度 */
}

.now-movies,
.coming-soon-movies,
.box-office-module,
.most-expected-module,
.top100-module {
  margin-bottom: 30px;
}

.section-title {
  font-size: 22px;
  margin-bottom: 20px;
  color: #333;
  text-align: left; /* 改为左对齐 */
  padding-left: 10px;
}

.movie-list {
  display: flex;
  flex-wrap: wrap;
  justify-content: space-between;
  margin: 0 -8px;
}

.movie-item {
  width: 22%; /* 稍微减小每个电影项的宽度 */
  margin: 0 1% 20px 1%; /* 添加左右边距 */
  position: relative;
  text-align: center;
  width: calc(25% - 16px);
  margin: 0 8px 16px;
}

.movie-cover {
  position: relative;
  border-radius: 5px;
  overflow: hidden;
  padding-bottom: 140%; /* 设置容器高宽比约为1.4:1，适合大多数电影海报 */
}

.movie-poster {
  position: absolute;
  top: 0;
  left: 0;
  width: 100%;
  height: 100%;
  object-fit: cover; /* 保持图片比例完整显示 */
  display: block;
  transition: transform 0.3s ease;
}

.movie-info {
  position: absolute;
  bottom: 0;
  left: 0;
  width: 100%;
  background: rgba(0, 0, 0, 0.7); /* 加深背景透明度 */
  color: #fff;
  display: flex;
  justify-content: space-between;
  align-items: center;
  padding: 0px 0px;
  font-size: 14px;
  height: 40px; /* 固定高度 */
}

/* 修改 movie-title 样式，确保长标题不换行且溢出显示省略号 */
.movie-title {
  flex: 1;
  text-align: left;
  white-space: nowrap;
  overflow: hidden;
  text-overflow: ellipsis;
  margin-right: 5px; /* 添加右边距，与评分保持距离 */
}

/* 修改 movie-score 和 movie-release-date 样式，设置最小宽度以正常显示数值 */
.movie-score,
.movie-release-date {
  flex: none; /* 改为none，防止被压缩 */
  text-align: right;
  min-width: 60px; /* 增加最小宽度 */
  white-space: nowrap; /* 防止换行 */
  color: #ffb400;  /* 评分显示为金黄色 */
  font-weight: 700; /* 加粗显示 */
  font-size: 16px;
}

.movie-score {
  flex: none;
  text-align: right;
  min-width: 10px;
  white-space: nowrap;
  color: #ffb400;
  font-weight: 700;
  font-size: 16px;
}

.movie-item:hover .movie-poster {
  transform: scale(1.1);
}

.box-office-list,
.most-expected-list,
.top100-list {
  list-style-type: none;
  padding: 0;
}

.box-office-list li,
.most-expected-list li,
.top100-list li {
  margin-bottom: 10px;
  font-size: 16px;
  color: #666;
}

.section-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  margin-bottom: 15px;
}

.all-link {
  color: #e74c3c;
  text-decoration: none;
  font-size: 16px;
}

/* 修改右侧内容样式 */
.ad-banner {
  margin-bottom: 30px;
}

.ad-banner img {
  width: 100%;
  border-radius: 8px;
}

/* 修改榜单样式 */
.box-office-module {
  background: #fff;
  border-radius: 8px;
  padding: 20px; /* 增加内边距 */
  box-shadow: 0 2px 8px rgba(0,0,0,0.1);
}

.box-office-list {
  list-style: none;
  padding: 0;
  margin: 0;
}

.box-office-item {
  display: flex;
  align-items: center;
  padding: 12px 10px;
  border-bottom: 1px solid #f5f5f5;
}

.box-office-item:last-child {
  border-bottom: none;
}

.ranking {
  width: 24px;
  font-size: 16px;
  color: #999;
  font-weight: bold;
  text-align: center;
}

.ranking.italic {
  font-style: italic;
}

.ranking.top3 {
  color: #ff5733;
}

.box-office-info {
  flex: 1;
  margin-left: 10px;
  display: flex;
  justify-content: space-between;
  align-items: center;
}

.box-office-info .movie-title {
  font-size: 16px; /* 增加字体大小 */
  color: #333;
  max-width: 200px; /* 增加最大宽度 */
  white-space: nowrap;
  overflow: hidden;
  text-overflow: ellipsis;
  padding: 0 10px; /* 添加内边距 */
}

.box-office-detail {
  display: flex;
  align-items: center;
}

.box-office {
  font-size: 16px; /* 增加字体大小 */
  color: #ff5733;
  margin-right: 8px;
  min-width: 80px; /* 设置最小宽度 */
  text-align: right;
}

.trend {
  width: 0;
  height: 0;
  border-left: 5px solid transparent;
  border-right: 5px solid transparent;
  margin-left: 5px;
}

.trend.up {
  border-bottom: 8px solid #ff5733;
}

.trend.down {
  border-top: 8px solid #green;
}

/* 修改最受期待列表样式 */
.most-expected-module {
  background: #fff;
  border-radius: 8px;
  padding: 20px;
  box-shadow: 0 2px 8px rgba(0,0,0,0.1);
}

.most-expected-list {
  list-style: none;
  padding: 0;
  margin: 0;
}

.movie-poster-small {
  width: 60px;
  height: 80px;
  border-radius: 4px;
  margin-right: 10px;
  object-fit: cover;
}

/* 删除原有的 most-expected-list 相关样式，使用 box-office-item 的样式 */
</style>