<template>
  <div class="box-office-container">
    <div class="stats-cards">
      <div class="stat-card total-box">
        <div class="stat-title">实时票房(万)</div>
        <div class="stat-value">{{ getTotalBoxOffice() }}</div>
        <div class="stat-trend positive">
          <span class="trend-icon">↑</span>
          <span>{{ calculateTrend() }}%</span>
        </div>
      </div>
      <div class="stat-card">
        <div class="stat-title">今日观影人次</div>
        <div class="stat-value">{{ calculateViewers() }}万</div>
        <div class="stat-subtitle">更新时间：{{ currentDate }}</div>
      </div>
      <div class="stat-card">
        <div class="stat-title">上映影片数</div>
        <div class="stat-value">{{ boxOfficeData.length }}</div>
        <div class="stat-subtitle">今日排片：{{ getTotalScreenings() }}场</div>
      </div>
    </div>

    <div class="dashboard-content">
      <div class="left-panel">
        <div class="box-office-table">
          <div class="table-header">
            <h2>实时票房排行</h2>
            <div class="table-filters">
              <button 
                v-for="(filter, index) in filters" 
                :key="index"
                @click="sortBy(filter.key)"
                :class="['filter-btn', { active: sortKey === filter.key }]"
              >
                {{ filter.label }}
              </button>
            </div>
          </div>

          <div class="movie-table">
            <div class="table-header-row">
              <div class="table-cell rank">排名</div>
              <div class="table-cell movie-name">影片名称</div>
              <div class="table-cell box-office">实时票房(万)</div>
              <div class="table-cell percent">票房占比</div>
              <div class="table-cell shows">排片场次</div>
              <div class="table-cell days">上映天数</div>
              <div class="table-cell total">总票房(万)</div>
            </div>

            <div class="table-body">
              <div 
                v-for="(movie, index) in sortedMovies" 
                :key="index"
                :class="['table-row', { 'highlight': index < 3 }]"
              >
                <div class="table-cell rank">
                  <span :class="['rank-num', `rank-${index + 1}`]">{{ index + 1 }}</span>
                </div>
                <div class="table-cell movie-name">
                  <span class="movie-title">{{ movie["影片"] }}</span>
                </div>
                <div class="table-cell box-office">
                  <span class="primary-text">{{ formatNumber(movie["综合票房"]) }}</span>
                </div>
                <div class="table-cell percent">{{ movie["票房占比"] }}</div>
                <div class="table-cell shows">{{ movie["排片场次占比"] }}</div>
                <div class="table-cell days">{{ movie["上映天数"] }}</div>
                <div class="table-cell total">{{ formatNumber(movie["总票房"]) }}</div>
              </div>
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
  name: "MovieBoxOffice",
  data() {
    return {
      currentDate: new Date().toLocaleDateString('zh-CN', {
        year: 'numeric',
        month: '2-digit',
        day: '2-digit',
        hour: '2-digit',
        minute: '2-digit'
      }),
      boxOfficeData: [],
      sortKey: '综合票房',
      sortOrder: 'desc',
      filters: [
        { key: '综合票房', label: '实时票房' },
        { key: '上映天数', label: '上映天数' },
        { key: '总票房', label: '累计票房' }
      ]
    };
  },
  mounted() {
    this.fetchBoxOfficeData();
  },
  computed: {
    sortedMovies() {
      return [...this.boxOfficeData].sort((a, b) => {
        const modifier = this.sortOrder === 'desc' ? -1 : 1;
        return modifier * (a[this.sortKey] - b[this.sortKey]);
      });
    }
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
    getTotalBoxOffice() {
      return this.boxOfficeData
        .reduce((sum, movie) => sum + parseFloat(movie['综合票房'] || 0), 0)
        .toFixed(2);
    },
    sortBy(key) {
      if (this.sortKey === key) {
        this.sortOrder = this.sortOrder === 'desc' ? 'asc' : 'desc';
      } else {
        this.sortKey = key;
        this.sortOrder = 'desc';
      }
    },
    formatNumber(num) {
      return parseFloat(num).toLocaleString('zh-CN', { 
        maximumFractionDigits: 2 
      });
    },
    calculateTrend() {
      return (Math.random() * 10).toFixed(2);
    },
    calculateViewers() {
      return (this.getTotalBoxOffice() * 0.3).toFixed(2);
    },
    getTotalScreenings() {
      return Math.floor(Math.random() * 50000 + 10000);
    }
  }
};
</script>

<style scoped>
.box-office-container {
  padding: 20px;
  background: linear-gradient(to bottom, #f8f9fa, #e9ecef);
  min-height: 100vh;
}

.stats-cards {
  display: grid;
  grid-template-columns: 2fr 1fr 1fr;
  gap: 20px;
  margin-bottom: 24px;
}

.stat-card {
  background: white;
  border-radius: 12px;
  padding: 24px;
  box-shadow: 0 4px 12px rgba(0,0,0,0.05);
  transition: transform 0.2s;
}

.stat-card:hover {
  transform: translateY(-2px);
}

.total-box {
  background: linear-gradient(135deg, #ff5733, #ff8c69);
  color: white;
}

.stat-title {
  font-size: 16px;
  color: inherit;
  margin-bottom: 12px;
  opacity: 0.9;
}

.stat-value {
  font-size: 32px;
  font-weight: bold;
  margin: 8px 0;
}

.stat-subtitle {
  font-size: 13px;
  color: #666;
  margin-top: 8px;
}

.stat-trend {
  display: flex;
  align-items: center;
  gap: 4px;
  font-size: 14px;
  margin-top: 8px;
}

.stat-trend.positive {
  color: #52c41a;
}

.dashboard-content {
  display: flex;
  gap: 24px;
}

.left-panel {
  flex: 1;
  background: white;
  border-radius: 12px;
  box-shadow: 0 4px 12px rgba(0,0,0,0.05);
  padding: 24px;
}

.table-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  margin-bottom: 24px;
}

.table-header h2 {
  font-size: 20px;
  color: #333;
}

.table-filters {
  display: flex;
  gap: 12px;
}

.filter-btn {
  padding: 6px 16px;
  border: 1px solid #e0e0e0;
  border-radius: 20px;
  background: white;
  color: #666;
  cursor: pointer;
  transition: all 0.3s;
}

.filter-btn.active {
  background: #ff5733;
  color: white;
  border-color: #ff5733;
}

.movie-table {
  width: 100%;
  border-radius: 8px;
  overflow: hidden;
}

.table-header-row {
  display: flex;
  background: #f8f9fa;
  padding: 12px 0;
  font-weight: 500;
  color: #666;
}

.table-row {
  display: flex;
  padding: 16px 0;
  border-bottom: 1px solid #f0f0f0;
  transition: background 0.3s;
}

.table-row:hover {
  background: #fff8f6;
}

.table-row.highlight {
  background: #fff8f6;
}

.table-cell {
  padding: 0 12px;
  display: flex;
  align-items: center;
}

.rank { width: 8%; }
.movie-name { width: 25%; }
.box-office { width: 15%; }
.percent { width: 12%; }
.shows { width: 15%; }
.days { width: 10%; }
.total { width: 15%; }

.rank-num {
  width: 24px;
  height: 24px;
  line-height: 24px;
  text-align: center;
  border-radius: 4px;
  font-weight: bold;
}

.rank-1 { background: #ff5733; color: white; }
.rank-2 { background: #ff8c69; color: white; }
.rank-3 { background: #ffa07a; color: white; }

.movie-title {
  font-weight: 500;
  color: #333;
}

.primary-text {
  color: #ff5733;
  font-weight: 500;
}
</style>
