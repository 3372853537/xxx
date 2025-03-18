<template>
  <nav class="navbar">
    <div class="container">
      <div class="left-section">
        <a href="/" class="logo">
          <img src="../assets/logo.png" alt="猫眼电影" />
        </a>
        <router-link 
          to="/movie-news" 
          class="news-link"
          :class="{ active: currentLink === 'news' }"
          @click="setActiveLink('news')"
        >
          资讯
        </router-link>
      </div>
      <ul class="nav-links">
        <li :class="{ active: currentLink === 'home' }">
          <router-link to="/" exact @click="setActiveLink('home')">首页</router-link>
        </li>
        <li :class="{ active: currentLink === 'movie' }">
          <router-link to="/movie-list" @click="setActiveLink('movie')">电影</router-link>
        </li>
        <li :class="{ active: currentLink === 'boxoffice' }">
          <router-link to="/movie-box-office" @click="setActiveLink('boxoffice')">票房</router-link>
        </li>
        <li :class="{ active: currentLink === 'show' }">
          <router-link to="/movie-prediction" @click="setActiveLink('show')">预测</router-link>
        </li>
      </ul>
      <div class="nav-actions">
        <div class="search-box">
          <input v-model="searchQuery" type="text" placeholder="搜索电影、演出" class="search-input" />
          <button class="search-button" @click="searchMovie">搜索</button>
        </div>
        <div class="user-menu" @click.stop="toggleDropdown">
          <img :src="userAvatar" alt="用户图标" class="user-icon" />
          <div v-show="dropdownOpen" class="user-dropdown">
            <template v-if="isLoggedIn">
              <a href="/user-detail" @click.stop="goToMyPage">我的主页</a>
              <a href="/user-login" @click.stop="logout">退出</a>
            </template>
            <template v-else>
              <a href="/user-login" @click.stop="login">登录</a>
            </template>
          </div>
        </div>
      </div>
    </div>
  </nav>
</template>

<script>
import { userAvatar } from '../store/userStore';
import { getLoginInfo, clearLoginInfo } from '../utils/auth';

export default {
  name: "NaBar",
  data() {
    return {
      currentLink: 'home', // 默认首页，但首页无active样式
      isLoggedIn: false, // 修改为false
      dropdownOpen: false, // 新增，下拉菜单显示状态
      searchQuery: "",  // 新增搜索输入绑定
      userAvatar, // 添加共享的头像状态
    };
  },
  methods: {
    setActiveLink(link) {
      this.currentLink = link;
    },
    toggleDropdown() {
      this.dropdownOpen = !this.dropdownOpen;
    },
    goToMyPage() {
      if (!this.isLoggedIn) {
        console.log('未登录，跳转到登录页');
        this.$router.push('/user-login');
        return;
      }
      console.log('跳转到用户主页');
      this.$router.push('/user-detail');
      this.dropdownOpen = false; // 点击后关闭下拉框
    },
    logout() {
      console.log('执行登出操作');
      clearLoginInfo();
      this.isLoggedIn = false;
      this.$router.push('/user-login');
      this.dropdownOpen = false; // 退出后关闭下拉框
    },
    searchMovie() {  // 新增搜索方法
      if (this.searchQuery.trim() === "") {
        alert("请输入电影标题")
      } else {
        this.$router.push({ name: 'MovieDetail', params: { title: this.searchQuery } });
      }
    },
    closeDropdown() {
      this.dropdownOpen = false;
    },
    login() {
      this.$router.push('/user-login');
      this.dropdownOpen = false;
    },
    checkLoginStatus() {
      const { username, isLoggedIn } = getLoginInfo();
      console.log('检查登录状态:', { username, isLoggedIn });
      this.isLoggedIn = isLoggedIn;
    }
  },
  created() {
    console.log('组件创建时检查登录状态');
    this.checkLoginStatus();
  },
  mounted() {
    console.log('组件挂载时检查登录状态');
    this.checkLoginStatus(); // 初始化时检查登录状态
    document.addEventListener('click', this.closeDropdown);
  },
  watch: {
    // 监听路由变化
    $route: {
      handler() {
        console.log('路由变化，重新检查登录状态');
        this.checkLoginStatus();
      },
      immediate: true
    }
  },
  beforeUnmount() {  // 将 beforeDestroy 改为 beforeUnmount
    document.removeEventListener('click', this.closeDropdown);
  }
};
</script>

<style scoped>
/* 调整整体导航栏样式，参照猫眼网站 */
.navbar {
  background-color: #ffffff; /* 改为白色背景 */
  padding: 10px 0; /* 调整顶部和底部的空隙 */
  border-bottom: 1px solid #cfc6c6; /* 新增黑色横线 */
}
.container {
  display: flex;
  justify-content: space-between;
  align-items: center;
  max-width: 1200px;
  margin: 0 auto;
  padding: 0 20px;
}
.logo {
  display: flex;
  align-items: center;
  text-decoration: none;
  height: 60px;
  padding: 5px 0;
  transition: opacity 0.3s ease;
}

.logo:hover {
  opacity: 0.8;
}

.logo img {
  height: 100%;
  width: auto;
  object-fit: contain;
  vertical-align: middle;
  filter: brightness(1.02); /* 略微提高亮度 */
}
.nav-links {
  list-style: none;
  display: flex;
  margin: 0;
  padding: 0;
}
.nav-links li {
  margin: 0 10px; /* 调整导航项间距 */
}
.nav-links a {
  color: #000;               /* 默认字体黑色 */
  font-size: 16px;           /* 字体大小 */
  text-decoration: none;
  padding-bottom: 5px;
  transition: color 0.3s ease, border-bottom 0.3s ease;
}
/* 修改后：仅对电影和预测激活样式应用黄色 */
.nav-links li.active a {
  color: #ffc107;            /* 点击进入时显示黄色 */
  border-bottom: 2px solid #ffc107;
}
.nav-actions {
  display: flex;
  align-items: center;
}
.search-box {
  display: flex;
  align-items: center;
  background-color: #ffffff; /* 搜索框白色背景 */
  border: 1px solid #000;    /* 黑色边框 */
  border-radius: 30px;
  overflow: hidden;
  margin-right: 10px; /* 调整搜索框右侧间距 */
}
.search-input {
  background: transparent;
  border: none;
  padding: 8px 10px;
  color: #000;               /* 搜索框字体改为黑色 */
  outline: none;
  width: 150px;
}
.search-button {
  background-color: #ffc107;
  border: none;
  padding: 8px 15px;
  cursor: pointer;
  color: #000;
}
.user-menu {
  position: relative;
}
.user-icon {
  height: 50px;
  width: 50px;
  border-radius: 50%;
  cursor: pointer;
}
/* 删除了 hover 相关规则，改为点击固定显示 */
/* .user-menu:hover .user-dropdown {
  display: block;
} */
.user-dropdown {
  display: block; /* 修改为 block，用 v-show 控制显示隐藏 */
  position: absolute;
  top: 45px;
  right: 0;
  background-color: #fff;
  border: 1px solid #ddd;
  border-radius: 4px;
  padding: 10px 0;
  box-shadow: 0 2px 8px rgba(0,0,0,0.15);
  min-width: 120px;
  z-index: 1000; /* 提高层级 */
}
.user-dropdown a {
  color: #333;
  text-decoration: none;
  display: block;
  padding: 8px 15px;
  transition: background-color 0.3s;
}
.user-dropdown a:hover {
  background-color: #f0f0f0;
}
.left-section {
  display: flex;
  align-items: center;
  gap: 20px;
}

.news-link {
  color: #000;
  font-size: 16px; /* 字体大小 */
  text-decoration: none;
  padding-bottom: 5px;
  transition: color 0.3s ease, border-bottom 0.3s ease;
}

.news-link.active {
  color: #ffc107;
  border-bottom: 2px solid #ffc107;
}
</style>