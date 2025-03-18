<template>
  <div class="login-container">
    <div v-if="isLogin" class="login-box">
      <h2 class="login-title">登录</h2>
      <div class="role-selector">
        <label>
          <input type="radio" v-model="role" value="user" /> 用户
        </label>
        <label>
          <input type="radio" v-model="role" value="admin" /> 管理员
        </label>
      </div>
      <form @submit.prevent="handleLogin">
        <div class="form-group">
          <label for="username">用户名</label>
          <input
            type="text"
            id="username"
            v-model="loginUsername"
            placeholder="请输入用户名"
            class="form-input"
          />
        </div>
        <div class="form-group">
          <label for="password">密码</label>
          <input
            type="password"
            id="password"
            v-model="loginPassword"
            placeholder="请输入密码"
            class="form-input"
          />
        </div>
        <div class="form-group">
          <button type="submit" class="login-button" :disabled="loading">登录</button>
        </div>
        <div class="form-group">
          <p @click="toggleToRegister">没有账号？去注册</p>
        </div>
        <div v-if="loading" class="loading-message">正在处理请求...</div>
        <div v-if="errorMessage" class="error-message">{{ errorMessage }}</div>
      </form>
    </div>
    <div v-else class="register-box">
      <h2 class="register-title">注册</h2>
      <div class="role-selector">
        <label>
          <input type="radio" v-model="role" value="user" checked disabled /> 用户
        </label>
      </div>
      <form @submit.prevent="handleRegister">
        <div class="form-group">
          <label for="registerUsername">用户名</label>
          <input
            type="text"
            id="registerUsername"
            v-model="registerUsername"
            placeholder="请输入用户名"
            class="form-input"
          />
        </div>
        <div class="form-group">
          <label for="registerPassword">密码</label>
          <input
            type="password"
            id="registerPassword"
            v-model="registerPassword"
            placeholder="请输入密码"
            class="form-input"
          />
        </div>
        <div class="form-group">
          <button type="submit" class="register-button" :disabled="loading">注册</button>
        </div>
        <div class="form-group">
          <p @click="toggleToLogin">已有账号？去登录</p>
        </div>
        <div v-if="loading" class="loading-message">正在处理请求...</div>
        <div v-if="errorMessage" class="error-message">{{ errorMessage }}</div>
      </form>
    </div>
  </div>
</template>

<script setup>
import { ref } from 'vue';
import { useRouter } from 'vue-router';
import axios from 'axios';
import { setLoginInfo } from '../utils/auth';

const router = useRouter();
const isLogin = ref(true);
const loginUsername = ref('');
const loginPassword = ref('');
const registerUsername = ref('');
const registerPassword = ref('');
const errorMessage = ref('');
const loading = ref(false);
const role = ref('user');

const toggleToRegister = () => {
  isLogin.value = false;
  errorMessage.value = '';
};

const toggleToLogin = () => {
  isLogin.value = true;
  errorMessage.value = '';
};

const handleRegister = async () => {
  if (!registerUsername.value || !registerPassword.value) {
    errorMessage.value = '用户名和密码不能为空';
    return;
  }

  loading.value = true;
  errorMessage.value = '';

  try {
    const response = await axios.post('http://localhost:5000/register', {
      username: registerUsername.value,
      password: registerPassword.value
    });

    if (response.status === 201) {
      alert('注册成功');
      toggleToLogin();
      registerUsername.value = '';
      registerPassword.value = '';
    }
  } catch (error) {
    if (error.response?.status === 409) {
      errorMessage.value = '用户名已存在';
    } else {
      errorMessage.value = error.response?.data?.message || '注册失败，请重试';
    }
  } finally {
    loading.value = false;
  }
};

const handleLogin = async () => {
  if (!loginUsername.value || !loginPassword.value) {
    errorMessage.value = '用户名和密码不能为空';
    return;
  }

  loading.value = true;
  errorMessage.value = '';

  try {
    const endpoint = role.value === 'admin' ? '/admin/login' : '/login';
    const response = await axios.post(`http://localhost:5000${endpoint}`, {
      username: loginUsername.value,
      password: loginPassword.value
    });

    if (response.status === 200) {
      sessionStorage.setItem('loginSuccess', 'true');
      sessionStorage.setItem('userRole', role.value);
      // 根据角色跳转到不同页面
      if (role.value === 'admin') {
        router.push('/admin-movie-system');  // 管理员界面
      } else {
        router.push('/');      // 用户界面
      }
      const username = '用户名'; // 从登录响应中获取
      setLoginInfo(username);
      // 删除下面这一行，避免统一跳转到用户主页
      // router.push('/'); // 登录成功后跳转
    }
  } catch (error) {
    if (error.response?.status === 401) {
      errorMessage.value = '用户名或密码错误';
    } else {
      errorMessage.value = error.response?.data?.message || '登录失败，请重试';
    }
  } finally {
    loading.value = false;
  }
};
</script>

<style scoped>
.login-container {
  display: flex;
  justify-content: center;
  align-items: center;
  min-height: 100vh;
  background-color: #f4f4f4;
}

.login-box,
.register-box {
  width: 360px;
  background-color: #fff;
  padding: 30px;
  border-radius: 5px;
  box-shadow: 0 0 10px rgba(0, 0, 0, 0.1);
}

.login-title,
.register-title {
  text-align: center;
  margin-bottom: 20px;
  font-size: 24px;
  color: #333;
}

.form-group {
  margin-bottom: 15px;
}

.form-group label {
  display: block; 
  margin-bottom: 5px;
  font-size: 16px;
  color: #666;
}

.form-input {
  width: 100%;
  padding: 10px;
  border: 1px solid #ccc;
  border-radius: 3px;
  outline: none;
  font-size: 16px;
}

.login-button,
.register-button {
  width: 100%;
  padding: 10px;
  background-color: #ffc107;
  border: none;
  border-radius: 3px;
  font-size: 16px;
  color: #000;
  cursor: pointer;
}

.error-message {
  color: red;
  text-align: center;
  margin-top: 10px;
}

.loading-message {
  text-align: center;
  color: #666;
  margin-top: 10px;
}

.login-button:disabled,
.register-button:disabled {
  background-color: #ccc;
  cursor: not-allowed;
}

.role-selector {
  margin-bottom: 15px;
  text-align: center;
}

.role-selector label {
  margin: 0 10px;
  cursor: pointer;
}

.role-selector input[type="radio"] {
  margin-right: 5px;
}
</style>