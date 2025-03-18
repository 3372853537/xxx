<template>
  <div class="profile-page">
    <div class="profile-banner"></div>
    <div class="profile-content">
      <header class="profile-header">
        <div class="avatar-container">
          <img class="avatar" :src="avatar" alt="用户头像" />
          <div v-if="isEditing" class="avatar-upload">
            <label for="avatar-input" class="upload-label">更换头像</label>
            <input
              id="avatar-input"
              type="file"
              accept="image/*"
              @change="handleAvatarChange"
              style="display: none"
            />
          </div>
        </div>
        <div class="info">
          <h2 class="username">
            <template v-if="isEditing">
              <input v-model="username" />
            </template>
            <template v-else>
              {{ username }}
            </template>
          </h2>
        </div>
      </header>
      <nav class="profile-nav">
        <ul>
          <li><a href="#">动态</a></li>
          <li><a href="#">影评</a></li>
          <li><a href="#">收藏</a></li>
          <li><a href="#">关注</a></li>
          <li><a href="#">粉丝</a></li>
        </ul>
      </nav>
      <section class="profile-info">
        <div v-if="!isEditing">
          <ul>
            <li>账号：{{ account }}</li>
            <li>邮箱：{{ email }}</li>
            <li>性别：{{ gender }}</li>
            <li>生日：{{ birthday }}</li>
            <li>生活状态：{{ status }}</li>
            <li>行业：{{ industry }}</li>
            <li>兴趣爱好：{{ hobbies }}</li>
            <li>个性签名：{{ signature }}</li>
          </ul>
          <button class="edit-btn" @click="editProfile">编辑资料</button>
        </div>
        <div v-else>
          <ul>
            <li>账号：<input v-model="account" /></li>
            <li>邮箱：<input v-model="email" /></li>
            <li>性别：
              <select v-model="gender">
                <option value="男">男</option>
                <option value="女">女</option>
                <option value="其他">其他</option>
              </select>
            </li>
            <li>生日：<input type="date" v-model="birthday" /></li>
            <li>生活状态：<input v-model="status" /></li>
            <li>行业：<input v-model="industry" /></li>
            <li>兴趣爱好：<input v-model="hobbies" /></li>
            <li>个性签名：<input v-model="signature" /></li>
          </ul>
          <button class="save-btn" @click="saveProfile">保存</button>
          <button class="cancel-btn" @click="cancelEdit">取消</button>
        </div>
      </section>
    </div>
  </div>
</template>

<script setup>
import { ref, onMounted } from 'vue'
import { saveUserProfile, getUserProfile } from '../utils/userProfile'
import { userAvatar } from '../store/userStore'

const isEditing = ref(false)
const account = ref('')
const email = ref('')
const username = ref('')
const gender = ref('')
const birthday = ref('')
const status = ref('')
const industry = ref('')
const hobbies = ref('')
const signature = ref('')
const avatar = ref(userAvatar)

// 初始化加载数据
onMounted(() => {
  const profile = getUserProfile()
  if (profile) {
    account.value = profile.account
    email.value = profile.email
    username.value = profile.username
    gender.value = profile.gender
    birthday.value = profile.birthday
    status.value = profile.status
    industry.value = profile.industry
    hobbies.value = profile.hobbies
    signature.value = profile.signature
    avatar.value = profile.avatar || userAvatar
  }
})

const handleAvatarChange = (event) => {
  const file = event.target.files[0]
  if (file) {
    const reader = new FileReader()
    reader.onload = (e) => {
      avatar.value = e.target.result
    }
    reader.readAsDataURL(file)
  }
}

function editProfile() {
  isEditing.value = true
}

function saveProfile() {
  const profile = {
    account: account.value,
    email: email.value,
    username: username.value,
    gender: gender.value,
    birthday: birthday.value,
    status: status.value,
    industry: industry.value,
    hobbies: hobbies.value,
    signature: signature.value,
    avatar: avatar.value
  }
  saveUserProfile(profile)
  isEditing.value = false
}

function cancelEdit() {
  isEditing.value = false
  // 在此处可重置数据为初始值
}
</script>

<style scoped>
.profile-page {
  position: relative;
  min-height: 100vh;
  background: #f2f2f2;
  display: flex;
  flex-direction: column;
  align-items: center;
}
.profile-banner {
  width: 100%;
  height: 120px; /* 增加banner高度 */
  background-image: url('https://via.placeholder.com/1200x200');
  background-size: cover;
  background-position: center;
}
.profile-content {
  width: 90%; /* 增加宽度占比 */
  max-width: 1200px; /* 增加最大宽度 */
  margin-top: -60px;
  background: #fff;
  border-radius: 12px;
  padding: 40px; /* 增加内边距 */
  box-shadow: 0 4px 12px rgba(0, 0, 0, 0.1);
}
.profile-header {
  display: flex;
  align-items: center;
  margin-bottom: 30px;
}
.avatar-container {
  position: relative;
  margin-right: 40px; /* 增加间距 */
}
.avatar {
  width: 150px; /* 增加头像尺寸 */
  height: 150px;
  border-radius: 50%;
  border: 6px solid #fff;
}
.avatar-upload {
  position: absolute;
  bottom: 0;
  left: 0;
  right: 0;
  background-color: rgba(0, 0, 0, 0.5);
  padding: 5px;
  text-align: center;
  cursor: pointer;
}
.upload-label {
  color: white;
  font-size: 14px;
  cursor: pointer;
}
.info {
  display: flex;
  flex-direction: column;
}
.username {
  font-size: 36px; /* 增加用户名字体大小 */
  font-weight: bold;
  margin: 0;
}
.user-level {
  font-size: 16px;
  color: #777;
}
.user-bio {
  font-size: 14px;
  color: #999;
  margin-top: 10px;
}
.profile-nav {
  margin-bottom: 30px;
}
.profile-nav ul {
  display: flex;
  list-style: none;
  padding: 0;
  margin: 0;
  justify-content: center; /* 导航居中 */
}
.profile-nav li {
  margin: 0 30px; /* 增加导航项间距 */
}
.profile-nav a {
  text-decoration: none;
  color: #333;
  font-size: 18px; /* 增加导航字体大小 */
  font-weight: 500;
}
.profile-nav a:hover {
  color: #f60;
}
.profile-info ul {
  list-style: none;
  padding: 0;
  max-width: 800px; /* 限制信息区域宽度 */
  margin: 0 auto; /* 信息区域居中 */
}
.profile-info li {
  margin-bottom: 20px; /* 增加信息项间距 */
  font-size: 18px; /* 增加信息字体大小 */
  padding: 10px 0;
  border-bottom: 1px solid #eee;
}
.edit-btn, .save-btn, .cancel-btn {
  padding: 12px 30px; /* 增加按钮大小 */
  font-size: 18px;
  background-color: #f60;
  color: #fff;
  border: none;
  border-radius: 6px;
  cursor: pointer;
  margin-right: 15px;
  transition: background-color 0.3s;
}
.edit-btn:hover, .save-btn:hover, .cancel-btn:hover {
  background-color: #e55a00;
}
input, select {
  padding: 8px 15px;
  font-size: 18px;
  border: 1px solid #ddd;
  border-radius: 6px;
  width: 300px; /* 增加输入框宽度 */
}
</style>