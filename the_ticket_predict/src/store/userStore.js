import { ref } from 'vue'
import { getUserProfile } from '../utils/userProfile'

// 从本地存储初始化头像
const profile = getUserProfile()
export const userAvatar = ref(profile?.avatar || 'https://via.placeholder.com/120')

export const updateUserAvatar = (newAvatarUrl) => {
  userAvatar.value = newAvatarUrl
}
