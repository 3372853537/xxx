// 设置登录信息
export const setLoginInfo = (username) => {
  localStorage.setItem('username', username);
  localStorage.setItem('loginTime', Date.now());
};

// 获取登录信息
export const getLoginInfo = () => {
  const username = localStorage.getItem('username');
  const loginTime = localStorage.getItem('loginTime');
  return {
    username,
    loginTime,
    isLoggedIn: !!username
  };
};

// 清除登录信息
export const clearLoginInfo = () => {
  localStorage.removeItem('username');
  localStorage.removeItem('loginTime');
};
