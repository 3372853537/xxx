<template>
  <footer class="footer">
    <!-- ...可能添加的logo或者其他内容... -->
    <div class="footer-content">
      <div class="links">
        <a href="https://www.maoyan.com" target="_blank">先知电影</a>
        <a href="/about">关于我们</a>
        <a href="/contact">联系我们</a>
        <!-- 新增内容 -->
        <a href="/careers">加入我们</a>
        <a href="/privacy">隐私政策</a>
      </div>
      <!-- 新增社交媒体区域 -->
      <div class="extra">
        <p>关注我们：</p>
        <div class="social">
          <a href="https://weibo.com" target="_blank">微博</a>
          <a href="https://wechat.com" target="_blank">微信</a>
          <a href="https://twitter.com" target="_blank">Twitter</a>
        </div>
      </div>
      <div class="copyright">
        © 2025 先知电影 版权信息. 版权所有.
      </div>
    </div>
  </footer>
</template>

<script>
export default {
  name: "Footer-Bar"
};
</script>

<style scoped>
.footer {
  background-color: #f7f7f7;
  border-top: 1px solid #ddd;
  padding: 20px 0;
  text-align: center;
  font-size: 14px;
  color: #333;
}
.footer-content {
  max-width: 1200px;
  margin: 0 auto;
}
.links a {
  margin: 0 10px;
  color: #333;
  text-decoration: none;
  transition: color 0.3s;
}
.links a:hover {
  color: #ffc107;
}
/* 新增社交媒体及附加区域样式 */
.extra {
  margin: 15px 0;
}
.extra p {
  margin: 0 0 5px;
}
.social a {
  margin: 0 8px;
  color: #333;
  text-decoration: none;
  transition: color 0.3s;
}
.social a:hover {
  color: #ffc107;
}
</style>
