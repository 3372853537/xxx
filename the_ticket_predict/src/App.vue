<template>
  <div id="app">
    <NavBar v-if="$route.meta.showNavbar" /> 
    <router-view /> <!-- 显示当前路由对应的页面组件 -->
    <Footer />
  </div>
</template>

<script setup>
import NavBar from "@/components/NavBar.vue";
import Footer from "./components/Footer.vue";
</script>
