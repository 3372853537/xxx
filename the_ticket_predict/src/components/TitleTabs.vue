<template>
  <div class="title-tabs">
    <div
      v-for="(tab, index) in tabs"
      :key="index"
      :class="{ active: currentTab === tab.value }"
      @click="changeTab(tab.value)"
    >
      {{ tab.label }}
    </div>
  </div>
</template>

<script>
export default {
  name: "TitleTabs",
  props: {
    currentTab: {
      type: String,
      default: "hot",
    },
  },
  data() {
    return {
      tabs: [
        { label: "正在热映", value: "hot" },
        { label: "即将上映", value: "coming" },
        { label: "Top100", value: "top100" }  // 新增的 Top100 标签
      ],
    };
  },
  methods: {
    changeTab(tab) {
      this.$emit("tab-change", tab);
    },
  },
};
</script>

<style scoped>
.title-tabs {
  display: flex;
  justify-content: flex-start; /* 改为靠左对齐 */
  margin-bottom: 20px;
  margin-left: 500px; /* 添加左边距 */
}

.title-tabs div {
  padding: 10px 20px;
  cursor: pointer;
  font-size: 18px;
  color: #666;
  transition: color 0.3s ease;
}

.title-tabs div.active {
  color: #333;
  border-bottom: 2px solid #ffc107;
}
</style>