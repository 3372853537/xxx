<template>
  <div class="admin-system">
    <h1>电影管理系统</h1>
    <div class="controls">
      <el-select v-model="currentCategory" placeholder="选择电影类别">
        <el-option label="正在上映" value="now"></el-option>
        <el-option label="即将上映" value="will"></el-option>
        <el-option label="TOP100" value="top100"></el-option>
      </el-select>
      <el-button type="primary" @click="showAddDialog">添加电影</el-button>
      <el-button @click="handleLogout">退出</el-button>
    </div>

    <!-- 电影列表 -->
    <el-table :data="movies" style="width: 100%">
      <el-table-column prop="id" label="ID" width="80"></el-table-column>
      <el-table-column prop="title" label="标题"></el-table-column>
      <el-table-column prop="release_time" label="上映时间"></el-table-column>
      <el-table-column label="海报" width="100">
        <template #default="scope">
          <el-image 
            style="width: 50px; height: 70px" 
            :src="getPoster(scope.row.poster)" 
            :preview-src-list="[scope.row.poster]">
          </el-image>
        </template>
      </el-table-column>
      <el-table-column label="操作" width="200">
        <template #default="scope">
          <el-button size="small" @click="handleEdit(scope.row)">编辑</el-button>
          <el-button 
            size="small" 
            type="danger" 
            @click="handleDelete(scope.row)">删除</el-button>
        </template>
      </el-table-column>
    </el-table>

    <!-- 添加/编辑对话框 -->
    <el-dialog 
      :title="dialogTitle" 
      v-model="dialogVisible"
      width="70%">
      <el-form :model="movieForm" label-width="120px">
        <el-form-item label="电影标题">
          <el-input v-model="movieForm.title"></el-input>
        </el-form-item>
        <el-form-item label="上映时间">
          <el-date-picker
            v-model="movieForm.release_time"
            type="datetime"
            placeholder="选择日期时间">
          </el-date-picker>
        </el-form-item>
        <el-form-item label="期待人数">
          <el-input v-model="movieForm.whis_count" type="number"></el-input>
        </el-form-item>
        <el-form-item label="海报URL">
          <el-input v-model="movieForm.poster"></el-input>
        </el-form-item>
        <el-form-item label="剧情简介">
          <el-input type="textarea" v-model="movieForm.plot" rows="4"></el-input>
        </el-form-item>
        <el-form-item label="导演海报">
          <el-input v-model="movieForm.directors_poster"></el-input>
        </el-form-item>
        <el-form-item label="导演">
          <el-tag
            v-for="(director, index) in movieForm.directors"
            :key="index"
            closable
            @close="removeDirector(index)"
            style="margin-right: 5px">
            {{ director }}
          </el-tag>
          <el-input
            v-if="inputVisible.director"
            ref="directorInput"
            v-model="inputValue.director"
            size="small"
            @keyup.enter="addDirector"
            @blur="addDirector"
            style="width: 100px">
          </el-input>
          <el-button v-else size="small" @click="showInput('director')">
            + 添加导演
          </el-button>
        </el-form-item>
        <el-form-item label="演员">
          <div v-for="(actor, index) in movieForm.actors" :key="index" style="margin-bottom: 10px">
            <el-input v-model="actor.actor" placeholder="演员名" style="width: 200px; margin-right: 10px"></el-input>
            <el-input v-model="actor.role" placeholder="角色" style="width: 200px; margin-right: 10px"></el-input>
            <el-input v-model="actor.actor_poster" placeholder="演员海报URL" style="width: 300px; margin-right: 10px"></el-input>
            <el-button type="danger" size="small" @click="removeActor(index)">删除</el-button>
          </div>
          <el-button type="primary" size="small" @click="addActor">添加演员</el-button>
        </el-form-item>
      </el-form>
      <template #footer>
        <span class="dialog-footer">
          <el-button @click="dialogVisible = false">取消</el-button>
          <el-button type="primary" @click="handleSubmit">确认</el-button>
        </span>
      </template>
    </el-dialog>
  </div>
</template>

<script>
import axios from 'axios';
import { ElMessage, ElMessageBox } from 'element-plus';
const images = require.context('@/assets', true, /\.(png|jpe?g|svg)$/);

export default {
  name: 'AdminMovieSystem',
  data() {
    return {
      currentCategory: 'now',
      movies: [],
      dialogVisible: false,
      dialogTitle: '添加电影',
      isEdit: false,
      movieForm: {
        title: '',
        release_time: '',
        whis_count: '',
        poster: '',
        plot: '',
        directors_poster: '',
        directors: [],
        actors: []
      },
      inputVisible: {
        director: false
      },
      inputValue: {
        director: ''
      }
    };
  },
  watch: {
    currentCategory: {
      handler() {
        this.fetchMovies();
      },
      immediate: true
    }
  },
  methods: {
    getPoster(path) {
      if (path.startsWith('../assets/')) {
        let imgPath = './' + path.slice(10);
        // 如果扩展名前缺少点号，则添加点号（例如将 "3.1jpg" 转为 "3.1.jpg"）
        imgPath = imgPath.replace(/(\d+)(jpe?g|png|svg)$/i, '$1.$2');
        return images(imgPath);
      }
      return path;
    },
    async fetchMovies() {
      try {
        const categoryMap = {
          now: 'nowmovies',
          will: 'willmovies',
          top100: 'top100movies'
        };
        
        const baseURL = 'http://localhost:5000';
        const url = `${baseURL}/admin/${categoryMap[this.currentCategory]}`;
        
        console.log('Fetching movies from:', url);
        
        const response = await axios.get(url, {
          headers: {
            'Content-Type': 'application/json'
          },
          withCredentials: true
        });

        console.log('Response:', response);

        if (response.data) {
          this.movies = response.data;
          console.log('Movies loaded:', this.movies.length);
        } else {
          console.warn('Empty response data');
          ElMessage.warning('获取到的电影列表为空');
          this.movies = [];
        }
      } catch (error) {
        console.error('获取电影列表错误:', error);
        // 详细的错误信息处理
        if (error.response) {
          // 服务器返回了错误状态码
          console.error('Error response:', error.response);
          console.error('Error status:', error.response.status);
          console.error('Error data:', error.response.data);
          ElMessage.error(`服务器错误 (${error.response.status}): ${error.response.data.error || '未知错误'}`);
        } else if (error.request) {
          // 请求已发出，但没有收到响应
          console.error('No response received:', error.request);
          ElMessage.error('服务器无响应，请检查网络连接或联系管理员');
        } else {
          // 请求配置出错
          console.error('Error config:', error.config);
          ElMessage.error('请求配置错误：' + error.message);
        }
        this.movies = [];
      }
    },
    showAddDialog() {
      this.isEdit = false;
      this.dialogTitle = '添加电影';
      this.resetForm();
      this.dialogVisible = true;
    },
    handleEdit(row) {
      this.isEdit = true;
      this.dialogTitle = '编辑电影';
      this.movieForm = JSON.parse(JSON.stringify(row));
      this.dialogVisible = true;
    },
    async handleDelete(row) {
      try {
        await ElMessageBox.confirm('确认删除该电影？', '警告', {
          type: 'warning'
        });
        const categoryMap = {
          now: 'nowmovies',
          will: 'willmovies',
          top100: 'top100movies'
        };
        
        const baseURL = 'http://localhost:5000';
        const url = `${baseURL}/admin/${categoryMap[this.currentCategory]}/delete/${row.id}`;
        
        console.log('Deleting movie:', url);
        
        const response = await axios.delete(url, {
          headers: {
            'Content-Type': 'application/json'
          },
          withCredentials: true
        });

        if (response.data && response.data.message) {
          ElMessage.success(response.data.message);
          await this.fetchMovies();
        } else {
          throw new Error('删除响应数据异常');
        }
      } catch (error) {
        console.error('删除错误:', error);
        if (error.response) {
          ElMessage.error(`删除失败: ${error.response.data?.error || '服务器错误'}`);
        } else if (error.request) {
          ElMessage.error('删除失败: 服务器无响应');
        } else if (error !== 'cancel') {
          ElMessage.error(`删除失败: ${error.message || '未知错误'}`);
        }
      }
    },
    async handleSubmit() {
      try {
        const categoryMap = {
          now: 'nowmovies',
          will: 'willmovies',
          top100: 'top100movies'
        };
        const baseURL = 'http://localhost:5000'; // 设置后端基础地址
        if (this.isEdit) {
          await axios.put(`${baseURL}/admin/${categoryMap[this.currentCategory]}/update/${this.movieForm.id}`, this.movieForm);
        } else {
          await axios.post(`${baseURL}/admin/${categoryMap[this.currentCategory]}/add`, this.movieForm);
        }
        ElMessage.success(`${this.isEdit ? '更新' : '添加'}成功`);
        this.dialogVisible = false;
        this.fetchMovies();
      } catch (error) {
        ElMessage.error(`${this.isEdit ? '更新' : '添加'}失败`);
      }
    },
    resetForm() {
      this.movieForm = {
        title: '',
        release_time: '',
        whis_count: '',
        poster: '',
        plot: '',
        directors_poster: '',
        directors: [],
        actors: []
      };
    },
    showInput(type) {
      this.inputVisible[type] = true;
      this.$nextTick(() => {
        this.$refs[`${type}Input`].focus();
      });
    },
    addDirector() {
      if (this.inputValue.director) {
        this.movieForm.directors.push(this.inputValue.director);
        this.inputValue.director = '';
      }
      this.inputVisible.director = false;
    },
    removeDirector(index) {
      this.movieForm.directors.splice(index, 1);
    },
    addActor() {
      this.movieForm.actors.push({
        actor: '',
        role: '',
        actor_poster: ''
      });
    },
    removeActor(index) {
      this.movieForm.actors.splice(index, 1);
    },
    handleLogout() {
      this.$router.push('/user-login');
    }
  }
};
</script>

<style scoped>
.admin-system {
  padding: 20px;
}

.controls {
  margin-bottom: 20px;
  display: flex;
  gap: 10px;
}

h1 {
  color: #333;
  margin-bottom: 30px;
}

.dialog-footer {
  display: flex;
  justify-content: flex-end;
  gap: 10px;
}
</style>