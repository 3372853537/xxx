// import { createRouter, createWebHistory } from "vue-router";
import MovieHome from "@/views/MovieHome.vue";
import MovieList from "@/views/MovieList.vue";
import { createRouter, createWebHistory } from "vue-router";
import MoviePrediction from "@/views/MoviePrediction.vue";
import UserLogin from "@/views/UserLogin.vue";
import UserDetail from "@/views/UserDetail.vue";
import MovieDetail from "@/views/MovieDetail.vue";
import MovieBoxOffice from "@/views/MovieBoxOffice.vue";
import MovieNews from "@/views/MovieNews.vue";
import AdminMovieSystem from "@/views/AdminMovieSystem.vue";
const routes = [
  {
    path: "/", 
    component: MovieHome,
    meta: {
      showNavbar: true
    }
  },
  { 
    path: "/movie-list", 
    component: MovieList,
    meta: {
      showNavbar: true
    }
   },
  {
    path: "/movie-prediction",
    component: MoviePrediction,
    meta: {
      showNavbar: true
    }
  },
  {
    path: "/user-login",
    component: UserLogin, 
    meta: {
    showNavbar: false
    }
  },
  {
    path: "/user-detail",
    component:UserDetail,
    meta: {
      showNavbar: true
    }
  },
  {
    path: "/movie-detail/:title",
    name: "MovieDetail",
    component: MovieDetail,
    meta: {
      showNavbar: true
    }
  },
  {
    path: "/movie-box-office",
    name: "MovieBoxOffice",
    component: MovieBoxOffice,
    meta: {
      showNavbar: true
    }
  },
  {
    path:"/movie-news",
    name:"MovieNews",
    component:MovieNews,
    meta: {
      showNavbar: true
    }
  },
  {
    path:"/admin-movie-system",
    name:"AdminMovieSystem",
    component:AdminMovieSystem
  }
];

const router = createRouter({
  history: createWebHistory(),
  routes,
});

export default router;
