#include <torch/extension.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <limits>
#include <numeric>
#include <vector>

namespace {

struct Vec3 { float x, y, z; };
struct Triangle { Vec3 a, b, c, centroid, lo, hi; };
struct Node { Vec3 lo, hi; int left = -1, right = -1, start = 0, count = 0; };

inline Vec3 vmin(Vec3 a, Vec3 b) { return {std::min(a.x,b.x), std::min(a.y,b.y), std::min(a.z,b.z)}; }
inline Vec3 vmax(Vec3 a, Vec3 b) { return {std::max(a.x,b.x), std::max(a.y,b.y), std::max(a.z,b.z)}; }
inline Vec3 sub(Vec3 a, Vec3 b) { return {a.x-b.x,a.y-b.y,a.z-b.z}; }
inline Vec3 cross(Vec3 a, Vec3 b) { return {a.y*b.z-a.z*b.y,a.z*b.x-a.x*b.z,a.x*b.y-a.y*b.x}; }
inline float dot(Vec3 a, Vec3 b) { return a.x*b.x+a.y*b.y+a.z*b.z; }
inline float axis(Vec3 a, int i) { return i == 0 ? a.x : (i == 1 ? a.y : a.z); }

class MeshBVH {
 public:
  MeshBVH(torch::Tensor vertices, torch::Tensor faces) {
    TORCH_CHECK(vertices.device().is_cpu() && faces.device().is_cpu(), "mesh tensors must be CPU");
    vertices = vertices.contiguous().to(torch::kFloat32);
    faces = faces.contiguous().to(torch::kInt64);
    TORCH_CHECK(vertices.dim()==2 && vertices.size(1)==3, "vertices must be Nx3");
    TORCH_CHECK(faces.dim()==2 && faces.size(1)==3, "faces must be Mx3");
    auto v = vertices.accessor<float,2>();
    auto f = faces.accessor<int64_t,2>();
    triangles_.reserve(faces.size(0));
    for (int64_t i=0;i<faces.size(0);++i) {
      Vec3 p[3];
      for (int j=0;j<3;++j) {
        auto index=f[i][j];
        TORCH_CHECK(index>=0 && index<vertices.size(0), "face index out of range");
        p[j]={v[index][0],v[index][1],v[index][2]};
      }
      Vec3 lo=vmin(p[0],vmin(p[1],p[2])), hi=vmax(p[0],vmax(p[1],p[2]));
      triangles_.push_back({p[0],p[1],p[2],{(p[0].x+p[1].x+p[2].x)/3.0f,(p[0].y+p[1].y+p[2].y)/3.0f,(p[0].z+p[1].z+p[2].z)/3.0f},lo,hi});
    }
    order_.resize(triangles_.size());
    std::iota(order_.begin(),order_.end(),0);
    nodes_.reserve(triangles_.size()*2);
    build(0, static_cast<int>(order_.size()));
  }

  std::vector<torch::Tensor> intersect(torch::Tensor origins, torch::Tensor directions) const {
    TORCH_CHECK(origins.device().is_cpu() && directions.device().is_cpu(), "rays must be CPU");
    origins=origins.contiguous().to(torch::kFloat32);
    directions=directions.contiguous().to(torch::kFloat32);
    TORCH_CHECK(origins.sizes()==directions.sizes() && origins.dim()==2 && origins.size(1)==3,"rays must be matching Nx3");
    auto near=torch::zeros({origins.size(0)},torch::kFloat32);
    auto far=torch::zeros({origins.size(0)},torch::kFloat32);
    auto counts=torch::zeros({origins.size(0)},torch::kInt32);
    auto o=origins.accessor<float,2>(); auto d=directions.accessor<float,2>();
    auto n=near.accessor<float,1>(); auto f=far.accessor<float,1>(); auto c=counts.accessor<int32_t,1>();
    #pragma omp parallel for schedule(dynamic, 64)
    for (int64_t ray=0;ray<origins.size(0);++ray) {
      Vec3 origin{o[ray][0],o[ray][1],o[ray][2]}, direction{d[ray][0],d[ray][1],d[ray][2]};
      std::vector<int> stack; stack.reserve(64); stack.push_back(0);
      std::vector<float> hits; hits.reserve(8);
      while (!stack.empty()) {
        int node_index=stack.back(); stack.pop_back();
        const Node& node=nodes_[node_index];
        if (!hit_aabb(origin,direction,node.lo,node.hi)) continue;
        if (node.count) {
          for (int i=0;i<node.count;++i) {
            float t;
            if (hit_triangle(origin,direction,triangles_[order_[node.start+i]],t)) hits.push_back(t);
          }
        } else { stack.push_back(node.left); stack.push_back(node.right); }
      }
      if (hits.empty()) continue;
      std::sort(hits.begin(),hits.end());
      std::vector<float> unique; unique.reserve(hits.size());
      for (float t:hits) {
        float eps=1e-5f*std::max(1.0f,std::abs(t));
        if (unique.empty() || std::abs(t-unique.back())>eps) unique.push_back(t);
      }
      c[ray]=static_cast<int32_t>(unique.size());
      n[ray]=unique.front(); f[ray]=unique.back();
    }
    return {near,far,counts};
  }

 private:
  std::vector<Triangle> triangles_; std::vector<int> order_; std::vector<Node> nodes_;
  int build(int start,int end) {
    int index=static_cast<int>(nodes_.size()); nodes_.push_back({});
    Vec3 lo{INFINITY,INFINITY,INFINITY},hi{-INFINITY,-INFINITY,-INFINITY};
    Vec3 clo=lo,chi=hi;
    for(int i=start;i<end;++i){const auto&t=triangles_[order_[i]];lo=vmin(lo,t.lo);hi=vmax(hi,t.hi);clo=vmin(clo,t.centroid);chi=vmax(chi,t.centroid);}
    nodes_[index].lo=lo;nodes_[index].hi=hi;
    if(end-start<=8){nodes_[index].start=start;nodes_[index].count=end-start;return index;}
    Vec3 extent=sub(chi,clo);int split=(extent.y>extent.x)?1:0;if(axis(extent,2)>axis(extent,split))split=2;
    int middle=(start+end)/2;
    std::nth_element(order_.begin()+start,order_.begin()+middle,order_.begin()+end,[&](int a,int b){return axis(triangles_[a].centroid,split)<axis(triangles_[b].centroid,split);});
    int left=build(start,middle),right=build(middle,end);nodes_[index].left=left;nodes_[index].right=right;return index;
  }
  static bool hit_aabb(Vec3 o,Vec3 d,Vec3 lo,Vec3 hi){
    float tmin=0.0f,tmax=std::numeric_limits<float>::infinity();
    for(int a=0;a<3;++a){float ov=axis(o,a),dv=axis(d,a),lv=axis(lo,a),hv=axis(hi,a);if(std::abs(dv)<1e-12f){if(ov<lv||ov>hv)return false;continue;}float t0=(lv-ov)/dv,t1=(hv-ov)/dv;if(t0>t1)std::swap(t0,t1);tmin=std::max(tmin,t0);tmax=std::min(tmax,t1);if(tmax<tmin)return false;}return true;
  }
  static bool hit_triangle(Vec3 o,Vec3 d,const Triangle&t,float&out){
    Vec3 e1=sub(t.b,t.a),e2=sub(t.c,t.a),p=cross(d,e2);float det=dot(e1,p);if(std::abs(det)<1e-8f)return false;float inv=1.0f/det;Vec3 s=sub(o,t.a);float u=dot(s,p)*inv;if(u<-1e-6f||u>1.000001f)return false;Vec3 q=cross(s,e1);float v=dot(d,q)*inv;if(v<-1e-6f||u+v>1.000001f)return false;float hit=dot(e2,q)*inv;if(hit<=1e-6f)return false;out=hit;return true;
  }
};

}  // namespace

PYBIND11_MODULE(TORCH_EXTENSION_NAME, module) {
  pybind11::class_<MeshBVH>(module,"MeshBVH")
      .def(pybind11::init<torch::Tensor,torch::Tensor>())
      .def("intersect",&MeshBVH::intersect);
}
