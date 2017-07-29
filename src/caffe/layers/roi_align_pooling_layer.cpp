#include <cfloat>
#include <vector>

#include "caffe/layers/roi_align_pooling_layer.hpp"

namespace caffe {

template <typename Dtype>
void ROIAlignPoolingLayer<Dtype>::LayerSetUp(
    const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  ROIAlignPoolingParameter roi_align_pool_param =
      this->layer_param_.roi_align_pooling_param();
  CHECK_GT(roi_align_pool_param.pooled_h(), 0)
      << "pooled_h must be > 0";
  CHECK_GT(roi_align_pool_param.pooled_w(), 0)
      << "pooled_w must be > 0";
  pooled_height_ = roi_align_pool_param.pooled_h();
  pooled_width_ = roi_align_pool_param.pooled_w();
  spatial_scale_ = roi_align_pool_param.spatial_scale();
  sample_num_ = roi_align_pool_param.sample_num();
}

template <typename Dtype>
void ROIAlignPoolingLayer<Dtype>::Reshape(
    const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  CHECK_EQ(bottom[1]->channels(), 5)
      << "roi input shape should be (R, 5) or (R, 5, 1, 1)";
  CHECK_EQ(bottom[1]->num() * bottom[1]->channels(), bottom[1]->count())
      << "roi input shape should be (R, 5) or (R, 5, 1, 1)";
  channels_ = bottom[0]->channels();
  height_ = bottom[0]->height();
  width_ = bottom[0]->width();
  top[0]->Reshape(bottom[1]->num(), channels_, pooled_height_,
      pooled_width_);
  argmax_pos_.Reshape(bottom[1]->num(), channels_,
      pooled_height_ * pooled_width_, 2);
}

template <typename Dtype>
void ROIAlignPoolingLayer<Dtype>::Forward_cpu(
    const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  NOT_IMPLEMENTED;
}

template <typename Dtype>
void ROIAlignPoolingLayer<Dtype>::Backward_cpu(
    const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  NOT_IMPLEMENTED;
}

#ifdef CPU_ONLY
STUB_GPU(ROIAlignPoolingLayer);
#endif

INSTANTIATE_CLASS(ROIAlignPoolingLayer);
REGISTER_LAYER_CLASS(ROIAlignPooling);

}  // namespace caffe
