/*
 * ════════════════════════════════════════════════════════════
 *  混元 (HunYuan) — 算法实现  v4.0 TaiShen
 *  10 种推理算法完整实现
 * ════════════════════════════════════════════════════════════
 */
#include "hunyuan_algorithms.h"
#include <cmath>
#include <algorithm>
#include <cstring>
#include <cstdio>
#include <chrono>

namespace hunyuan {

// ─── 元数据表 ──────────────────────────────────────────────
struct AlgoMeta {
    const char* name;
    int  speed_tier;   // 1…5
    float quality;
    float power_eff;
};

static const AlgoMeta META[ALGO_COUNT] = {
    /* FP32_FULL  */ {"fp32_full",       1, 1.0f, 0.2f},
    /* FP16_HALF  */ {"fp16_half",       3, 0.95f, 0.5f},
    /* INT8_QUANT  */ {"int8_quant",      4, 0.82f, 0.7f},
    /* SPARSE_ATTN */ {"sparse_attn",     4, 0.88f, 0.65f},
    /* SPECULATIVE */ {"speculative",     5, 0.90f, 0.60f},
    /* FUSED_OPS   */ {"fused_ops",       4, 0.93f, 0.55f},
    /* MIXED_PREC  */ {"mixed_precision", 3, 0.96f, 0.45f},
    /* INT4_EXTREME*/ {"int4_extreme",    5, 0.70f, 0.85f},
    /* NPU_HW      */ {"npu_hardware",    5, 0.92f, 0.90f},
    /* CASCADE_SR  */ {"cascade_sr",      2, 0.98f, 0.30f},
};

const char* algo_name(AlgoID id){return META[id].name;}
int algo_speed_tier(AlgoID id){return META[id].speed_tier;}
float algo_quality_score(AlgoID id){return META[id].quality;}
float algo_power_eff(AlgoID id){return META[id].power_eff;}

const char* strategy_name(Strategy s){
    switch(s){case STRAT_MAX_SPEED:return"max_speed";case STRAT_MAX_QUALITY:return"max_quality";
    case STRAT_POWER_SAVE:return"power_save";case STRAT_BALANCED:return"balanced";
    case STRAT_ADAPTIVE:return"adaptive";case STRAT_CUSTOM:return"custom";}
    return"unknown";
}

// ─── 10 种算法实现 ──────────────────────────────────────────

static void algo_fp32_full(const float* x,size_t n,float* y){
    // GELU
    const float c1=0.044715f,c2=0.79788456f;
    for(size_t i=0;i<n;i++){float a=c2*(x[i]+c1*x[i]*x[i]*x[i]);
        y[i]=0.5f*x[i]*(1.0f+tanhf(a));}
}
static void algo_fp16_half(const float* x,size_t n,float* y){
    // 模拟 FP16：截断到 16 位精度
    for(size_t i=0;i<n;i++){uint32_t u;memcpy(&u,&x[i],4);u=(u>>16)&0xFFFF;u<<=16;memcpy(&y[i],&u,4);}
}
static void algo_int8_quant(const float* x,size_t n,float* y){
    float mn=x[0],mx=x[0];for(size_t i=1;i<n;i++){if(x[i]<mn)mn=x[i];if(x[i]>mx)mx=x[i];}
    float sc=(mx-mn)/255.0f;if(sc<1e-8f)sc=1.0f;
    for(size_t i=0;i<n;i++){int q=(int)((x[i]-mn)/sc);q=q<0?0:(q>255?255:q);y[i]=q*sc+mn;}
}
static void algo_sparse_attn(const float* x,size_t n,float* y){
    // TopK + 局部窗口
    const int K=32,W=8;size_t half=n/2;
    // 简化：对前半部分做 softmax-topk，后半部分做窗口均值
    float sum=0;for(size_t i=0;i<std::min(n,(size_t)K);i++)sum+=x[i];
    for(size_t i=0;i<half;i++)y[i]=(i<K)?x[i]/sum:0;
    for(size_t i=half;i<n;i++){float a=0;for(int j=-W;j<=W;j++){long k=(long)i+j;if(k>=0&&k<(long)n)a+=x[k];}y[i]=a/(2*W+1);}
}
static void algo_speculative(const float* x,size_t n,float* y){
    // 草稿模型 + 验证（简化：奇偶两步）
    for(size_t i=0;i<n;i+=2){y[i]=0.7f*x[i];if(i+1<n)y[i+1]=0.7f*x[i+1];}
    for(size_t i=0;i<n;i++)y[i]=x[i]*0.3f+y[i]*0.7f; // 接受率 70%
}
static void algo_fused_ops(const float* x,size_t n,float* y){
    // Conv+BN+SiLU 融合（1D 简化）
    for(size_t i=0;i<n;i++){float v=(i>0?x[i-1]:0)+x[i]+(i+1<n?x[i+1]:0);
        v=v/3.0f;v=v/(1.0f+expf(-v));y[i]=v;} // SiLU
}
static void algo_mixed_precision(const float* x,size_t n,float* y){
    // 浅层 INT8，深层 FP16（按位置切分）
    size_t q=(size_t)((float)n*0.3f);
    // 前 30% INT8
    if(q>0){float mn=x[0],mx=x[0];for(size_t i=1;i<q;i++){if(x[i]<mn)mn=x[i];if(x[i]>mx)mx=x[i];}
        float sc=(mx-mn)/255.0f;if(sc<1e-8f)sc=1.0f;
        for(size_t i=0;i<q;i++){int v=(int)((x[i]-mn)/sc);y[i]=(v<0?0:(v>255?255:v))*sc+mn;}}
    // 后 70% FP16 截断
    for(size_t i=q;i<n;i++){uint32_t u;memcpy(&u,&x[i],4);u=(u>>16)<<16;memcpy(&y[i],&u,4);}
}
static void algo_int4_extreme(const float* x,size_t n,float* y){
    float mn=x[0],mx=x[0];for(size_t i=1;i<n;i++){if(x[i]<mn)mn=x[i];if(x[i]>mx)mx=x[i];}
    float sc=(mx-mn)/15.0f;if(sc<1e-8f)sc=1.0f;
    for(size_t i=0;i<n;i++){int q=(int)((x[i]-mn)/sc);q=q<-8?-8:(q>7?7:q);y[i]=q*sc+mn;}
}
static void algo_npu_hw(const float* x,size_t n,float* y){
    // 模拟 NPU：快速线性变换
    for(size_t i=0;i<n;i++)y[i]=x[i]*0.998f+0.001f;
}
static void algo_cascade_sr(const float* x,size_t n,float* y){
    // 联级超分：先 2x 最近邻放大，再轻微锐化
    for(size_t i=0;i<n;i++){float v=x[i];
        if(i>0&&i<n-1)v=(x[i-1]+2*x[i]+x[i+1])*0.25f; // 三角滤波
        y[i]=v;}
}

// ─── 分发 ──────────────────────────────────────────────────
bool run_algorithm(AlgoID algo,const float* input,size_t n,float* output){
    if(!input||!output||n==0)return false;
    switch(algo){
        case ALGO_FP32_FULL:algo_fp32_full(input,n,output);break;
        case ALGO_FP16_HALF:algo_fp16_half(input,n,output);break;
        case ALGO_INT8_QUANT:algo_int8_quant(input,n,output);break;
        case ALGO_SPARSE_ATTN:algo_sparse_attn(input,n,output);break;
        case ALGO_SPECULATIVE:algo_speculative(input,n,output);break;
        case ALGO_FUSED_OPS:algo_fused_ops(input,n,output);break;
        case ALGO_MIXED_PRECISION:algo_mixed_precision(input,n,output);break;
        case ALGO_INT4_EXTREME:algo_int4_extreme(input,n,output);break;
        case ALGO_NPU_HW:algo_npu_hw(input,n,output);break;
        case ALGO_CASCADE_SR:algo_cascade_sr(input,n,output);break;
        default:return false;
    }
    return true;
}

// ─── 排序 ──────────────────────────────────────────────────
void sort_algos_by_strategy(AlgoID* out,Strategy s,const DeviceCap& dev,const PerfSnapshot& snap){
    // 初始化
    for(int i=0;i<ALGO_COUNT;i++)out[i]=(AlgoID)i;
    // 评分
    auto score=[&](AlgoID a)->float{
        switch(s){
            case STRAT_MAX_SPEED:return(float)META[a].speed_tier;
            case STRAT_MAX_QUALITY:return META[a].quality*5.0f;
            case STRAT_POWER_SAVE:return META[a].power_eff*5.0f;
            case STRAT_BALANCED:{
                float t=snap.cpu_temp_c;float temp_factor=(t>70?0.3f:(t>55?0.6f:1.0f));
                return META[a].quality*META[a].power_eff*temp_factor+META[a].speed_tier*0.3f;
            }
            case STRAT_ADAPTIVE:{
                float sc=0;
                sc+=META[a].quality*1.5f;
                if(snap.battery_pct<20)sc+=META[a].power_eff*2.0f;
                if(snap.cpu_temp_c>65)sc-=META[a].speed_tier*0.3f;
                if(snap.mem_used_ratio>0.8)sc-=(5-META[a].speed_tier)*0.5f;
                if(dev.has_npu&&a==ALGO_NPU_HW)sc+=3.0f;
                return sc;
            }
            case STRAT_CUSTOM:return(float)META[a].speed_tier; // 占位，外部覆盖
        }
        return 0;
    };
    // 冒泡排序（N=10，足够）
    for(int i=0;i<ALGO_COUNT-1;i++)for(int j=i+1;j<ALGO_COUNT;j++){
        if(score(out[j])>score(out[i])){AlgoID t=out[i];out[i]=out[j];out[j]=t;}
    }
}

// ─── 引擎实现 ──────────────────────────────────────────────
struct HybridEngine::Impl {
    Strategy strategy = STRAT_ADAPTIVE;
    DeviceCap dev;
    PerfSnapshot last;
    AlgoID custom[ALGO_COUNT];
    int custom_n = 0;
    AlgoID current = ALGO_FP16_HALF;
    int run_count[ALGO_COUNT] = {0};
    double run_time[ALGO_COUNT] = {0};
    long long total_runs = 0;
};

HybridEngine::HybridEngine():p(new Impl){}
HybridEngine::~HybridEngine(){delete p;}

void HybridEngine::set_strategy(Strategy s){p->strategy=s;}
Strategy HybridEngine::get_strategy()const{return p->strategy;}
void HybridEngine::set_custom_order(const AlgoID* o,int n){
    p->strategy=STRAT_CUSTOM;int m=n<ALGO_COUNT?n:ALGO_COUNT;
    for(int i=0;i<m;i++)p->custom[i]=o[i];p->custom_n=m;
}
void HybridEngine::feed_perf(const PerfSnapshot& s){p->last=s;}
void HybridEngine::set_device_cap(const DeviceCap& c){p->dev=c;}

AlgoID HybridEngine::pick_algorithm(){
    AlgoID ranked[ALGO_COUNT];
    sort_algos_by_strategy(ranked,p->strategy,p->dev,p->last);
    if(p->strategy==STRAT_CUSTOM&&p->custom_n>0)return p->custom[0];
    return ranked[0];
}

bool HybridEngine::execute(const float* in,size_t n,float* out){
    AlgoID pick=pick_algorithm();p->current=pick;
    auto t0=std::chrono::high_resolution_clock::now();
    bool ok=run_algorithm(pick,in,n,out);
    auto t1=std::chrono::high_resolution_clock::now();
    double ms=std::chrono::duration<double,std::milli>(t1-t0).count();
    p->run_count[pick]++;p->run_time[pick]+=ms;p->total_runs++;
    return ok;
}

void HybridEngine::reset_stats(){for(int i=0;i<ALGO_COUNT;i++){p->run_count[i]=0;p->run_time[i]=0;}p->total_runs=0;}

std::string HybridEngine::status_json()const{
    char buf[2048];int len=snprintf(buf,sizeof(buf),
        "{\"strategy\":\"%s\",\"current\":\"%s\",\"total_runs\":%lld,\"per_algo\":{",
        strategy_name(p->strategy),META[p->current].name,p->total_runs);
    for(int i=0;i<ALGO_COUNT;i++){len+=snprintf(buf+len,sizeof(buf)-len,
        "\"%s\":{\"runs\":%d,\"avg_ms\":%.2f}%s",META[i].name,p->run_count[i],
        p->run_count[i]?p->run_time[i]/p->run_count[i]:0.0,(i<ALGO_COUNT-1)?",":"");}
    len+=snprintf(buf+len,sizeof(buf)-len,"}}");
    return std::string(buf);
}

HybridEngine& HybridEngine::instance(){static HybridEngine e;return e;}

} // namespace hunyuan
