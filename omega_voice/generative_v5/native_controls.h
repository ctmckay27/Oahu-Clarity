/* Mari's explicitly selected single-request CPU trajectory adapter.
 * Bank directions are learned from measured same-speaker acoustic contrasts.
 * No direction prose, named emotional presets, or speaker changes enter here.
 * Binary MTR1: u32 B,L,D,F; f32 bank[B,L,D]; f32 weights[F,B].
 */
#include <stdint.h>
#include <math.h>
static float *mari_bank=NULL,*mari_weights=NULL;
static uint32_t mari_B=0,mari_L=0,mari_D=0,mari_F=0;
static int mari_trajectory_init(qwen_tts_ctx_t *ctx) {
    const char *p=getenv("MARI_TRAJECTORY");
    if (!p || !*p) return 0;
    if (mari_bank || ctx->ml_steer) {
        fprintf(stderr,"MARI: trajectory is single-request and excludes other steering\n");return -1;
    }
    FILE *f=fopen(p,"rb"); uint32_t head[5];
    if (!f) {perror("MARI_TRAJECTORY");return -1;}
    if(fread(head,4,5,f)!=5 || head[0]!=0x3152544d || head[1]<1 || head[1]>16 ||
       head[2]!=(uint32_t)ctx->config.num_layers+1 || head[3]!=(uint32_t)ctx->config.hidden_size ||
       head[4]<1 || head[4]>8192) {fclose(f);fprintf(stderr,"MARI: invalid trajectory header\n");return -1;}
    mari_B=head[1];mari_L=head[2];mari_D=head[3];mari_F=head[4];
    size_t nb=(size_t)mari_B*mari_L*mari_D,nw=(size_t)mari_F*mari_B;
    mari_bank=malloc(nb*sizeof(float));mari_weights=malloc(nw*sizeof(float));
    if(!mari_bank||!mari_weights){fclose(f);return -1;}
    int ok=fread(mari_bank,4,nb,f)==nb && fread(mari_weights,4,nw,f)==nw && fgetc(f)==EOF;
    fclose(f);
    /* Bit checks remain valid under the engine's -ffast-math build. */
    for(size_t i=0;i<nb && ok;i++){uint32_t bits;memcpy(&bits,mari_bank+i,4);if((bits&0x7f800000u)==0x7f800000u||fabsf(mari_bank[i])>100)ok=0;}
    for(size_t i=0;i<nw && ok;i++){uint32_t bits;memcpy(&bits,mari_weights+i,4);if((bits&0x7f800000u)==0x7f800000u||fabsf(mari_weights[i])>2)ok=0;}
    if(!ok){fprintf(stderr,"MARI: malformed, nonfinite or out-of-envelope trajectory\n");return -1;}
    ctx->ml_steer=calloc((size_t)mari_L*mari_D,sizeof(float));if(!ctx->ml_steer)return -1;
    ctx->ml_steer_layers=mari_L;ctx->ml_steer_dim=mari_D;
    ctx->ml_steer_l0=0;ctx->ml_steer_l1=ctx->config.num_layers-1;
    ctx->ml_steer_weight=1.0f;ctx->ml_steer_w_eff=0.0f;
    fprintf(stderr,"MARI_NATIVE_TRAJECTORY banks=%u layers=%u dim=%u frames=%u; CPU single-request; no instruct\n",mari_B,mari_L,mari_D,mari_F);
    return 0;
}
static void mari_trajectory_step(qwen_tts_ctx_t *ctx,int frame) {
    if(!mari_bank)return;
    size_t ld=(size_t)mari_L*mari_D;memset(ctx->ml_steer,0,ld*sizeof(float));
    ctx->ml_steer_w_eff=0.0f;
    if(frame<0||(uint32_t)frame>=mari_F)return;
    for(uint32_t b=0;b<mari_B;b++) {
        float w=mari_weights[(size_t)frame*mari_B+b];if(w==0)continue;
        const float *v=mari_bank+(size_t)b*ld;
        for(size_t i=0;i<ld;i++)ctx->ml_steer[i]+=w*v[i];
        ctx->ml_steer_w_eff=1.0f;
    }
}
