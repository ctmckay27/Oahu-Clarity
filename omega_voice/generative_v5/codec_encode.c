/* Diagnostic codec extraction through the project's native encoder API.
 * This does not extract/change speaker identity or perform TTS generation.
 */
#include "qwen_tts.h"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
int main(int argc,char **argv) {
 if(argc!=4){fprintf(stderr,"usage: mari_codec_encode MODEL WAV CODES\n");return 2;}
 float *y=NULL;int n=0,sr=0;
 if(qwen_read_wav(argv[2],&y,&n,&sr)!=0 || sr!=24000 || n<2400 || n>24000*120){fprintf(stderr,"invalid calibration audio\n");return 3;}
 for(int i=0;i<n;i++)if(!isfinite(y[i]))return 3;
 qwen_tts_ctx_t *ctx=qwen_tts_load_ex(argv[1],0,0,0);if(!ctx)return 4;
 if(qwen_speech_encoder_load(ctx)!=0){fprintf(stderr,"codec encoder unavailable; no fallback\n");return 5;}
 int *codes=NULL,frames=0;
 if(qwen_speech_encoder_encode(ctx,y,n,&codes,&frames)!=0 || frames<=0)return 6;
 FILE *f=fopen(argv[3],"w");if(!f)return 7;
 for(int i=0;i<frames;i++){
  for(int j=0;j<16;j++){int c=codes[i*16+j];if(c<0||c>2047)return 8;fprintf(f,"%d%s",c,j==15?"\n":" ");}
 }
 int failed=fflush(f)!=0||ferror(f);if(fclose(f)!=0)failed=1;
 if(failed)return 9;
 fprintf(stderr,"MARI_CODEC_EXTRACT frames=%d channels=16 identity_unchanged=true\n",frames);
 free(codes);free(y);qwen_tts_unload(ctx);return 0;
}
