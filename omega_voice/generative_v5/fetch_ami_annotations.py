"""Retrieve selected public AMI annotations using verified ZIP byte ranges."""
import argparse,pathlib,requests,struct,io,zipfile,zlib,binascii,hashlib,json

def main():
 ap=argparse.ArgumentParser();ap.add_argument('output');a=ap.parse_args();p=pathlib.Path(a.output);p.mkdir(parents=True,exist_ok=True)
 url='https://groups.inf.ed.ac.uk/ami/AMICorpusAnnotations/ami_public_manual_1.6.2.zip';session=requests.Session();ranges=[]
 def get(spec):
  r=session.get(url,headers={'Range':'bytes='+spec},timeout=60);r.raise_for_status()
  if r.status_code!=206:raise ValueError('range retrieval not supported; no silent full download')
  ranges.append({'request':spec,'content_range':r.headers.get('Content-Range'),'etag':r.headers.get('ETag'),'sha256':hashlib.sha256(r.content).hexdigest()});return r.content,r.headers['Content-Range']
 tail,content_range=get('-65536');total=int(content_range.split('/')[-1]);end=tail.rfind(b'PK\x05\x06')
 if end<0:raise ValueError('ZIP end record absent')
 fields=struct.unpack_from('<4s4H2IH',tail,end);size,offset=fields[5:7]
 central,_=get(f'{offset}-{offset+size-1}');mem=io.BytesIO(b'\0'*total);mem.seek(offset);mem.write(central);mem.seek(total-len(tail));mem.write(tail)
 archive=zipfile.ZipFile(mem);selected=[x for x in archive.infolist() if ('ES2002a' in x.filename and any(k in x.filename for k in ['words','dialogueActs'])) or 'da-types' in x.filename or 'meetings.xml' in x.filename]
 files=[]
 for item in selected:
  start=item.header_offset;body,_=get(f'{start}-{min(total-1,start+30+len(item.filename.encode())+65536+item.compress_size)}')
  if body[:4]!=b'PK\x03\x04':raise ValueError('bad local header')
  name_len,extra_len=struct.unpack_from('<HH',body,26);begin=30+name_len+extra_len;compressed=body[begin:begin+item.compress_size]
  data=zlib.decompress(compressed,-15) if item.compress_type==8 else compressed
  if len(data)!=item.file_size or binascii.crc32(data)&0xffffffff!=item.CRC:raise ValueError('ZIP file integrity mismatch')
  relative=pathlib.PurePosixPath(item.filename)
  if relative.is_absolute() or '..' in relative.parts:raise ValueError('unsafe archive path')
  out=p/'annotations'/relative;out.parent.mkdir(parents=True,exist_ok=True);out.write_bytes(data)
  files.append({'path':str(relative),'sha256':hashlib.sha256(data).hexdigest(),'zip_crc32':item.CRC,'bytes':len(data)});print(relative,flush=True)
 (p/'RANGE_ANNOTATION_PROVENANCE.json').write_text(json.dumps({'source':url,'version':'1.6.2','license':'CC BY 4.0','archive_total_bytes':total,'files':files,'range_requests':ranges,'role':'human evaluator reference; not Mari identity source'},indent=2)+'\n')
if __name__=='__main__':main()
