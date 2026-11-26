import React, {useEffect, useMemo, useState} from 'react';
import {createRoot} from 'react-dom/client';
import {MultiTrackAudioMixer, audioTrackDragMime} from './MultiTrackAudioMixer';
import {useTranslations, reportMixerError} from './adapter';
import type {OutputFile} from './types';

type Track = {file:OutputFile; src:string};
type Payload = {files:Array<{url:string; orig_name?:string}|null>; language:string; status?:string; theme?:string};
const roles=['final','converted_vocals','original_vocals','lead_vocals','backing_vocals','accompaniment','accompaniment_without_harmony'];
const labelKeys=['audioStemFinal','audioStemConvertedVocals','audioStemOriginalVocals','audioStemLeadVocals','audioStemBackingVocals','audioStemAccompaniment','audioStemAccompanimentWithoutHarmony'];
const colors=[
  'var(--track-color-0)',
  'var(--track-color-1)',
  'var(--track-color-2)',
  'var(--track-color-3)',
  'var(--track-color-4)',
  'var(--track-color-5)',
  'var(--track-color-6)',
];

function applyTheme(theme?: string) {
  let next = theme;
  if (next !== 'light' && next !== 'dark') {
    try {
      const saved = localStorage.getItem('ai-rvc-theme');
      if (saved === 'light' || saved === 'dark') next = saved;
    } catch {}
  }
  if (next !== 'light' && next !== 'dark') {
    next = window.matchMedia && window.matchMedia('(prefers-color-scheme: dark)').matches ? 'dark' : 'light';
  }
  document.documentElement.dataset.theme = next;
  document.documentElement.style.colorScheme = next;
}

function Result({payload}:{payload:Payload}) {
  const t=useTranslations();
  const [added,setAdded]=useState<string[]>([]);
  const [error,setError]=useState('');
  const [mixerGeneration,setMixerGeneration]=useState(0);
  const all=useMemo<Track[]>(()=>payload.files.flatMap((file,index)=>{
    if(!file) return [];
    if(!file.url) throw new Error(t('invalidFile'));
    const url=new URL(file.url,document.baseURI);
    if(url.origin!==new URL(document.baseURI).origin || !['http:','https:'].includes(url.protocol)) throw new Error(t('invalidFile'));
    return [{file:{name:file.orig_name || `${roles[index]}.wav`, s3_key:roles[index], stem_key:roles[index]},src:url.href}];
  }),[payload,t]);
  const vocalsOnly=!payload.files[0] && !payload.files[5];
  const base=useMemo(()=>all.filter(x=>(vocalsOnly?['converted_vocals','original_vocals']:['converted_vocals','accompaniment']).includes(x.file.stem_key)).map(x=>x.file.s3_key),[all,vocalsOnly]);
  const selected=useMemo(()=>all.filter(x=>base.includes(x.file.s3_key)||added.includes(x.file.s3_key)),[all,base,added]);
  const available=useMemo(()=>all.filter(x=>!selected.includes(x)),[all,selected]);
  useEffect(()=>{
    const fail=(event:Event)=>setError((event as CustomEvent).detail);
    window.addEventListener('rvc-player-error',fail);
    return ()=>window.removeEventListener('rvc-player-error',fail);
  },[]);
  useEffect(()=>{
    // A standalone comparison player and the mixer must never sound together.
    const stopMixer=(event:Event)=>{
      if(event.target instanceof HTMLAudioElement && event.target.closest('.output-card')) {
        document.querySelectorAll<HTMLAudioElement>('.output-card audio').forEach(a=>{if(a!==event.target)a.pause();});
        window.dispatchEvent(new Event('rvc-pause-mixer'));
      }
    };
    document.addEventListener('play',stopMixer,true);
    return ()=>document.removeEventListener('play',stopMixer,true);
  },[]);
  if(!all.length) {
    return (
      <div className="player-empty">
        <span className="player-empty-icon" aria-hidden="true">
          <svg viewBox="0 0 24 24" width="20" height="20" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
            <path d="M2 10v3" /><path d="M6 6v11" /><path d="M10 3v18" /><path d="M14 8v7" /><path d="M18 5v13" /><path d="M22 10v3" />
          </svg>
        </span>
        <p className="player-empty-text">{t(payload.status==='processing'?'processing':'empty')}</p>
      </div>
    );
  }
  return <>
    {error ? <div role="alert" className="player-error"><strong>{t('failed')}</strong><p>{t('errorAdvice')}</p><details><summary>{t('errorDetails')}</summary>{error}</details><button onClick={()=>{setError('');setMixerGeneration(v=>v+1);}}>{t('retry')}</button></div> :
    <MultiTrackAudioMixer key={mixerGeneration} tracks={selected} availableTracks={available}
      initialMutedTrackIds={vocalsOnly?['original_vocals',...added]:added} removableTrackIds={added}
      onAddTrack={id=>setAdded(prev=>prev.includes(id)?prev:[...prev,id])}
      onRemoveTrack={id=>setAdded(prev=>prev.filter(x=>x!==id))}
      stemLabels={vocalsOnly?{original_vocals:t('audioStemInputVocals')}:undefined}
      description={t(vocalsOnly?'vocalsDescription':'description')} />}
    <p className="player-note">{t('previewNote')}</p>
    <div className="output-grid">{all.map(track=>{
      const index=roles.indexOf(track.file.stem_key), label=t(vocalsOnly && index===2?'audioStemInputVocals':labelKeys[index]);
      const canAdd=available.includes(track);
      return <article key={track.file.s3_key} className={`output-card ${(index===0 || (vocalsOnly && index===1))?'hero':''}`} style={{'--track-color':colors[index]} as React.CSSProperties}
        draggable={canAdd} onDragStart={e=>{e.dataTransfer.setData(audioTrackDragMime,track.file.s3_key);}}>
        <header><strong>{label}</strong><a className="output-download" href={track.src} download={track.file.name}>{t('download')}</a></header>
        <p>{track.file.name}</p>
        <audio controls preload="metadata" src={track.src} aria-label={label} onError={()=>reportMixerError(`${label}：${t('mediaFailed')}`)} />
        {canAdd&&<button type="button" className="output-add-btn" onClick={()=>setAdded(prev=>prev.includes(track.file.s3_key)?prev:[...prev,track.file.s3_key])}>{t('audioMixerAddTrack',{name:label})}</button>}
      </article>;
    })}</div>
  </>;
}

class Boundary extends React.Component<{children:React.ReactNode},{error:string}> {
  state={error:''};
  static getDerivedStateFromError(error:Error){return {error:error.message};}
  render(){
    const t=useTranslations();
    return this.state.error?<div role="alert" className="player-error"><strong>{t('failed')}</strong><details><summary>{t('errorDetails')}</summary>{this.state.error}</details></div>:this.props.children;
  }
}

const root=createRoot(document.getElementById('root')!);
let generation=0;
function render(payload:Payload, preserveSession=false){
  if (payload.theme) applyTheme(payload.theme);
  document.documentElement.lang=payload.language==='en_US'?'en':'zh';
  if(!preserveSession) generation++;
  root.render(<Boundary key={generation}><Result payload={payload}/></Boundary>);
}
applyTheme();
window.addEventListener('message',event=>{
  if(event.source!==parent||event.origin!==new URL(document.baseURI).origin)return;
  if(event.data?.type==='rvc-theme'){
    applyTheme(event.data.theme);
    return;
  }
  if(!['rvc-tracks','rvc-language'].includes(event.data?.type))return;
  render(event.data.payload,event.data.type==='rvc-language');
});
document.addEventListener('wheel',event=>{
  if(event.target instanceof HTMLInputElement && ['range','number'].includes(event.target.type)) event.preventDefault();
},{passive:false,capture:true});
render({files:[],language:document.documentElement.lang==='en'?'en_US':'zh_CN'});
new ResizeObserver(()=>parent.postMessage({type:'rvc-player-height',height:document.body.scrollHeight},new URL(document.baseURI).origin)).observe(document.body);
parent.postMessage({type:'rvc-player-ready'},new URL(document.baseURI).origin);
