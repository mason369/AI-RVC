"use client";

import {
  useCallback,
  useEffect,
  useMemo,
  useRef,
  useState,
  type CSSProperties,
} from "react";
import { useTranslations } from "./adapter";
import {
  ChevronsLeftRight,
  GripHorizontal,
  Headphones,
  Pause,
  Play,
  Plus,
  RotateCcw,
  Volume2,
  VolumeX,
  X,
} from "lucide-react";
import { cn, reportMixerError } from "./adapter";
import type { OutputFile } from "./types";

interface AudioMixerTrackInput {
  file: OutputFile;
  src: string;
  waveformSrc?: string;
  waveformPeaksSrc?: string;
}

interface AudioMixerTrack extends AudioMixerTrackInput {
  id: string;
  label: string;
  color: string;
}

interface MultiTrackAudioMixerProps {
  tracks: AudioMixerTrackInput[];
  availableTracks?: AudioMixerTrackInput[];
  initialMutedTrackIds?: string[];
  removableTrackIds?: string[];
  onAddTrack?: (trackId: string) => void;
  onRemoveTrack?: (trackId: string) => void;
  stemLabels?: Partial<Record<string, string>>;
  title?: string;
  contextLabel?: string;
  description?: string;
  className?: string;
}

type OutputFileWithStem = OutputFile & { stem_key?: string | null };
type WaveformPeaksEnvelope = {
  peaks?: unknown;
  mins?: unknown;
  maxs?: unknown;
  duration_seconds?: unknown;
  durationSeconds?: unknown;
  data?: {
    peaks?: unknown;
    mins?: unknown;
    maxs?: unknown;
    duration_seconds?: unknown;
    durationSeconds?: unknown;
  };
};
const stemLabelTranslationKeys = {
  vocals: "audioStemVocals",
  vocals_with_harmony: "audioStemVocalsWithHarmony",
  vocals_without_harmony: "audioStemVocalsWithoutHarmony",
  original_vocals: "audioStemOriginalVocals",
  converted_vocals: "audioStemConvertedVocals",
  lead_vocals: "audioStemLeadVocals",
  backing_vocals: "audioStemBackingVocals",
  back_vocals: "audioStemBackingVocals",
  accompaniment: "audioStemAccompaniment",
  accompaniment_without_harmony: "audioStemAccompanimentWithoutHarmony",
  no_vocals: "audioStemAccompanimentWithoutHarmony",
  accompaniment_with_harmony: "audioStemAccompanimentWithHarmony",
  instrumental: "audioStemInstrumental",
  drums: "audioStemDrums",
  bass: "audioStemBass",
  other: "audioStemOther",
  guitar: "audioStemGuitar",
  piano: "audioStemPiano",
  final: "audioStemFinal",
  speech: "audioStemSpeech",
} as const;
const builtInStemLabels: Partial<Record<string, string>> = {
  kick: "Kick",
  snare: "Snare",
  toms: "Toms",
  cymbals: "Cymbals",
};
type StemLabelTranslationKey =
  (typeof stemLabelTranslationKeys)[keyof typeof stemLabelTranslationKeys];
const standaloneHarmonyFallbackLabels = new Set([
  "harmony",
  "和声",
  "和聲",
  "ハーモニー",
  "화음",
]);
interface TimeRange {
  start: number;
  end: number;
}
type AudioGainGraph = {
  context: AudioContext;
  sources: Record<string, MediaElementAudioSourceNode>;
  gains: Record<string, GainNode>;
};

const defaultTimelineSeconds = 30;
const maxTrackOffsetSeconds = 60;
const minTrackGainDb = -2.5;
const maxTrackGainDb = 2.5;
const trackGainDbStep = 0.1;
const fallbackMinTimelineZoom = 0.65;
const absoluteMinTimelineZoom = 0.001;
const maxTimelineZoom = 6;
const timelinePixelsPerSecond = 18;
const minTimelineWidth = 720;
const maxTimelineWidth = 120000;
const minTimelineViewportWidth = 1;
const timelineFitPadding = 56;
const playbackBufferAheadSeconds = 0.55;
const mediaReadyStateCanPlay = 3;
const waveformSampleCount = 2048;
const trackColors = [
  "var(--color-accent)",
  "var(--tool-color-voice)",
  "var(--color-success)",
  "var(--tool-color-midi)",
  "var(--tool-color-image)",
  "var(--color-warning)",
];
const emptyTrackIds: string[] = [];
type WaveformLoadResult = {
  peaks: number[];
  mins: number[] | null;
  maxs: number[] | null;
  durationSeconds: number | null;
};
type WaveformCacheValue = WaveformLoadResult | Promise<WaveformLoadResult>;
const waveformCache = new Map<string, WaveformCacheValue>();
export const audioTrackDragMime = "application/x-telknet-audio-track";

export function clearAudioMixerWaveformCacheForTests() {
  if (process.env.NODE_ENV !== "production") {
    waveformCache.clear();
  }
}

function finiteDuration(value: number | undefined) {
  return typeof value === "number" && Number.isFinite(value) && value > 0
    ? value
    : null;
}

function clamp(value: number, min: number, max: number) {
  if (!Number.isFinite(value)) return min;
  return Math.min(max, Math.max(min, value));
}

function sameTimeRanges(left: TimeRange[], right: TimeRange[]) {
  return (
    left.length === right.length &&
    left.every(
      (range, index) =>
        Math.abs(range.start - right[index].start) < 0.001 &&
        Math.abs(range.end - right[index].end) < 0.001,
    )
  );
}

function mergeTimeRanges(ranges: TimeRange[]) {
  const sorted = ranges
    .filter((range) => range.end - range.start > 0.01)
    .sort((left, right) => left.start - right.start);
  const merged: TimeRange[] = [];

  sorted.forEach((range) => {
    const previous = merged.at(-1);
    if (previous && range.start <= previous.end + 0.12) {
      previous.end = Math.max(previous.end, range.end);
      return;
    }
    merged.push({ ...range });
  });

  return merged;
}

function mediaTimeRangesToArray(ranges: TimeRanges) {
  const next: TimeRange[] = [];
  for (let index = 0; index < ranges.length; index += 1) {
    next.push({
      start: ranges.start(index),
      end: ranges.end(index),
    });
  }
  return next;
}

function roundOffset(value: number) {
  return Math.round(value * 20) / 20;
}

function formatTime(value: number) {
  if (!Number.isFinite(value) || value < 0) return "--:--";
  const totalSeconds = Math.floor(value);
  const minutes = Math.floor(totalSeconds / 60);
  const seconds = totalSeconds % 60;
  return `${minutes}:${seconds.toString().padStart(2, "0")}`;
}

function formatOffset(value: number) {
  if (Math.abs(value) < 0.005) return "0.00s";
  return `${value > 0 ? "+" : ""}${value.toFixed(2)}s`;
}

function roundGainDb(value: number) {
  return Math.round(value * 10) / 10;
}

function formatGainDb(value: number) {
  const roundedValue = Math.abs(value) < 0.05 ? 0 : roundGainDb(value);
  return `${roundedValue > 0 ? "+" : ""}${roundedValue.toFixed(1)} dB`;
}

function gainDbToLinearGain(value: number) {
  return 10 ** (clamp(value, minTrackGainDb, maxTrackGainDb) / 20);
}

function timelineFitWidth(viewportWidth: number) {
  if (viewportWidth <= 0) return minTimelineWidth;
  return Math.max(minTimelineViewportWidth, viewportWidth - timelineFitPadding);
}

function minZoomForTimeline(duration: number, viewportWidth: number) {
  if (viewportWidth <= 0) return fallbackMinTimelineZoom;
  const rawFitZoom =
    timelineFitWidth(viewportWidth) /
    Math.max(duration * timelinePixelsPerSecond, 1);
  return clamp(rawFitZoom, absoluteMinTimelineZoom, 1);
}

function timelineWidthForZoom(
  duration: number,
  zoom: number,
  viewportWidth = 0,
) {
  const minWidth =
    viewportWidth > 0
      ? Math.min(minTimelineWidth, timelineFitWidth(viewportWidth))
      : minTimelineWidth;
  return Math.round(
    clamp(duration * timelinePixelsPerSecond * zoom, minWidth, maxTimelineWidth),
  );
}

function audioHasBufferedTime(
  audio: HTMLAudioElement,
  localTime: number,
  duration: number | null,
) {
  if (duration != null && localTime >= duration - 0.08) return true;
  if (audio.readyState < mediaReadyStateCanPlay) return false;

  const targetEnd =
    duration == null
      ? localTime + playbackBufferAheadSeconds
      : Math.min(duration, localTime + playbackBufferAheadSeconds);
  const buffered = audio.buffered;

  if (buffered.length === 0) return audio.readyState >= 4;

  for (let index = 0; index < buffered.length; index += 1) {
    if (buffered.start(index) <= localTime + 0.05 && buffered.end(index) >= targetEnd) {
      return true;
    }
  }

  return false;
}

function seekAudioToTime(audio: HTMLAudioElement, time: number) {
  try {
    if (typeof audio.fastSeek === "function") {
      audio.fastSeek(time);
    } else {
      audio.currentTime = time;
    }
    return true;
  } catch {
    return false;
  }
}

function setAudioVolume(audio: HTMLAudioElement, volume: number) {
  if (Math.abs(audio.volume - volume) > 0.001) {
    audio.volume = volume;
  }
}

function setAudioMuted(audio: HTMLAudioElement, muted: boolean) {
  if (audio.muted !== muted) {
    audio.muted = muted;
  }
}

function setGainValue(gain: GainNode, value: number) {
  if (Math.abs(gain.gain.value - value) > 0.0001) {
    gain.gain.value = value;
  }
}

function pauseAudio(audio: HTMLAudioElement) {
  if (audio.paused) return;
  try {
    audio.pause();
  } catch {
    // Tests and some locked-down browsers may not implement pause fully.
  }
}

function requestAudioLoad(audio: HTMLAudioElement) {
  if (
    typeof navigator !== "undefined" &&
    navigator.userAgent.toLowerCase().includes("jsdom")
  ) {
    return;
  }
  try {
    audio.load();
  } catch {
    // Some test/browser media shims do not implement load.
  }
}

function audioHasAttachedSource(audio: HTMLAudioElement) {
  return Boolean(audio.getAttribute("src"));
}

function attachAudioSource(audio: HTMLAudioElement, src: string, reload = false) {
  const nextSrc = reload ? mediaReloadSource(src) : src;
  const absoluteNextSrc = absoluteMediaSource(nextSrc);
  if (audioHasAttachedSource(audio) && audio.src === absoluteNextSrc) {
    return false;
  }

  audio.src = nextSrc;
  return true;
}

function absoluteMediaSource(src: string) {
  if (typeof window === "undefined") return src;
  try {
    return new URL(src, window.location.href).toString();
  } catch {
    return src;
  }
}

function mediaReloadSource(src: string) {
  if (typeof window === "undefined") return src;
  try {
    const url = new URL(src, window.location.href);
    if (
      url.origin === window.location.origin &&
      url.pathname.includes("/api/v1/tasks/") &&
      url.searchParams.get("redirect") !== "false"
    ) {
      url.searchParams.set("_audioReload", String(Date.now()));
    }
    return url.toString();
  } catch {
    return src;
  }
}

function audioContextConstructor() {
  const audioWindow = window as Window &
    typeof globalThis & {
      webkitAudioContext?: typeof AudioContext;
    };
  return audioWindow.AudioContext ?? audioWindow.webkitAudioContext ?? null;
}

function scaleWaveformBounds(mins: number[], maxs: number[]) {
  const maxAbs = Math.max(
    0.001,
    ...mins.map((value) => Math.abs(value)),
    ...maxs.map((value) => Math.abs(value)),
  );
  return {
    mins: mins.map((value) => clamp(value / maxAbs, -1, 1)),
    maxs: maxs.map((value) => clamp(value / maxAbs, -1, 1)),
  };
}

function extractWaveformPeaks(audioBuffer: AudioBuffer) {
  const channelCount = Math.max(1, audioBuffer.numberOfChannels);
  const bucketSize = Math.max(1, Math.floor(audioBuffer.length / waveformSampleCount));
  const mins: number[] = [];
  const maxs: number[] = [];
  const rawPeaks = Array.from({ length: waveformSampleCount }, (_, bucketIndex) => {
    const start = bucketIndex * bucketSize;
    const end =
      bucketIndex === waveformSampleCount - 1
        ? audioBuffer.length
        : Math.min(audioBuffer.length, start + bucketSize);
    const stride = Math.max(1, Math.floor((end - start) / 220));
    let minSample = 0;
    let maxSample = 0;

    for (let sampleIndex = start; sampleIndex < end; sampleIndex += stride) {
      let mixedSample = 0;
      for (let channelIndex = 0; channelIndex < channelCount; channelIndex += 1) {
        const channel = audioBuffer.getChannelData(channelIndex);
        mixedSample += channel[sampleIndex] ?? 0;
      }
      mixedSample /= channelCount;
      minSample = Math.min(minSample, mixedSample);
      maxSample = Math.max(maxSample, mixedSample);
    }

    mins.push(minSample);
    maxs.push(maxSample);
    return Math.max(Math.abs(minSample), Math.abs(maxSample));
  });
  const maxPeak = Math.max(...rawPeaks, 0.001);
  const bounds = scaleWaveformBounds(mins, maxs);
  return {
    peaks: rawPeaks.map((peak) => clamp(Math.sqrt(peak / maxPeak) * 94, 6, 94)),
    mins: bounds.mins,
    maxs: bounds.maxs,
  };
}

function steppedWaveformPath(
  tops: number[],
  bottoms: number[],
) {
  const pointCount = Math.min(tops.length, bottoms.length);
  if (pointCount === 0) return "";
  const maxX = 1000;
  const bucketWidth = maxX / pointCount;
  const topPoints: string[] = [];
  const bottomPoints: string[] = [];

  for (let index = 0; index < pointCount; index += 1) {
    const x0 = index * bucketWidth;
    const x1 = index === pointCount - 1 ? maxX : (index + 1) * bucketWidth;
    const topY = tops[index];
    const bottomY = bottoms[index];
    topPoints.push(`${x0.toFixed(2)} ${topY.toFixed(2)}`);
    topPoints.push(`${x1.toFixed(2)} ${topY.toFixed(2)}`);
    bottomPoints.push(`${x1.toFixed(2)} ${bottomY.toFixed(2)}`);
    bottomPoints.push(`${x0.toFixed(2)} ${bottomY.toFixed(2)}`);
  }

  return `M ${topPoints.join(" L ")} L ${bottomPoints.reverse().join(" L ")} Z`;
}

function waveformSvgPath(waveform: WaveformLoadResult) {
  const minCount = waveform.mins?.length ?? 0;
  const maxCount = waveform.maxs?.length ?? 0;
  if (minCount > 0 && minCount === maxCount) {
    const tops = waveform.maxs!.map((value) =>
      clamp(50 - clamp(value, -1, 1) * 47, 2, 98),
    );
    const bottoms = waveform.mins!.map((value) =>
      clamp(50 - clamp(value, -1, 1) * 47, 2, 98),
    );
    return steppedWaveformPath(
      tops.map((top, index) => Math.min(top, bottoms[index] - 1)),
      bottoms,
    );
  }

  if (waveform.peaks.length === 0) return "";
  const tops = waveform.peaks.map((peak) => {
    const halfHeight = clamp(peak * 0.5, 2, 47);
    return 50 - halfHeight;
  });
  const bottoms = waveform.peaks.map((peak) => {
    const halfHeight = clamp(peak * 0.5, 2, 47);
    return 50 + halfHeight;
  });
  return steppedWaveformPath(tops, bottoms);
}

function normalizeWaveformNumberArray(
  candidate: unknown,
  min: number,
  max: number,
) {
  if (!Array.isArray(candidate)) return null;
  const values = candidate
    .map((value) => Number(value))
    .filter((value) => Number.isFinite(value))
    .map((value) => clamp(value, min, max));
  return values.length > 0 ? values : null;
}

function waveformDataFromEnvelope(
  body: WaveformPeaksEnvelope,
): WaveformLoadResult | null {
  const peaksCandidate = Array.isArray(body.peaks) ? body.peaks : body.data?.peaks;
  const peaks = normalizeWaveformNumberArray(peaksCandidate, 0, 100);
  if (!peaks) return null;
  const minsCandidate = Array.isArray(body.mins) ? body.mins : body.data?.mins;
  const maxsCandidate = Array.isArray(body.maxs) ? body.maxs : body.data?.maxs;
  const mins = normalizeWaveformNumberArray(minsCandidate, -1, 1);
  const maxs = normalizeWaveformNumberArray(maxsCandidate, -1, 1);
  const normalizedMins = mins && maxs && mins.length === maxs.length ? mins : null;
  const normalizedMaxs = mins && maxs && mins.length === maxs.length ? maxs : null;

  const durationCandidate =
    body.duration_seconds ??
    body.durationSeconds ??
    body.data?.duration_seconds ??
    body.data?.durationSeconds;
  return {
    peaks,
    mins: normalizedMins,
    maxs: normalizedMaxs,
    durationSeconds: finiteDuration(Number(durationCandidate)),
  };
}

async function loadWaveformPeaks(src: string) {
  const t = useTranslations();
  const cached = waveformCache.get(src);
  if (cached) return cached;

  const pending = (async () => {
    const fetchUrl = src;
    const response = await fetch(fetchUrl, {
      credentials: "same-origin",
      cache: "force-cache",
    });
    if (!response.ok) {
      throw new Error(t("waveformHttpFailed", {status: response.status}));
    }

    const contentType = response.headers?.get("content-type") ?? "";
    if (
      contentType.includes("application/json") ||
      src.toLowerCase().includes(".waveform.json")
    ) {
      const body = (await response.json()) as WaveformPeaksEnvelope;
      const waveform = waveformDataFromEnvelope(body);
      if (waveform) return waveform;
      throw new Error(t("waveformDataInvalid"));
    }

    const AudioContextConstructor = audioContextConstructor();
    if (!AudioContextConstructor) {
      throw new Error(t("audioContextFailed"));
    }

    const encodedAudio = await response.arrayBuffer();
    const audioContext = new AudioContextConstructor();
    try {
      const decodedAudio = await audioContext.decodeAudioData(encodedAudio);
      return {
        ...extractWaveformPeaks(decodedAudio),
        durationSeconds: finiteDuration(decodedAudio.duration),
      };
    } finally {
      if (audioContext.state !== "closed") {
        void audioContext.close().catch(() => {});
      }
    }
  })();

  waveformCache.set(src, pending);
  try {
    const waveform = await pending;
    waveformCache.set(src, waveform);
    return waveform;
  } catch (error) {
    waveformCache.delete(src);
    throw error;
  }
}

function fileStemLabel(
  file: OutputFile,
  translateStemLabel: (key: StemLabelTranslationKey) => string,
  stemLabels?: Partial<Record<string, string>>,
) {
  const stemKey = ((file as OutputFileWithStem).stem_key ?? "")
    .trim()
    .toLowerCase();
  if (stemKey) {
    const explicitLabel = stemLabels?.[stemKey];
    if (explicitLabel) return explicitLabel;

    const builtInLabel = builtInStemLabels[stemKey];
    if (builtInLabel) return builtInLabel;

    const translationKey =
      stemLabelTranslationKeys[
        stemKey as keyof typeof stemLabelTranslationKeys
      ];
    if (translationKey) return translateStemLabel(translationKey);
  }

  const baseName = file.name.replace(/\.[^.]+$/, "");
  const parts = baseName.split(/[-_]/).map((part) => part.trim()).filter(Boolean);
  const fallbackLabel = parts.at(-1) ?? baseName;
  if (standaloneHarmonyFallbackLabels.has(fallbackLabel.toLowerCase())) {
    return translateStemLabel("audioStemBackingVocals");
  }
  return fallbackLabel;
}

function buildRulerMarks(duration: number, width: number) {
  const intervals = [1, 2, 5, 10, 15, 30, 60, 120, 300];
  const targetInterval = duration / Math.max(1, Math.min(6, Math.floor(width / 80)));
  const interval =
    intervals.find((candidate) => candidate >= targetInterval) ??
    intervals[intervals.length - 1];
  const marks: number[] = [];
  for (let value = 0; value <= duration + 0.001; value += interval) {
    if (value === 0 || ((duration - value) / duration) * width >= 48) marks.push(value);
  }
  if (marks.at(-1) !== duration) marks.push(duration);
  return marks;
}

function keyedRecord<T>(
  trackIds: string[],
  previous: Record<string, T>,
  defaultValue: T,
) {
  return Object.fromEntries(
    trackIds.map((trackId) => [trackId, previous[trackId] ?? defaultValue]),
  );
}

export function MultiTrackAudioMixer({
  tracks: inputTracks,
  availableTracks: inputAvailableTracks = [],
  initialMutedTrackIds = emptyTrackIds,
  removableTrackIds = [],
  onAddTrack,
  onRemoveTrack,
  stemLabels,
  title,
  contextLabel,
  description,
  className,
}: MultiTrackAudioMixerProps) {
  const t = useTranslations("tasks");
  const mixerTitle = title ?? t("audioMixerTitle");
  const translateStemLabel = useCallback(
    (key: StemLabelTranslationKey) => t(key),
    [t],
  );
  const tracks = useMemo<AudioMixerTrack[]>(
    () =>
      inputTracks.map((track, index) => ({
        ...track,
        id: track.file.s3_key,
        label: fileStemLabel(track.file, translateStemLabel, stemLabels),
        color: trackColors[index % trackColors.length],
      })),
    [inputTracks, stemLabels, translateStemLabel],
  );
  const availableTracks = useMemo<AudioMixerTrack[]>(
    () =>
      inputAvailableTracks.map((track, index) => ({
        ...track,
        id: track.file.s3_key,
        label: fileStemLabel(track.file, translateStemLabel, stemLabels),
        color: trackColors[(tracks.length + index) % trackColors.length],
      })),
    [inputAvailableTracks, stemLabels, tracks.length, translateStemLabel],
  );
  const availableTrackIds = useMemo(
    () => new Set(availableTracks.map((track) => track.id)),
    [availableTracks],
  );
  const removableTrackIdSet = useMemo(
    () => new Set(removableTrackIds),
    [removableTrackIds],
  );
  const trackIdsKey = tracks.map((track) => track.id).join("\n");
  const initialMutedTrackIdsKey = initialMutedTrackIds.join("\n");
  const initialMutedTrackIdSet = useMemo(() => {
    const ids = initialMutedTrackIdsKey ? initialMutedTrackIdsKey.split("\n") : [];
    return new Set(ids);
  }, [initialMutedTrackIdsKey]);
  const audioElementTracks = useMemo(
    () =>
      inputTracks.map((track) => ({
        id: track.file.s3_key,
        src: track.src,
        mediaSrc: track.src,
        waveformSrc: track.waveformSrc ?? track.src,
        waveformPeaksSrc: track.waveformPeaksSrc,
      })),
    [inputTracks],
  );
  const audioElementTracksKey = audioElementTracks
    .map(
      (track) =>
        `${track.id}\t${track.src}\t${track.mediaSrc}\t${track.waveformSrc}\t${track.waveformPeaksSrc ?? ""}`,
    )
    .join("\n");

  const audioElementsRef = useRef<Record<string, HTMLAudioElement>>({});
  const audioGainGraphRef = useRef<AudioGainGraph | null>(null);
  const recoverAudioElementRef = useRef<Record<string, () => void>>({});
  const pendingSeekTimesRef = useRef<Record<string, number>>({});
  const animationFrameRef = useRef<number | null>(null);
  const bufferRetryTimeoutRef = useRef<number | null>(null);
  const timelineScrollRef = useRef<HTMLDivElement | null>(null);
  const progressFillRef = useRef<HTMLSpanElement | null>(null);
  const progressThumbRef = useRef<HTMLSpanElement | null>(null);
  const playheadRef = useRef<HTMLButtonElement | null>(null);
  const isPlayingRef = useRef(false);
  const isBufferingRef = useRef(false);
  const currentTimeRef = useRef(0);
  const playbackStartRef = useRef(0);
  const playbackStartedAtRef = useRef(0);
  const timelineDurationRef = useRef(defaultTimelineSeconds);
  const resumePlaybackIfBufferedRef = useRef<() => void>(() => {});
  const dragRef = useRef<{
    id: string;
    startX: number;
    startOffset: number;
    pxPerSecond: number;
  } | null>(null);
  const zoomDragRef = useRef<{
    pointerId: number;
    startX: number;
    startZoom: number;
    scroller: HTMLDivElement;
  } | null>(null);
  const playheadDragRef = useRef<{
    pointerId: number;
    scroller: HTMLDivElement;
    handle: HTMLElement;
  } | null>(null);

  const [durations, setDurations] = useState<Record<string, number>>({});
  const [offsets, setOffsets] = useState<Record<string, number>>({});
  const [trackGainDb, setTrackGainDb] = useState<Record<string, number>>({});
  const [mutedIds, setMutedIds] = useState<Set<string>>(
    () => new Set(initialMutedTrackIds),
  );
  const [soloIds, setSoloIds] = useState<Set<string>>(() => new Set());
  const [currentTime, setCurrentTime] = useState(0);
  const [isPlaying, setIsPlaying] = useState(false);
  const [isBuffering, setIsBuffering] = useState(false);
  const [isTrackDropActive, setIsTrackDropActive] = useState(false);
  const [zoom, setZoom] = useState(1);
  const userZoomRef = useRef(false);
  const [trackBufferedRanges, setTrackBufferedRanges] = useState<Record<string, TimeRange[]>>({});
  const [waveformPeaks, setWaveformPeaks] = useState<Record<string, WaveformLoadResult>>({});
  const [timelineViewportWidth, setTimelineViewportWidth] = useState(0);
  const [timelineScrollerElement, setTimelineScrollerElement] =
    useState<HTMLDivElement | null>(null);
  const transportRoutingKey = useMemo(() => {
    const offsetKey = tracks
      .map((track) => `${track.id}:${offsets[track.id] ?? 0}`)
      .join(",");

    return offsetKey;
  }, [offsets, tracks]);
  const audibilityRoutingKey = useMemo(() => {
    const mutedKey = [...mutedIds].sort().join(",");
    const soloKey = [...soloIds].sort().join(",");

    return `${mutedKey}\n${soloKey}`;
  }, [mutedIds, soloIds]);
  const setTimelineScrollElement = useCallback((node: HTMLDivElement | null) => {
    timelineScrollRef.current = node;
    setTimelineScrollerElement(node);
  }, []);
  const applyPlaybackPositionStyle = useCallback((timelineTime: number) => {
    const duration = Math.max(timelineDurationRef.current, 0.001);
    const percent = `${clamp(timelineTime / duration, 0, 1) * 100}%`;

    if (progressFillRef.current) {
      progressFillRef.current.style.width = percent;
    }
    if (progressThumbRef.current) {
      progressThumbRef.current.style.left = percent;
    }
    if (playheadRef.current) {
      playheadRef.current.style.left = percent;
    }
  }, []);
  const readLivePlaybackTime = useCallback(() => {
    if (!isPlayingRef.current || isBufferingRef.current) {
      return clamp(currentTimeRef.current, 0, timelineDurationRef.current);
    }

    const elapsed =
      (window.performance.now() - playbackStartedAtRef.current) / 1000;
    return clamp(
      playbackStartRef.current + Math.max(0, elapsed),
      0,
      timelineDurationRef.current,
    );
  }, []);
  const anchorPlaybackClock = useCallback(
    (timelineTime: number, options?: { commitState?: boolean }) => {
      const nextTime = clamp(timelineTime, 0, timelineDurationRef.current);
      currentTimeRef.current = nextTime;
      playbackStartRef.current = nextTime;
      playbackStartedAtRef.current = window.performance.now();
      applyPlaybackPositionStyle(nextTime);
      if (options?.commitState) {
        setCurrentTime(nextTime);
      }
      return nextTime;
    },
    [applyPlaybackPositionStyle],
  );

  const timelineDuration = useMemo(() => {
    const maxTrackEnd = tracks.reduce((max, track) => {
      const duration = finiteDuration(durations[track.id]) ?? defaultTimelineSeconds;
      const offset = offsets[track.id] ?? 0;
      const end = Math.max(0.5, offset + duration);
      return Math.max(max, end);
    }, 0);

    return Math.max(0.5, currentTime, maxTrackEnd);
  }, [currentTime, durations, offsets, tracks]);

  const minTimelineZoom = useMemo(
    () => minZoomForTimeline(timelineDuration, timelineViewportWidth),
    [timelineDuration, timelineViewportWidth],
  );
  const timelineWidth = timelineWidthForZoom(
    timelineDuration,
    zoom,
    timelineViewportWidth,
  );
  const rulerMarks = useMemo(
    () => buildRulerMarks(timelineDuration, timelineWidth),
    [timelineDuration, timelineWidth],
  );
  useEffect(() => {
    timelineDurationRef.current = timelineDuration;
    applyPlaybackPositionStyle(currentTimeRef.current);
  }, [applyPlaybackPositionStyle, timelineDuration]);

  useEffect(() => {
    applyPlaybackPositionStyle(currentTime);
  }, [applyPlaybackPositionStyle, currentTime]);

  useEffect(() => {
    setZoom((previousZoom) =>
      !userZoomRef.current || previousZoom < minTimelineZoom
        ? minTimelineZoom
        : Math.min(previousZoom, maxTimelineZoom),
    );
  }, [minTimelineZoom]);

  useEffect(() => {
    const scroller = timelineScrollerElement;
    if (!scroller) return;

    const updateViewportWidth = () => {
      setTimelineViewportWidth(scroller.clientWidth);
    };

    updateViewportWidth();

    if (typeof ResizeObserver === "undefined") {
      window.addEventListener("resize", updateViewportWidth);
      return () => {
        window.removeEventListener("resize", updateViewportWidth);
      };
    }

    const observer = new ResizeObserver(updateViewportWidth);
    observer.observe(scroller);

    return () => {
      observer.disconnect();
    };
  }, [timelineScrollerElement]);

  useEffect(() => {
    const currentIds = trackIdsKey ? trackIdsKey.split("\n") : [];
    const currentIdSet = new Set(currentIds);
    pendingSeekTimesRef.current = Object.fromEntries(
      Object.entries(pendingSeekTimesRef.current).filter(([id]) =>
        currentIdSet.has(id),
      ),
    );

    setOffsets((previous) => keyedRecord(currentIds, previous, 0));
    setTrackGainDb((previous) => keyedRecord(currentIds, previous, 0));
    setMutedIds((previous) => {
      const next = new Set([...previous].filter((id) => currentIdSet.has(id)));
      initialMutedTrackIdSet.forEach((id) => {
        if (currentIdSet.has(id)) next.add(id);
      });
      return next;
    });
    setSoloIds((previous) => {
      return new Set([...previous].filter((id) => currentIdSet.has(id)));
    });
    setTrackBufferedRanges((previous) =>
      Object.fromEntries(
        Object.entries(previous).filter(([id]) => currentIdSet.has(id)),
      ),
    );
  }, [initialMutedTrackIdSet, trackIdsKey]);

  useEffect(() => {
    const currentIds = new Set(audioElementTracks.map((track) => track.id));
    setWaveformPeaks((previous) =>
      Object.fromEntries(
        Object.entries(previous).filter(([trackId]) => currentIds.has(trackId)),
      ),
    );

    if (typeof fetch !== "function") return;

    let cancelled = false;

    audioElementTracks.forEach((track) => {
      void (async () => {
        try {
          let waveform: WaveformLoadResult;
          try {
            waveform = await loadWaveformPeaks(
              track.waveformPeaksSrc ?? track.waveformSrc,
            );
          } catch (error) {
            if (!track.waveformPeaksSrc) throw error;
            waveform = await loadWaveformPeaks(track.waveformSrc);
          }
          if (cancelled) return;
          setWaveformPeaks((previous) => {
            if (previous[track.id] === waveform) return previous;
            return { ...previous, [track.id]: waveform };
          });
          const durationSeconds = waveform.durationSeconds;
          if (durationSeconds != null) {
            setDurations((previous) =>
              previous[track.id] === durationSeconds
                ? previous
                : { ...previous, [track.id]: durationSeconds },
            );
          }
        } catch (error) {
          if (!cancelled) reportMixerError(error);
        }
      })();
    });

    return () => {
      cancelled = true;
    };
  }, [audioElementTracks, audioElementTracksKey]);

  const disposeAudioGainGraph = useCallback(() => {
    const graph = audioGainGraphRef.current;
    if (!graph) return;

    Object.values(graph.sources).forEach((source) => {
      try {
        source.disconnect();
      } catch {
        // The node may already be disconnected during a media element refresh.
      }
    });
    Object.values(graph.gains).forEach((gain) => {
      try {
        gain.disconnect();
      } catch {
        // The node may already be disconnected during a media element refresh.
      }
    });
    if (graph.context.state !== "closed") {
      void graph.context.close().catch(() => {});
    }
    audioGainGraphRef.current = null;
  }, []);

  const ensureAudioGainGraph = useCallback(() => {
    const AudioContextConstructor = audioContextConstructor();
    if (!AudioContextConstructor) { reportMixerError(new Error(t("audioContextFailed"))); return null; }

    let graph = audioGainGraphRef.current;
    if (!graph || graph.context.state === "closed") {
      graph = {
        context: new AudioContextConstructor(),
        sources: {},
        gains: {},
      };
      audioGainGraphRef.current = graph;
    }

    tracks.forEach((track) => {
      if (graph.sources[track.id]) return;
      const audio = audioElementsRef.current[track.id];
      if (!audio) return;

      try {
        const source = graph.context.createMediaElementSource(audio);
        const gain = graph.context.createGain();
        source.connect(gain);
        gain.connect(graph.context.destination);
        graph.sources[track.id] = source;
        graph.gains[track.id] = gain;
      } catch (error) {
        reportMixerError(error);
      }
    });

    return graph;
  }, [tracks]);

  const resumeAudioGainGraph = useCallback(() => {
    const context = audioGainGraphRef.current?.context;
    if (context?.state === "suspended") {
      void context.resume().catch(reportMixerError);
    }
  }, []);

  const applyTrackGain = useCallback(
    (
      trackId: string,
      audio: HTMLAudioElement,
      gainDb: number,
      isAudible = true,
    ) => {
      const clampedGainDb = clamp(roundGainDb(gainDb), minTrackGainDb, maxTrackGainDb);
      const linearGain = isAudible ? gainDbToLinearGain(clampedGainDb) : 0;
      const gainNode = audioGainGraphRef.current?.gains[trackId];

      if (gainNode) {
        setGainValue(gainNode, linearGain);
        setAudioVolume(audio, 1);
        setAudioMuted(audio, false);
        return;
      }

      setAudioVolume(audio, clamp(linearGain, 0, 1));
      setAudioMuted(audio, false);
    },
    [],
  );

  useEffect(() => {
    const elements: Record<string, HTMLAudioElement> = {};
    const cleanupListeners: Array<() => void> = [];

    audioElementTracks.forEach((track) => {
      const audio = document.createElement("audio");
      audio.crossOrigin = "anonymous";
      audio.preload = "none";
      audio.setAttribute("preload", "none");

      const scheduleLocalBufferRetry = () => {
        if (bufferRetryTimeoutRef.current != null) return;
        bufferRetryTimeoutRef.current = window.setTimeout(() => {
          bufferRetryTimeoutRef.current = null;
          resumePlaybackIfBufferedRef.current();
        }, 180);
      };

      const recoverMediaSource = () => {
        reportMixerError(new Error(t("mediaTrackFailed", {name: track.label, code: audio.error?.code ?? audio.networkState})));
      };
      recoverAudioElementRef.current[track.id] = recoverMediaSource;

      const updateDuration = () => {
        const duration = finiteDuration(audio.duration);
        if (!duration) return;
        setDurations((previous) =>
          previous[track.id] === duration
            ? previous
            : { ...previous, [track.id]: duration },
        );
        const pendingSeekTime = pendingSeekTimesRef.current[track.id];
        if (Number.isFinite(pendingSeekTime)) {
          const clampedSeekTime = clamp(pendingSeekTime, 0, duration);
          if (seekAudioToTime(audio, clampedSeekTime)) {
            delete pendingSeekTimesRef.current[track.id];
          }
        }
      };
      const updateBufferedRanges = () => {
        const nextRanges = mediaTimeRangesToArray(audio.buffered);
        if (nextRanges.length === 0) return;
        setTrackBufferedRanges((previous) => {
          const merged = mergeTimeRanges([
            ...(previous[track.id] ?? []),
            ...nextRanges,
          ]);
          if (sameTimeRanges(previous[track.id] ?? [], merged)) {
            return previous;
          }
          return { ...previous, [track.id]: merged };
        });
      };
      const updateMediaState = () => {
        updateDuration();
        updateBufferedRanges();
      };
      const bufferEvents = [
        "loadedmetadata",
        "durationchange",
        "loadeddata",
        "canplay",
        "canplaythrough",
        "progress",
        "seeked",
        "timeupdate",
      ] as const;

      bufferEvents.forEach((eventName) =>
        audio.addEventListener(eventName, updateMediaState),
      );
      audio.addEventListener("error", recoverMediaSource);
      audio.addEventListener("stalled", scheduleLocalBufferRetry);
      audio.addEventListener("waiting", scheduleLocalBufferRetry);
      cleanupListeners.push(() => {
        bufferEvents.forEach((eventName) =>
          audio.removeEventListener(eventName, updateMediaState),
        );
        audio.removeEventListener("error", recoverMediaSource);
        audio.removeEventListener("stalled", scheduleLocalBufferRetry);
        audio.removeEventListener("waiting", scheduleLocalBufferRetry);
        delete recoverAudioElementRef.current[track.id];
      });
      updateMediaState();
      elements[track.id] = audio;
    });

    audioElementsRef.current = elements;

    return () => {
      disposeAudioGainGraph();
      cleanupListeners.forEach((cleanup) => cleanup());
      Object.values(elements).forEach((audio) => {
        pauseAudio(audio);
        audio.removeAttribute("src");
      });
      audioElementsRef.current = {};
      recoverAudioElementRef.current = {};
    };
  }, [audioElementTracksKey, audioElementTracks, disposeAudioGainGraph]);

  useEffect(() => {
    const needsWebAudioGain = tracks.some(
      (track) => (trackGainDb[track.id] ?? 0) > 0.0001,
    );
    if (needsWebAudioGain) {
      ensureAudioGainGraph();
      resumeAudioGainGraph();
    }
  }, [
    ensureAudioGainGraph,
    resumeAudioGainGraph,
    trackGainDb,
    tracks,
  ]);

  const pauseAllAudio = useCallback(() => {
    Object.values(audioElementsRef.current).forEach((audio) => {
      pauseAudio(audio);
    });
  }, []);

  const applyAudioAudibility = useCallback(() => {
    const hasSolo = soloIds.size > 0;

    tracks.forEach((track) => {
      const audio = audioElementsRef.current[track.id];
      if (!audio) return;

      const isAudible =
        !mutedIds.has(track.id) && (!hasSolo || soloIds.has(track.id));
      applyTrackGain(track.id, audio, trackGainDb[track.id] ?? 0, isAudible);
    });
  }, [applyTrackGain, mutedIds, soloIds, tracks, trackGainDb]);

  const syncAudioElements = useCallback(
    (timelineTime: number, shouldPlay: boolean) => {
      tracks.forEach((track) => {
        const audio = audioElementsRef.current[track.id];
        if (!audio) return;

        const offset = offsets[track.id] ?? 0;
        const localTime = timelineTime - offset;
        const duration = finiteDuration(audio.duration) ?? durations[track.id] ?? null;
        const clampedLocalTime = duration
          ? clamp(localTime, 0, duration)
          : Math.max(0, localTime);
        const isInRange =
          localTime >= 0 && (!duration || localTime <= duration - 0.04);

        if (shouldPlay && isInRange) {
          attachAudioSource(audio, track.src);
        }

        if (
          Number.isFinite(clampedLocalTime) &&
          Math.abs(audio.currentTime - clampedLocalTime) > 0.18
        ) {
          if (seekAudioToTime(audio, clampedLocalTime)) {
            delete pendingSeekTimesRef.current[track.id];
          } else {
            pendingSeekTimesRef.current[track.id] = clampedLocalTime;
          }
        }

        if (shouldPlay && isInRange) {
          if (audio.paused) {
            try {
              const playResult = audio.play();
              if (typeof playResult?.catch === "function") {
                void playResult.catch((error) => { if (error.name !== "AbortError") reportMixerError(error); });
              }
            } catch (error) {
              reportMixerError(error);
            }
          }
        } else if (!audio.paused) {
          pauseAudio(audio);
        }
      });
    },
    [durations, offsets, tracks],
  );

  const syncAudioElementsRef = useRef(syncAudioElements);
  useEffect(() => {
    syncAudioElementsRef.current = syncAudioElements;
  }, [syncAudioElements]);

  const cancelAnimationFrameLoop = useCallback(() => {
    if (animationFrameRef.current != null) {
      window.cancelAnimationFrame(animationFrameRef.current);
      animationFrameRef.current = null;
    }
  }, []);

  const clearBufferRetry = useCallback(() => {
    if (bufferRetryTimeoutRef.current != null) {
      window.clearTimeout(bufferRetryTimeoutRef.current);
      bufferRetryTimeoutRef.current = null;
    }
  }, []);

  const scheduleBufferRetry = useCallback(() => {
    if (bufferRetryTimeoutRef.current != null) return;
    bufferRetryTimeoutRef.current = window.setTimeout(() => {
      bufferRetryTimeoutRef.current = null;
      resumePlaybackIfBufferedRef.current();
    }, 180);
  }, []);

  const scrollTimelineToTime = useCallback(
    (timelineTime: number, mode: "playback" | "center" | "start" = "center") => {
      const scroller = timelineScrollRef.current;
      if (!scroller || scroller.clientWidth <= 0) return;

      const duration = Math.max(timelineDurationRef.current, 0.001);
      const timelineScrollWidth = Math.max(scroller.scrollWidth, scroller.clientWidth);
      const timeX = clamp(timelineTime / duration, 0, 1) * timelineScrollWidth;
      const maxScrollLeft = Math.max(0, timelineScrollWidth - scroller.clientWidth);

      if (mode === "start") {
        scroller.scrollLeft = 0;
        return;
      }

      if (mode === "center") {
        scroller.scrollLeft = clamp(
          timeX - scroller.clientWidth / 2,
          0,
          maxScrollLeft,
        );
        return;
      }

      const targetScrollLeft = clamp(
        timeX - scroller.clientWidth * 0.5,
        0,
        maxScrollLeft,
      );
      const delta = targetScrollLeft - scroller.scrollLeft;
      if (Math.abs(delta) < 0.75) {
        scroller.scrollLeft = targetScrollLeft;
        return;
      }

      scroller.scrollLeft = clamp(
        scroller.scrollLeft + delta * 0.24,
        0,
        maxScrollLeft,
      );
    },
    [],
  );

  const isTimelineBufferedForPlayback = useCallback(
    (timelineTime: number) => {
      const hasSolo = soloIds.size > 0;

      return tracks.every((track) => {
        const audio = audioElementsRef.current[track.id];
        if (!audio) return true;

        const offset = offsets[track.id] ?? 0;
        const localTime = timelineTime - offset;
        const duration = finiteDuration(audio.duration) ?? durations[track.id] ?? null;
        const isInRange =
          localTime >= 0 && (!duration || localTime <= duration - 0.04);
        const isAudible =
          !mutedIds.has(track.id) && (!hasSolo || soloIds.has(track.id));

        if (!isInRange || !isAudible) return true;

        return audioHasBufferedTime(audio, localTime, duration);
      });
    },
    [durations, mutedIds, offsets, soloIds, tracks],
  );

  const requestTimelineBufferAtTime = useCallback(
    (timelineTime: number) => {
      tracks.forEach((track) => {
        const audio = audioElementsRef.current[track.id];
        if (!audio) return;

        const offset = offsets[track.id] ?? 0;
        const localTime = timelineTime - offset;
        const duration = finiteDuration(audio.duration) ?? durations[track.id] ?? null;
        const isInRange =
          localTime >= 0 && (!duration || localTime <= duration - 0.04);
        if (!isInRange) return;

        audio.preload = "auto";
        const attachedSource = attachAudioSource(audio, track.src);
        if (audio.error) {
          recoverAudioElementRef.current[track.id]?.();
        }
        if (attachedSource || audio.networkState === HTMLMediaElement.NETWORK_EMPTY) {
          requestAudioLoad(audio);
        }

        const clampedLocalTime = duration
          ? clamp(localTime, 0, duration)
          : Math.max(0, localTime);
        if (
          Number.isFinite(clampedLocalTime) &&
          !audioHasBufferedTime(audio, clampedLocalTime, duration) &&
          Math.abs(audio.currentTime - clampedLocalTime) > 0.01
        ) {
          if (!seekAudioToTime(audio, clampedLocalTime)) {
            pendingSeekTimesRef.current[track.id] = clampedLocalTime;
          }
        }
      });
    },
    [durations, offsets, tracks],
  );

  const pauseForBuffer = useCallback(
    (timelineTime = currentTimeRef.current) => {
      const frozenTime = clamp(timelineTime, 0, timelineDurationRef.current);
      currentTimeRef.current = frozenTime;
      playbackStartRef.current = frozenTime;
      playbackStartedAtRef.current = window.performance.now();
      applyPlaybackPositionStyle(frozenTime);
      setCurrentTime(frozenTime);
      isBufferingRef.current = true;
      setIsBuffering(true);
      cancelAnimationFrameLoop();
      requestTimelineBufferAtTime(frozenTime);
      pauseAllAudio();
      syncAudioElementsRef.current(frozenTime, false);
      scheduleBufferRetry();
    },
    [
      applyPlaybackPositionStyle,
      cancelAnimationFrameLoop,
      pauseAllAudio,
      requestTimelineBufferAtTime,
      scheduleBufferRetry,
    ],
  );

  const animatePlayback = useCallback(() => {
    if (!isPlayingRef.current) return;

    const duration = timelineDurationRef.current;
    const elapsed =
      (window.performance.now() - playbackStartedAtRef.current) / 1000;
    const nextTime = clamp(playbackStartRef.current + elapsed, 0, duration);

    if (!isTimelineBufferedForPlayback(nextTime)) {
      pauseForBuffer(currentTimeRef.current);
      return;
    }

    isBufferingRef.current = false;
    setIsBuffering(false);
    clearBufferRetry();
    currentTimeRef.current = nextTime;
    applyPlaybackPositionStyle(nextTime);
    setCurrentTime(nextTime);
    syncAudioElementsRef.current(nextTime, true);
    scrollTimelineToTime(nextTime, "playback");

    if (nextTime >= duration - 0.02) {
      isPlayingRef.current = false;
      setIsPlaying(false);
      isBufferingRef.current = false;
      setIsBuffering(false);
      pauseAllAudio();
      return;
    }

    animationFrameRef.current = window.requestAnimationFrame(animatePlayback);
  }, [
    applyPlaybackPositionStyle,
    clearBufferRetry,
    isTimelineBufferedForPlayback,
    pauseAllAudio,
    pauseForBuffer,
    scrollTimelineToTime,
  ]);

  const resumePlaybackIfBuffered = useCallback(() => {
    if (!isPlayingRef.current || !isBufferingRef.current) return;

    const resumeAt = currentTimeRef.current;
    syncAudioElementsRef.current(resumeAt, false);

    if (!isTimelineBufferedForPlayback(resumeAt)) {
      scheduleBufferRetry();
      return;
    }

    clearBufferRetry();
    isBufferingRef.current = false;
    setIsBuffering(false);
    playbackStartRef.current = resumeAt;
    playbackStartedAtRef.current = window.performance.now();
    syncAudioElementsRef.current(resumeAt, true);
    cancelAnimationFrameLoop();
    animationFrameRef.current = window.requestAnimationFrame(animatePlayback);
  }, [
    animatePlayback,
    cancelAnimationFrameLoop,
    clearBufferRetry,
    isTimelineBufferedForPlayback,
    scheduleBufferRetry,
  ]);

  useEffect(() => {
    resumePlaybackIfBufferedRef.current = resumePlaybackIfBuffered;
  }, [resumePlaybackIfBuffered]);
  const requestTimelineBufferAtTimeRef = useRef(requestTimelineBufferAtTime);
  const isTimelineBufferedForPlaybackRef = useRef(isTimelineBufferedForPlayback);
  const pauseForBufferRef = useRef(pauseForBuffer);
  const animatePlaybackRef = useRef(animatePlayback);

  useEffect(() => {
    requestTimelineBufferAtTimeRef.current = requestTimelineBufferAtTime;
  }, [requestTimelineBufferAtTime]);

  useEffect(() => {
    isTimelineBufferedForPlaybackRef.current = isTimelineBufferedForPlayback;
  }, [isTimelineBufferedForPlayback]);

  useEffect(() => {
    pauseForBufferRef.current = pauseForBuffer;
  }, [pauseForBuffer]);

  useEffect(() => {
    animatePlaybackRef.current = animatePlayback;
  }, [animatePlayback]);

  useEffect(() => {
    applyAudioAudibility();
  }, [applyAudioAudibility]);

  const realignPlaybackForControlGesture = useCallback(() => {
    if (!isPlayingRef.current) return;

    anchorPlaybackClock(readLivePlaybackTime(), {
      commitState: true,
    });
  }, [anchorPlaybackClock, readLivePlaybackTime]);

  useEffect(() => {
    const shouldPlay = isPlayingRef.current;
    const timelineTime = shouldPlay
      ? anchorPlaybackClock(readLivePlaybackTime(), { commitState: true })
      : currentTimeRef.current;

    if (shouldPlay || isBufferingRef.current) {
      requestTimelineBufferAtTimeRef.current(timelineTime);
    }

    if (!shouldPlay) {
      syncAudioElementsRef.current(timelineTime, false);
      return;
    }

    if (isBufferingRef.current) {
      syncAudioElementsRef.current(timelineTime, false);
      scheduleBufferRetry();
      return;
    }

    if (!isTimelineBufferedForPlaybackRef.current(timelineTime)) {
      pauseForBufferRef.current(timelineTime);
      return;
    }

    clearBufferRetry();
    isBufferingRef.current = false;
    setIsBuffering(false);
    syncAudioElementsRef.current(timelineTime, true);
    cancelAnimationFrameLoop();
    animationFrameRef.current = window.requestAnimationFrame(() => {
      animatePlaybackRef.current();
    });
  }, [
    anchorPlaybackClock,
    cancelAnimationFrameLoop,
    clearBufferRetry,
    readLivePlaybackTime,
    scheduleBufferRetry,
    transportRoutingKey,
  ]);

  useEffect(() => {
    if (!isPlayingRef.current) return;

    const timelineTime = anchorPlaybackClock(readLivePlaybackTime(), {
      commitState: true,
    });

    if (isBufferingRef.current) {
      scheduleBufferRetry();
      return;
    }

    if (!isTimelineBufferedForPlaybackRef.current(timelineTime)) {
      pauseForBufferRef.current(timelineTime);
    }
  }, [
    anchorPlaybackClock,
    audibilityRoutingKey,
    readLivePlaybackTime,
    scheduleBufferRetry,
  ]);

  useEffect(() => {
    const events = [
      "canplay",
      "canplaythrough",
      "loadeddata",
      "progress",
      "seeked",
      "playing",
    ] as const;
    const retryPlayback = () => {
      if (!isPlayingRef.current || !isBufferingRef.current) return;
      resumePlaybackIfBufferedRef.current();
    };
    const elements = Object.values(audioElementsRef.current);

    elements.forEach((audio) => {
      events.forEach((eventName) => audio.addEventListener(eventName, retryPlayback));
    });

    return () => {
      elements.forEach((audio) => {
        events.forEach((eventName) =>
          audio.removeEventListener(eventName, retryPlayback),
        );
      });
    };
  }, [audioElementTracksKey]);

  const setTimelinePosition = useCallback(
    (
      value: number,
      options?: {
        scrollMode?: "playback" | "center" | "start";
        syncPlayback?: boolean;
        requestBuffer?: boolean;
      },
    ) => {
      const nextTime = clamp(value, 0, timelineDurationRef.current);
      currentTimeRef.current = nextTime;
      playbackStartRef.current = nextTime;
      playbackStartedAtRef.current = window.performance.now();
      applyPlaybackPositionStyle(nextTime);
      setCurrentTime(nextTime);
      scrollTimelineToTime(nextTime, options?.scrollMode ?? "center");

      const shouldPlay = options?.syncPlayback ?? isPlayingRef.current;
      syncAudioElementsRef.current(nextTime, false);
      if (options?.requestBuffer ?? true) {
        requestTimelineBufferAtTime(nextTime);
      }

      if (!shouldPlay) return;

      if (!isTimelineBufferedForPlayback(nextTime)) {
        pauseForBuffer(nextTime);
        return;
      }

      clearBufferRetry();
      isBufferingRef.current = false;
      setIsBuffering(false);
      syncAudioElementsRef.current(nextTime, true);
      cancelAnimationFrameLoop();
      animationFrameRef.current = window.requestAnimationFrame(animatePlayback);
    },
    [
      animatePlayback,
      cancelAnimationFrameLoop,
      clearBufferRetry,
      applyPlaybackPositionStyle,
      isTimelineBufferedForPlayback,
      pauseForBuffer,
      requestTimelineBufferAtTime,
      scrollTimelineToTime,
    ],
  );

  const setTimelineZoom = useCallback(
    (
      value: number | ((previousZoom: number) => number),
      options?: { anchorClientX?: number; scroller?: HTMLDivElement | null },
    ) => {
      userZoomRef.current = true;
      setZoom((previousZoom) => {
        const rawNextZoom =
          typeof value === "function" ? value(previousZoom) : value;
        const nextZoom = clamp(rawNextZoom, minTimelineZoom, maxTimelineZoom);
        if (Math.abs(nextZoom - previousZoom) < 0.001) return previousZoom;

        const scroller = options?.scroller ?? timelineScrollRef.current;
        if (scroller) {
          const rect = scroller.getBoundingClientRect();
          const anchorX =
            options?.anchorClientX == null
              ? scroller.clientWidth / 2
              : clamp(
                  options.anchorClientX - rect.left,
                  0,
                  Math.max(rect.width, 0),
                );
          const duration = timelineDurationRef.current;
          const previousWidth = timelineWidthForZoom(
            duration,
            previousZoom,
            scroller.clientWidth,
          );
          const nextWidth = timelineWidthForZoom(
            duration,
            nextZoom,
            scroller.clientWidth,
          );
          const anchorRatio =
            (scroller.scrollLeft + anchorX) / Math.max(previousWidth, 1);

          window.requestAnimationFrame(() => {
            const maxScrollLeft = Math.max(0, nextWidth - scroller.clientWidth);
            scroller.scrollLeft = clamp(
              anchorRatio * nextWidth - anchorX,
              0,
              maxScrollLeft,
            );
          });
        }

        return nextZoom;
      });
    },
    [minTimelineZoom],
  );

  const startPlayback = useCallback(() => {
    const duration = timelineDurationRef.current;
    const startAt = currentTimeRef.current >= duration - 0.02 ? 0 : currentTimeRef.current;
    isPlayingRef.current = true;
    setIsPlaying(true);
    ensureAudioGainGraph();
    resumeAudioGainGraph();
    applyAudioAudibility();
    playbackStartRef.current = startAt;
    playbackStartedAtRef.current = window.performance.now();
    currentTimeRef.current = startAt;
    applyPlaybackPositionStyle(startAt);
    setCurrentTime(startAt);
    syncAudioElementsRef.current(startAt, false);
    requestTimelineBufferAtTime(startAt);
    scrollTimelineToTime(startAt, "playback");

    if (!isTimelineBufferedForPlayback(startAt)) {
      pauseForBuffer(startAt);
      return;
    }

    clearBufferRetry();
    isBufferingRef.current = false;
    setIsBuffering(false);
    syncAudioElementsRef.current(startAt, true);
    cancelAnimationFrameLoop();
    animationFrameRef.current = window.requestAnimationFrame(animatePlayback);
  }, [
    animatePlayback,
    applyPlaybackPositionStyle,
    applyAudioAudibility,
    cancelAnimationFrameLoop,
    clearBufferRetry,
    ensureAudioGainGraph,
    isTimelineBufferedForPlayback,
    pauseForBuffer,
    requestTimelineBufferAtTime,
    resumeAudioGainGraph,
    scrollTimelineToTime,
  ]);

  const pausePlayback = useCallback(() => {
    isPlayingRef.current = false;
    setIsPlaying(false);
    isBufferingRef.current = false;
    setIsBuffering(false);
    clearBufferRetry();
    cancelAnimationFrameLoop();
    pauseAllAudio();
  }, [cancelAnimationFrameLoop, clearBufferRetry, pauseAllAudio]);

  const resetMixerState = useCallback(() => {
    isPlayingRef.current = false;
    setIsPlaying(false);
    isBufferingRef.current = false;
    setIsBuffering(false);
    clearBufferRetry();
    cancelAnimationFrameLoop();
    pauseAllAudio();
    setOffsets(Object.fromEntries(tracks.map((track) => [track.id, 0])));
    setTrackGainDb(Object.fromEntries(tracks.map((track) => [track.id, 0])));
    setMutedIds(
      new Set(
        tracks
          .map((track) => track.id)
          .filter((trackId) => initialMutedTrackIdSet.has(trackId)),
      ),
    );
    setSoloIds(new Set());
    setTimelineZoom(minTimelineZoom);
    userZoomRef.current = false;
    setTimelinePosition(0, {
      scrollMode: "start",
      syncPlayback: false,
      requestBuffer: false,
    });
    window.requestAnimationFrame(() => {
      scrollTimelineToTime(0, "start");
    });
  }, [
    cancelAnimationFrameLoop,
    clearBufferRetry,
    pauseAllAudio,
    initialMutedTrackIdSet,
    scrollTimelineToTime,
    setTimelinePosition,
    setTimelineZoom,
    tracks,
  ]);

  useEffect(() => {
    return () => {
      isPlayingRef.current = false;
      cancelAnimationFrameLoop();
      clearBufferRetry();
      pauseAllAudio();
    };
  }, [cancelAnimationFrameLoop, clearBufferRetry, pauseAllAudio]);

  const updateTrackOffset = useCallback((trackId: string, value: number) => {
    const nextOffset = clamp(roundOffset(value), -maxTrackOffsetSeconds, maxTrackOffsetSeconds);
    setOffsets((previous) => ({ ...previous, [trackId]: nextOffset }));
  }, []);

  const beginClipDrag = useCallback(
    (event: React.PointerEvent<HTMLButtonElement>, trackId: string) => {
      if (event.altKey) return;
      const timeline = event.currentTarget.closest("[data-audio-mixer-timeline]");
      const width = timeline?.getBoundingClientRect().width ?? timelineWidth;
      const pxPerSecond = Math.max(1, width / timelineDurationRef.current);
      dragRef.current = {
        id: trackId,
        startX: event.clientX,
        startOffset: offsets[trackId] ?? 0,
        pxPerSecond,
      };
      if (typeof event.currentTarget.setPointerCapture === "function") {
        event.currentTarget.setPointerCapture(event.pointerId);
      }
      event.preventDefault();
    },
    [offsets, timelineWidth],
  );

  const timelineTimeFromClientX = useCallback(
    (clientX: number, scroller: HTMLDivElement) => {
      const timeline = scroller.querySelector<HTMLElement>(
        "[data-audio-mixer-timeline]",
      );
      const timelineRect = timeline?.getBoundingClientRect();
      const scrollerRect = scroller.getBoundingClientRect();
      const width =
        timelineRect && timelineRect.width > 0
          ? timelineRect.width
          : Math.max(scroller.scrollWidth, 1);
      const safeClientX = Number.isFinite(clientX) ? clientX : scrollerRect.left;
      const left =
        timelineRect && timelineRect.width > 0
          ? safeClientX - timelineRect.left
          : safeClientX - scrollerRect.left + scroller.scrollLeft;
      const duration = Number.isFinite(timelineDurationRef.current)
        ? timelineDurationRef.current
        : defaultTimelineSeconds;

      return clamp(
        (left / Math.max(width, 1)) * duration,
        0,
        duration,
      );
    },
    [],
  );

  const autoScrollTimelineDrag = useCallback(
    (clientX: number, scroller: HTMLDivElement) => {
      const rect = scroller.getBoundingClientRect();
      const edgeSize = Math.min(72, Math.max(36, scroller.clientWidth * 0.18));
      const maxScrollLeft = Math.max(0, scroller.scrollWidth - scroller.clientWidth);
      let nextScrollLeft = scroller.scrollLeft;

      if (clientX > rect.right - edgeSize) {
        nextScrollLeft += (clientX - (rect.right - edgeSize)) * 0.5 + 4;
      } else if (clientX < rect.left + edgeSize) {
        nextScrollLeft -= (rect.left + edgeSize - clientX) * 0.5 + 4;
      }

      if (nextScrollLeft !== scroller.scrollLeft) {
        scroller.scrollLeft = clamp(nextScrollLeft, 0, maxScrollLeft);
      }
    },
    [],
  );

  const beginPlayheadDrag = useCallback(
    (event: React.PointerEvent<HTMLElement>) => {
      if (event.button > 0) return;
      const scroller = timelineScrollRef.current;
      if (!scroller) return;

      playheadDragRef.current = {
        pointerId: event.pointerId,
        scroller,
        handle: event.currentTarget,
      };
      if (typeof event.currentTarget.setPointerCapture === "function") {
        event.currentTarget.setPointerCapture(event.pointerId);
      }
      setTimelinePosition(timelineTimeFromClientX(event.clientX, scroller), {
        scrollMode: "playback",
      });
      event.preventDefault();
      event.stopPropagation();
    },
    [setTimelinePosition, timelineTimeFromClientX],
  );

  const beginTimelineZoomDrag = useCallback(
    (event: React.PointerEvent<HTMLDivElement>) => {
      if (!event.altKey || event.button > 0) return;
      zoomDragRef.current = {
        pointerId: event.pointerId,
        startX: event.clientX,
        startZoom: zoom,
        scroller: event.currentTarget,
      };
      if (typeof event.currentTarget.setPointerCapture === "function") {
        event.currentTarget.setPointerCapture(event.pointerId);
      }
      event.preventDefault();
    },
    [zoom],
  );

  const handleTimelineWheel = useCallback(
    (event: WheelEvent, scroller: HTMLDivElement) => {
      const wheelUnit =
        event.deltaMode === 1
          ? 16
          : event.deltaMode === 2
            ? Math.max(scroller.clientWidth, 1)
            : 1;
      const deltaX = event.deltaX * wheelUnit;
      const deltaY = event.deltaY * wheelUnit;

      if (event.altKey || event.ctrlKey || event.metaKey) {
        const dominantDelta =
          Math.abs(deltaY) >= Math.abs(deltaX) ? deltaY : deltaX;
        if (dominantDelta === 0) return;
        event.preventDefault();
        setTimelineZoom((previousZoom) => previousZoom * Math.exp(-dominantDelta * 0.0015), {
          anchorClientX: event.clientX,
          scroller,
        });
        return;
      }

      if (event.shiftKey) {
        const scrollDelta = Math.abs(deltaX) > Math.abs(deltaY) ? deltaX : deltaY;
        if (scrollDelta === 0) return;
        event.preventDefault();
        scroller.scrollLeft += scrollDelta;
      }
    },
    [setTimelineZoom],
  );

  useEffect(() => {
    const scroller = timelineScrollRef.current;
    if (!scroller) return;

    const onWheel = (event: WheelEvent) => {
      handleTimelineWheel(event, scroller);
    };

    scroller.addEventListener("wheel", onWheel, { passive: false });
    return () => {
      scroller.removeEventListener("wheel", onWheel);
    };
  }, [handleTimelineWheel]);

  const handleTimelineKeyDown = useCallback(
    (event: React.KeyboardEvent<HTMLDivElement>) => {
      if (event.target !== event.currentTarget) return;

      if (event.key === "+" || event.key === "=") {
        event.preventDefault();
        setTimelineZoom((previousZoom) => previousZoom + 0.1, {
          scroller: event.currentTarget,
        });
        return;
      }

      if (event.key === "-" || event.key === "_") {
        event.preventDefault();
        setTimelineZoom((previousZoom) => previousZoom - 0.1, {
          scroller: event.currentTarget,
        });
        return;
      }

      if (event.key === "0") {
        event.preventDefault();
        setTimelineZoom(1, { scroller: event.currentTarget });
        return;
      }

      if (event.shiftKey && (event.key === "ArrowLeft" || event.key === "ArrowRight")) {
        event.preventDefault();
        const direction = event.key === "ArrowRight" ? 1 : -1;
        event.currentTarget.scrollLeft += direction * event.currentTarget.clientWidth * 0.8;
      }
    },
    [setTimelineZoom],
  );

  const updateTimelineZoomDrag = useCallback(
    (clientX: number) => {
      const zoomDrag = zoomDragRef.current;
      if (!zoomDrag) return false;
      const deltaX = clientX - zoomDrag.startX;
      setTimelineZoom(zoomDrag.startZoom * Math.exp(deltaX / 260), {
        anchorClientX: zoomDrag.startX,
        scroller: zoomDrag.scroller,
      });
      return true;
    },
    [setTimelineZoom],
  );

  const handleTimelinePointerMove = useCallback(
    (event: React.PointerEvent<HTMLDivElement>) => {
      if (!updateTimelineZoomDrag(event.clientX)) return;
      event.preventDefault();
    },
    [updateTimelineZoomDrag],
  );

  useEffect(() => {
    const onPointerMove = (event: PointerEvent) => {
      if (updateTimelineZoomDrag(event.clientX)) {
        event.preventDefault();
        return;
      }

      const playheadDrag = playheadDragRef.current;
      if (playheadDrag) {
        autoScrollTimelineDrag(event.clientX, playheadDrag.scroller);
        setTimelinePosition(
          timelineTimeFromClientX(event.clientX, playheadDrag.scroller),
          { scrollMode: "playback" },
        );
        event.preventDefault();
        return;
      }

      const drag = dragRef.current;
      if (!drag) return;
      const deltaSeconds = (event.clientX - drag.startX) / drag.pxPerSecond;
      updateTrackOffset(drag.id, drag.startOffset + deltaSeconds);
    };
    const onPointerUp = (event: PointerEvent) => {
      const zoomDrag = zoomDragRef.current;
      if (
        zoomDrag &&
        zoomDrag.pointerId === event.pointerId &&
        typeof zoomDrag.scroller.releasePointerCapture === "function"
      ) {
        try {
          zoomDrag.scroller.releasePointerCapture(event.pointerId);
        } catch {
          // The pointer may already be released by the browser.
        }
      }
      zoomDragRef.current = null;
      const playheadDrag = playheadDragRef.current;
      if (
        playheadDrag &&
        playheadDrag.pointerId === event.pointerId &&
        typeof playheadDrag.handle.releasePointerCapture === "function"
      ) {
        try {
          playheadDrag.handle.releasePointerCapture(event.pointerId);
        } catch {
          // The pointer may already be released by the browser.
        }
      }
      playheadDragRef.current = null;
      dragRef.current = null;
    };

    window.addEventListener("pointermove", onPointerMove);
    window.addEventListener("pointerup", onPointerUp);
    window.addEventListener("pointercancel", onPointerUp);
    return () => {
      window.removeEventListener("pointermove", onPointerMove);
      window.removeEventListener("pointerup", onPointerUp);
      window.removeEventListener("pointercancel", onPointerUp);
    };
  }, [
    autoScrollTimelineDrag,
    setTimelinePosition,
    timelineTimeFromClientX,
    updateTimelineZoomDrag,
    updateTrackOffset,
  ]);

  const toggleMuted = useCallback((trackId: string) => {
    realignPlaybackForControlGesture();
    setMutedIds((previous) => {
      const next = new Set(previous);
      if (next.has(trackId)) next.delete(trackId);
      else next.add(trackId);
      return next;
    });
  }, [realignPlaybackForControlGesture]);

  const toggleSolo = useCallback((trackId: string) => {
    realignPlaybackForControlGesture();
    setSoloIds((previous) => {
      const next = new Set(previous);
      if (next.has(trackId)) next.delete(trackId);
      else next.add(trackId);
      return next;
    });
  }, [realignPlaybackForControlGesture]);

  const addAvailableTrack = useCallback(
    (trackId: string) => {
      if (!availableTrackIds.has(trackId)) return;
      onAddTrack?.(trackId);
    },
    [availableTrackIds, onAddTrack],
  );

  const beginAvailableTrackDrag = useCallback(
    (event: React.DragEvent<HTMLButtonElement>, trackId: string) => {
      event.dataTransfer.effectAllowed = "copy";
      event.dataTransfer.setData(audioTrackDragMime, trackId);
    },
    [],
  );

  const handleTrackDragOver = useCallback(
    (event: React.DragEvent<HTMLDivElement>) => {
      if (!onAddTrack) return;
      if (!Array.from(event.dataTransfer.types).includes(audioTrackDragMime)) {
        return;
      }
      event.preventDefault();
      event.dataTransfer.dropEffect = "copy";
      setIsTrackDropActive(true);
    },
    [onAddTrack],
  );

  const handleTrackDrop = useCallback(
    (event: React.DragEvent<HTMLDivElement>) => {
      const trackId = event.dataTransfer.getData(audioTrackDragMime);
      if (!trackId) return;
      event.preventDefault();
      setIsTrackDropActive(false);
      addAvailableTrack(trackId);
    },
    [addAvailableTrack],
  );

  useEffect(() => {
    window.addEventListener("rvc-pause-mixer", pausePlayback);
    return () => window.removeEventListener("rvc-pause-mixer", pausePlayback);
  }, [pausePlayback]);

  if (tracks.length < 2) return null;

  return (
    <section
      role="region"
      aria-label={t("audioMixerTimeline")}
      onDragLeave={() => setIsTrackDropActive(false)}
      onDragOver={handleTrackDragOver}
      onDrop={handleTrackDrop}
      className={cn(
        "rounded-[var(--radius-card)] border border-[var(--color-border)] bg-[color-mix(in_oklch,var(--color-bg)_72%,var(--color-surface-1))] p-3 shadow-[var(--shadow-sm)] sm:p-4",
        className,
      )}
    >
      <header className="flex flex-col gap-3">
        <div className="flex flex-col gap-3 sm:flex-row sm:items-start sm:justify-between">
          <div className="flex min-w-0 items-center gap-2">
            <span className="inline-flex h-8 w-8 shrink-0 items-center justify-center rounded-[var(--radius-input)] bg-[color-mix(in_oklch,var(--color-accent)_13%,transparent)] text-[var(--color-accent)]">
              <ChevronsLeftRight size={16} strokeWidth={1.8} aria-hidden />
            </span>
            <div className="min-w-0">
              <h5 className="truncate text-sm font-semibold text-[var(--color-text-primary)]">
                {mixerTitle}
              </h5>
              <p className="text-xs text-[var(--color-text-muted)]">
                {t("audioMixerSummary", { count: tracks.length })}
              </p>
            </div>
          </div>

          <div className="flex flex-wrap items-center gap-2">
            <button
              type="button"
              aria-label={isPlaying ? t("audioMixerPause") : t("audioMixerPlay")}
              title={isPlaying ? t("audioMixerPause") : t("audioMixerPlay")}
              onClick={() => { document.querySelectorAll<HTMLAudioElement>(".output-card audio").forEach(audio => audio.pause()); if (isPlaying) pausePlayback(); else startPlayback(); }}
              aria-busy={isBuffering || undefined}
              className={cn(
                "inline-flex h-9 w-9 items-center justify-center rounded-[var(--radius-input)] bg-[var(--color-accent)] text-[var(--color-on-accent)] transition hover:bg-[var(--color-accent-hover)] focus:outline-none focus:ring-2 focus:ring-[color-mix(in_oklch,var(--color-accent)_28%,transparent)]",
                isBuffering ? "animate-pulse" : null,
              )}
            >
              {isPlaying ? (
                <Pause size={16} strokeWidth={1.9} aria-hidden />
              ) : (
                <Play size={16} strokeWidth={1.9} aria-hidden />
              )}
            </button>
            <button
              type="button"
              aria-label={t("audioMixerRestart")}
              title={t("audioMixerRestart")}
              onClick={resetMixerState}
              className="inline-flex h-9 w-9 items-center justify-center rounded-[var(--radius-input)] border border-[var(--color-border)] bg-[var(--color-surface-1)] text-[var(--color-text-primary)] transition hover:border-[var(--color-accent)] hover:text-[var(--color-accent)] focus:outline-none focus:ring-2 focus:ring-[color-mix(in_oklch,var(--color-accent)_28%,transparent)]"
            >
              <RotateCcw size={15} strokeWidth={1.8} aria-hidden />
            </button>
            <button
              type="button"
              aria-label={t("audioMixerResetOffsets")}
              title={t("audioMixerResetOffsets")}
              onClick={resetMixerState}
              className="inline-flex h-9 w-9 items-center justify-center rounded-[var(--radius-input)] border border-[var(--color-border)] bg-[var(--color-surface-1)] text-[var(--color-text-primary)] transition hover:border-[var(--color-accent)] hover:text-[var(--color-accent)] focus:outline-none focus:ring-2 focus:ring-[color-mix(in_oklch,var(--color-accent)_28%,transparent)]"
            >
              <GripHorizontal size={15} strokeWidth={1.8} aria-hidden />
            </button>
          </div>
        </div>
        {contextLabel ? (
          <p
            data-testid="audio-mixer-context-label"
            className="rounded-[var(--radius-input)] border border-[color-mix(in_oklch,var(--color-accent)_34%,var(--color-border))] bg-[color-mix(in_oklch,var(--color-accent)_11%,var(--color-surface-1))] px-3 py-2 text-xs font-semibold leading-5 text-[var(--color-text-primary)]"
            title={contextLabel}
          >
            {contextLabel}
          </p>
        ) : null}
        {description ? (
          <p
            data-testid="audio-mixer-description"
            className="flex items-start gap-2 rounded-[var(--radius-input)] border border-[color-mix(in_oklch,var(--color-accent)_26%,var(--color-border))] bg-[color-mix(in_oklch,var(--color-accent)_8%,var(--color-surface-1))] px-3 py-2 text-xs leading-5 text-[var(--color-text-secondary)]"
          >
            <Headphones
              size={14}
              strokeWidth={1.9}
              aria-hidden
              className="mt-0.5 shrink-0 text-[var(--color-accent)]"
            />
            <span className="min-w-0">{description}</span>
          </p>
        ) : null}
      </header>

      {availableTracks.length > 0 ? (
        <div
          className="mt-3 flex flex-wrap gap-2"
          aria-label={t("audioMixerAvailableTracks")}
        >
          {availableTracks.map((track) => (
            <button
              key={track.id}
              type="button"
              draggable
              aria-label={t("audioMixerAddTrack", { name: track.label })}
              title={t("audioMixerAddTrack", { name: track.label })}
              onClick={() => addAvailableTrack(track.id)}
              onDragStart={(event) => beginAvailableTrackDrag(event, track.id)}
              className="inline-flex h-8 max-w-full items-center gap-2 rounded-[var(--radius-input)] border border-dashed border-[color-mix(in_oklch,var(--color-border)_72%,var(--color-accent))] bg-[var(--color-surface-1)] px-2.5 text-xs font-medium text-[var(--color-text-secondary)] transition hover:border-[var(--color-accent)] hover:text-[var(--color-accent)] focus:outline-none focus:ring-2 focus:ring-[color-mix(in_oklch,var(--color-accent)_28%,transparent)]"
              style={{ "--track-color": track.color } as CSSProperties}
            >
              <Plus size={13} strokeWidth={1.9} aria-hidden />
              <span className="h-2 w-2 shrink-0 rounded-full bg-[var(--track-color)]" />
              <span className="truncate">{track.label}</span>
            </button>
          ))}
        </div>
      ) : null}

      <div className="mt-3 grid gap-2 sm:grid-cols-[minmax(12rem,1fr)_minmax(16rem,2fr)] sm:items-center">
        <div className="font-mono text-xs text-[var(--color-text-muted)]">
          <span className="text-[var(--color-text-primary)]">
            {formatTime(currentTime)}
          </span>
          <span> / {formatTime(timelineDuration)}</span>
        </div>
        <div className="grid grid-cols-[1fr_auto] items-center gap-3">
          <div className="relative h-6 min-w-0">
            <div
              aria-hidden="true"
              className="pointer-events-none absolute inset-x-0 top-1/2 h-1 -translate-y-1/2 rounded-full bg-[color-mix(in_oklch,var(--color-border-strong)_55%,transparent)]"
            >
              <span
                ref={progressFillRef}
                className="absolute left-0 top-0 h-full rounded-full bg-[var(--color-accent)]"
                style={{
                  width: `${(currentTime / timelineDuration) * 100}%`,
                }}
              />
            </div>
            <input
              type="range"
              min={0}
              max={timelineDuration}
              step={0.05}
              value={currentTime}
              aria-label={t("audioMixerTimeline")}
              onChange={(event) => setTimelinePosition(Number(event.currentTarget.value))}
              className="peer absolute inset-0 z-20 h-full w-full cursor-pointer opacity-0"
            />
            <span
              ref={progressThumbRef}
              aria-hidden="true"
              data-testid="audio-mixer-timeline-progress-thumb"
              className="pointer-events-none absolute top-1/2 z-10 h-3.5 w-3.5 -translate-x-1/2 -translate-y-1/2 rounded-full border-2 border-[var(--color-accent)] bg-[var(--color-surface-1)] shadow-[var(--shadow-sm)] transition-[scale,box-shadow] duration-[var(--duration-fast)] ease-[var(--easing-standard)] peer-hover:scale-[1.15] peer-focus-visible:scale-[1.15] peer-focus-visible:shadow-[var(--shadow-focus)]"
              style={{
                left: `${(currentTime / timelineDuration) * 100}%`,
              }}
            />
          </div>
          <label className="flex items-center gap-2 text-xs text-[var(--color-text-muted)]">
            <span className="sr-only">{t("audioMixerZoom")}</span>
            <span aria-hidden className="font-medium">x{zoom < 0.1 ? zoom.toFixed(2) : zoom.toFixed(1)}</span>
            <input
              type="range"
              min={minTimelineZoom}
              max={maxTimelineZoom}
              step={0.05}
              value={zoom}
              aria-label={t("audioMixerZoom")}
              onChange={(event) => setTimelineZoom(Number(event.currentTarget.value))}
              className="tn-range"
              style={
                {
                  "--tn-range-progress":
                    ((zoom - minTimelineZoom) /
                      (maxTimelineZoom - minTimelineZoom)) *
                    100,
                  width: "5rem",
                  flex: "none",
                } as CSSProperties
              }
            />
          </label>
        </div>
      </div>

      <div
        className={cn(
          "mt-4 overflow-hidden rounded-[var(--radius-input)] border border-[var(--color-border)] transition",
          isTrackDropActive
            ? "border-[var(--color-accent)] ring-2 ring-[color-mix(in_oklch,var(--color-accent)_22%,transparent)]"
            : null,
        )}
        onDragLeave={() => setIsTrackDropActive(false)}
        onDragOver={handleTrackDragOver}
        onDrop={handleTrackDrop}
      >
        <div className="grid grid-cols-[8.5rem_minmax(0,1fr)] sm:grid-cols-[minmax(9.5rem,13rem)_minmax(0,1fr)]">
          <div className="border-r border-[var(--color-border)] bg-[var(--color-surface-1)]">
            <div className="h-7 border-b border-[var(--color-border)]" />
            {tracks.map((track) => {
              const isMuted = mutedIds.has(track.id);
              const isSolo = soloIds.has(track.id);
              const gainDb = trackGainDb[track.id] ?? 0;
              const offset = offsets[track.id] ?? 0;

              return (
                <div
                  key={track.id}
                  className="rvc-track-control h-[92px] border-b border-[var(--color-border)] p-2 last:border-b-0"
                  style={{ "--track-color": track.color } as CSSProperties}
                >
                  <div className="grid min-w-0 grid-cols-[auto_minmax(0,1fr)_auto] items-center gap-2">
                    <span className="h-2.5 w-2.5 shrink-0 rounded-full bg-[var(--track-color)]" />
                    <span
                      className="truncate text-xs font-semibold text-[var(--color-text-primary)]"
                      title={track.file.name}
                    >
                      {track.label}
                    </span>
                    {removableTrackIdSet.has(track.id) ? (
                      <button
                        type="button"
                        aria-label={t("audioMixerRemoveTrack", { name: track.label })}
                        title={t("audioMixerRemoveTrack", { name: track.label })}
                        onClick={() => onRemoveTrack?.(track.id)}
                        className="inline-flex h-5 w-5 shrink-0 items-center justify-center rounded-[var(--radius-input)] text-[var(--color-text-muted)] transition-colors duration-[var(--duration-fast)] ease-[var(--easing-standard)] hover:bg-[var(--color-surface-2)] hover:text-[var(--color-accent)] focus:outline-none focus:ring-2 focus:ring-[color-mix(in_oklch,var(--color-accent)_28%,transparent)]"
                      >
                        <X size={12} strokeWidth={1.9} aria-hidden />
                      </button>
                    ) : null}
                  </div>
                  <div className="rvc-gain-row mt-2 flex items-center gap-1.5">
                    <button
                      type="button"
                      aria-label={
                        isMuted
                          ? t("audioMixerUnmute", { name: track.label })
                          : t("audioMixerMute", { name: track.label })
                      }
                      aria-pressed={isMuted}
                      title={
                        isMuted
                          ? t("audioMixerUnmute", { name: track.label })
                          : t("audioMixerMute", { name: track.label })
                      }
                      onClick={() => toggleMuted(track.id)}
                      className={cn(
                        "inline-flex h-7 w-7 items-center justify-center rounded-[var(--radius-input)] border transition-colors duration-[var(--duration-fast)] ease-[var(--easing-standard)] focus:outline-none focus:ring-2 focus:ring-[color-mix(in_oklch,var(--color-accent)_28%,transparent)]",
                        isMuted
                          ? "border-[var(--color-accent)] bg-[color-mix(in_oklch,var(--color-accent)_13%,transparent)] text-[var(--color-accent)]"
                          : "border-[var(--color-border)] bg-[var(--color-bg)] text-[var(--color-text-secondary)] hover:border-[var(--color-accent)] hover:text-[var(--color-accent)]",
                      )}
                    >
                      {isMuted ? (
                        <VolumeX size={14} strokeWidth={1.9} aria-hidden />
                      ) : (
                        <Volume2 size={14} strokeWidth={1.9} aria-hidden />
                      )}
                    </button>
                    <button
                      type="button"
                      aria-label={
                        isSolo
                          ? t("audioMixerUnsolo", { name: track.label })
                          : t("audioMixerSolo", { name: track.label })
                      }
                      aria-pressed={isSolo}
                      title={
                        isSolo
                          ? t("audioMixerUnsolo", { name: track.label })
                          : t("audioMixerSolo", { name: track.label })
                      }
                      onClick={() => toggleSolo(track.id)}
                      className={cn(
                        "inline-flex h-7 w-7 items-center justify-center rounded-[var(--radius-input)] border transition-colors duration-[var(--duration-fast)] ease-[var(--easing-standard)] focus:outline-none focus:ring-2 focus:ring-[color-mix(in_oklch,var(--color-accent)_28%,transparent)]",
                        isSolo
                          ? "border-[var(--color-success)] bg-[color-mix(in_oklch,var(--color-success)_14%,transparent)] text-[var(--color-success)]"
                          : "border-[var(--color-border)] bg-[var(--color-bg)] text-[var(--color-text-secondary)] hover:border-[var(--color-success)] hover:text-[var(--color-success)]",
                      )}
                    >
                      <Headphones size={14} strokeWidth={1.9} aria-hidden />
                    </button>
                    <input
                      type="range"
                      min={minTrackGainDb}
                      max={maxTrackGainDb}
                      step={trackGainDbStep}
                      value={gainDb}
                      aria-label={t("audioMixerVolume", { name: track.label })}
                      aria-valuetext={formatGainDb(gainDb)}
                      onChange={(event) => {
                        const nextGainDb = clamp(
                          roundGainDb(Number(event.currentTarget.value)),
                          minTrackGainDb,
                          maxTrackGainDb,
                        );
                        setTrackGainDb((previous) => ({
                          ...previous,
                          [track.id]: nextGainDb,
                        }));
                      }}
                      className="tn-range min-w-0 flex-1"
                      style={
                        {
                          "--tn-range-progress":
                            ((gainDb - minTrackGainDb) /
                              (maxTrackGainDb - minTrackGainDb)) *
                            100,
                        } as CSSProperties
                      }
                    />
                    <span className="w-12 shrink-0 text-right font-mono text-[10px] text-[var(--color-text-muted)]">
                      {formatGainDb(gainDb)}
                    </span>
                  </div>
                  <div className="mt-2 grid grid-cols-[1fr_auto] items-center gap-2">
                    <input
                      type="range"
                      min={-maxTrackOffsetSeconds}
                      max={maxTrackOffsetSeconds}
                      step={0.05}
                      value={offset}
                      aria-label={t("audioMixerOffset", { name: track.label })}
                      onChange={(event) =>
                        updateTrackOffset(track.id, Number(event.currentTarget.value))
                      }
                      className="tn-range min-w-0"
                      style={
                        {
                          "--tn-range-progress":
                            ((offset + maxTrackOffsetSeconds) /
                              (maxTrackOffsetSeconds * 2)) *
                            100,
                        } as CSSProperties
                      }
                    />
                    <span className="w-12 text-right font-mono text-[10px] text-[var(--color-text-muted)]">
                      {formatOffset(offset)}
                    </span>
                  </div>
                </div>
              );
            })}
          </div>

          <div
            ref={setTimelineScrollElement}
            role="group"
            tabIndex={0}
            aria-label={t("audioMixerTimeline")}
            aria-keyshortcuts="+ - 0 Shift+ArrowLeft Shift+ArrowRight"
            title={t("audioMixerTimelineShortcuts")}
            data-testid="audio-mixer-timeline-scroller"
            onPointerDown={beginTimelineZoomDrag}
            onPointerMove={handleTimelinePointerMove}
            onKeyDown={handleTimelineKeyDown}
            className="min-w-0 overflow-x-auto bg-[var(--color-bg)] focus:outline-none focus:ring-2 focus:ring-inset focus:ring-[color-mix(in_oklch,var(--color-accent)_28%,transparent)]"
          >
            <div
              className="relative"
              data-audio-mixer-timeline
              style={{ width: `${timelineWidth}px` }}
            >
              <div className="relative h-7 border-b border-[var(--color-border)] bg-[var(--color-surface-2)]">
                {rulerMarks.map((mark) => (
                  <span
                    key={mark}
                    className="absolute top-0 h-full border-l border-[var(--color-border)] pl-1 text-[10px] leading-7 text-[var(--color-text-muted)]"
                    style={{ left: `${(mark / timelineDuration) * 100}%` }}
                  >
                    {formatTime(mark)}
                  </span>
                ))}
              </div>
              <div className="relative">
                <button
                  ref={playheadRef}
                  type="button"
                  aria-label={t("audioMixerTimeline")}
                  title={formatTime(currentTime)}
                  data-testid="audio-mixer-playhead"
                  onPointerDown={beginPlayheadDrag}
                  className="absolute bottom-0 top-0 z-30 w-6 -translate-x-1/2 cursor-ew-resize touch-none bg-transparent p-0 focus:outline-none focus:ring-2 focus:ring-inset focus:ring-[color-mix(in_oklch,var(--color-accent)_32%,transparent)]"
                  style={{
                    left: `${(currentTime / timelineDuration) * 100}%`,
                  }}
                >
                  <span
                    aria-hidden="true"
                    className="absolute bottom-0 left-1/2 top-0 w-0.5 -translate-x-1/2 bg-[var(--color-accent)] shadow-[0_0_0_1px_color-mix(in_oklch,var(--color-accent)_35%,transparent)]"
                  />
                  <span
                    aria-hidden="true"
                    className="absolute left-1/2 top-1 h-3 w-3 -translate-x-1/2 rounded-[3px] border border-[var(--color-accent)] bg-[var(--color-bg)] shadow-[var(--shadow-sm)]"
                  />
                </button>
                {tracks.map((track) => {
                  const duration =
                    finiteDuration(durations[track.id]) ?? defaultTimelineSeconds;
                  const offset = offsets[track.id] ?? 0;
                  const start = Math.max(0, offset);
                  const end = Math.max(start + 0.4, offset + duration);
                  const left = clamp((start / timelineDuration) * 100, 0, 100);
                  const width = clamp(((end - start) / timelineDuration) * 100, 2, 100);
                  const waveform = waveformPeaks[track.id];
                  const waveformPath = waveform ? waveformSvgPath(waveform) : "";
                  const hasDecodedWaveform = waveformPath.length > 0;
                  const visibleLocalStart = Math.max(0, start - offset);
                  const visibleDuration = Math.max(0.01, end - start);
                  const bufferedRanges = trackBufferedRanges[track.id] ?? [];

                  return (
                    <div
                      key={track.id}
                      className="rvc-track-lane relative h-[92px] border-b border-[var(--color-border)] bg-[linear-gradient(90deg,color-mix(in_oklch,var(--color-border)_36%,transparent)_1px,transparent_1px)] bg-[length:56px_100%] last:border-b-0"
                    >
                      <button
                        type="button"
                        aria-label={t("audioMixerClip", { name: track.label })}
                        title={t("audioMixerClip", { name: track.label })}
                        onPointerDown={(event) => beginClipDrag(event, track.id)}
                        className="absolute top-1/2 flex h-16 min-w-0 -translate-y-1/2 cursor-ew-resize flex-col justify-start overflow-hidden rounded-[var(--radius-input)] border px-3 py-2 text-left shadow-[var(--shadow-sm)] transition hover:brightness-105 focus:outline-none focus:ring-2 focus:ring-[color-mix(in_oklch,var(--color-accent)_28%,transparent)]"
                        style={
                          {
                            "--track-color": track.color,
                            left: `${left}%`,
                            width: `${width}%`,
                            borderColor:
                              "color-mix(in oklch, var(--track-color) 52%, transparent)",
                            background:
                              "linear-gradient(180deg, color-mix(in oklch, var(--track-color) 22%, var(--color-surface-1)), color-mix(in oklch, var(--track-color) 10%, var(--color-surface-1)))",
                          } as CSSProperties
                        }
                      >
                        <span
                          aria-hidden="true"
                          className="pointer-events-none absolute inset-y-0 left-0 right-0 overflow-hidden rounded-[inherit]"
                        >
                          {bufferedRanges.map((range, index) => {
                            const rangeStart = clamp(
                              range.start - visibleLocalStart,
                              0,
                              visibleDuration,
                            );
                            const rangeEnd = clamp(
                              range.end - visibleLocalStart,
                              0,
                              visibleDuration,
                            );
                            if (rangeEnd - rangeStart <= 0.01) return null;
                            return (
                              <span
                                key={`${track.id}-buffer-${range.start}-${range.end}-${index}`}
                                data-testid="audio-mixer-track-buffered-range"
                                className="absolute bottom-0 top-0 bg-[color-mix(in_oklch,var(--color-text-muted)_20%,var(--color-surface-1))]"
                                style={{
                                  left: `${(rangeStart / visibleDuration) * 100}%`,
                                  width: `${((rangeEnd - rangeStart) / visibleDuration) * 100}%`,
                                }}
                              />
                            );
                          })}
                        </span>
                        <span
                          aria-hidden="true"
                          data-testid="audio-mixer-waveform"
                          data-waveform-source={hasDecodedWaveform ? "decoded" : "pending"}
                          className={cn(
                            "absolute inset-x-2 bottom-2 top-2 overflow-hidden [mask-image:linear-gradient(90deg,transparent,black_6%,black_94%,transparent)]",
                            hasDecodedWaveform ? "opacity-95" : "opacity-45",
                          )}
                        >
                          {hasDecodedWaveform ? (
                            <svg
                              viewBox="0 0 1000 100"
                              preserveAspectRatio="none"
                              className="h-full w-full"
                            >
                              <path
                                d={waveformPath}
                                className="fill-[var(--track-color)]"
                              />
                              <path
                                d={waveformPath}
                                className="fill-[var(--track-color)] opacity-35 blur-[2px]"
                              />
                            </svg>
                          ) : (
                            <span
                              data-testid="audio-mixer-waveform-pending"
                              className="absolute left-3 right-3 top-1/2 h-px -translate-y-1/2 rounded-full bg-[var(--track-color)] opacity-45"
                            />
                          )}
                        </span>
                        <span className="relative z-10 flex min-w-0 max-w-full items-center gap-2 rounded-[var(--radius-input)] bg-[color-mix(in_oklch,var(--color-surface-1)_76%,transparent)] px-1.5 py-0.5">
                          <GripHorizontal
                            size={13}
                            strokeWidth={1.9}
                            aria-hidden
                            className="shrink-0 text-[var(--color-text-muted)]"
                          />
                          <span className="truncate text-xs font-semibold text-[var(--color-text-primary)]">
                            {track.label}
                          </span>
                        </span>
                      </button>
                    </div>
                  );
                })}
              </div>
            </div>
          </div>
        </div>
      </div>
    </section>
  );
}
