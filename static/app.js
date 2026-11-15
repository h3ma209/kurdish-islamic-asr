/**
 * Kurdish Quran Tafsir & Whisper Subtitles SPA
 */

const ICONS = {
  play: `<svg class="ui-icon" width="13" height="13" viewBox="0 0 24 24" fill="currentColor"><polygon points="6 3 20 12 6 21 6 3"/></svg>`,
  pause: `<svg class="ui-icon" width="13" height="13" viewBox="0 0 24 24" fill="currentColor"><rect x="6" y="4" width="4" height="16" rx="1"/><rect x="14" y="4" width="4" height="16" rx="1"/></svg>`,
  volume: `<svg class="ui-icon" width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><polygon points="11 5 6 9 2 9 2 15 6 15 11 19 11 5"/><path d="M15.54 8.46a5 5 0 0 1 0 7.07"/><path d="M19.07 4.93a10 10 0 0 1 0 14.14"/></svg>`,
  clock: `<svg class="ui-icon" width="11" height="11" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><circle cx="12" cy="12" r="10"/><polyline points="12 6 12 12 16 14"/></svg>`,
  mic: `<svg class="ui-icon" width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M12 2a3 3 0 0 0-3 3v7a3 3 0 0 0 6 0V5a3 3 0 0 0-3-3Z"/><path d="M19 10v2a7 7 0 0 1-14 0v-2"/><line x1="12" y1="19" x2="12" y2="22"/></svg>`,
  sparkles: `<svg class="ui-icon" width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="m12 3-1.912 5.813a2 2 0 0 1-1.275 1.275L3 12l5.813 1.912a2 2 0 0 1 1.275 1.275L12 21l1.912-5.813a2 2 0 0 1 1.275-1.275L21 12l-5.813-1.912a2 2 0 0 1-1.275-1.275L12 3Z"/></svg>`,
  book: `<svg class="ui-icon" width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M4 19.5v-15A2.5 2.5 0 0 1 6.5 2H20v20H6.5a2.5 2.5 0 0 1-2.5-2.5Z"/><path d="M6 2v20"/><path d="M4 19.5A2.5 2.5 0 0 1 6.5 17H20"/></svg>`,
  download: `<svg class="ui-icon" width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"/><polyline points="7 10 12 15 17 10"/><line x1="12" y1="15" x2="12" y2="3"/></svg>`,
  waveform: `<svg class="ui-icon" width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M2 10v4"/><path d="M6 7v10"/><path d="M10 4v16"/><path d="M14 6v12"/><path d="M18 9v6"/><path d="M22 11v2"/></svg>`,
};

const state = {
  surahs: [],
  filter: "juz_amma",
  searchQuery: "",
  activeSurahNum: 1,
  activeSurahData: null,
  playbackMode: "idle", // 'idle' | 'verse_tafsir' | 'verse_full' | 'all_tafsir_queue' | 'full_surah'
  activeAyah: null,
  segmentStopTime: null,
  tafsirQueue: [],
  pollTimer: null,
  selectedTafsir: localStorage.getItem("kurdish_tafsir_edition") || "asan",
  classicalTafsirs: {}, // cache { [edition]: { [surahNum]: { [ayahNum]: text } } }
};

const TAFSIR_NAMES = {
  asan: "تەفسیری ئاسان",
  raman: "تەفسیری ڕامان",
  puxta: "تەفسیری پوختە",
  rebar: "تەفسیری ڕێبەر",
  hazhar: "وەرگێڕانی هەژار",
  tawhid: "تەفسیری تەوحید",
  roshn: "تەفسیری ڕۆشن",
  maisar: "تەفسیری مویەسەر (بە کوردی)",
  mokhtasar: "تەفسیری موختەسەر (بە کوردی)",
  sanahi: "تەفسیری ساناهی (بادینی)",
  zhin: "تەفسیری ژین (بادینی)",
  runahi: "تەفسیری ڕووناهی (بادینی)",
  ibn_kathir: "تفسير ابن كثير (ئیبن کەسیر)",
  tabari: "تفسير الطبري (تەبەری)",
  qurtubi: "تفسير القرطبي (قورتوبی)",
  saadi: "تفسير السعدي (سەعدی)",
};

async function loadSelectedTafsirIfNeeded() {
  const ed = state.selectedTafsir;
  const surahNum = state.activeSurahNum;
  if (["ibn_kathir", "tabari", "qurtubi", "saadi"].includes(ed)) {
    if (!state.classicalTafsirs[ed]) state.classicalTafsirs[ed] = {};
    if (!state.classicalTafsirs[ed][surahNum]) {
      try {
        const res = await fetch(`/api/tafsir/${ed}/${surahNum}`);
        const data = await res.json();
        state.classicalTafsirs[ed][surahNum] = data.ayahs || {};
      } catch (err) {
        console.warn("Could not load classical tafsir:", ed, err);
      }
    }
  }
}

function getActiveTafsirText(verse) {
  if (!verse) return "";
  const ed = state.selectedTafsir;
  if (["ibn_kathir", "tabari", "qurtubi", "saadi"].includes(ed)) {
    const cached = state.classicalTafsirs[ed]?.[state.activeSurahNum];
    if (cached) {
      return cached[String(verse.number_in_surah)] || cached[verse.number_in_surah] || "";
    }
    return "";
  }
  if (verse.kurdish_tafsirs && verse.kurdish_tafsirs[ed]) {
    return verse.kurdish_tafsirs[ed];
  }
  return verse.kurdish_text || "";
}

function getActiveTafsirName() {
  return TAFSIR_NAMES[state.selectedTafsir] || "تەفسیر";
}

function cleanTafsirForTTS(text) {
  if (!text) return "";
  return text
    .replace(/\{[^}]*\}/g, " ")
    .replace(/﴾[^﴿]*﴿/g, " ")
    .replace(/\s+/g, " ")
    .trim();
}

const els = {
  selectTafsirEdition: document.getElementById("select-tafsir-edition"),
  tafsirActivePill: document.getElementById("tafsir-active-pill"),
  surahList: document.getElementById("surah-list"),
  surahSearch: document.getElementById("surah-search"),
  tabBtns: document.querySelectorAll(".tab-btn"),
  video: document.getElementById("main-video"),
  subtitleTrack: document.getElementById("main-subtitle-track"),
  playbackBadge: document.getElementById("playback-mode-badge"),
  playbackStatusPill: document.getElementById("playback-status-pill"),
  playbackSurahPill: document.getElementById("playback-surah-pill"),
  activeSurahNum: document.getElementById("active-surah-num"),
  activeSurahTitle: document.getElementById("active-surah-title"),
  activeSurahMeta: document.getElementById("active-surah-meta"),
  ytSourceLink: document.getElementById("yt-source-link"),
  downloadVttLink: document.getElementById("download-vtt-link"),
  downloadSrtLink: document.getElementById("download-srt-link"),
  liveCueRole: document.getElementById("live-cue-role"),
  liveCueText: document.getElementById("live-cue-text"),
  btnPlayAllTafsir: document.getElementById("btn-play-all-tafsir"),
  btnPlayFullSurah: document.getElementById("btn-play-full-surah"),
  btnStopPlayback: document.getElementById("btn-stop-playback"),
  togglePreciseSlice: document.getElementById("toggle-precise-slice"),
  toggleAutoNext: document.getElementById("toggle-auto-next"),
  toggleShowWhisper: document.getElementById("toggle-show-whisper"),
  processBanner: document.getElementById("process-banner"),
  processBannerTitle: document.getElementById("process-banner-title"),
  processBannerDesc: document.getElementById("process-banner-desc"),
  btnTriggerProcess: document.getElementById("btn-trigger-process"),
  versesContainer: document.getElementById("verses-container"),
  clonerInputText: document.getElementById("cloner-input-text"),
  clonerEngineSelect: document.getElementById("cloner-engine-select"),
  btnSynthesizeCustom: document.getElementById("btn-synthesize-custom"),
  btnStopGeneration: document.getElementById("btn-stop-generation"),
  clonerStatusText: document.getElementById("cloner-status-text"),
  clonerOutputAudio: document.getElementById("cloner-output-audio"),
  clonerPhoneticPreview: document.getElementById("cloner-phonetic-preview"),
};

function fmtTime(sec) {
  if (sec == null || isNaN(sec)) return "0:00";
  const m = Math.floor(sec / 60);
  const s = (sec % 60).toFixed(1).padStart(4, "0");
  return `${m}:${s}`;
}

async function fetchSurahList() {
  const res = await fetch("/api/surahs");
  const data = await res.json();
  state.surahs = data.surahs || [];
  renderSurahSidebar();
}

function renderSurahSidebar() {
  const q = state.searchQuery.trim().toLowerCase();
  const filtered = state.surahs.filter((s) => {
    if (state.filter === "juz_amma" && !s.is_juz_amma) return false;
    if (state.filter === "ready" && !s.whisper_aligned) return false;
    if (!q) return true;
    return (
      String(s.number).includes(q) ||
      (s.name_english || "").toLowerCase().includes(q) ||
      (s.name_translation || "").toLowerCase().includes(q) ||
      (s.name_arabic || "").includes(q)
    );
  });

  if (filtered.length === 0) {
    els.surahList.innerHTML = `<div class="loading-state">No matching Surahs found.</div>`;
    return;
  }

  els.surahList.innerHTML = filtered
    .map((s) => {
      const isActive = s.number === state.activeSurahNum;
      let badgeClass = "idle";
      let badgeText = "Cloud";
      if (s.whisper_aligned) {
        badgeClass = "ready";
        badgeText = "Audio + Subtitles";
      } else if (s.video_downloaded) {
        badgeClass = "downloaded";
        badgeText = "Audio Ready";
      }
      return `
        <div class="surah-item ${isActive ? "active" : ""}" data-surah="${s.number}">
          <div class="surah-item-left">
            <div class="surah-num-circle">${s.number}</div>
            <div class="surah-item-titles">
              <h4>${s.name_english}</h4>
              <p>${s.number_of_ayahs} Ayahs • ${s.name_translation}</p>
            </div>
          </div>
          <div class="surah-item-right">
            <span class="surah-ar-name">${s.name_arabic}</span>
            <span class="status-dot ${badgeClass}">${badgeText}</span>
          </div>
        </div>
      `;
    })
    .join("");

  els.surahList.querySelectorAll(".surah-item").forEach((item) => {
    item.addEventListener("click", () => {
      const num = parseInt(item.dataset.surah, 10);
      loadSurah(num);
    });
  });
}

async function loadSurah(surahNum) {
  stopPlayback();
  state.activeSurahNum = surahNum;
  renderSurahSidebar();

  const res = await fetch(`/api/surah/${surahNum}`);
  const data = await res.json();
  state.activeSurahData = data;

  // Update Header
  els.activeSurahNum.textContent = `Surah ${data.surah_number}`;
  els.activeSurahTitle.textContent = `${data.name_english} — ${data.name_arabic}`;
  const statusLabel = data.whisper_aligned
    ? `${(data.segments || []).length} Subtitle Cues • ${fmtTime(data.duration)}`
    : "Audio On-Demand";
  els.activeSurahMeta.textContent = `${data.number_of_ayahs} Verses • ${data.revelation_type} • ${statusLabel}`;

  if (els.playbackSurahPill) {
    els.playbackSurahPill.textContent = `Surah ${data.surah_number}`;
  }
  if (els.playbackStatusPill) {
    els.playbackStatusPill.textContent = data.whisper_aligned ? "Audio Ready" : "Text Only";
  }

  if (data.youtube_url) {
    els.ytSourceLink.href = data.youtube_url;
    els.ytSourceLink.style.display = "inline-flex";
  } else {
    els.ytSourceLink.style.display = "none";
  }

  if (data.vtt_file) {
    els.downloadVttLink.href = data.vtt_file;
    els.downloadVttLink.style.display = "inline-flex";
    els.downloadSrtLink.href = data.srt_file;
    els.downloadSrtLink.style.display = "inline-flex";
  } else {
    els.downloadVttLink.style.display = "none";
    els.downloadSrtLink.style.display = "none";
  }

  // Setup Audio Track
  if (data.video_file && data.video_downloaded) {
    els.video.src = data.video_file;
    if (data.vtt_file && els.subtitleTrack) {
      els.subtitleTrack.src = data.vtt_file;
    } else if (els.subtitleTrack) {
      els.subtitleTrack.removeAttribute("src");
    }
    els.video.load();
  } else {
    els.video.removeAttribute("src");
    els.video.load();
  }

  // Show or hide on-demand process banner
  updateProcessBanner(data);
  await loadSelectedTafsirIfNeeded();
  renderVerses();
}

function updateProcessBanner(data) {
  if (data.whisper_aligned) {
    els.processBanner.classList.add("hidden");
    if (state.pollTimer) {
      clearInterval(state.pollTimer);
      state.pollTimer = null;
    }
    return;
  }

  els.processBanner.classList.remove("hidden");
  const st = data.processing_state || "idle";
  if (st === "downloading" || st === "transcribing") {
    els.processBannerTitle.textContent =
      st === "downloading"
        ? `Downloading Surah ${data.surah_number} Audio...`
        : `Running Local Whisper Alignment on Surah ${data.surah_number}...`;
    els.processBannerDesc.textContent = "This page will automatically refresh once Whisper alignment finishes.";
    els.btnTriggerProcess.disabled = true;
    els.btnTriggerProcess.textContent = "Processing...";
    if (!state.pollTimer) {
      state.pollTimer = setInterval(async () => {
        await fetchSurahList();
        const r = await fetch(`/api/surah/${state.activeSurahNum}`);
        const d = await r.json();
        if (d.whisper_aligned || (d.processing_state && d.processing_state.startsWith("error"))) {
          clearInterval(state.pollTimer);
          state.pollTimer = null;
          loadSurah(state.activeSurahNum);
        } else {
          updateProcessBanner(d);
        }
      }, 3000);
    }
  } else {
    els.processBannerTitle.textContent = `Audio & Subtitles Not Yet Extracted for Surah ${data.surah_number}`;
    els.processBannerDesc.textContent =
      "Download recitation audio and run local Apple Silicon Whisper alignment for this Surah.";
    els.btnTriggerProcess.disabled = false;
    els.btnTriggerProcess.innerHTML = `${ICONS.download}<span>Extract Audio & Subtitles</span>`;
  }
}

function getVerseTafsirWindow(verse) {
  const useSub = els.togglePreciseSlice.checked && verse.shared_block && verse.tafsir_sub_start != null;
  const start = useSub ? verse.tafsir_sub_start : verse.tafsir_start;
  const end = useSub ? verse.tafsir_sub_end : verse.tafsir_end;
  return { start, end };
}

function renderVerses() {
  const data = state.activeSurahData;
  if (!data || !data.verses) {
    els.versesContainer.innerHTML = "";
    return;
  }

  const showWhisper = els.toggleShowWhisper.checked;
  const aligned = data.whisper_aligned;

  els.versesContainer.innerHTML = data.verses
    .map((v) => {
      const ayahNum = v.number_in_surah;
      const win = aligned ? getVerseTafsirWindow(v) : { start: null, end: null };
      const sharedInfo =
        aligned && v.shared_block && v.shared_verses && v.shared_verses.length > 1
          ? `<span class="shared-pill" title="Grouped verses ${v.shared_verses.join(", ")}">Grouped ${v.shared_verses[0]}–${v.shared_verses[v.shared_verses.length - 1]}</span>`
          : "";

      const timeInfo = aligned
        ? `<span class="timestamp-pill" id="ts-pill-${ayahNum}">${ICONS.clock} <span>${fmtTime(win.start)} – ${fmtTime(win.end)}</span></span>`
        : `<span class="timestamp-pill">Audio pending</span>`;

      const whisperHtml = aligned
        ? `
          <div class="verse-whisper-box ${showWhisper ? "" : "hidden"}" id="whisper-box-${ayahNum}">
            <div class="whisper-header">
              <div class="whisper-title">
                ${ICONS.waveform}
                <span>Whisper Voice Transcript (${fmtTime(v.tafsir_start)} – ${fmtTime(v.tafsir_end)})</span>
              </div>
              <div class="nudge-controls">
                <span class="nudge-field-tag">Start</span>
                <button class="nudge-btn" data-ayah="${ayahNum}" data-field="start" data-delta="-0.5">-0.5s</button>
                <button class="nudge-btn" data-ayah="${ayahNum}" data-field="start" data-delta="0.5">+0.5s</button>
                <span class="nudge-sep">|</span>
                <span class="nudge-field-tag">End</span>
                <button class="nudge-btn" data-ayah="${ayahNum}" data-field="end" data-delta="-0.5">-0.5s</button>
                <button class="nudge-btn" data-ayah="${ayahNum}" data-field="end" data-delta="0.5">+0.5s</button>
              </div>
            </div>
            <div class="whisper-voice-text" dir="rtl">${v.whisper_tafsir || "(Kurdish Tafsir Audio Segment)"}</div>
          </div>
        `
        : "";

      return `
        <article class="verse-card" id="verse-card-${ayahNum}" data-ayah="${ayahNum}">
          <div class="verse-card-header">
            <div class="verse-meta-left">
              <span class="ayah-key-badge">${data.surah_number}:${ayahNum}</span>
              ${timeInfo}
              ${sharedInfo}
            </div>
            <div class="verse-actions">
              <button class="btn-play-tafsir" id="btn-tafsir-${ayahNum}" data-ayah="${ayahNum}" ${!aligned ? "disabled" : ""} title="Play Kurdish Tafsir">
                ${ICONS.volume}
                <span class="btn-label">Tafsir</span>
              </button>
              <button class="btn-play-arabic" id="btn-full-${ayahNum}" data-ayah="${ayahNum}" ${!aligned ? "disabled" : ""} title="Play Arabic Recitation + Kurdish Tafsir">
                ${ICONS.play}
                <span>Recitation</span>
              </button>
              <button class="btn-clone-verse" id="btn-clone-${ayahNum}" data-ayah="${ayahNum}" title="Synthesize Kurdish tafsir in narrator voice">
                ${ICONS.sparkles}
                <span>Clone</span>
              </button>
            </div>
          </div>

          <div class="verse-arabic" dir="rtl">${v.arabic_text}</div>
          ${
            getActiveTafsirText(v)
              ? `
                <div class="verse-tafsir-card" dir="rtl">
                  <div class="verse-tafsir-header">
                    <span class="verse-tafsir-chip">
                      ${ICONS.book}
                      <span>${getActiveTafsirName()}</span>
                    </span>
                  </div>
                  <div class="verse-tafsir-text">${getActiveTafsirText(v)}</div>
                </div>
              `
              : `
                <div class="verse-tafsir-card empty" dir="rtl">
                  <div class="verse-tafsir-text-muted">تەفسیر لەگەڵ ئایەتی پێشوودا تۆمارکراوە — کلیک لەسەر Tafsir بکە بۆ بیستن.</div>
                </div>
              `
          }
          ${whisperHtml}
        </article>
      `;
    })
    .join("");

  // Bind Play Kurdish Tafsir buttons
  els.versesContainer.querySelectorAll(".btn-play-tafsir").forEach((btn) => {
    btn.addEventListener("click", () => {
      const ayahNum = parseInt(btn.dataset.ayah, 10);
      if (state.activeAyah === ayahNum && state.playbackMode === "verse_tafsir" && !els.video.paused) {
        els.video.pause();
        updateHighlightUI();
        return;
      }
      playVerseTafsir(ayahNum);
    });
  });

  // Bind Play Arabic + Kurdish buttons
  els.versesContainer.querySelectorAll(".btn-play-arabic").forEach((btn) => {
    btn.addEventListener("click", () => {
      const ayahNum = parseInt(btn.dataset.ayah, 10);
      playVerseFull(ayahNum);
    });
  });

  // Bind Zero-Shot Clone Voice buttons per verse
  els.versesContainer.querySelectorAll(".btn-clone-verse").forEach((btn) => {
    btn.addEventListener("click", async () => {
      const ayahNum = parseInt(btn.dataset.ayah, 10);
      const verse = (state.activeSurahData?.verses || []).find((v) => v.number_in_surah === ayahNum);
      const rawText = getActiveTafsirText(verse);
      const activeText = cleanTafsirForTTS(rawText);
      if (!verse || !activeText) return;
      switchInspectorTab("studio");
      els.clonerInputText.value = activeText;
      await runZeroShotClone(activeText, btn);
    });
  });

  // Bind timestamp fine-tune nudge buttons
  els.versesContainer.querySelectorAll(".nudge-btn").forEach((btn) => {
    btn.addEventListener("click", async () => {
      const ayahNum = parseInt(btn.dataset.ayah, 10);
      const field = btn.dataset.field;
      const delta = parseFloat(btn.dataset.delta);
      await nudgeTimestamp(ayahNum, field, delta);
    });
  });
}

function playVerseTafsir(ayahNum, isQueue = false) {
  const data = state.activeSurahData;
  if (!data || !data.whisper_aligned) return;
  const verse = data.verses.find((v) => v.number_in_surah === ayahNum);
  if (!verse) return;

  if (!isQueue) {
    state.tafsirQueue = [];
  }

  const win = getVerseTafsirWindow(verse);
  state.playbackMode = isQueue ? "all_tafsir_queue" : "verse_tafsir";
  state.activeAyah = ayahNum;
  state.segmentStopTime = win.end;

  if (els.playbackStatusPill) els.playbackStatusPill.textContent = "Tafsir Audio";
  if (els.playbackSurahPill) els.playbackSurahPill.textContent = `Ayah ${data.surah_number}:${ayahNum}`;
  els.playbackBadge.textContent = `Ayah ${data.surah_number}:${ayahNum} • Kurdish Tafsir (${fmtTime(win.start)} – ${fmtTime(win.end)})`;
  els.video.currentTime = Math.max(0, win.start);
  els.video.play();

  scrollToVerseCard(ayahNum);
  updateHighlightUI();
}

function playVerseFull(ayahNum) {
  const data = state.activeSurahData;
  if (!data || !data.whisper_aligned) return;
  const verse = data.verses.find((v) => v.number_in_surah === ayahNum);
  if (!verse) return;

  state.tafsirQueue = [];
  state.playbackMode = "verse_full";
  state.activeAyah = ayahNum;
  state.segmentStopTime = verse.end_time;

  if (els.playbackStatusPill) els.playbackStatusPill.textContent = "Recitation + Tafsir";
  if (els.playbackSurahPill) els.playbackSurahPill.textContent = `Ayah ${data.surah_number}:${ayahNum}`;
  els.playbackBadge.textContent = `Ayah ${data.surah_number}:${ayahNum} • Recitation + Tafsir (${fmtTime(verse.start_time)} – ${fmtTime(verse.end_time)})`;
  els.video.currentTime = Math.max(0, verse.start_time);
  els.video.play();

  scrollToVerseCard(ayahNum);
  updateHighlightUI();
}

function playAllKurdishTafsirOnly() {
  const data = state.activeSurahData;
  if (!data || !data.whisper_aligned || !data.verses.length) return;

  // Build deduplicated queue of tafsir blocks (unless Split Grouped Verses is checked)
  const splitSub = els.togglePreciseSlice.checked;
  const seenStarts = new Set();
  const queue = [];
  for (const v of data.verses) {
    if (!splitSub) {
      const key = `${v.tafsir_start}-${v.tafsir_end}`;
      if (seenStarts.has(key)) continue;
      seenStarts.add(key);
    }
    queue.push(v.number_in_surah);
  }
  state.tafsirQueue = queue.slice(1);
  playVerseTafsir(queue[0], true);
}

function playFullSurahVideo() {
  const data = state.activeSurahData;
  if (!data || !data.video_downloaded) return;
  state.tafsirQueue = [];
  state.playbackMode = "full_surah";
  state.segmentStopTime = null;
  state.activeAyah = 1;
  if (els.playbackStatusPill) els.playbackStatusPill.textContent = "Full Surah Audio";
  els.playbackBadge.textContent = `Full Surah Audio Playback`;
  els.video.currentTime = 0;
  els.video.play();
  updateHighlightUI();
}

function stopPlayback() {
  els.video.pause();
  state.playbackMode = "idle";
  state.activeAyah = null;
  state.segmentStopTime = null;
  state.tafsirQueue = [];
  if (els.playbackStatusPill) els.playbackStatusPill.textContent = "Idle";
  els.playbackBadge.textContent = "Idle • Select Ayah to Listen";
  updateHighlightUI();
}

function scrollToVerseCard(ayahNum) {
  const card = document.getElementById(`verse-card-${ayahNum}`);
  if (card) {
    card.scrollIntoView({ behavior: "smooth", block: "center" });
  }
}

function updateHighlightUI() {
  document.querySelectorAll(".verse-card").forEach((card) => {
    const num = parseInt(card.dataset.ayah, 10);
    const btn = document.getElementById(`btn-tafsir-${num}`);

    card.classList.remove("playing-tafsir", "playing-arabic");
    if (btn) {
      btn.classList.remove("active");
      btn.innerHTML = `${ICONS.volume}<span class="btn-label">Tafsir</span>`;
    }

    if (num === state.activeAyah && !els.video.paused) {
      if (state.playbackMode === "verse_tafsir" || state.playbackMode === "all_tafsir_queue") {
        card.classList.add("playing-tafsir");
        if (btn) {
          btn.classList.add("active");
          btn.innerHTML = `${ICONS.pause}<span class="btn-label">Playing</span>`;
        }
      } else {
        card.classList.add("playing-arabic");
      }
    }
  });
}

async function nudgeTimestamp(ayahNum, field, delta) {
  const data = state.activeSurahData;
  if (!data) return;
  const verse = data.verses.find((v) => v.number_in_surah === ayahNum);
  if (!verse) return;

  if (field === "start") {
    verse.tafsir_start = Math.max(0, +(verse.tafsir_start + delta).toFixed(2));
    verse.tafsir_sub_start = verse.tafsir_start;
  } else {
    verse.tafsir_end = Math.max(verse.tafsir_start + 0.5, +(verse.tafsir_end + delta).toFixed(2));
    verse.tafsir_sub_end = verse.tafsir_end;
  }

  const pill = document.getElementById(`ts-pill-${ayahNum}`);
  const win = getVerseTafsirWindow(verse);
  if (pill) {
    pill.innerHTML = `${ICONS.clock} <span>${fmtTime(win.start)} – ${fmtTime(win.end)}</span>`;
  }

  await fetch(`/api/surah/${data.surah_number}/verse/${ayahNum}/timestamps`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      tafsir_start: verse.tafsir_start,
      tafsir_end: verse.tafsir_end,
    }),
  });

  playVerseTafsir(ayahNum);
}

// Monitor video timeupdate for segment stop, live Whisper subtitles, and active verse tracking
els.video.addEventListener("timeupdate", () => {
  const t = els.video.currentTime;
  const data = state.activeSurahData;
  if (!data) return;

  // Update live Whisper subtitle cue box
  if (data.segments && data.segments.length) {
    const seg = data.segments.find((s) => t >= s.start && t <= s.end);
    if (seg) {
      const isArabic = seg.role === "arabic";
      els.liveCueRole.textContent = isArabic ? "Arabic Recitation" : "Kurdish Tafsir";
      els.liveCueRole.className = `cue-role-pill ${isArabic ? "arabic" : ""}`;
      els.liveCueText.textContent = seg.text;
    }
  }

  // Check if segment reached its stop boundary
  if (state.segmentStopTime != null && t >= state.segmentStopTime) {
    if (state.playbackMode === "all_tafsir_queue" && state.tafsirQueue.length > 0) {
      const nextAyah = state.tafsirQueue.shift();
      playVerseTafsir(nextAyah, true);
      return;
    }
    if (els.toggleAutoNext.checked && state.activeAyah != null) {
      const nextAyah = state.activeAyah + 1;
      const exists = data.verses.some((v) => v.number_in_surah === nextAyah);
      if (exists) {
        if (state.playbackMode === "verse_tafsir") {
          playVerseTafsir(nextAyah);
        } else {
          playVerseFull(nextAyah);
        }
        return;
      }
    }
    stopPlayback();
    return;
  }

  // In full_surah mode, highlight whichever verse contains current timestamp
  if (state.playbackMode === "full_surah" && data.verses) {
    const curVerse = data.verses.find((v) => t >= v.start_time && t <= v.end_time);
    if (curVerse && curVerse.number_in_surah !== state.activeAyah) {
      state.activeAyah = curVerse.number_in_surah;
      scrollToVerseCard(state.activeAyah);
      updateHighlightUI();
    }
  }
});

els.video.addEventListener("play", updateHighlightUI);
els.video.addEventListener("pause", updateHighlightUI);

// Event Listeners
els.tabBtns.forEach((btn) => {
  btn.addEventListener("click", () => {
    els.tabBtns.forEach((b) => b.classList.remove("active"));
    btn.classList.add("active");
    state.filter = btn.dataset.filter;
    renderSurahSidebar();
  });
});

els.surahSearch.addEventListener("input", (e) => {
  state.searchQuery = e.target.value;
  renderSurahSidebar();
});

els.btnPlayAllTafsir.addEventListener("click", playAllKurdishTafsirOnly);
els.btnPlayFullSurah.addEventListener("click", playFullSurahVideo);
els.btnStopPlayback.addEventListener("click", stopPlayback);

els.togglePreciseSlice.addEventListener("change", renderVerses);
els.toggleShowWhisper.addEventListener("change", () => {
  const show = els.toggleShowWhisper.checked;
  document.querySelectorAll(".verse-whisper-box").forEach((box) => {
    box.classList.toggle("hidden", !show);
  });
});

els.btnTriggerProcess.addEventListener("click", async () => {
  if (!state.activeSurahNum) return;
  await fetch(`/api/process/${state.activeSurahNum}`, { method: "POST" });
  const r = await fetch(`/api/surah/${state.activeSurahNum}`);
  const d = await r.json();
  updateProcessBanner(d);
});

let activeTtsAbortController = null;

function stopSpeechGeneration() {
  if (activeTtsAbortController) {
    activeTtsAbortController.abort();
    activeTtsAbortController = null;
  }
  if (els.clonerOutputAudio) {
    els.clonerOutputAudio.pause();
    els.clonerOutputAudio.currentTime = 0;
  }
  els.clonerStatusText.textContent = "Speech generation stopped.";
  els.btnSynthesizeCustom.disabled = false;
  document.querySelectorAll(".btn-clone-verse").forEach((b) => {
    b.disabled = false;
    b.innerHTML = `${ICONS.sparkles}<span>Clone</span>`;
  });
}

if (els.btnStopGeneration) {
  els.btnStopGeneration.addEventListener("click", stopSpeechGeneration);
}

async function runZeroShotClone(text, triggerBtn = null) {
  const cleanText = (text || "").trim();
  if (!cleanText) return;

  stopPlayback();
  stopSpeechGeneration(); // Cancel any in-flight synthesis
  activeTtsAbortController = new AbortController();

  const engine = els.clonerEngineSelect ? els.clonerEngineSelect.value : "audio8_finetuned";
  const origBtnLabel = triggerBtn ? triggerBtn.innerHTML : null;
  if (triggerBtn) {
    triggerBtn.disabled = true;
    triggerBtn.innerHTML = `${ICONS.waveform}<span>Synthesizing...</span>`;
  }
  els.btnSynthesizeCustom.disabled = true;
  els.clonerStatusText.textContent = "Synthesizing native Sorani speech...";

  try {
    const res = await fetch("/api/tts/clone", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ text: cleanText, engine }),
      signal: activeTtsAbortController.signal,
    });
    const data = await res.json();
    if (!res.ok || !data.ok) {
      throw new Error(data.detail || "TTS cloning failed");
    }

    els.clonerOutputAudio.src = data.audio_url;
    els.clonerOutputAudio.classList.remove("hidden");
    els.clonerOutputAudio.play();

    const modeName =
      engine === "audio8_finetuned"
        ? "Fine-Tuned Quran Narrator (Audio8)"
        : engine === "mms_native"
        ? "Pure MMS Sorani"
        : "Sorani + Cloned Narrator Timbre";
    const cacheTag = data.cached ? "Cached instant" : `${data.duration_sec || "?"}s audio in ${data.elapsed_sec || "?"}s`;
    els.clonerStatusText.textContent = `${modeName} Ready (${cacheTag})`;
    if (data.phonetic_text) {
      els.clonerPhoneticPreview.textContent = `Normalized Sorani (ckb): ${data.phonetic_text}`;
      els.clonerPhoneticPreview.classList.remove("hidden");
    }
  } catch (err) {
    if (err.name === "AbortError") {
      els.clonerStatusText.textContent = "Speech generation stopped.";
    } else {
      els.clonerStatusText.textContent = `Error: ${err.message}`;
    }
  } finally {
    activeTtsAbortController = null;
    els.btnSynthesizeCustom.disabled = false;
    if (triggerBtn) {
      triggerBtn.disabled = false;
      triggerBtn.innerHTML = origBtnLabel;
    }
  }
}

if (els.btnSynthesizeCustom) {
  els.btnSynthesizeCustom.addEventListener("click", () => {
    runZeroShotClone(els.clonerInputText.value);
  });
}

// Tafsir Edition Selection Listener
if (els.selectTafsirEdition) {
  els.selectTafsirEdition.value = state.selectedTafsir;
  if (els.tafsirActivePill) {
    const pillText = els.tafsirActivePill.querySelector(".tafsir-pill-text");
    if (pillText) {
      pillText.textContent = getActiveTafsirName();
    } else {
      els.tafsirActivePill.innerHTML = `${ICONS.book}<span class="tafsir-pill-text">${getActiveTafsirName()}</span>`;
    }
  }
  els.selectTafsirEdition.addEventListener("change", async (e) => {
    state.selectedTafsir = e.target.value;
    localStorage.setItem("kurdish_tafsir_edition", state.selectedTafsir);
    if (els.tafsirActivePill) {
      const pillText = els.tafsirActivePill.querySelector(".tafsir-pill-text");
      if (pillText) {
        pillText.textContent = getActiveTafsirName();
      } else {
        els.tafsirActivePill.innerHTML = `${ICONS.book}<span class="tafsir-pill-text">${getActiveTafsirName()}</span>`;
      }
    }
    await loadSelectedTafsirIfNeeded();
    renderVerses();
  });
}

// Inspector Tab Navigation
function switchInspectorTab(tabId) {
  document.querySelectorAll(".inspector-tab-btn").forEach((btn) => {
    btn.classList.toggle("active", btn.dataset.tab === tabId);
  });
  const panePlayback = document.getElementById("pane-playback");
  const paneStudio = document.getElementById("pane-studio");
  if (panePlayback) panePlayback.classList.toggle("hidden", tabId !== "playback");
  if (paneStudio) paneStudio.classList.toggle("hidden", tabId !== "studio");
}

document.querySelectorAll(".inspector-tab-btn").forEach((btn) => {
  btn.addEventListener("click", () => {
    switchInspectorTab(btn.dataset.tab);
  });
});

// Initialize SPA
(async function init() {
  await fetchSurahList();
  await loadSurah(1);
})();
