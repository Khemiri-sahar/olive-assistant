/*
 * recorder.js
 * Compatibility shim for audio recording actions now implemented in app.js.
 * Kept intentionally to avoid broken imports in future refactors.
 */

"use strict";

function toggleAudioRecording() {
  if (
    typeof window !== "undefined" &&
    typeof window.toggleRecording === "function"
  ) {
    return window.toggleRecording();
  }
  throw new Error(
    "toggleRecording() is not available on window. Ensure app.js is loaded first.",
  );
}

function stopAudioRecording() {
  if (
    typeof window !== "undefined" &&
    typeof window.stopRecording === "function"
  ) {
    return window.stopRecording();
  }
  throw new Error(
    "stopRecording() is not available on window. Ensure app.js is loaded first.",
  );
}

if (typeof window !== "undefined") {
  window.toggleAudioRecording = toggleAudioRecording;
  window.stopAudioRecording = stopAudioRecording;
}

if (typeof window !== "undefined") {
  window.toggleAudioRecording = toggleAudioRecording;
  window.stopAudioRecording = stopAudioRecording;
}
