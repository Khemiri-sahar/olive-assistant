/*
 * camera.js
 * Compatibility shim for camera actions now implemented in app.js.
 * Kept intentionally to avoid broken imports in future refactors.
 */

"use strict";

function startCameraCapture() {
  if (
    typeof window !== "undefined" &&
    typeof window.startCamera === "function"
  ) {
    return window.startCamera();
  }
  throw new Error(
    "startCamera() is not available on window. Ensure app.js is loaded first.",
  );
}

function stopCameraCapture() {
  if (
    typeof window !== "undefined" &&
    typeof window.stopCamera === "function"
  ) {
    return window.stopCamera();
  }
  throw new Error(
    "stopCamera() is not available on window. Ensure app.js is loaded first.",
  );
}

if (typeof window !== "undefined") {
  window.startCameraCapture = startCameraCapture;
  window.stopCameraCapture = stopCameraCapture;
}

if (typeof window !== "undefined") {
  window.startCameraCapture = startCameraCapture;
  window.stopCameraCapture = stopCameraCapture;
}
