/**
 * sw.js — Service Worker for Olive Assistant PWA
 *
 * Strategy: network-first for everything.
 * Cache is only used as fallback when offline.
 */

const CACHE_NAME = 'olive-assistant-v3';

const STATIC_ASSETS = [
  '/',
  '/index.html',
  '/manifest.json',
  '/src/style.css',
  '/src/app.js',
  '/src/camera.js',
  '/src/recorder.js',
];

// ── Install ────────────────────────────────────────────────────────────────────
self.addEventListener('install', event => {
  event.waitUntil(
    caches.open(CACHE_NAME).then(cache => cache.addAll(STATIC_ASSETS))
  );
  self.skipWaiting();
});

// ── Activate — delete all old caches ──────────────────────────────────────────
self.addEventListener('activate', event => {
  event.waitUntil(
    caches.keys().then(keys =>
      Promise.all(keys.filter(k => k !== CACHE_NAME).map(k => caches.delete(k)))
    )
  );
  self.clients.claim();
});

// ── Fetch — network-first, cache as offline fallback ──────────────────────────
self.addEventListener('fetch', event => {
  const { request } = event;
  const url = new URL(request.url);

  // Skip non-GET and cross-origin (fonts, etc.)
  if (request.method !== 'GET' || url.origin !== location.origin) return;

  // API calls: network only, no caching
  if (url.pathname.startsWith('/api/') || url.pathname === '/health') return;

  // Everything else: network-first
  event.respondWith(
    fetch(request)
      .then(response => {
        const clone = response.clone();
        caches.open(CACHE_NAME).then(cache => cache.put(request, clone));
        return response;
      })
      .catch(() => caches.match(request))
  );
});
