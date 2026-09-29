/**
 * SonicSentinel AI - Progressive Web App Service Worker
 * Enables offline shell caching and native background alert push notifications.
 */

const CACHE_NAME = 'sonicsentinel-v1.2';
const STATIC_ASSETS = [
  '/',
  '/manifest.json',
  '/static/css/style.css',
  '/static/js/app.js',
  '/static/js/radar.js',
  '/static/icons/icon-192.svg',
  '/static/icons/icon-512.svg'
];

// Install: Cache Shell Assets
self.addEventListener('install', (event) => {
  event.waitUntil(
    caches.open(CACHE_NAME).then((cache) => {
      console.log('[SonicSentinel SW] Caching tactical shell assets...');
      return cache.addAll(STATIC_ASSETS);
    }).then(() => self.skipWaiting())
  );
});

// Activate: Cleanup Old Caches
self.addEventListener('activate', (event) => {
  event.waitUntil(
    caches.keys().then((keys) => {
      return Promise.all(
        keys.map((key) => {
          if (key !== CACHE_NAME) {
            console.log('[SonicSentinel SW] Clearing obsolete cache:', key);
            return caches.delete(key);
          }
        })
      );
    }).then(() => self.clients.claim())
  );
});

// Fetch Strategy: Network First with Cache Fallback for shell
self.addEventListener('fetch', (event) => {
  // Pass API and WebSocket calls directly to network
  if (event.request.url.includes('/api/') || event.request.url.includes('/ws/')) {
    return;
  }

  event.respondWith(
    fetch(event.request)
      .then((response) => {
        if (response && response.status === 200 && response.type === 'basic') {
          const responseClone = response.clone();
          caches.open(CACHE_NAME).then((cache) => cache.put(event.request, responseClone));
        }
        return response;
      })
      .catch(() => caches.match(event.request).then((cached) => cached || caches.match('/')))
  );
});

// Push & Background Notification Listener
self.addEventListener('push', (event) => {
  let payload = {
    title: 'CRITICAL ACOUSTIC THREAT DETECTED',
    body: 'High-threat acoustic event registered by SonicSentinel AI.',
    icon: '/static/icons/icon-192.svg',
    badge: '/static/icons/icon-192.svg',
    tag: 'acoustic-threat',
    vibrate: [200, 100, 200, 100, 400],
    data: { url: '/' }
  };

  if (event.data) {
    try {
      const data = event.data.json();
      payload.title = data.title || payload.title;
      payload.body = data.body || payload.body;
      if (data.category) {
        payload.body = `[${data.severity || 'ALERT'}] ${data.category} detected @ Azimuth ${data.azimuth}° | Conf: ${data.confidence}%`;
      }
    } catch (e) {
      payload.body = event.data.text();
    }
  }

  event.waitUntil(
    self.registration.showNotification(payload.title, {
      body: payload.body,
      icon: payload.icon,
      badge: payload.badge,
      vibrate: payload.vibrate,
      tag: payload.tag,
      renotify: true,
      data: payload.data
    })
  );
});

// Notification Click Handler: Focus or open dashboard
self.addEventListener('notificationclick', (event) => {
  event.notification.close();
  event.waitUntil(
    clients.matchAll({ type: 'window', includeUncontrolled: true }).then((windowClients) => {
      for (let client of windowClients) {
        if (client.url.includes(self.registration.scope) && 'focus' in client) {
          return client.focus();
        }
      }
      if (clients.openWindow) {
        return clients.openWindow('/');
      }
    })
  );
});
