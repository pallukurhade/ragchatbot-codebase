# Frontend Changes — Dark/Light Theme Toggle

## Summary

Added a dark/light theme toggle button that lets users switch between the existing dark theme and a new light theme. The preference is saved in `localStorage` and applied on page load with no flash.

---

## Files Changed

### `frontend/index.html`

- Added `<button id="themeToggle">` fixed to the top-right corner of the viewport.
- Button contains two inline SVGs: a **sun icon** (shown in dark mode) and a **moon icon** (shown in light mode), controlled purely via CSS.
- Both icons carry `aria-hidden="true"`; the button itself has `aria-label="Toggle light/dark theme"` and a matching `title` for tooltip/screen-reader support.
- Bumped cache-busting query strings: `style.css?v=15`, `script.js?v=13`.

### `frontend/style.css`

- **Light theme block** — Added `html[data-theme="light"] { … }` overriding every CSS custom property that differs between themes:

  | Variable | Dark | Light |
  |---|---|---|
  | `--background` | `#0f172a` | `#f8fafc` |
  | `--surface` | `#1e293b` | `#ffffff` |
  | `--surface-hover` | `#334155` | `#f1f5f9` |
  | `--text-primary` | `#f1f5f9` | `#0f172a` |
  | `--text-secondary` | `#94a3b8` | `#64748b` |
  | `--border-color` | `#334155` | `#e2e8f0` |
  | `--assistant-message` | `#374151` | `#f1f5f9` |
  | `--welcome-bg` | `#1e3a5f` | `#eff6ff` |
  | `--welcome-border` | `#2563eb` | `#93c5fd` |
  | `--shadow` | 30% black | 8% black |

- **Smooth transitions** — Added `transition: background-color 0.25s ease, color 0.25s ease, border-color 0.25s ease, box-shadow 0.25s ease` to all structural elements (`body`, `.sidebar`, `.chat-messages`, `.message-content`, `#chatInput`, `.stat-item`, `.source-pill`, `.suggested-item`, etc.). Intentionally excluded elements with existing keyframe animations to avoid conflicts.

- **Toggle button styles** — `.theme-toggle`: `position: fixed; top: 1rem; right: 1rem; z-index: 200`, circular 40 × 40 px, uses `--surface`/`--border-color`/`--text-secondary` variables so it adapts automatically to both themes. Hover scales up 1.08×; active scales down 0.95×; focus shows a `--focus-ring` outline.

- **Icon visibility rules** — CSS rules hide the sun in light mode and hide the moon in dark mode (or when no `data-theme` attribute is present, which is the default dark state).

### `frontend/script.js`

- **Instant theme init** — An IIFE at the top of the file (before `DOMContentLoaded`) reads `localStorage.getItem('theme')` and sets `data-theme="light"` on `<html>` if needed. This runs synchronously before paint, preventing a flash of the wrong theme.

- **`toggleTheme()`** — Reads the current `data-theme` attribute, flips to the opposite value, removes the attribute entirely for dark (default) to avoid redundancy, and saves the new value to `localStorage`.

- **Event wiring** — `themeToggle.addEventListener('click', toggleTheme)` added inside `setupEventListeners()`.

---

## Verification

Tested manually via Playwright against the running server:

| Step | Result |
|---|---|
| Page loads in dark mode (default) | Sun icon visible, dark background (`#0f172a`) |
| Click toggle → light mode | Moon icon, light background (`#f8fafc`), `data-theme="light"` set, `localStorage.theme = "light"` |
| Click toggle → dark mode | Sun icon, dark background, `data-theme` attribute removed, `localStorage.theme = "dark"` |
| Keyboard: focus button + Enter | Theme toggled correctly |
| Page reload with `theme = "light"` in storage | Light theme applied immediately on load, no flash |
