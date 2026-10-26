// UMAP -> grid morph animation, drawn with Observable Plot.
//
// Shared by notebooks/umap_to_grid.py (via wigglystuff ObservablePlot) and the
// exported Observable notebook cell. This file is a single JS *expression*
// that evaluates to a DOM node, with these variables in scope:
//
//   Plot, d3                  - Observable Plot and d3
//   width, height             - available size in px, including `chrome`
//   chrome                    - px above the plot for summary + controls
//   points                    - [{sx, sy, dx, dy, color, img, label}, ...]
//                               start (sx, sy) and end (dx, dy) in mosaic pixels
//   cols, rows                - grid size
//   cellW, cellH              - cell size in mosaic pixels
//   canvasW, canvasH          - mosaic canvas size in pixels
//   fitWidth                  - OPTIONAL; if true, fill the available width
//                               (width/height are then only the initial size)
//   imgAspect, meanDist       - OPTIONAL; if defined, a summary line is shown
//                               (image aspect ratio; mean displacement in cell diagonals)
//   model                     - OPTIONAL; an anywidget model with a synced integer
//                               `selected` trait makes points clickable: clicking sets
//                               `selected` to the point's `id` (or its index if no id;
//                               click again to clear, -1), and the selected point is
//                               outlined
(() => {
  const root = document.createElement("div");
  if (typeof meanDist !== "undefined" && meanDist != null) {
    const nThumbs = points.filter((p) => p.img).length;
    const summary = document.createElement("div");
    summary.style.cssText = "font:13px sans-serif;margin-bottom:6px;color:#444";
    summary.textContent = [
      nThumbs ? `${nThumbs} thumbnails` : `${points.length} points`,
      `grid ${cols}×${rows}`,
      `image aspect ${(+imgAspect).toFixed(2)} (cells ${cellW}×${cellH}px)`,
      `mean displacement ${(+meanDist).toFixed(2)} cell diagonals`,
    ].join(" · ");
    root.append(summary);
  }
  const controls = document.createElement("div");
  controls.style.cssText =
    "display:flex;gap:8px;align-items:center;margin-bottom:6px;font:13px sans-serif";
  const play = document.createElement("button");
  play.textContent = "▶";
  const slider = Object.assign(document.createElement("input"), {
    type: "range", min: 0, max: 1, step: 0.005, value: 0,
  });
  slider.style.width = "300px";
  const label = document.createElement("span");
  const lines = Object.assign(document.createElement("input"), { type: "checkbox" });
  const linesLabel = document.createElement("label");
  linesLabel.append(lines, " displacement lines");
  // how thumbnails fill their cells: cover (crop to fill, like the mosaic)
  // or fit (whole image, letterboxed); neither distorts the image
  const fitMode = document.createElement("select");
  fitMode.append(new Option("cover", "slice"), new Option("fit", "meet"));
  const fitLabel = document.createElement("label");
  fitLabel.append("thumbnails: ", fitMode);
  const resetZoom = document.createElement("button");
  resetZoom.textContent = "⟲ Reset zoom";
  resetZoom.title = "scroll to zoom, drag to pan, double-click to reset";
  // explicit button styling, since host pages (e.g. marimo) may reset
  // button styles so it reads as plain text
  const styleButton = (b) => {
    b.style.cssText =
      "padding:3px 10px;border:1px solid #888;border-radius:4px;" +
      "background:#f3f3f3;color:#222;font:13px sans-serif;cursor:pointer;" +
      "box-shadow:0 1px 1px rgba(0,0,0,0.15)";
    b.onmouseenter = () => (b.style.background = "#e2e2e2");
    b.onmouseleave = () => (b.style.background = "#f3f3f3");
  };
  styleButton(resetZoom);
  // endpoint buttons on either side of the slider: click to animate to
  // that view (handlers set below, with the animation)
  const endButton = (text) => {
    const b = document.createElement("button");
    b.textContent = text;
    b.title = `animate to ${text} view`;
    styleButton(b);
    return b;
  };
  const toUmap = endButton("UMAP");
  const toGrid = endButton("Grid");
  controls.append(play, toUmap, slider, toGrid, label, linesLabel);
  if (points.some((p) => p.img)) controls.append(fitLabel);
  controls.append(resetZoom);
  const holder = document.createElement("div");
  holder.style.cssText = "overflow:hidden;cursor:grab";
  root.append(controls, holder);

  const ease = (t) => 0.5 - 0.5 * Math.cos(Math.PI * t);
  // all coordinates are in mosaic pixels: canvasW x canvasH, cells cellW x cellH
  const xs = d3.range(cols + 1).map((i) => i * cellW);
  const ys = d3.range(rows + 1).map((i) => i * cellH);
  // fit the canvas into the available space, preserving its aspect ratio;
  // recomputed by layout() when entering/leaving fullscreen. With `fill`, the
  // plot takes the whole available area and the canvas is centered in it, so
  // zooming/panning can use the full screen; x0/x1, y0/y1 are the visible
  // bounds in mosaic pixels.
  let k, plotW, plotH, cellPxW, cellPxH, x0, x1, y0, y1;
  function fitTo(availW, availH, fill = false) {
    k = Math.min(availW / canvasW, availH / canvasH);
    plotW = Math.round(fill ? availW : canvasW * k);
    plotH = Math.round(fill ? availH : canvasH * k);
    const padX = (plotW / k - canvasW) / 2, padY = (plotH / k - canvasH) / 2;
    [x0, x1, y0, y1] = [-padX, canvasW + padX, -padY, canvasH + padY];
    cellPxW = cellW * k;
    cellPxH = cellH * k;
  }
  const useImg = points.some((p) => p.img);

  // OPTIONAL click-to-select, synced through the widget model
  const selectable =
    typeof model !== "undefined" && model && typeof model.get("selected") === "number";
  const pointId = (p, i) => (p.id ?? i);
  // a click ends a drag too; only count clicks that didn't move
  let downAt = null;
  // Plot render transform: attach click handlers to the drawn elements
  // (`index` maps each element to its datum)
  const clickable = (index, scales, values, dimensions, context, next) => {
    const g = next(index, scales, values, dimensions, context);
    g.querySelectorAll("image, circle, rect").forEach((el, j) => {
      const i = index[j];
      el.style.cursor = "pointer";
      el.addEventListener("click", (event) => {
        if (downAt && Math.hypot(event.clientX - downAt[0], event.clientY - downAt[1]) > 4) return;
        const id = pointId(points[i], i);
        model.set("selected", model.get("selected") === id ? -1 : id);
        model.save_changes();
      });
    });
    return g;
  };

  // lazy thumbnails: an image is only loaded once its point is on screen
  // (plot scrolled into view, within the zoomed region), and kept after
  // that; until then the point is drawn as a colored placeholder.
  // SVG images don't support loading="lazy", so this is done by hand.
  const loaded = new Set();
  let inView = false;
  // add newly visible points to `loaded`; true if any were added
  function loadVisible(pos) {
    if (!useImg || !inView) return false;
    // visible bounds in mosaic pixels, from the zoom transform, plus a
    // cell of margin so images are ready as they scroll in
    const toX = (px) => x0 + (px - zoomT.x) / zoomT.k / k;
    const toY = (py) => y0 + (py - zoomT.y) / zoomT.k / k;
    const [vx0, vx1] = [toX(0) - cellW, toX(plotW) + cellW];
    const [vy0, vy1] = [toY(0) - cellH, toY(plotH) + cellH];
    let added = false;
    pos.forEach((p, i) => {
      if (!loaded.has(i) && p.img && p.x >= vx0 && p.x <= vx1 && p.y >= vy0 && p.y <= vy1) {
        loaded.add(i);
        added = true;
      }
    });
    return added;
  }

  function draw(t) {
    const e = ease(t);
    const pos = points.map((p) => ({
      ...p,
      x: (1 - e) * p.sx + e * p.dx,
      y: (1 - e) * p.sy + e * p.dy,
    }));
    loadVisible(pos);
    pos.forEach((p, i) => { if (!loaded.has(i)) p.img = null; });
    const size = 0.25 + 0.75 * e;
    const marks = [
      // grid lines limited to the canvas (the plot may extend past it)
      Plot.ruleX(xs, { y1: 0, y2: canvasH, stroke: "#ccc", strokeOpacity: e }),
      Plot.ruleY(ys, { x1: 0, x2: canvasW, stroke: "#ccc", strokeOpacity: e }),
    ];
    if (lines.checked)
      marks.push(Plot.link(points, {
        x1: "sx", y1: "sy", x2: "dx", y2: "dy", stroke: "color", strokeOpacity: 0.4,
      }));
    // placeholders for thumbnails not loaded yet (off screen)
    if (useImg)
      marks.push(Plot.rect(pos, {
        filter: (p) => p.img === null,
        x1: (p) => p.x - (cellW * size) / 2, x2: (p) => p.x + (cellW * size) / 2,
        y1: (p) => p.y - (cellH * size) / 2, y2: (p) => p.y + (cellH * size) / 2,
        fill: "color", fillOpacity: 0.5, title: "label",
        ...(selectable ? { render: clickable } : {}),
      }));
    marks.push(useImg
      ? Plot.image(pos, {
          x: "x", y: "y", src: "img", filter: (p) => p.img !== null,
          width: cellPxW * size, height: cellPxH * size, title: "label",
          preserveAspectRatio: `xMidYMid ${fitMode.value}`,
          ...(selectable ? { render: clickable } : {}),
        })
      : Plot.dot(pos, {
          x: "x", y: "y", fill: "color",
          r: 2 + (Math.min(cellPxW, cellPxH) / 2 - 2) * e, title: "label",
          ...(selectable ? { render: clickable } : {}),
        }));
    if (selectable) {
      // outline the selected point, at the size it's drawn
      const s = model.get("selected");
      // half-size in mosaic pixels: image cell, or dot radius (px) + margin
      const r = (2 + (Math.min(cellPxW, cellPxH) / 2 - 2) * e + 2) / k;
      const hw = useImg ? (cellW * size) / 2 : r;
      const hh = useImg ? (cellH * size) / 2 : r;
      const selPos = pos.filter((p, i) => pointId(p, i) === s);
      marks.push(Plot.rect(selPos, {
        x1: (p) => p.x - hw, x2: (p) => p.x + hw,
        y1: (p) => p.y - hh, y2: (p) => p.y + hh,
        stroke: "#1f77b4", strokeWidth: 3, fill: "none",
      }));
    }
    const plot = Plot.plot({
      width: plotW, height: plotH, margin: 0,
      x: { domain: [x0, x1], axis: null },
      // row 0 at top, like the mosaic
      y: { domain: [y0, y1], reverse: true, axis: null },
      marks,
    });
    applyZoom(plot);
    holder.replaceChildren(plot);
    label.textContent = `t = ${t.toFixed(2)}`;
  }

  // zoom/pan: the plot is redrawn on every animation frame, so keep the
  // current transform and reapply it to each new plot as a CSS transform
  let zoomT = d3.zoomIdentity;
  function applyZoom(plot = holder.firstChild) {
    if (!plot) return;
    plot.style.transformOrigin = "0 0";
    plot.style.transform = `translate(${zoomT.x}px,${zoomT.y}px) scale(${zoomT.k})`;
  }
  const zoom = d3.zoom()
    .scaleExtent([1, 40])
    .on("zoom", (event) => { zoomT = event.transform; applyZoom(); })
    // after zooming/panning, load thumbnails that came into view
    .on("end", () => redrawIfNewlyVisible());
  const sel = d3.select(holder).call(zoom).on("dblclick.zoom", null);
  holder.addEventListener("pointerdown", (event) => { downAt = [event.clientX, event.clientY]; }, true);
  const reset = () => sel.call(zoom.transform, d3.zoomIdentity);
  holder.ondblclick = reset;
  resetZoom.onclick = reset;

  // size the plot for the current space: the given width/height normally, or
  // the whole screen when the cell output is fullscreen (e.g. marimo's
  // fullscreen button). The widget wrapper pins its width, so unpin it too.
  // anywidgets render inside a shadow root, so fs.contains(root) is false
  // even when this plot is the one shown fullscreen; any fullscreen element
  // is treated as ours (if it isn't, this plot isn't visible anyway)
  const fullscreenEl = () =>
    document.fullscreenElement || document.webkitFullscreenElement || null;
  // OPTIONAL `fitWidth`: if true, fill the available width (height follows
  // the canvas aspect, capped to fit in the window) instead of `width`
  const fitWidthMode = typeof fitWidth !== "undefined" && !!fitWidth;
  const wrapperEl = () =>
    root.parentElement && root.parentElement.closest(".observable-plot-container");
  let pinned = null;
  let lastAvailW = null;
  function layout() {
    const fs = fullscreenEl();
    const wrapper = wrapperEl();
    if (fs) {
      if (wrapper && pinned === null) {
        pinned = wrapper.style.width;
        wrapper.style.width = "100%";
      }
      const pad = 16;
      // available height: the screen minus everything else in the fullscreen
      // output (widget controls, notebook controls in the same cell). Measured
      // as content height minus the plot, since marimo may center the content.
      const kids = [...fs.children].map((c) => c.getBoundingClientRect());
      const contentH = kids.length
        ? Math.max(...kids.map((r) => r.bottom)) - Math.min(...kids.map((r) => r.top))
        : holder.offsetHeight;
      const otherH = Math.max(0, contentH - holder.offsetHeight);
      fitTo(fs.clientWidth - 2 * pad, fs.clientHeight - otherH - 2 * pad, true);
    } else if (fitWidthMode && wrapper && wrapper.parentElement) {
      // the widget pins the wrapper to `width`; let it fill the widget instead
      if (pinned === null) pinned = wrapper.style.width;
      wrapper.style.width = "100%";
      const availW = wrapper.parentElement.clientWidth || width;
      lastAvailW = availW;
      const availH = Math.max(200, window.innerHeight - root.offsetHeight + holder.offsetHeight - 40);
      fitTo(availW, Math.min(availW * canvasH / canvasW, availH));
    } else {
      if (wrapper && pinned !== null) {
        wrapper.style.width = pinned;
        pinned = null;
      }
      fitTo(width, height - chrome);
    }
    holder.style.width = `${plotW}px`;
    holder.style.height = `${plotH}px`;
    zoom.extent([[0, 0], [plotW, plotH]]).translateExtent([[0, 0], [plotW, plotH]]);
    zoomT = d3.zoomIdentity;
    sel.property("__zoom", d3.zoomIdentity);
    draw(+slider.value);
  }
  // fullscreen changes resize the element after the event; wait a frame
  // (marimo has no invalidation hook, so drop listeners once the plot is replaced)
  const onFullscreen = () => {
    if (!root.isConnected) return cleanup();
    requestAnimationFrame(layout);
  };
  document.addEventListener("fullscreenchange", onFullscreen);
  document.addEventListener("webkitfullscreenchange", onFullscreen);
  // also refit while fullscreen if the window is resized
  const onResize = () => {
    if (!root.isConnected) return cleanup();
    if (fullscreenEl()) layout();
  };
  window.addEventListener("resize", onResize);
  // in fitWidth mode, refit when the available width changes (window resize,
  // sidebar toggled, ...); the plot isn't attached yet on first run, so this
  // also does the initial fit once it is
  let ro = null;
  if (fitWidthMode && typeof ResizeObserver !== "undefined") {
    let pending = false;
    ro = new ResizeObserver(() => {
      if (pending) return;
      pending = true;
      requestAnimationFrame(() => {
        pending = false;
        if (!root.isConnected) return cleanup();
        if (fullscreenEl()) return;
        const wrapper = wrapperEl();
        const w = wrapper && wrapper.parentElement && wrapper.parentElement.clientWidth;
        if (w && w !== lastAvailW) layout();
      });
    });
    // observe once mounted (root is built detached, then swapped in)
    const attach = () => {
      const wrapper = wrapperEl();
      if (wrapper && wrapper.parentElement) ro.observe(wrapper.parentElement);
      else if (!attach.tries || attach.tries++ < 50) requestAnimationFrame(attach);
    };
    attach.tries = 1;
    requestAnimationFrame(attach);
  }
  const cleanup = () => {
    document.removeEventListener("fullscreenchange", onFullscreen);
    document.removeEventListener("webkitfullscreenchange", onFullscreen);
    window.removeEventListener("resize", onResize);
    if (ro) ro.disconnect();
    if (io) io.disconnect();
    if (selectable) model.off("change:selected", onSelected);
  };

  let raf = null, dir = 1;
  function step() {
    let t = +slider.value + dir * 0.008;
    if (t >= 1) { t = 1; stop(); dir = -1; }
    if (t <= 0) { t = 0; stop(); dir = 1; }
    slider.value = t;
    draw(t);
    if (raf) raf = requestAnimationFrame(step);
  }
  function stop() {
    cancelAnimationFrame(raf);
    raf = null;
    play.textContent = "▶";
  }
  play.onclick = () => {
    if (raf) return stop();
    play.textContent = "⏸";
    raf = requestAnimationFrame(step);
  };
  // animate toward one end (0 = UMAP, 1 = Grid) from the current position
  const animateTo = (end) => {
    stop();
    if (+slider.value === end) return;
    dir = end ? 1 : -1;
    play.textContent = "⏸";
    raf = requestAnimationFrame(step);
  };
  toUmap.onclick = () => animateTo(0);
  toGrid.onclick = () => animateTo(1);
  slider.oninput = () => { stop(); draw(+slider.value); };
  lines.onchange = () => draw(+slider.value);
  fitMode.onchange = () => draw(+slider.value);
  // redraw when the selection changes (from a click here or from Python)
  const onSelected = () => {
    if (!root.isConnected) return;
    if (!raf) draw(+slider.value);
  };
  if (selectable) model.on("change:selected", onSelected);
  // redraw (when not animating; animation redraws every frame anyway) if
  // more thumbnails are now visible
  function redrawIfNewlyVisible() {
    if (raf || !root.isConnected) return;
    const e = ease(+slider.value);
    const pos = points.map((p) => ({
      ...p, x: (1 - e) * p.sx + e * p.dx, y: (1 - e) * p.sy + e * p.dy,
    }));
    if (loadVisible(pos)) draw(+slider.value);
  }
  // only load thumbnails once the plot is scrolled into view
  let io = null;
  if (useImg && typeof IntersectionObserver !== "undefined") {
    io = new IntersectionObserver((entries) => {
      if (!entries.some((en) => en.isIntersecting) || inView) return;
      inView = true;
      io.disconnect();
      redrawIfNewlyVisible();
    }, { rootMargin: "200px" });
    io.observe(holder);
  } else {
    inView = true;
  }
  // in Observable, stop animating when the cell re-runs (e.g. on resize)
  if (typeof invalidation !== "undefined") invalidation.then(() => { stop(); cleanup(); });
  layout();
  return root;
})()
