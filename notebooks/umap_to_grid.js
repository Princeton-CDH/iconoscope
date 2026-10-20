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
  resetZoom.textContent = "reset zoom";
  resetZoom.title = "scroll to zoom, drag to pan, double-click to reset";
  // endpoint labels on either side of the slider
  const endLabel = (text) => {
    const s = document.createElement("span");
    s.textContent = text;
    s.style.color = "#666";
    return s;
  };
  controls.append(play, endLabel("UMAP"), slider, endLabel("Grid"), label, linesLabel);
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

  function draw(t) {
    const e = ease(t);
    const pos = points.map((p) => ({
      ...p,
      x: (1 - e) * p.sx + e * p.dx,
      y: (1 - e) * p.sy + e * p.dy,
    }));
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
    marks.push(useImg
      ? Plot.image(pos, {
          x: "x", y: "y", src: "img",
          width: cellPxW * size, height: cellPxH * size, title: "label",
          preserveAspectRatio: `xMidYMid ${fitMode.value}`,
        })
      : Plot.dot(pos, {
          x: "x", y: "y", fill: "color",
          r: 2 + (Math.min(cellPxW, cellPxH) / 2 - 2) * e, title: "label",
        }));
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
    .on("zoom", (event) => { zoomT = event.transform; applyZoom(); });
  const sel = d3.select(holder).call(zoom).on("dblclick.zoom", null);
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
  slider.oninput = () => { stop(); draw(+slider.value); };
  lines.onchange = () => draw(+slider.value);
  fitMode.onchange = () => draw(+slider.value);
  // in Observable, stop animating when the cell re-runs (e.g. on resize)
  if (typeof invalidation !== "undefined") invalidation.then(() => { stop(); cleanup(); });
  layout();
  return root;
})()
