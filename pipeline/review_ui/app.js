(() => {
  const STAMP_LABEL = {
    correct: "APPROVED",
    rejected: "REJECTED",
    edited: "AMENDED",
  };

  let pool = [];
  let index = 0;
  let editing = false;
  let justFlagged = false; // true right after this session sets a status, to trigger the stamp animation

  const el = {
    image: document.getElementById("object-image"),
    captionText: document.getElementById("caption-text"),
    captionInput: document.getElementById("caption-input"),
    hitemText: document.getElementById("hitem-text"),
    hitemInput: document.getElementById("hitem-input"),
    meta: document.getElementById("card-meta"),
    stamp: document.getElementById("stamp"),
    progressCount: document.getElementById("progress-count"),
    progressFill: document.getElementById("progress-fill"),
    btnBack: document.getElementById("btn-back"),
    btnForward: document.getElementById("btn-forward"),
    btnReject: document.getElementById("btn-reject"),
    btnEdit: document.getElementById("btn-edit"),
    btnSave: document.getElementById("btn-save"),
    btnCancel: document.getElementById("btn-cancel"),
  };

  async function loadPool() {
    const res = await fetch("/api/pool");
    pool = await res.json();
    const firstUnreviewed = pool.findIndex((e) => !e.review_status);
    index = firstUnreviewed === -1 ? pool.length - 1 : firstUnreviewed;
    render();
  }

  async function saveEntry(entry, fields) {
    const res = await fetch("/api/pool", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ image_id: entry.image_id, ...fields }),
    });
    if (!res.ok) {
      const err = await res.json().catch(() => ({}));
      alert(`Save failed: ${err.error || res.status}`);
      return false;
    }
    Object.assign(entry, fields);
    return true;
  }

  function render() {
    const entry = pool[index];

    el.image.src = "/" + entry.local_path;
    el.image.alt = entry.h_item_en || entry.subject || "";

    el.captionText.textContent = entry.caption_ar || "";
    el.hitemText.textContent = entry.h_item_ar || "";
    el.captionInput.value = entry.caption_ar || "";
    el.hitemInput.value = entry.h_item_ar || "";

    el.meta.textContent = [entry.category, entry.country, entry.medium]
      .filter(Boolean)
      .join(" · ");

    renderStamp(entry.review_status);

    el.progressCount.textContent = `${index + 1} / ${pool.length}`;
    el.progressFill.style.width = `${((index + 1) / pool.length) * 100}%`;

    el.btnBack.disabled = index === 0;
    el.btnForward.disabled = index === pool.length - 1;

    setEditing(false);
    justFlagged = false;
  }

  function renderStamp(status) {
    el.stamp.classList.remove(
      "visible", "status-correct", "status-rejected", "status-edited", "stamp-press"
    );
    if (!status) {
      el.stamp.textContent = "";
      return;
    }

    el.stamp.textContent = STAMP_LABEL[status] || status;
    el.stamp.classList.add("visible", `status-${status}`);

    if (justFlagged) {
      // restart the animation even if the same class was already present
      void el.stamp.offsetWidth;
      el.stamp.classList.add("stamp-press");
    }
  }

  function setEditing(on) {
    editing = on;
    el.captionText.hidden = on;
    el.hitemText.hidden = on;
    el.captionInput.hidden = !on;
    el.hitemInput.hidden = !on;
    el.btnEdit.hidden = on;
    el.btnReject.hidden = on;
    el.btnSave.hidden = !on;
    el.btnCancel.hidden = !on;
    if (on) el.captionInput.focus();
  }

  function goBack() {
    if (editing || index === 0) return;
    index -= 1;
    render();
  }

  async function goForward() {
    if (editing || index === pool.length - 1) return;
    const entry = pool[index];
    if (!entry.review_status) {
      justFlagged = true;
      const ok = await saveEntry(entry, { review_status: "correct" });
      if (!ok) { justFlagged = false; return; }
    }
    index += 1;
    render();
  }

  async function reject() {
    if (editing) return;
    const entry = pool[index];
    justFlagged = true;
    const ok = await saveEntry(entry, { review_status: "rejected" });
    if (!ok) { justFlagged = false; return; }
    if (index < pool.length - 1) index += 1;
    render();
  }

  async function saveEdit() {
    const entry = pool[index];
    justFlagged = true;
    const ok = await saveEntry(entry, {
      review_status: "edited",
      caption_ar: el.captionInput.value,
      h_item_ar: el.hitemInput.value,
    });
    if (!ok) { justFlagged = false; return; }
    if (index < pool.length - 1) index += 1;
    render();
  }

  el.btnBack.addEventListener("click", goBack);
  el.btnForward.addEventListener("click", goForward);
  el.btnReject.addEventListener("click", reject);
  el.btnEdit.addEventListener("click", () => setEditing(true));
  el.btnCancel.addEventListener("click", () => render());
  el.btnSave.addEventListener("click", saveEdit);

  document.addEventListener("keydown", (e) => {
    if (editing) return; // let typing in the textareas behave normally
    if (e.key === "ArrowLeft") { e.preventDefault(); goBack(); }
    if (e.key === "ArrowRight") { e.preventDefault(); goForward(); }
  });

  loadPool();
})();
