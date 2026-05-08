document.addEventListener('click', function (e) {
    const btn = e.target.closest('[data-target]');
    if (!btn) return;
    // If this is a tab button, ignore here — tabs have their own click handler.
    if (btn.classList && btn.classList.contains('fg-tab')) return;
    const id = btn.getAttribute('data-target');
    const el = document.getElementById(id);
    if (!el) return;
    if (el.style.display === 'none') {
        el.style.display = 'block';
    } else {
        el.style.display = 'none';
    }
});

// hide all trade lists by default for long logs
document.addEventListener('DOMContentLoaded', function () {
    document.querySelectorAll('.fg-collapsible').forEach(function (el) {
        el.style.display = 'block';
    });

    // Tabs: symbol panels only. Holdings remains visible.
    const panels = document.querySelectorAll('[id^="tab-symbol-"]');
    const tabs = document.querySelectorAll('.fg-tab');
    function showSymbolPanel(id) {
        // show only the requested symbol panel
        panels.forEach(p => {
            p.style.display = (p.id === id) ? 'block' : 'none';
        });
    }

    // default: hide all symbol panels
    panels.forEach(p => p.style.display = 'none');

    tabs.forEach(function (t) {
        t.addEventListener('click', function (ev) {
            // ensure button doesn't try to submit if placed in a form
            ev.preventDefault();
            const target = t.getAttribute('data-target');
            if (target) {
                showSymbolPanel(target);
                // persist active symbol in the URL as ?symbol=<name>
                try {
                    const params = new URLSearchParams(window.location.search);
                    const name = (t.textContent || '').trim();
                    if (name) {
                        params.set('symbol', name);
                    } else {
                        params.delete('symbol');
                    }
                    const newUrl = window.location.pathname + (params.toString() ? ('?' + params.toString()) : '') + window.location.hash;
                    history.replaceState(null, '', newUrl);
                } catch (e) {
                    // ignore if URLSearchParams isn't supported
                }
            }
        });
    });

    // Trade filtering handlers (per-symbol controls)
    function applyFilterForIndex(idx) {
        const agentSelect = document.getElementById('trade-agent-select-' + idx);
        const textInput = document.getElementById('trade-filter-text-' + idx);
        const sideInputs = document.getElementsByName('trade-side-' + idx);
        const agent = agentSelect ? agentSelect.value : '';
        const text = textInput ? textInput.value.trim().toLowerCase() : '';
        let side = 'all';
        for (const si of sideInputs) { if (si.checked) { side = si.value; break; } }

        const table = document.querySelector('.trades-table[data-index="' + idx + '"]');
        if (!table) return;
        const rows = table.querySelectorAll('tbody tr');
        rows.forEach(r => {
            const buyer = (r.getAttribute('data-buyer') || '').toString();
            const seller = (r.getAttribute('data-seller') || '').toString();
            const price = (r.getAttribute('data-price') || '').toString();
            let visible = true;

            // agent filter
            if (agent) {
                if (side === 'buy' && buyer !== agent) visible = false;
                else if (side === 'sell' && seller !== agent) visible = false;
                else if (side === 'all' && buyer !== agent && seller !== agent) visible = false;
            }

            // text filter across buyer/seller/price
            if (visible && text) {
                const hay = (buyer + ' ' + seller + ' ' + price + ' ' + r.textContent).toLowerCase();
                if (hay.indexOf(text) === -1) visible = false;
            }

            r.style.display = visible ? '' : 'none';
        });
    }

    // wire apply/clear buttons
    document.querySelectorAll('.trade-filter-apply').forEach(btn => {
        btn.addEventListener('click', e => {
            const idx = btn.getAttribute('data-index');
            applyFilterForIndex(idx);
        });
    });
    document.querySelectorAll('.trade-filter-clear').forEach(btn => {
        btn.addEventListener('click', e => {
            const idx = btn.getAttribute('data-index');
            const agentSelect = document.getElementById('trade-agent-select-' + idx);
            const textInput = document.getElementById('trade-filter-text-' + idx);
            const sideInputs = document.getElementsByName('trade-side-' + idx);
            if (agentSelect) agentSelect.value = '';
            if (textInput) textInput.value = '';
            for (const si of sideInputs) si.checked = si.value === 'all';
            applyFilterForIndex(idx);
        });
    });

    // Navigation handlers for snapshot stepping (first / prev / next / last)
    // Navigate by parsing the numeric counter in the current filename and incrementing/decrementing it.
    function parseMaxStep() {
        const params = new URLSearchParams(window.location.search);
        const s = params.get('maxstep');
        if (s) {
            const n = parseInt(s, 10);
            if (!Number.isNaN(n) && n >= 0) return n;
        }
        const FG = window.FG_NAV || null;
        if (FG && typeof FG.total === 'number' && FG.total > 0) return FG.total - 1;
        return null;
    }

    function parseMinStep() {
        const params = new URLSearchParams(window.location.search);
        const s = params.get('minstep');
        if (s) {
            const n = parseInt(s, 10);
            if (!Number.isNaN(n) && n >= 0) return n;
        }
        // default to 0
        return 0;
    }

    function navigateByDelta(delta) {
        const curRaw = window.location.pathname.split('/').pop() || '';
        const m = curRaw.match(/(\d+)(?=\.html$)/);
        if (!m) {
            alert('No numeric snapshot index found in filename (expected e.g. 00003.html)');
            return;
        }
        const width = m[1].length;
        const curNum = parseInt(m[1], 10);
        const minStep = parseMinStep();
        const maxStep = parseMaxStep();
        const target = curNum + delta;
        if (target < minStep) {
            showToast('At first snapshot');
            return;
        }
        if (maxStep !== null && target > maxStep) {
            showToast('At last snapshot');
            return;
        }
        const newName = String(target).padStart(width, '0') + '.html';
        // replace the numeric group in the filename with the new one
        const newUrl = window.location.href.replace(m[1] + '.html', newName);
        window.location.href = newUrl;
    }

    function navigateToIndex(index) {
        const curRaw = window.location.pathname.split('/').pop() || '';
        const m = curRaw.match(/(\d+)(?=\.html$)/);
        if (!m) {
            alert('No numeric snapshot index found in filename (expected e.g. 00003.html)');
            return;
        }
        const width = m[1].length;
        const minStep = parseMinStep();
        const maxStep = parseMaxStep();
        if (index < minStep) {
            showToast('At first snapshot');
            return;
        }
        if (maxStep !== null && index > maxStep) {
            showToast('At last snapshot');
            return;
        }
        const newName = String(index).padStart(width, '0') + '.html';
        const newUrl = window.location.href.replace(m[1] + '.html', newName);
        window.location.href = newUrl;
    }

    function showToast(msg, duration = 1800) {
        const el = document.getElementById('fg-toast');
        if (!el) return;
        el.textContent = msg;
        el.classList.add('show');
        clearTimeout(el._hideTimer);
        el._hideTimer = setTimeout(() => {
            el.classList.remove('show');
        }, duration);
    }

    document.querySelectorAll('.fg-nav .fg-nav-btn').forEach(btn => {
        btn.addEventListener('click', function (ev) {
            // buttons won't navigate by default but still prevent any default actions
            ev.preventDefault();
            if (btn.classList.contains('disabled')) return;
            const action = (btn.getAttribute('data-action') || '').toLowerCase();
            const target = btn.getAttribute('data-target') || '';

            // If a direct filename was provided at render time, prefer that for first/last
            // if (target) {
            //     if (action === 'first' || action === 'last') {
            //         window.location.href = target;
            //         return;
            //     }
            // }

            if (action === 'prev') {
                navigateByDelta(-1);
            } else if (action === 'next') {
                navigateByDelta(1);
            } else if (action === 'first') {
                const min = parseMinStep();
                navigateToIndex(min);
            } else if (action === 'last') {
                const max = parseMaxStep();
                if (max !== null) {
                    navigateToIndex(max);
                }
            }
        });

        // Adjust nav button enabled/disabled state based on current index and maxstep (URL param or FG_NAV)
        function getCurrentIndex() {
            const FG = window.FG_NAV || null;
            if (FG && typeof FG.index === 'number') return FG.index;
            const curRaw = window.location.pathname.split('/').pop() || '';
            const m = curRaw.match(/(\d+)(?=\.html$)/);
            if (!m) return null;
            return parseInt(m[1], 10);
        }

        (function adjustNavButtonStates() {
            const cur = getCurrentIndex();
            const max = parseMaxStep();
            const btnPrev = document.querySelector('.fg-nav .fg-nav-btn[data-action="prev"]');
            const btnNext = document.querySelector('.fg-nav .fg-nav-btn[data-action="next"]');
            const btnFirst = document.querySelector('.fg-nav .fg-nav-btn[data-action="first"]');
            const btnLast = document.querySelector('.fg-nav .fg-nav-btn[data-action="last"]');
            const stepLabel = document.getElementById('fg-step-label');

            if (cur === null) {
                // If we can't determine the index, show unknowns and disable prev/next
                if (stepLabel) stepLabel.textContent = 'Step ? of ?';
                if (btnPrev) btnPrev.classList.add('disabled');
                if (btnFirst) btnFirst.classList.add('disabled');
                if (btnNext) btnNext.classList.add('disabled');
                if (btnLast) btnLast.classList.add('disabled');
                return;
            }

            const displayTotal = (max !== null) ? (max) : '?';
            if (stepLabel) stepLabel.textContent = 'Step ' + String(cur) + ' of ' + displayTotal;

            // Prev/First enabled only if current > 0
            if (btnPrev) {
                if (cur > 0) btnPrev.classList.remove('disabled'); else btnPrev.classList.add('disabled');
            }
            if (btnFirst) {
                if (cur > 0) btnFirst.classList.remove('disabled'); else btnFirst.classList.add('disabled');
            }

            // Next/Last enabled only if max is known and current < max
            if (btnNext) {
                if (max !== null && cur < max) btnNext.classList.remove('disabled'); else btnNext.classList.add('disabled');
            }
            if (btnLast) {
                if (max !== null && cur < max) btnLast.classList.remove('disabled'); else btnLast.classList.add('disabled');
            }
        })();

        // Restore active symbol from URL param (symbol=<name> or symbol_index=<n>)
        (function restoreActiveSymbol() {
            const params = new URLSearchParams(window.location.search);
            const symParam = params.get('symbol');
            const symIndexParam = params.get('symbol_index');
            if (symParam) {
                const wanted = symParam.trim();
                // find a tab whose label matches (case-sensitive exact)
                for (const tb of tabs) {
                    if ((tb.textContent || '').trim() === wanted) {
                        const target = tb.getAttribute('data-target');
                        if (target) { showSymbolPanel(target); return; }
                    }
                }
            }
            if (symIndexParam) {
                const idx = parseInt(symIndexParam, 10);
                if (!Number.isNaN(idx) && idx >= 0 && idx < panels.length) {
                    const id = 'tab-symbol-' + idx;
                    showSymbolPanel(id);
                    return;
                }
            }
            // no symbol requested: keep all panels hidden (default)
        })();
    });

    // Keyboard navigation: Left/Right for prev/next, Home/End for first/last
    document.addEventListener('keydown', function (ev) {
        // Ignore if user is typing in an input, textarea, or contentEditable
        const active = document.activeElement;
        const tag = active && active.tagName;
        if (tag === 'INPUT' || tag === 'TEXTAREA' || (active && active.isContentEditable)) return;
        // Ignore when modifier keys are pressed to avoid conflicts
        if (ev.ctrlKey || ev.altKey || ev.metaKey) return;

        if (ev.key === 'ArrowLeft') {
            ev.preventDefault();
            navigateByDelta(-1);
        } else if (ev.key === 'ArrowRight') {
            ev.preventDefault();
            navigateByDelta(1);
        } else if (ev.key === 'Home') {
            ev.preventDefault();
            const min = parseMinStep();
            navigateToIndex(min);
        } else if (ev.key === 'End') {
            ev.preventDefault();
            const max = parseMaxStep();
            if (max !== null) navigateToIndex(max);
        }
    });
});
