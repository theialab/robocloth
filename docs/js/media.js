/**
 * Video playback on the landing page.
 *
 *  - #launch-video (top of the page) autoplays muted + looping. The "Sound on"
 *    pill unmutes it and restarts it from the beginning so the voice-over is
 *    heard in full; the pill hides while sound is on. It is paused when it is
 *    scrolled out of view only while still muted (a visitor listening to the
 *    voice-over can keep reading).
 *  - video[data-play-in-view] are preload="metadata" (nothing big is fetched
 *    up front); they start muted when >= 25 % visible and pause when they
 *    leave the viewport.
 *
 * A video the visitor paused by hand is never restarted automatically.
 */
(function () {
    function tryPlay(v) {
        // once a video actually starts, let the browser buffer it whole: scrubbing then never
        // waits on a range request (the files are 3-8 MB)
        if (v.preload !== "auto") v.preload = "auto";
        var p = v.play();
        if (p && p.catch) p.catch(function () { /* blocked (e.g. low-power mode) — controls remain */ });
    }

    function trackUserPause(v) {
        v._autoPausing = false;
        v._userPaused = false;
        v.addEventListener("pause", function () {
            if (!v._autoPausing) v._userPaused = true;
            v._autoPausing = false;
        });
        v.addEventListener("play", function () { v._userPaused = false; });
    }

    function autoPause(v) {
        if (!v.paused) { v._autoPausing = true; v.pause(); }
    }

    function playInView(v, mayPause) {
        trackUserPause(v);
        if (!("IntersectionObserver" in window)) { tryPlay(v); return; }
        new IntersectionObserver(function (entries) {
            entries.forEach(function (e) {
                if (e.isIntersecting) {
                    if (!v._userPaused) tryPlay(v);
                } else if (!mayPause || mayPause()) {
                    autoPause(v);
                }
            });
        }, { threshold: 0.25 }).observe(v);
    }

    // Launch video + sound button.
    // Voiced videos autoplay muted; a "Sound on" pill restarts them with sound. Captions
    // (text tracks) show only while muted -- the method video has its captions burned in.
    function soundControl(video, btn, mayPause) {
        if (!video) return;
        video.muted = true;
        tryPlay(video);
        playInView(video, mayPause || function () { return video.muted; });
        if (!btn) return;
        var captions = function () {
            var tt = video.textTracks;
            for (var i = 0; tt && i < tt.length; i++) tt[i].mode = video.muted ? "showing" : "hidden";
        };
        var sync = function () { btn.hidden = !video.muted; captions(); };
        btn.addEventListener("click", function () {
            video.muted = false;
            video.volume = 1;
            try { video.currentTime = 0; } catch (e) { /* not seekable yet */ }
            tryPlay(video);
        });
        video.addEventListener("volumechange", sync);
        sync();
    }
    soundControl(document.getElementById("launch-video"), document.getElementById("launch-sound"));
    var methodVideo = document.getElementById("method-video");
    soundControl(methodVideo, document.getElementById("method-sound"), null);

    // Lazy videos further down.
    document.querySelectorAll("video[data-play-in-view]").forEach(function (v) {
        if (v === methodVideo) return;             // handled above (sound button)
        v.muted = true;
        playInView(v, null);
    });
})();
